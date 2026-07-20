"""
refresh_dashboard.py
=====================================================================
Pulls the latest NPS survey data from Metabase's REST API, runs the
sentiment + two-level VOC classification, and regenerates the dashboard
HTML file in place (auth gate and design untouched).

Run this from GitHub Actions (see .github/workflows/refresh-nps-dashboard.yml)
or locally for testing.

Required environment variables (set as GitHub Secrets in Actions):
  METABASE_URL       e.g. https://your-company.metabaseapp.com
  METABASE_USERNAME  Metabase login email
  METABASE_PASSWORD  Metabase login password
  METABASE_DB_ID     Database id for the Databricks connection (integer)

These are Metabase credentials only. No GitHub token is needed here —
the Action commits back to its own repo using GitHub's built-in
GITHUB_TOKEN, which this script never touches.
"""

import os
import json
from datetime import datetime, timezone

import requests

METABASE_URL = os.environ["METABASE_URL"].rstrip("/")
METABASE_USERNAME = os.environ["METABASE_USERNAME"]
METABASE_PASSWORD = os.environ["METABASE_PASSWORD"]
METABASE_DB_ID = int(os.environ["METABASE_DB_ID"])

TEMPLATE_PATH = "dashboard_template.html"
OUTPUT_PATH = "nps_voc_dashboard.html"


# =============================================================================
# 1. METABASE AUTH + QUERY HELPERS
# =============================================================================

def get_session_token():
    resp = requests.post(
        f"{METABASE_URL}/api/session",
        json={"username": METABASE_USERNAME, "password": METABASE_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def run_query(session_token, sql):
    resp = requests.post(
        f"{METABASE_URL}/api/dataset",
        headers={"X-Metabase-Session": session_token},
        json={
            "type": "native",
            "native": {"query": sql},
            "database": METABASE_DB_ID,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Metabase query error: {data['error']}")
    cols = [c["name"] for c in data["data"]["cols"]]
    rows = data["data"]["rows"]
    return [dict(zip(cols, row)) for row in rows]


# =============================================================================
# 2. THE THREE EXTRACTION QUERIES (same as nps_sentiment_extraction.sql)
# =============================================================================

DAILY_NPS_SQL = """
WITH parsed AS (
  SELECT id, survey_status, created_at,
         from_json(response, 'array<struct<ans:array<string>,text:string>>') as items
  FROM pop.cx_nps_user_response
  WHERE response IS NOT NULL AND trim(response) != '' AND response != '[]'
),
score_rows AS (
  SELECT id, survey_status, created_at, CAST(item.ans[0] AS INT) as score
  FROM parsed LATERAL VIEW explode(items) t AS item
  WHERE item.text = 'score'
)
SELECT date(created_at) as day,
       count(*) as responses,
       round(avg(score), 2) as avg_score,
       sum(case when score >= 9 then 1 else 0 end) as promoters,
       sum(case when score >= 7 and score <= 8 then 1 else 0 end) as passives,
       sum(case when score <= 6 then 1 else 0 end) as detractors
FROM score_rows
GROUP BY date(created_at)
ORDER BY day
"""

THEME_TAGS_SQL = """
WITH parsed AS (
  SELECT id, survey_status, created_at,
         from_json(response, 'array<struct<ans:array<string>,text:string>>') as items
  FROM pop.cx_nps_user_response
  WHERE response IS NOT NULL AND trim(response) != '' AND response != '[]'
),
qrows AS (
  SELECT id, survey_status, date(created_at) as day, item.text as qtext, item.ans as ans
  FROM parsed LATERAL VIEW explode(items) t AS item
  WHERE item.text != 'score'
    AND item.text != 'What is the single most important thing POP Club UPI could do to improve your experience?'
),
tags AS (
  SELECT id, day, qtext, trim(tag) as tag
  FROM qrows LATERAL VIEW explode(ans) t2 AS tag
)
SELECT day,
       CASE qtext
         WHEN 'What did you love about POP?' THEN 'love'
         WHEN 'What made you feel this way about POP UPI?' THEN 'passive'
         WHEN 'What did not go well with your experience on POP?' THEN 'detract'
       END as q,
       tag,
       count(*) as cnt
FROM tags
GROUP BY day, qtext, tag
ORDER BY day, qtext, cnt DESC
"""

FREETEXT_SQL = """
WITH parsed AS (
  SELECT id, survey_status, created_at,
         from_json(response, 'array<struct<ans:array<string>,text:string>>') as items
  FROM pop.cx_nps_user_response
  WHERE response IS NOT NULL AND trim(response) != '' AND response != '[]'
),
scores AS (
  SELECT id, CAST(item.ans[0] AS INT) as score
  FROM parsed LATERAL VIEW explode(items) t AS item
  WHERE item.text = 'score'
),
freetext AS (
  SELECT p.id, date(p.created_at) as day, p.survey_status, item.ans[0] as txt
  FROM parsed p LATERAL VIEW explode(items) t AS item
  WHERE item.text = 'What is the single most important thing POP Club UPI could do to improve your experience?'
)
SELECT f.day, f.survey_status, s.score, f.txt
FROM freetext f LEFT JOIN scores s ON f.id = s.id
WHERE f.txt IS NOT NULL AND trim(f.txt) != ''
ORDER BY f.day
"""


# =============================================================================
# 3. CLASSIFICATION (mirrors nps_voc_classifier.py — kept inline so this
#    script has no local-file dependency beyond the template)
# =============================================================================

JUNK_EXACT = {
    "test", "testing", "tedt", "tetst", "check", "ghi", "b", "bhhh", "v gg fxu",
    "gcy", "jvjf", "gsusb", "hzjziakus zip", "bsjissg", "hi kopal", "na", "i'm",
    "ok", "ovifufuco", ".. ok", "ui,", ",",
}

def is_junk(text):
    low = text.lower().strip()
    if len(low) == 0:
        return True
    if "gibberish repeated text" in low:
        return True
    if "lorem ipsum" in low:
        return True
    if low in JUNK_EXACT:
        return True
    return False

POS_WORDS = [
    "great", "good", "amazing", "excellent", "awesome", "nice", "love", "best",
    "super", "wonderful", "happy", "perfect", "easy", "worthy", "cool", "fast",
    "fasta", "trustworthy", "wowww", "very good", "all good", "superb",
    "outstanding", "satisfied", "recommend",
]
NEG_WORDS = [
    "worst", "bad", "scam", "fake", "not working", "shit", "poor", "unfair",
    "deceived", "disgusting", "disappearing", "not available", "cancel",
    "damaged", "failed", "slow", "worse", "issue", "problem", "buggy",
    "not usable", "not withdrawable", "valueless", "not credited",
    "not received", "not receive", "not able", "unable", "hang", "lag",
    "robotic", "unhappy", "removed - inappropriate",
]

def classify_sentiment(text, score):
    low = text.lower()
    pos_hit = any(w in low for w in POS_WORDS)
    neg_hit = any(w in low for w in NEG_WORDS)
    if neg_hit and not pos_hit:
        return "Negative"
    if pos_hit and not neg_hit:
        return "Positive"
    if pos_hit and neg_hit:
        return "Mixed"
    if score is None:
        return "Neutral"
    if score >= 9:
        return "Positive"
    if score <= 6:
        return "Negative"
    return "Neutral"

RULES = [
    ("Credit Card", "Application / Approval", [
        "credit card application", "applied for a credit card", "under review",
        "reject me for credit card", "not getting pop credit card",
        "approving the credit cards", "credit card application is rejected",
    ]),
    ("Credit Card", "Bill Payment", [
        "credit card bill", "card bill payment", "card repayment",
        "credit card payment not working", "card bill not showing",
        "advans bill", "mark credit card bill as paid",
    ]),
    ("Credit Card", "Rewards on Card", [
        "credit card rewards", "card rewards", "opo credit card from tnx",
        "rupay credit card",
    ]),
    ("Credit Card", "Card Features / Setup", [
        "card number", "emi", "debit card option was removed",
        "card to be saved", "different phone number",
    ]),
    ("Credit Card", "General", ["credit card"]),
    ("Recharges & Bill Payments", "Missing Bill Categories", [
        "electricity", "dth", "broadband", "postpaid", "d2h", "utility",
        "recharge", "lpg", "gas booking",
    ]),
    ("Recharges & Bill Payments", "Bill Payment Failures", [
        "bill payment", "postpaid bill is not able", "bill not showing",
    ]),
    ("UPI", "Payment Failures", [
        "payment failed", "payment is pending", "transaction failed",
        "declined by bank", "debited", "money debited", "did not redirected",
        "not redirected", "withdrawal is not possible",
    ]),
    ("UPI", "Speed / Performance", [
        "upi is slow", "slow to open", "app loading", "opening very slow",
        "app hanging", "takes more time to connect", "faster app browsing",
        "internet connection could be better",
    ]),
    ("UPI", "Scanner / QR", [
        "scan", "qr", "scanner", "camera has bug", "barcode", "bar code",
    ]),
    ("UPI", "Bank Linking / Registration", [
        "upi pin", "set upi pin", "upi registration", "linking", "anacity",
        "fingerprint configuration",
    ]),
    ("UPI", "General", ["upi", "gpay", "google pay", "paytm"]),
    ("Shop", "Delivery / Shipping", [
        "deliver", "shipping", "shipment", "dilevery", "pick up", "collected",
    ]),
    ("Shop", "Order Cancellations / Refunds", [
        "order was cancelled", "order cancelled", "refund",
        "return the product", "not pick up", "exchange a wrong bought product",
        "return policy",
    ]),
    ("Shop", "Product Quality", [
        "damaged", "quality", "fake brand", "poor quality", "broken seal",
    ]),
    ("Shop", "Catalog / Brand Variety", [
        "add more brands", "more brands", "brand catalogue", "not relevant",
        "variety", "involve more brands", "add brand", "expand brand",
    ]),
    ("Shop", "Pricing / Discounts", [
        "prices are almost the same", "products rates",
        "expensive than official website", "discount", "50%off", "50% off",
    ]),
    ("Shop", "General", ["pop shop", "shop section", "order ", "product "]),
    ("Others", "Rewards / Pop Coins Program", [
        "reward", "cashback", "cash back", "pop coin", "popcoin", "coin",
        "coupon", "voucher", "redeem",
    ]),
    ("Others", "Customer Support", [
        "customer support", "customer care", "care team", "support team",
        "chat bot", "chat box", "resolution", "raise a ticket",
        "raised an issue", "executive", "no proper help", "no reply",
    ]),
    ("Others", "App UI / Navigation", [
        "navigat", "interface", "screen", "layout", "clutter", "ui should be",
        "hard to use", "easy to use", "easy to navigate", "search option",
        "dark-only theme", "ux",
    ]),
    ("Others", "App Reliability / Performance", [
        "buggy", "bug", "crash", "not working", "not reliable", "glitch",
        "freeze", "hang", "laggy", "lags", "app is very laggy",
    ]),
    ("Others", "Referral Program", ["refer", "referral"]),
    ("Others", "General Praise / Feedback", [
        "great", "good", "amazing", "excellent", "awesome", "nice", "love",
        "best", "super", "wonderful", "happy", "perfect", "worthy", "superb",
    ]),
]

def classify_theme(text, junk):
    if junk:
        return [{"l1": "Others", "l2": "Junk / Not meaningful"}]
    low = text.lower()
    hits, seen = [], set()
    for l1, l2, keywords in RULES:
        if any(kw in low for kw in keywords):
            key = (l1, l2)
            if key not in seen:
                hits.append({"l1": l1, "l2": l2})
                seen.add(key)
    l1_with_specific = {h["l1"] for h in hits if h["l2"] != "General"}
    hits = [h for h in hits if not (h["l2"] == "General" and h["l1"] in l1_with_specific)]
    if not hits:
        hits = [{"l1": "Others", "l2": "General / Uncategorized"}]
    return hits

def classify_comment(day, score, text):
    text = text.strip()
    junk = is_junk(text)
    return {
        "day": day,
        "score": score,
        "text": text,
        "junk": junk,
        "sentiment": "Neutral" if junk else classify_sentiment(text, score),
        "voc": classify_theme(text, junk),
    }


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def normalize_tag(tag):
    return tag.replace("relaible", "reliable")

def to_day_str(v):
    # Metabase returns dates as ISO strings like "2026-07-20T00:00:00Z"
    return str(v)[:10]

def main():
    print("Authenticating with Metabase...")
    token = get_session_token()

    print("Fetching daily NPS aggregates...")
    daily_raw = run_query(token, DAILY_NPS_SQL)
    daily = [{
        "day": to_day_str(r["day"]),
        "responses": r["responses"],
        "avg_score": r["avg_score"],
        "promoters": r["promoters"],
        "passives": r["passives"],
        "detractors": r["detractors"],
    } for r in daily_raw]

    print("Fetching checkbox reason tallies...")
    theme_raw = run_query(token, THEME_TAGS_SQL)
    theme_tags = [{
        "day": to_day_str(r["day"]),
        "q": r["q"],
        "tag": normalize_tag(r["tag"]),
        "cnt": r["cnt"],
    } for r in theme_raw]

    print("Fetching open-text comments...")
    freetext_raw = run_query(token, FREETEXT_SQL)
    freetext = [
        classify_comment(to_day_str(r["day"]), r["score"], r["txt"])
        for r in freetext_raw
    ]

    print(f"Got {len(daily)} days, {len(theme_tags)} tag rows, {len(freetext)} comments")

    bundle = {
        "daily": daily,
        "themeTags": theme_tags,
        "freetext": freetext,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
    bundle_json = json.dumps(bundle, separators=(",", ":"))

    print(f"Rendering template from {TEMPLATE_PATH}...")
    with open(TEMPLATE_PATH) as f:
        html = f.read()

    now_str = datetime.now(timezone.utc).strftime("%b %d, %Y, %I:%M %p UTC")
    html = html.replace("__BUNDLE_JSON_PLACEHOLDER__", bundle_json)
    html = html.replace("__GENERATED_AT_PLACEHOLDER__", f"Snapshot generated {now_str}")

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
