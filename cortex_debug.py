import os
from datetime import date, timedelta
import json
import requests
import base64
import time

# ── CONFIG ──
FITBIT_CLIENT_ID     = os.environ["FITBIT_CLIENT_ID"]
FITBIT_CLIENT_SECRET = os.environ["FITBIT_CLIENT_SECRET"]
TOKEN_FILE           = "fitbit_tokens.json"

# ── AUTH HELPER ──
def load_tokens():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return json.load(f)
    return None

def save_tokens(tokens):
    tokens["saved_at"] = time.time()
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)
    return tokens

def refresh_tokens(tokens):
    creds = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        "https://api.fitbit.com/oauth2/token",
        headers={"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]}
    )
    resp.raise_for_status()
    return save_tokens(resp.json())

def get_headers():
    tokens = load_tokens()
    if not tokens:
        raise Exception("No Fitbit tokens found. Run bootstrap first.")
    # Refresh if > 7h 55min
    if time.time() - tokens.get("saved_at", 0) > (tokens.get("expires_in", 28800) - 300):
        tokens = refresh_tokens(tokens)
    return {"Authorization": f"Bearer {tokens['access_token']}"}

# ── CLIENT ──
BASE = "https://api.fitbit.com"
yesterday = (date.today() - timedelta(days=1)).isoformat()
headers = get_headers()

endpoints = {
    "sleep": f"/1.2/user/-/sleep/date/{yesterday}.json",
    "heart_rate": f"/1/user/-/activities/heart/date/{yesterday}/1d.json",
    "hrv": f"/1/user/-/hrv/date/{yesterday}.json",
    "spo2": f"/1/user/-/spo2/date/{yesterday}.json",
    "activity": f"/1/user/-/activities/date/{yesterday}.json",
    "vo2max": f"/1/user/-/cardioscore/date/{yesterday}.json",
}

for name, path in endpoints.items():
    r = requests.get(BASE + path, headers=headers)
    print(f"\n{name.upper()} (status {r.status_code}):")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
