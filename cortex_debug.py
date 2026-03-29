import os
import json
import requests
from datetime import datetime, timedelta

# 1. TOKEN HANDLING (Using a refresh-first approach for GitHub Actions)
REFRESH_TOKEN = os.environ.get("FITBIT_REFRESH_TOKEN")
CLIENT_ID = os.environ.get("FITBIT_CLIENT_ID")
CLIENT_SECRET = os.environ.get("FITBIT_CLIENT_SECRET")

def get_valid_headers():
    # In a real run, you'd call the refresh endpoint here to get a fresh Access Token
    # For this debug script, we assume access_token is provided or refreshed
    access_token = os.environ.get("FITBIT_ACCESS_TOKEN") 
    return {"Authorization": f"Bearer {access_token}"}

headers = get_valid_headers()

# 2. DATE LOGIC (Correctly Aligned)
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.now().strftime("%Y-%m-%d")

def fetch_fitbit(endpoint_url):
    r = requests.get(endpoint_url, headers=headers)
    if r.status_code == 401:
        return {"error": "expired"}
    if r.status_code != 200:
        return {}
    return r.json()

# --- EXECUTION ---

# Activity (Yesterday)
activity = fetch_fitbit(f"https://api.fitbit.com/1/user/-/activities/date/{yesterday}.json")
steps = activity.get("summary", {}).get("steps", 0)

# Sleep (Today - using version 1.2)
sleep = fetch_fitbit(f"https://api.fitbit.com/1.2/user/-/sleep/date/{today}.json")
sleep_summary = sleep.get("summary", {})
# Better to use totalMinutesAsleep than duration (which includes being awake in bed)
sleep_mins = sleep_summary.get("totalMinutesAsleep", 0) 

# HRV (Today)
hrv = fetch_fitbit(f"https://api.fitbit.com/1/user/-/hrv/date/{today}.json")
hrv_entries = hrv.get("hrv", [])
# Safely get RMSSD without crashing if list is empty
hrv_val = hrv_entries[0].get("value", {}).get("dailyRmssd") if hrv_entries else "N/A"

# --- OUTPUT ---
print(f"--- CORTEX DEBUG ---")
print(f"Date Context: Activity={yesterday} | Recovery={today}")
print(f"Steps: {steps}")
print(f"Sleep: {sleep_mins} mins asleep")
print(f"HRV: {hrv_val} ms")

if hrv_val == "N/A":
    print("Note: HRV requires a high-quality sleep log. Ensure watch is snug.")
