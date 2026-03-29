import os
import json
import requests
from datetime import datetime, timedelta

# 1. CLEANER TOKEN HANDLING
# Suggestion: Store as a JSON string in GitHub Secrets, not a Python dict string
tokens_raw = os.environ.get("FITBIT_TOKENS")
if not tokens_raw:
    raise Exception("Missing FITBIT_TOKENS environment variable.")

tokens = json.loads(tokens_raw) 
access_token = tokens.get("access_token")

headers = {"Authorization": f"Bearer {access_token}"}

# 2. THE DATE LOGIC
# For Activities (Steps): Use yesterday (00:00 - 23:59)
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
# For Sleep/HRV: Use 'today' because the sleep log 'date' is the day you WAKE UP.
today = datetime.now().strftime("%Y-%m-%d")

def fetch_fitbit(endpoint_url):
    response = requests.get(endpoint_url, headers=headers)
    if response.status_code == 401:
        print("!! Token Expired. Need to run refresh_token logic.")
        return None
    return response.json()

# --- FETCH ACTIVITY (Yesterday) ---
activity_data = fetch_fitbit(f"https://api.fitbit.com/1/user/-/activities/date/{yesterday}.json")
summary = activity_data.get("summary", {})

# --- FETCH SLEEP (Today's morning wake-up) ---
# This includes duration, efficiency, and stages 
sleep_data = fetch_fitbit(f"https://api.fitbit.com/1.2/user/-/sleep/date/{today}.json")

# --- FETCH HRV (Today's morning wake-up) ---
# Cortex needs RMSSD for the Recovery Status [cite: 53, 227]
hrv_data = fetch_fitbit(f"https://api.fitbit.com/1/user/-/hrv/date/{today}.json")

# DEBUG PRINT
print(f"--- CORTEX DEBUG (Target Date: {yesterday}) ---")
print(f"Steps: {summary.get('steps')}")
print(f"Active Zone Minutes: {summary.get('activeZoneMinutes')}")

if sleep_data and sleep_data.get('sleep'):
    main_sleep = sleep_data['sleep'][0]
    print(f"Sleep Duration: {main_sleep.get('duration') / 60000:.2f} mins")
    print(f"Sleep Efficiency: {main_sleep.get('efficiency')}%")
else:
    print("Sleep: No data found for today yet (Sync your watch!)")

if hrv_data and hrv_data.get('hrv'):
    latest_hrv = hrv_data['hrv'][0]['value']['dailyRmssd']
    print(f"HRV (RMSSD): {latest_hrv}")
