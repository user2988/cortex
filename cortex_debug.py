import os
import json
import requests
from datetime import datetime, timedelta

# 1. SETUP HEADERS & SETTINGS
# Get this from your environment/secrets
access_token = os.environ.get("FITBIT_ACCESS_TOKEN")

# 2. DEFINE THE FUNCTION FIRST
def fetch_fitbit(endpoint_url):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept-Language": "en_US"  # Forces Fitbit to use your profile's Timezone
    }
    response = requests.get(endpoint_url, headers=headers)
    
    if response.status_code == 401:
        print("!! Unauthorized: Token is expired or scopes are missing.")
        return {}
    if response.status_code != 200:
        print(f"!! Error {response.status_code}: {response.text}")
        return {}
        
    return response.json()

# 3. DEFINE THE DATES
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.now().strftime("%Y-%m-%d")

# 4. EXECUTE THE FETCHES (Now that the function is defined)
print(f"--- CORTEX DEBUG ---")
print(f"Syncing for: Activity={yesterday} | Recovery={today}")

# Steps (Yesterday)
activity_data = fetch_fitbit(f"https://api.fitbit.com/1/user/-/activities/date/{yesterday}.json")
steps = activity_data.get("summary", {}).get("steps", 0)

# Sleep (Today)
sleep_data = fetch_fitbit(f"https://api.fitbit.com/1.2/user/-/sleep/date/{today}.json")
sleep_list = sleep_data.get('sleep', [])

# HRV (Today)
hrv_data = fetch_fitbit(f"https://api.fitbit.com/1/user/-/hrv/date/{today}.json")
hrv_list = hrv_data.get('hrv', [])

# 5. PRINT THE RESULTS
print(f"Steps: {steps}")

if sleep_list:
    # Fitbit returns the most recent sleep first
    main_sleep = sleep_list[0]
    print(f"Sleep Duration: {main_sleep.get('minutesAsleep')} mins")
    print(f"Sleep Efficiency: {main_sleep.get('efficiency')}%")
else:
    print("Sleep: No data found. (Check 'today' date vs Fitbit App sync time)")

if hrv_list:
    print(f"HRV (RMSSD): {hrv_list[0]['value']['dailyRmssd']}")
else:
    print("HRV: No data found.")



