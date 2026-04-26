"""
Cortex — Debug Script
Quick sanity-check for Fitbit API connectivity and token validity.
Requires FITBIT_ACCESS_TOKEN env var (raw token, no OAuth flow).
"""

import os
import requests
from datetime import datetime, timedelta

access_token = os.environ.get("FITBIT_ACCESS_TOKEN")

def fetch(url):
    r = requests.get(url, headers={
        "Authorization": f"Bearer {access_token}",
        "Accept-Language": "en_US",
    })
    if r.status_code == 401:
        print("Unauthorized — token expired or missing scopes.")
        return {}
    if not r.ok:
        print(f"Error {r.status_code}: {r.text}")
        return {}
    return r.json()

yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
today     = datetime.now().strftime("%Y-%m-%d")

print(f"--- CORTEX DEBUG ---")
print(f"Activity date: {yesterday}  |  Recovery date: {today}")

activity = fetch(f"https://api.fitbit.com/1/user/-/activities/date/{yesterday}.json")
sleep    = fetch(f"https://api.fitbit.com/1.2/user/-/sleep/date/{today}.json")
hrv      = fetch(f"https://api.fitbit.com/1/user/-/hrv/date/{today}.json")

print(f"Steps:            {activity.get('summary', {}).get('steps', 'n/a')}")

sleep_list = sleep.get("sleep", [])
if sleep_list:
    s = sleep_list[0]
    print(f"Sleep duration:   {s.get('minutesAsleep')} min")
    print(f"Sleep efficiency: {s.get('efficiency')}%")
else:
    print("Sleep: no data (check sync time)")

hrv_list = hrv.get("hrv", [])
if hrv_list:
    print(f"HRV RMSSD:        {hrv_list[0]['value']['dailyRmssd']}")
else:
    print("HRV: no data")
