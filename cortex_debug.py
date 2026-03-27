import os, json, requests
from datetime import date

# Load the Fitbit tokens from GitHub secret
tokens = json.loads(os.environ["FITBIT_TOKENS"])
access_token = tokens["access_token"]

# Get yesterday's date
yesterday = (date.today()).isoformat()

# Define endpoints
endpoints = {
    "sleep": f"/1.2/user/-/sleep/date/{yesterday}.json",
    "heart": f"/1/user/-/activities/heart/date/{yesterday}/1d.json",
    "hrv": f"/1/user/-/hrv/date/{yesterday}.json",
    "spo2": f"/1/user/-/spo2/date/{yesterday}.json",
    "activity": f"/1/user/-/activities/date/{yesterday}.json",
}

headers = {"Authorization": f"Bearer {access_token}"}

for name, path in endpoints.items():
    r = requests.get(f"https://api.fitbit.com{path}", headers=headers)
    print(f"\n{name.upper()}:")
    print(json.dumps(r.json(), indent=2))
