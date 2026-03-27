import os
import requests
from datetime import datetime, timedelta

# Get Fitbit token from secrets / environment
FITBIT_TOKENS = os.environ.get("FITBIT_TOKENS")  # should contain {"access_token": "...", "refresh_token": "..."}
access_token = FITBIT_TOKENS and eval(FITBIT_TOKENS)["access_token"]

if not access_token:
    raise Exception("No Fitbit access token found.")

# Get yesterday's date
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

url = f"https://api.fitbit.com/1/user/-/activities/date/{yesterday}.json"
headers = {"Authorization": f"Bearer {access_token}"}

resp = requests.get(url, headers=headers)
data = resp.json()

# Print steps, calories, distances, floors in hours where applicable
summary = data.get("summary", {})
print("Steps:", summary.get("steps"))
print("Calories Out:", summary.get("caloriesOut"))
print("BMR Calories:", summary.get("caloriesBMR"))
print("Activity Calories:", summary.get("activityCalories"))
print("Floors:", summary.get("floors"))
print("Distances (km):", {d["activity"]: d["distance"] for d in summary.get("distances", [])})
print("Sedentary Minutes:", summary.get("sedentaryMinutes"))
print("Lightly Active Minutes:", summary.get("lightlyActiveMinutes"))
print("Fairly Active Minutes:", summary.get("fairlyActiveMinutes"))
print("Very Active Minutes:", summary.get("veryActiveMinutes"))
