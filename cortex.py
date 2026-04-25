"""
Cortex — Biometric Data Pipeline
Fetches Fitbit metrics daily and writes them to PostgreSQL.
Runs automatically via GitHub Actions at 11:00am EST.
"""

import os
import json
import time
import base64
import hashlib
import secrets
import webbrowser
import requests
import psycopg2

from cronometer import run_nutrition_pipeline

from datetime import date, datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

FITBIT_CLIENT_ID     = os.environ["FITBIT_CLIENT_ID"]
FITBIT_CLIENT_SECRET = os.environ["FITBIT_CLIENT_SECRET"]
FITBIT_REDIRECT_URI  = "http://localhost:8080/callback"
TOKEN_FILE           = "fitbit_tokens.json"

DATABASE_URL         = os.environ["DATABASE_URL"]


# ─────────────────────────────────────────────────────────────
# PART 1 — FITBIT AUTH
# ─────────────────────────────────────────────────────────────

class FitbitAuth:
    AUTH_URL  = "https://www.fitbit.com/oauth2/authorize"
    TOKEN_URL = "https://api.fitbit.com/oauth2/token"
    SCOPES    = "sleep heartrate activity oxygen_saturation cardio_fitness respiratory_rate profile"

    def __init__(self):
        self.tokens = self._load_tokens()

    def _load_tokens(self):
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r") as f:
                return json.load(f)
        return None

    def _save_tokens(self, tokens):
        tokens["saved_at"] = time.time()
        with open(TOKEN_FILE, "w") as f:
            json.dump(tokens, f, indent=2)
        self.tokens = tokens

    def needs_refresh(self):
        if not self.tokens:
            return True
        age = time.time() - self.tokens.get("saved_at", 0)
        return age > (self.tokens.get("expires_in", 28800) - 300)

    def refresh(self):
        print("Refreshing Fitbit tokens...")
        creds = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
        resp  = requests.post(self.TOKEN_URL, headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        }, data={
            "grant_type":    "refresh_token",
            "refresh_token": self.tokens["refresh_token"],
        })
        if resp.status_code == 200:
            self._save_tokens(resp.json())
            print("Tokens refreshed.")
        else:
            print(f"Token refresh failed: {resp.text}")
            resp.raise_for_status()

    def get_headers(self):
        if not self.tokens:
            self.bootstrap_locally()
        if self.needs_refresh():
            self.refresh()
        return {"Authorization": f"Bearer {self.tokens['access_token']}"}

    def bootstrap_locally(self):
        """Run once locally to generate the first token file."""
        print("--- LOCAL BOOTSTRAP MODE ---")
        verifier  = secrets.token_urlsafe(64)[:128]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state     = secrets.token_urlsafe(16)

        params = {
            "client_id":             FITBIT_CLIENT_ID,
            "response_type":         "code",
            "scope":                 self.SCOPES,
            "redirect_uri":          FITBIT_REDIRECT_URI,
            "code_challenge":        challenge,
            "code_challenge_method": "S256",
            "state":                 state,
        }

        url = f"{self.AUTH_URL}?{urlencode(params)}"
        print(f"Opening browser: {url}")
        webbrowser.open(url)

        auth_code = [None]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                qs = parse_qs(urlparse(self.path).query)
                if "code" in qs:
                    auth_code[0] = qs["code"][0]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Authorized! Close this tab.")
            def log_message(self, *args):
                pass

        HTTPServer(("localhost", 8080), Handler).handle_request()
        if auth_code[0]:
            self._exchange_code(auth_code[0], verifier)

    def _exchange_code(self, code, verifier):
        creds = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
        resp  = requests.post(self.TOKEN_URL, headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        }, data={
            "code":          code,
            "grant_type":    "authorization_code",
            "redirect_uri":  FITBIT_REDIRECT_URI,
            "code_verifier": verifier,
        })
        resp.raise_for_status()
        self._save_tokens(resp.json())


# ─────────────────────────────────────────────────────────────
# PART 2 — FITBIT DATA FETCHING
# ─────────────────────────────────────────────────────────────

class FitbitClient:
    BASE = "https://api.fitbit.com"

    def __init__(self, auth: FitbitAuth):
        self.auth = auth

    def _get(self, path):
        r = requests.get(f"{self.BASE}{path}", headers=self.auth.get_headers())
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"Rate limited — waiting {wait}s...")
            time.sleep(wait)
            return self._get(path)
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} for url: {r.url}\nBody: {r.text}",
                response=r,
            )
        return r.json()

    def fetch_sleep(self, d):
        data    = self._get(f"/1.2/user/-/sleep/date/{d}.json")
        summary = data.get("summary", {})
        stages  = summary.get("stages", {})
        main    = data.get("sleep", [{}])[0]
        return {
            "sleep_minutes":           summary.get("totalMinutesAsleep"),
            "time_in_bed":             summary.get("totalTimeInBed"),
            "sleep_efficiency":        main.get("efficiency"),
            "stage_deep":              stages.get("deep"),
            "stage_rem":               stages.get("rem"),
            "stage_light":             stages.get("light"),
            "stage_wake":              stages.get("wake"),
        }

    def fetch_heart_rate(self, d):
        data  = self._get(f"/1/user/-/activities/heart/date/{d}/1d.json")
        value = data.get("activities-heart", [{}])[0].get("value", {})
        return {"resting_heart_rate": value.get("restingHeartRate")}

    def fetch_hrv(self, d):
        data     = self._get(f"/1/user/-/hrv/date/{d}.json")
        hrv_list = data.get("hrv", [])
        if not hrv_list:
            return {"hrv_rmssd": None, "hrv_deep_rmssd": None}
        value = hrv_list[0]["value"]
        return {
            "hrv_rmssd":      value.get("dailyRmssd"),
            "hrv_deep_rmssd": value.get("deepRmssd"),
        }

    def fetch_spo2(self, d):
        data  = self._get(f"/1/user/-/spo2/date/{d}.json")
        value = data.get("value", {})
        # Fall back to previous date — Fitbit sometimes keys SpO2 to sleep-start date
        if not value:
            prev  = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
            data  = self._get(f"/1/user/-/spo2/date/{prev}.json")
            value = data.get("value", {})
        return {
            "spo2_avg": value.get("avg"),
            "spo2_min": value.get("min"),
            "spo2_max": value.get("max"),
        }

    def fetch_breathing_rate(self, d):
        data    = self._get(f"/1/user/-/br/date/{d}.json")
        br_list = data.get("br", [])
        return {"respiratory_rate": br_list[0]["value"]["breathingRate"] if br_list else None}

    def fetch_activity(self, d):
        data    = self._get(f"/1/user/-/activities/date/{d}.json")
        summary = data.get("summary", {})
        return {
            "steps":                  summary.get("steps"),
            "calories_out":           summary.get("caloriesOut"),
            "very_active_minutes":    summary.get("veryActiveMinutes"),
            "fairly_active_minutes":  summary.get("fairlyActiveMinutes"),
            "lightly_active_minutes": summary.get("lightlyActiveMinutes"),
            "sedentary_minutes":      summary.get("sedentaryMinutes"),
            "distance_km":            next((d["distance"] for d in summary.get("distances", []) if d["activity"] == "total"), None),
        }

    def fetch_vo2max(self, d):
        data  = self._get(f"/1/user/-/cardioscore/date/{d}.json")
        score = data.get("cardioScore", [])
        if not score:
            return {"vo2_max": None}
        raw = score[0]["value"].get("vo2Max")
        if raw is None:
            return {"vo2_max": None}
        # Returns a range e.g. "46-50" before a GPS run — store midpoint
        if isinstance(raw, str) and "-" in raw:
            lo, hi = raw.split("-")
            return {"vo2_max": (float(lo) + float(hi)) / 2}
        return {"vo2_max": float(raw)}

    def fetch_hr_zones(self, d):
        data     = self._get(f"/1/user/-/activities/heart/date/{d}/1d.json")
        value    = data.get("activities-heart", [{}])[0].get("value", {})
        hr_zones = {z["name"]: z["minutes"] for z in value.get("heartRateZones", [])}
        return {
            "time_in_fat_burn_min": hr_zones.get("Fat Burn"),
            "time_in_cardio_min":   hr_zones.get("Cardio"),
            "time_in_peak_min":     hr_zones.get("Peak"),
        }

    def fetch_azm(self, d):
        data    = self._get(f"/1/user/-/activities/active-zone-minutes/date/{d}/1d.json")
        entries = data.get("activities-active-zone-minutes", [])
        if not entries:
            return {"active_zone_minutes": None}
        return {"active_zone_minutes": entries[0].get("value", {}).get("activeZoneMinutes")}



# ─────────────────────────────────────────────────────────────
# PART 3 — POSTGRESQL STORAGE
# ─────────────────────────────────────────────────────────────

def store_biometrics(record):
    sql = """
        INSERT INTO biometrics (
            date,
            sleep_duration_min, sleep_efficiency_pct,
            deep_sleep_min, rem_sleep_min, light_sleep_min, awake_min,
            time_in_bed_min,
            hrv_ms, hrv_deep_rmssd, rhr_bpm,
            spo2_avg_pct, spo2_min_pct, spo2_max_pct, respiratory_rate,
            steps, active_zone_min, very_active_min, fairly_active_min, lightly_active_min,
            sedentary_min, calories_burned, distance_km, vo2_max,
            time_in_fat_burn_min, time_in_cardio_min, time_in_peak_min
        ) VALUES (
            %(date)s,
            %(sleep_minutes)s, %(sleep_efficiency)s,
            %(stage_deep)s, %(stage_rem)s, %(stage_light)s, %(stage_wake)s,
            %(time_in_bed)s,
            %(hrv_rmssd)s, %(hrv_deep_rmssd)s, %(resting_heart_rate)s,
            %(spo2_avg)s, %(spo2_min)s, %(spo2_max)s, %(respiratory_rate)s,
            %(steps)s, %(active_zone_minutes)s, %(very_active_minutes)s, %(fairly_active_minutes)s, %(lightly_active_minutes)s,
            %(sedentary_minutes)s, %(calories_out)s, %(distance_km)s, %(vo2_max)s,
            %(time_in_fat_burn_min)s, %(time_in_cardio_min)s, %(time_in_peak_min)s
        )
        ON CONFLICT (date) DO UPDATE SET
            sleep_duration_min      = COALESCE(biometrics.sleep_duration_min,   EXCLUDED.sleep_duration_min),
            sleep_efficiency_pct    = COALESCE(biometrics.sleep_efficiency_pct, EXCLUDED.sleep_efficiency_pct),
            deep_sleep_min          = COALESCE(biometrics.deep_sleep_min,       EXCLUDED.deep_sleep_min),
            rem_sleep_min           = COALESCE(biometrics.rem_sleep_min,        EXCLUDED.rem_sleep_min),
            light_sleep_min         = COALESCE(biometrics.light_sleep_min,      EXCLUDED.light_sleep_min),
            awake_min               = COALESCE(biometrics.awake_min,            EXCLUDED.awake_min),
            time_in_bed_min         = COALESCE(biometrics.time_in_bed_min,      EXCLUDED.time_in_bed_min),
            hrv_ms                  = COALESCE(biometrics.hrv_ms,               EXCLUDED.hrv_ms),
            hrv_deep_rmssd          = COALESCE(biometrics.hrv_deep_rmssd,       EXCLUDED.hrv_deep_rmssd),
            rhr_bpm                 = COALESCE(biometrics.rhr_bpm,              EXCLUDED.rhr_bpm),
            spo2_avg_pct            = COALESCE(biometrics.spo2_avg_pct,         EXCLUDED.spo2_avg_pct),
            spo2_min_pct            = COALESCE(biometrics.spo2_min_pct,         EXCLUDED.spo2_min_pct),
            spo2_max_pct            = COALESCE(biometrics.spo2_max_pct,         EXCLUDED.spo2_max_pct),
            respiratory_rate        = COALESCE(biometrics.respiratory_rate,     EXCLUDED.respiratory_rate),
            steps                   = COALESCE(biometrics.steps,                EXCLUDED.steps),
            active_zone_min         = COALESCE(biometrics.active_zone_min,      EXCLUDED.active_zone_min),
            very_active_min         = COALESCE(biometrics.very_active_min,      EXCLUDED.very_active_min),
            fairly_active_min       = COALESCE(biometrics.fairly_active_min,    EXCLUDED.fairly_active_min),
            lightly_active_min      = COALESCE(biometrics.lightly_active_min,   EXCLUDED.lightly_active_min),
            sedentary_min           = COALESCE(biometrics.sedentary_min,        EXCLUDED.sedentary_min),
            calories_burned         = COALESCE(biometrics.calories_burned,      EXCLUDED.calories_burned),
            distance_km             = COALESCE(biometrics.distance_km,          EXCLUDED.distance_km),
            vo2_max                 = COALESCE(biometrics.vo2_max,              EXCLUDED.vo2_max),
            time_in_fat_burn_min    = COALESCE(biometrics.time_in_fat_burn_min, EXCLUDED.time_in_fat_burn_min),
            time_in_cardio_min      = COALESCE(biometrics.time_in_cardio_min,   EXCLUDED.time_in_cardio_min),
            time_in_peak_min        = COALESCE(biometrics.time_in_peak_min,     EXCLUDED.time_in_peak_min);
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "date":                    record.get("date"),
                    "sleep_minutes":           record.get("sleep_minutes"),
                    "sleep_efficiency":        record.get("sleep_efficiency"),
                    "stage_deep":              record.get("stage_deep"),
                    "stage_rem":               record.get("stage_rem"),
                    "stage_light":             record.get("stage_light"),
                    "stage_wake":              record.get("stage_wake"),
                    "time_in_bed":             record.get("time_in_bed"),
                    "hrv_rmssd":               record.get("hrv_rmssd"),
                    "hrv_deep_rmssd":          record.get("hrv_deep_rmssd"),
                    "resting_heart_rate":      record.get("resting_heart_rate"),
                    "spo2_avg":                record.get("spo2_avg"),
                    "spo2_min":                record.get("spo2_min"),
                    "spo2_max":                record.get("spo2_max"),
                    "respiratory_rate":        record.get("respiratory_rate"),
                    "steps":                   record.get("steps"),
                    "active_zone_minutes":     record.get("active_zone_minutes"),
                    "very_active_minutes":     record.get("very_active_minutes"),
                    "fairly_active_minutes":   record.get("fairly_active_minutes"),
                    "lightly_active_minutes":  record.get("lightly_active_minutes"),
                    "sedentary_minutes":       record.get("sedentary_minutes"),
                    "calories_out":            record.get("calories_out"),
                    "distance_km":             record.get("distance_km"),
                    "vo2_max":                 record.get("vo2_max"),
                    "time_in_fat_burn_min":    record.get("time_in_fat_burn_min"),
                    "time_in_cardio_min":      record.get("time_in_cardio_min"),
                    "time_in_peak_min":        record.get("time_in_peak_min"),
                })
        print(f"Stored {record['date']} in PostgreSQL.")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# PART 4 — PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline():
    today_str     = datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    auth   = FitbitAuth()
    client = FitbitClient(auth)

    print(f"Syncing {yesterday_str}...")

    def safe_fetch(name, fn, *args):
        try:
            result = fn(*args)
            print(f"  {name} OK")
            return result
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            return {}

    # Activity from yesterday — full day is complete by morning
    # Sleep/recovery from today — Fitbit keys overnight data to wake-up date
    activity  = safe_fetch("activity",       client.fetch_activity,       yesterday_str)
    azm       = safe_fetch("azm",            client.fetch_azm,            yesterday_str)
    hr_zones  = safe_fetch("hr_zones",       client.fetch_hr_zones,       yesterday_str)
    sleep     = safe_fetch("sleep",          client.fetch_sleep,          today_str)
    hrv       = safe_fetch("hrv",            client.fetch_hrv,            today_str)
    rhr       = safe_fetch("rhr",            client.fetch_heart_rate,     today_str)
    spo2      = safe_fetch("spo2",           client.fetch_spo2,           today_str)
    br        = safe_fetch("breathing_rate", client.fetch_breathing_rate, today_str)
    vo2max    = safe_fetch("vo2max",         client.fetch_vo2max,         today_str)
    # Record keyed to activity date — sleep/recovery is for the overnight period ending today
    record = {
        "date": yesterday_str,
        **activity, **azm, **hr_zones, **sleep, **hrv, **rhr, **spo2, **br, **vo2max
    }

    store_biometrics(record)

    print(f"Syncing nutrition for {yesterday_str}...")
    run_nutrition_pipeline(yesterday_str)

    print("DONE.")


if __name__ == "__main__":
    run_pipeline()
