"""
Cortex — Personal Performance Intelligence
Fetches Fitbit biometrics → stores in Pinecone → Claude analysis → morning email
Runs automatically via GitHub Actions at 8:30am EST daily
"""

import os
import json
import time
import base64
import hashlib
import secrets
import webbrowser
import smtplib
import re
import requests
import anthropic
import numpy as np

from datetime import date, datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pinecone import Pinecone, ServerlessSpec

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — all secrets loaded from environment variables
# ─────────────────────────────────────────────────────────────

FITBIT_CLIENT_ID     = os.environ["FITBIT_CLIENT_ID"]
FITBIT_CLIENT_SECRET = os.environ["FITBIT_CLIENT_SECRET"]
FITBIT_REDIRECT_URI  = "http://localhost:8080/callback"
TOKEN_FILE           = "fitbit_tokens.json"

PINECONE_API_KEY     = os.environ["PINECONE_API_KEY"]
INDEX_NAME           = "fitness-metrics"

ANTHROPIC_API_KEY    = os.environ["ANTHROPIC_API_KEY"]

EMAIL_SENDER         = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD       = os.environ["EMAIL_PASSWORD"]
EMAIL_RECIPIENT      = os.environ["EMAIL_RECIPIENT"]


# ─────────────────────────────────────────────────────────────
# PART 1 — FITBIT AUTH
# ─────────────────────────────────────────────────────────────

class FitbitAuth:
    AUTH_URL  = "https://www.fitbit.com/oauth2/authorize"
    TOKEN_URL = "https://api.fitbit.com/oauth2/token"
    SCOPES    = "sleep heartrate activity oxygen_saturation cardio_fitness profile"

    def __init__(self):
        self.tokens = self._load_tokens()

    def is_authenticated(self):
        """Checks if we have a token at all."""
        return self.tokens is not None
    
    def _load_tokens(self):
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                return json.load(f)
        return None

    def _save_tokens(self, tokens):
        # We add a timestamp so we know when the 8-hour window started
        tokens["saved_at"] = time.time()
        with open(TOKEN_FILE, "w") as f:
            json.dump(tokens, f, indent=2)
        self.tokens = tokens

    def needs_refresh(self):
        if not self.tokens: return True
        # Refresh 5 minutes before the 8-hour (28800s) window closes
        age = time.time() - self.tokens.get("saved_at", 0)
        return age > (self.tokens.get("expires_in", 28800) - 300)

    def refresh(self):
        """Automated refresh logic for GitHub Actions"""
        print("Refreshing Fitbit tokens...")
        creds = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
        
        resp = requests.post(self.TOKEN_URL, headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        }, data={
            "grant_type":    "refresh_token",
            "refresh_token": self.tokens["refresh_token"],
        })
        
        if resp.status_code == 200:
            self._save_tokens(resp.json())
            print("Tokens successfully rotated and saved to disk.")
        else:
            print(f"FAILED REFRESH: {resp.text}")
            resp.raise_for_status()

    def get_headers(self):
        """The entry point for the API client"""
        if not self.tokens:
            # This triggers if you haven't done the first manual login yet
            self.bootstrap_locally()
            
        if self.needs_refresh():
            self.refresh()
            
        return {"Authorization": f"Bearer {self.tokens['access_token']}"}

    def bootstrap_locally(self):
        """Run this once on your laptop to generate the first token file"""
        print("--- LOCAL BOOTSTRAP MODE ---")
        import secrets, hashlib, webbrowser
        from http.server import HTTPServer, BaseHTTPRequestHandler
        from urllib.parse import urlparse, parse_qs, urlencode

        verifier = secrets.token_urlsafe(64)[:128]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        
        params = {
            "client_id": FITBIT_CLIENT_ID,
            "response_type": "code",
            "scope": self.SCOPES,
            "redirect_uri": FITBIT_REDIRECT_URI,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        
        url = f"{self.AUTH_URL}?{urlencode(params)}"
        print(f"Opening browser to authorize: {url}")
        webbrowser.open(url)

        auth_code = [None]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                qs = parse_qs(urlparse(self.path).query)
                if "code" in qs: auth_code[0] = qs["code"][0]
                self.send_response(200); self.end_headers()
                self.wfile.write(b"Success! Close this tab.")
        
        server = HTTPServer(("localhost", 8080), Handler)
        server.handle_request()
        
        if auth_code[0]:
            self._exchange_code(auth_code[0], verifier)

    def _exchange_code(self, code, verifier):
        creds = base64.b64encode(f"{FITBIT_CLIENT_ID}:{FITBIT_CLIENT_SECRET}".encode()).decode()
        resp = requests.post(self.TOKEN_URL, headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        }, data={
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": FITBIT_REDIRECT_URI,
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
        r.raise_for_status()
        return r.json()

    def fetch_sleep(self, d):
        data    = self._get(f"/1.2/user/-/sleep/date/{d}.json")
        summary = data.get("summary", {})
        stages  = summary.get("stages", {})
        entries = data.get("sleep", [])
        main    = entries[0] if entries else {}
        return {
            "sleep_minutes":            summary.get("totalMinutesAsleep"),
            "time_in_bed":              summary.get("totalTimeInBed"),
            "sleep_score":              main.get("efficiency"),
            "stage_deep":               stages.get("deep"),
            "stage_rem":                stages.get("rem"),
            "stage_light":              stages.get("light"),
            "stage_wake":               stages.get("wake"),
            "sleep_onset_latency_min":  main.get("minutesToFallAsleep"),
        }

    def fetch_heart_rate(self, d):
        data  = self._get(f"/1/user/-/activities/heart/date/{d}/1d.json")
        value = data.get("activities-heart", [{}])[0].get("value", {})
        return {"resting_heart_rate": value.get("restingHeartRate")}

    def fetch_hrv(self, d):
        data     = self._get(f"/1/user/-/hrv/date/{d}.json")
        hrv_list = data.get("hrv", [])
        return {"hrv_rmssd": hrv_list[0]["value"]["dailyRmssd"] if hrv_list else None}

    def fetch_spo2(self, d):
        data  = self._get(f"/1/user/-/spo2/date/{d}.json")
        value = data.get("value", {})
        return {
            "spo2_avg": value.get("avg"),
            "spo2_min": value.get("min"),
        }

    def fetch_breathing_rate(self, d):
        data    = self._get(f"/1/user/-/br/date/{d}.json")
        br_list = data.get("br", [])
        return {"respiratory_rate": br_list[0]["value"]["breathingRate"] if br_list else None}

    def fetch_activity(self, d):
        data    = self._get(f"/1/user/-/activities/date/{d}.json")
        summary = data.get("summary", {})
        hr_zones = {z["name"]: z["minutes"] for z in summary.get("heartRateZones", [])}
        return {
            "steps":                   summary.get("steps"),
            "calories_out":            summary.get("caloriesOut"),
            "active_zone_minutes":     summary.get("activeZoneMinutes", {}).get("totalMinutes"),
            "very_active_minutes":     summary.get("veryActiveMinutes"),
            "fairly_active_minutes":   summary.get("fairlyActiveMinutes"),
            "sedentary_minutes":       summary.get("sedentaryMinutes"),
            "distance_km":             next((d["distance"] for d in summary.get("distances", []) if d["activity"] == "total"), None),
            "time_in_fat_burn_min":    hr_zones.get("Fat Burn"),
            "time_in_cardio_min":      hr_zones.get("Cardio"),
            "time_in_peak_min":        hr_zones.get("Peak"),
        }

    def fetch_vo2max(self, d):
        data  = self._get(f"/1/user/-/cardioscore/date/{d}.json")
        score = data.get("cardioScore", [])
        return {"vo2_max": score[0]["value"].get("vo2Max") if score else None}

    def fetch_day(self, target_date=None):
        if target_date is None:
            target_date = (date.today() - timedelta(days=1)).isoformat()
        print(f"Fetching {target_date}...")
        record = {"date": target_date}
        for name, fn in [
            ("sleep",      self.fetch_sleep),
            ("heart_rate", self.fetch_heart_rate),
            ("hrv",        self.fetch_hrv),
            ("spo2",       self.fetch_spo2),
            ("activity",   self.fetch_activity),
            ("vo2max",     self.fetch_vo2max),
        ]:
            try:
                record.update(fn(target_date))
                print(f"  {name} OK")
            except Exception as e:
                print(f"  {name} failed: {e}")
        return record

    def fetch_range(self, days=30):
        records = []
        for i in range(1, days + 1):
            d = (date.today() - timedelta(days=i)).isoformat()
            try:
                records.append(self.fetch_day(d))
                time.sleep(0.5)
            except Exception as e:
                print(f"Skipped {d}: {e}")
        return records


# ─────────────────────────────────────────────────────────────
# PART 3 — POSTGRESQL STORAGE
# ─────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")

def store_biometrics(record):
    if not DATABASE_URL:
        print("DATABASE_URL not set — skipping Postgres write.")
        return

    import psycopg2

    sql = """
        INSERT INTO biometrics (
            date,
            sleep_duration_min, sleep_efficiency_pct,
            deep_sleep_min, rem_sleep_min, light_sleep_min, awake_min, time_in_bed_min,
            sleep_onset_latency_min,
            hrv_ms, rhr_bpm, spo2_avg_pct, spo2_min_pct, respiratory_rate,
            steps, active_zone_min, very_active_min, fairly_active_min,
            sedentary_min, calories_burned, distance_km, vo2_max,
            time_in_fat_burn_min, time_in_cardio_min, time_in_peak_min
        ) VALUES (
            %(date)s,
            %(sleep_minutes)s, %(sleep_score)s,
            %(stage_deep)s, %(stage_rem)s, %(stage_light)s, %(stage_wake)s, %(time_in_bed)s,
            %(sleep_onset_latency_min)s,
            %(hrv_rmssd)s, %(resting_heart_rate)s, %(spo2_avg)s, %(spo2_min)s, %(respiratory_rate)s,
            %(steps)s, %(active_zone_minutes)s, %(very_active_minutes)s, %(fairly_active_minutes)s,
            %(sedentary_minutes)s, %(calories_out)s, %(distance_km)s, %(vo2_max)s,
            %(time_in_fat_burn_min)s, %(time_in_cardio_min)s, %(time_in_peak_min)s
        )
        ON CONFLICT (date) DO UPDATE SET
            sleep_duration_min      = EXCLUDED.sleep_duration_min,
            sleep_efficiency_pct    = EXCLUDED.sleep_efficiency_pct,
            deep_sleep_min          = EXCLUDED.deep_sleep_min,
            rem_sleep_min           = EXCLUDED.rem_sleep_min,
            light_sleep_min         = EXCLUDED.light_sleep_min,
            awake_min               = EXCLUDED.awake_min,
            time_in_bed_min         = EXCLUDED.time_in_bed_min,
            sleep_onset_latency_min = EXCLUDED.sleep_onset_latency_min,
            hrv_ms                  = EXCLUDED.hrv_ms,
            rhr_bpm                 = EXCLUDED.rhr_bpm,
            spo2_avg_pct            = EXCLUDED.spo2_avg_pct,
            spo2_min_pct            = EXCLUDED.spo2_min_pct,
            respiratory_rate        = EXCLUDED.respiratory_rate,
            steps                   = EXCLUDED.steps,
            active_zone_min         = EXCLUDED.active_zone_min,
            very_active_min         = EXCLUDED.very_active_min,
            fairly_active_min       = EXCLUDED.fairly_active_min,
            sedentary_min           = EXCLUDED.sedentary_min,
            calories_burned         = EXCLUDED.calories_burned,
            distance_km             = EXCLUDED.distance_km,
            vo2_max                 = EXCLUDED.vo2_max,
            time_in_fat_burn_min    = EXCLUDED.time_in_fat_burn_min,
            time_in_cardio_min      = EXCLUDED.time_in_cardio_min,
            time_in_peak_min        = EXCLUDED.time_in_peak_min;
    """

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "date":                     record.get("date"),
                    "sleep_minutes":            record.get("sleep_minutes"),
                    "sleep_score":              record.get("sleep_score"),
                    "stage_deep":               record.get("stage_deep"),
                    "stage_rem":                record.get("stage_rem"),
                    "stage_light":              record.get("stage_light"),
                    "stage_wake":               record.get("stage_wake"),
                    "time_in_bed":              record.get("time_in_bed"),
                    "sleep_onset_latency_min":  record.get("sleep_onset_latency_min"),
                    "hrv_rmssd":                record.get("hrv_rmssd"),
                    "resting_heart_rate":       record.get("resting_heart_rate"),
                    "spo2_avg":                 record.get("spo2_avg"),
                    "spo2_min":                 record.get("spo2_min"),
                    "respiratory_rate":         record.get("respiratory_rate"),
                    "steps":                    record.get("steps"),
                    "active_zone_minutes":      record.get("active_zone_minutes"),
                    "very_active_minutes":      record.get("very_active_minutes"),
                    "fairly_active_minutes":    record.get("fairly_active_minutes"),
                    "sedentary_minutes":        record.get("sedentary_minutes"),
                    "calories_out":             record.get("calories_out"),
                    "distance_km":              record.get("distance_km"),
                    "vo2_max":                  record.get("vo2_max"),
                    "time_in_fat_burn_min":     record.get("time_in_fat_burn_min"),
                    "time_in_cardio_min":       record.get("time_in_cardio_min"),
                    "time_in_peak_min":         record.get("time_in_peak_min"),
                })
        print(f"Stored {record['date']} in PostgreSQL.")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# PART 4 — PINECONE STORAGE & RETRIEVAL
# ─────────────────────────────────────────────────────────────

METRIC_RANGES = {
    "sleep_minutes":       (240, 600),
    "sleep_score":         (0,   100),
    "stage_deep":          (0,   120),
    "stage_rem":           (0,   180),
    "hrv_rmssd":           (10,  100),
    "resting_heart_rate":  (40,  100),
    "spo2_avg":            (90,  100),
    "steps":               (0,   20000),
    "active_zone_minutes": (0,   120),
    "calories_out":        (1500, 4000),
    "vo2_max":             (20,  60),
    "distance_km":         (0,   20),
}
METRIC_KEYS = list(METRIC_RANGES.keys())


def normalise(value, min_val, max_val):
    if value is None:
        return 0.5
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def metrics_to_vector(record):
    return [normalise(record.get(key), *METRIC_RANGES[key]) for key in METRIC_KEYS]


def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=len(METRIC_KEYS), # Automatically scales to your METRIC_KEYS list
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
    return pc.Index(INDEX_NAME)

def store_day(index, record):
    vector   = metrics_to_vector(record)
    metadata = {k: (v if v is not None else -1) for k, v in record.items() if k != "date"}
    metadata["date"] = record["date"]
    index.upsert(vectors=[{"id": record["date"], "values": vector, "metadata": metadata}])
    print(f"Stored {record['date']} in Pinecone.")


def get_recent_days(index, n=7):
    ids     = [(date.today() - timedelta(days=i)).isoformat() for i in range(1, n + 1)]
    result  = index.fetch(ids=ids)
    records = []
    for id_ in ids:
        if id_ in result["vectors"]:
            records.append(result["vectors"][id_]["metadata"])
    return records


def get_rolling_summary(index, n=7):
    records = get_recent_days(index, n)
    if not records:
        return "No historical data available yet."
    lines = [f"Last {len(records)} days of metrics:\n"]
    for r in records:
        lines.append(
            f"  {r.get('date','?')} | "
            f"Sleep: {r.get('sleep_minutes','?')}min | "
            f"HRV: {r.get('hrv_rmssd','?')}ms | "
            f"RHR: {r.get('resting_heart_rate','?')}bpm | "
            f"SpO2: {r.get('spo2_avg','?')}% | "
            f"Steps: {r.get('steps','?')} | "
            f"AZM: {r.get('active_zone_minutes','?')}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# PART 4 — CLAUDE ANALYSIS
# ─────────────────────────────────────────────────────────────

USER_PROFILE = """
PERSONAL PROFILE:
- Age: 23
- Bodyweight: 180 lbs (81.6 kg)
- Training experience: Intermediate (1-3 years lifting)

GOALS:
- Primary: Hypertrophy (clean bulk — slight caloric surplus, minimal fat gain)
- Cardiovascular: Daily cardio to support heart health and bring borderline blood pressure to normal range
- Long-term: Establish consistent HRV and RHR baselines in the healthy range

TRAINING SCHEDULE:
- 4 days/week lifting: Legs, Back, Chest, Arms rotation (hypertrophy rep ranges 8-12)
- Daily: 10,000 steps minimum
- No full rest days — active recovery on non-lifting days

LIFESTYLE CONTEXT:
- High stress, demanding schedule — cortisol management is relevant to recovery interpretation
- Sedentary desk job — steps and active zone minutes are especially important to monitor
- Target sleep: 8+ hours — flag any night under 7.5 hours as a recovery risk

SUPPLEMENTS:
- Creatine — supports strength and muscle volumization, may slightly elevate RHR in some individuals
- Protein powder — dietary support for muscle protein synthesis
- Magnesium — supports sleep quality and parasympathetic recovery; note if sleep efficiency is low
- Caffeine / coffee — factor into sleep efficiency interpretation; flag if poor sleep may be caffeine-related

NUTRITION:
- Clean bulk: slight caloric surplus
- Prioritise protein intake for hypertrophy
- Note if caloric strategy should be adjusted based on recovery trends

BLOOD PRESSURE:
- Currently borderline — slightly above normal range
- Key interventions: consistent Zone 2 cardio, stress management, sleep quality, HRV improvement
- Flag any metrics that are working against BP goal
"""

def build_prompt(today_metrics, rolling_summary, avg_hrv, avg_rhr):
    today = safe_dict(today_metrics)

    return f"""
{USER_PROFILE}

7-DAY BASELINES (REFERENCE):
- Average HRV: {avg_hrv} ms
- Average Resting Heart Rate: {avg_rhr} bpm

TODAY'S METRICS ({today.get('date')}):
- Sleep: {today.get('sleep_minutes', 'N/A')} min | Efficiency: {today.get('sleep_score', 'N/A')}%
- Sleep Stages: Deep {today.get('stage_deep', 'N/A')}min | REM {today.get('stage_rem', 'N/A')}min | Light {today.get('stage_light', 'N/A')}min
- HRV (RMSSD): {today.get('hrv_rmssd', 'N/A')} ms
- Resting Heart Rate: {today.get('resting_heart_rate', 'N/A')} bpm
- SpO2: {today.get('spo2_avg', 'N/A')}%
- Steps: {today.get('steps', 'N/A')}
- Active Zone Minutes: {today.get('active_zone_minutes', 'N/A')}
- Calories Out: {today.get('calories_out', 'N/A')}
- Distance: {today.get('distance_km', 'N/A')} km
- VO2 Max: {today.get('vo2_max', 'N/A')}

ROLLING HISTORY:
{rolling_summary}

Write my morning briefing with the following sections in this exact order.

Recovery & Activity Summary — ALWAYS INCLUDE
Rate recovery as Excellent, Good, Moderate, or Poor in the opening sentence. Then write one flowing paragraph that covers yesterday's activity and last night's recovery together — what the data shows, how it connects to recent trends, and what it means for the goals (cardiovascular health and blood pressure). Use the numbers but make them mean something. Clear and direct, not a list of observations.

Risk Flags — CONDITIONAL
Silently check every threshold below against today's exact metrics. If ANY single condition is met, you MUST include this section. Do not use judgment — these are hard rules.

Individual thresholds:
- Sleep was under 300 minutes (5 hours) → MUST flag
- HRV is 25% or more below the 7-day average ({avg_hrv}) → MUST flag
- RHR is 10+ bpm above the 7-day average ({avg_rhr}) → MUST flag
- SpO2 dropped below 93% → MUST flag
- HRV has declined for 3 or more consecutive days → MUST flag

Multi-metric flag — flag if 3 or more of the following are simultaneously true:
- HRV is 20% or more below your 7-day average
- RHR is 8+ bpm above your 7-day average
- Sleep under 5.5 hours (330 minutes)
- SpO2 below 94%

If triggered, write one concise paragraph explaining what the data is showing and why it matters. If zero conditions are met, do not include this section at all — not even a heading.

TONE: Data-informed and direct. No emojis. No markdown bold. Prose only. Write like a knowledgeable analyst reading the numbers, not a coach giving orders.
"""

def safe_dict(data):
    return data if isinstance(data, dict) else {}

def get_analysis(today_metrics, rolling_summary, avg_hrv, avg_rhr):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system="You are Cortex, a personal biometric intelligence system. Write in prose. No emojis.",
        messages=[
            {
                "role": "user",
                "content": build_prompt(today_metrics, rolling_summary, avg_hrv, avg_rhr)
            }
        ]
    )
    
    # This specifically finds the TextBlock and skips the ThinkingBlock
    for block in message.content:
        if hasattr(block, 'text'):
            return block.text
            
    return "Error: No text response generated."

# ─────────────────────────────────────────────────────────────
# PART 5 — EMAIL DELIVERY
# ─────────────────────────────────────────────────────────────

def format_analysis_to_html(analysis):
    sections = [
        "Recovery & Activity Summary",
        "Risk Flags",
    ]
    present_sections = [s for s in sections if s in analysis]
    html_sections = ""

    for i, section in enumerate(present_sections):
        try:
            start = analysis.index(section) + len(section)
            end = analysis.index(present_sections[i + 1]) if i + 1 < len(present_sections) else len(analysis)
            content = analysis[start:end].strip().replace("**", "").strip()

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            content_html = "".join([
                f'<p style="font-size:15px;color:#1a1a1a;line-height:1.8;margin:0 0 12px">{p}</p>'
                for p in paragraphs
            ])

            html_sections += f"""
            <tr>
              <td style="padding:28px 0 0">
                <p style="font-size:13px;font-weight:600;color:#000;margin:0 10px;font-family:Arial,sans-serif">{section.upper()}</p>
                <div style="border-top:2px solid #000;padding-top:14px">
                  {content_html}
                </div>
              </td>
            </tr>
            """
        except ValueError:
            continue
            
    return html_sections


def build_email_html(analysis, briefing_date):
    sections_html = format_analysis_to_html(analysis)
    try:
        d = datetime.strptime(briefing_date, "%Y-%m-%d")
        formatted_date = d.strftime("%A, %B %-d, %Y")
    except Exception:
        formatted_date = briefing_date

    return f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#ffffff;font-family:Georgia,serif">
      <table width="100%" cellpadding="0" cellspacing="0" style="background:#ffffff">
        <tr>
          <td align="center" style="padding:40px 20px">
            <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%">
              <tr>
                <td align="center" style="padding-bottom:24px">
                  <div style="display:inline-block;background:#000000;padding:10px 28px">
                    <span style="font-family:Arial,sans-serif;font-size:22px;font-weight:700;color:#ffffff;letter-spacing:3px">CORTEX</span>
                  </div>
                </td>
              </tr>
              <tr>
                <td align="center" style="padding-bottom:8px">
                  <h1 style="font-family:Georgia,serif;font-size:42px;font-weight:700;color:#000000;margin:0;line-height:1.1">Morning Briefing</h1>
                </td>
              </tr>
              <tr>
                <td align="center" style="padding-bottom:24px">
                  <p style="font-family:Arial,sans-serif;font-size:14px;font-weight:600;color:#000;margin:0">{formatted_date}</p>
                </td>
              </tr>
              <tr>
                <td style="border-bottom:2px solid #000;padding-bottom:24px"></td>
              </tr>
              <tr>
                <td align="center" style="padding:16px 0 8px">
                  <p style="font-family:Arial,sans-serif;font-size:11px;color:#999;margin:0;letter-spacing:1px;text-transform:uppercase">Personal Performance Intelligence</p>
                </td>
              </tr>
              <tr>
                <td>
                  <table width="100%" cellpadding="0" cellspacing="0">
                    {sections_html}
                  </table>
                </td>
              </tr>
              <tr>
                <td style="border-top:1px solid #e0e0e0;padding-top:24px;margin-top:40px">
                  <p style="font-family:Arial,sans-serif;font-size:11px;color:#aaa;text-align:center;margin:0;line-height:1.8">
                    Cortex — Personal Performance Intelligence<br>
                    Generated daily from your Fitbit biometrics
                  </p>
                </td>
              </tr>
            </table>
          </td>
        </tr>
      </table>
    </body>
    </html>
    """

def send_email(subject, analysis, briefing_date):
    # 1. Prepare the Multi-Part Message (Text + HTML)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECIPIENT

    # Attach Plain Text for backup
    msg.attach(MIMEText(analysis, "plain"))
    
    # Generate and Attach the High-End CORTEX HTML
    html_body = build_email_html(analysis, briefing_date)
    msg.attach(MIMEText(html_body, "html"))

    print(f"Connecting to Gmail SSL (Port 465)...")
    try:
        # 2. Execute the Secure Handshake
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            # Login using the 16-character App Password
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            
            # Send the multi-part message
            server.send_message(msg)
            
        print("✅ SUCCESS: Briefing accepted by Gmail. Check your 'Sent' folder.")
        
    except smtplib.SMTPAuthenticationError:
        print("❌ AUTH FAILED: Check your App Password or 2-Step Auth status.")
    except Exception as e:
        print(f"❌ SMTP ERROR: {e}")
        
# --- Run the pipeline ---

def run_morning_pipeline():
    today_str     = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    auth   = FitbitAuth()
    client = FitbitClient(auth)

    # Fetch all metrics — activity from yesterday, recovery metrics from today
    # Each fetch is wrapped individually so one failure doesn't crash the pipeline
    print(f"Syncing {today_str}...")
    def safe_fetch(name, fn, *args):
        try:
            result = fn(*args)
            print(f"  {name} OK")
            return result
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            return {}

    activity = safe_fetch("activity",       client.fetch_activity,       yesterday_str)
    sleep    = safe_fetch("sleep",          client.fetch_sleep,          today_str)
    hrv      = safe_fetch("hrv",            client.fetch_hrv,            today_str)
    rhr      = safe_fetch("rhr",            client.fetch_heart_rate,     today_str)
    spo2     = safe_fetch("spo2",           client.fetch_spo2,           today_str)
    br       = safe_fetch("breathing_rate", client.fetch_breathing_rate, today_str)
    vo2max   = safe_fetch("vo2max",         client.fetch_vo2max,         today_str)

    # Record is keyed to the activity date (yesterday).
    # Sleep/HRV/RHR/SpO2 are from the following night — the 1-day lag recovery window.
    combined_record = {
        "date": yesterday_str,
        **activity, **sleep, **hrv, **rhr, **spo2, **br, **vo2max
    }

    # PostgreSQL — store biometrics
    store_biometrics(combined_record)

    # Pinecone — store today's record, then retrieve history
    # Wrapped in try/except: Pinecone is deprecated in v2 and index may have stale config
    try:
        index  = init_pinecone()
        store_day(index, combined_record)
        recent = get_recent_days(index, n=7)
    except Exception as e:
        print(f"Pinecone unavailable ({e}) — skipping vector store and history retrieval.")
        index  = None
        recent = []
    hrv_vals = [r.get("hrv_rmssd")         for r in recent if (r.get("hrv_rmssd")         or -1) > 0]
    rhr_vals = [r.get("resting_heart_rate") for r in recent if (r.get("resting_heart_rate") or -1) > 0]

    avg_hrv = round(sum(hrv_vals) / len(hrv_vals), 1) if hrv_vals else "N/A"
    avg_rhr = round(sum(rhr_vals) / len(rhr_vals), 1) if rhr_vals else "N/A"

    rolling_summary = get_rolling_summary(index) if index else "No historical data available yet."

    # Generate analysis
    print("Calling Opus 4.6...")
    analysis = get_analysis(combined_record, rolling_summary, avg_hrv, avg_rhr)

    # Deliver
    print("Sending email...")
    send_email(f"Cortex — {yesterday_str}", analysis, yesterday_str)
    print("DONE.")

if __name__ == "__main__":
    run_morning_pipeline()
