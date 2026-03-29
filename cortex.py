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

# Workout context — passed in via GitHub Actions input or environment variable
LAST_SESSION         = os.environ.get("LAST_SESSION", "")
SESSION_NOTES        = os.environ.get("SESSION_NOTES", "")


# ─────────────────────────────────────────────────────────────
# PART 1 — FITBIT AUTH
# ─────────────────────────────────────────────────────────────

class FitbitAuth:
    AUTH_URL  = "https://www.fitbit.com/oauth2/authorize"
    TOKEN_URL = "https://api.fitbit.com/oauth2/token"
    SCOPES    = "sleep heartrate activity oxygen_saturation profile"

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
# PART 1 — FITBIT DATA FETCHING
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
        return {
            "sleep_minutes": summary.get("totalMinutesAsleep"),
            "time_in_bed":   summary.get("totalTimeInBed"),
            "sleep_score":   entries[0].get("efficiency") if entries else None,
            "stage_deep":    stages.get("deep"),
            "stage_rem":     stages.get("rem"),
            "stage_light":   stages.get("light"),
            "stage_wake":    stages.get("wake"),
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
        data = self._get(f"/1/user/-/spo2/date/{d}.json")
        return {"spo2_avg": data.get("value", {}).get("avg")}

    def fetch_activity(self, d):
        data    = self._get(f"/1/user/-/activities/date/{d}.json")
        summary = data.get("summary", {})
        return {
            "steps":                   summary.get("steps"),
            "calories_out":            summary.get("caloriesOut"),
            "active_zone_minutes":     summary.get("activeZoneMinutes", {}).get("totalMinutes"),
            "very_active_minutes":     summary.get("veryActiveMinutes"),
            "fairly_active_minutes":   summary.get("fairlyActiveMinutes"),
            "sedentary_minutes":       summary.get("sedentaryMinutes"),
            "distance_km":             next((d["distance"] for d in summary.get("distances", []) if d["activity"] == "total"), None),
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
# PART 2 — PINECONE RAG
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
# PART 3 — CLAUDE ANALYSIS
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
- 4 days/week lifting in this exact rotation: Legs, Back, Chest, Arms (1 hr each, hypertrophy rep ranges 8-12)
- Based on yesterday's logged session, determine which muscle group is due today
- Daily: 10,000 steps minimum regardless of training day
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

SYSTEM_PROMPT = """You are a personal fitness coach and sports science analyst with deep expertise in:
- HRV interpretation and recovery science
- Hypertrophy training programming
- Blood pressure reduction through exercise
- Sleep quality and its impact on performance
- Interpreting Fitbit wearable data

CORE ANALYTIC RULES:
1. STEP DEFICIT: If yesterday's steps are under 10,000, you must prioritize movement in the Action Items.
2. BP MANAGEMENT: If today's RHR is 5% or more above the 7-day average, you must recommend Zone 2 cardio for blood pressure management.
3. RECOVERY MISMATCH: If Sleep is high but HRV is low, prioritize CNS fatigue in your 'Full Picture' analysis.

You write in a direct, professional tone. No emojis. No fluff.
Write in clear prose paragraphs, not bullet points, except for the Action Items section.
Your analysis should read like it came from a highly informed human coach, not an AI chatbot.
Do not use asterisks or any markdown formatting anywhere in your response. Write section headers as plain text only."""


def build_prompt(today_metrics, rolling_summary, workout_context=""):

    return f"""
{USER_PROFILE}

YESTERDAY'S TRAINING:
{workout_context if workout_context else "No session logged."}

TODAY'S METRICS ({today_metrics.get('date')}):
- Sleep: {today_metrics.get('sleep_minutes', 'N/A')} min | Efficiency: {today_metrics.get('sleep_score', 'N/A')}%
- Sleep Stages: Deep {today_metrics.get('stage_deep', 'N/A')}min | REM {today_metrics.get('stage_rem', 'N/A')}min | Light {today_metrics.get('stage_light', 'N/A')}min
- HRV (RMSSD): {today_metrics.get('hrv_rmssd', 'N/A')} ms
- Resting Heart Rate: {today_metrics.get('resting_heart_rate', 'N/A')} bpm
- SpO2: {today_metrics.get('spo2_avg', 'N/A')}%
- Steps: {today_metrics.get('steps', 'N/A')}
- Active Zone Minutes: {today_metrics.get('active_zone_minutes', 'N/A')}
- Calories Out: {today_metrics.get('calories_out', 'N/A')}
- Distance: {today_metrics.get('distance_km', 'N/A')} km
- VO2 Max: {today_metrics.get('vo2_max', 'N/A')}

ROLLING HISTORY:
{rolling_summary}

Write my morning briefing with the following sections in this exact order. Only include sections marked as active for today.

Recovery Status — ALWAYS INCLUDE
Rate recovery as Excellent, Good, Moderate, or Poor. One sentence on the rating, then 2 sentences explaining why in plain terms. Use the numbers but make them mean something.

Training Recommendation — ALWAYS INCLUDE
Should I train hard, train light, or recover today? Which muscle group is due based on yesterday's logged session. Keep it practical and specific — what to do, how hard to push, and why. One short paragraph. No jargon.

The Full Picture — ALWAYS INCLUDE
This is the connective tissue of the briefing. Write one flowing paragraph that ties together what happened last night, how it connects to recent trends, and what it means for the goals — muscle building, cardiovascular health, and blood pressure. Use the data to tell a story, not list observations. A 20-year-old should read this and immediately understand what is going on with their body and why it matters. Always include a specific note on blood pressure progress. Keep it conversational but intelligent. No bullet points.

Risk Flags — CONDITIONAL
Before writing any other section, silently check every threshold below against today's exact metrics. If ANY single condition is met, you MUST include this section. Do not interpret or use judgment — these are hard rules.

Individual thresholds:
- Sleep was under 300 minutes (5 hours) → MUST flag
- HRV is 25% or more below the 7-day average → MUST flag
- RHR is 10+ bpm above the 7-day average → MUST flag
- SpO2 dropped below 93% → MUST flag
- HRV has declined for 3 or more consecutive days → MUST flag

Multi-metric flag — flag if 3 or more of the following are simultaneously true:
- HRV is 20% or more below your 7-day average
- RHR is 8+ bpm above your 7-day average
- Sleep under 5.5 hours (330 minutes)
- SpO2 below 94%

If triggered, write one concise paragraph explaining what the data is showing and why it matters today. If zero conditions are met, do not include this section at all — not even a heading.

Action Items — ALWAYS INCLUDE
Exactly 4 specific things I should do today. Numbered list. No emojis. Terse and direct. 

LOGIC FOR THIS SECTION:
- If steps < 10,000, Item #1 must be a specific time to walk today.
- If RHR is elevated >5% vs History, Item #2 must be a 20-min Zone 2 session for BP.
- Always include one item for cardiovascular health.
- If data is 'N/A', provide high-quality general coaching based on the goals.

TONE: Write like a knowledgeable coach who has access to your biometric data. Smart and data-informed, but clear and direct. Never sacrifice clarity for technical precision. No emojis. No markdown bold. Prose only except Action Items.
"""

# Helper to prevent unpacking crashes
def safe_dict(data):
    return data if isinstance(data, dict) else {}

def get_analysis(today_metrics, rolling_summary, workout_context=""):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-3-5-opus-20240229", 
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(today_metrics, rolling_summary, workout_context)}]
    )
    return message.content[0].text

# ─────────────────────────────────────────────────────────────
# PART 3 — EMAIL DELIVERY
# ─────────────────────────────────────────────────────────────

def format_analysis_to_html(analysis):
    sections = [
        "Recovery Status",
        "Training Recommendation",
        "The Full Picture",
        "Risk Flags",
        "Action Items",
    ]
    present_sections = [s for s in sections if s in analysis]
    html_sections    = ""

    for i, section in enumerate(present_sections):
        start = analysis.index(section) + len(section)
        end   = analysis.index(present_sections[i + 1]) if i + 1 < len(present_sections) else len(analysis)
        content = analysis[start:end].strip().replace("**", "").strip()

        if section == "Action Items":
            items = re.findall(r'\d+\.?\s+(.+?)(?=\d+\.|$)', content, re.DOTALL)
            items = [item.strip() for item in items if item.strip()]
            if items:
                list_html    = "".join([
                    f'<tr><td style="padding:10px 0;border-bottom:1px solid #f0f0f0;font-size:15px;color:#1a1a1a;line-height:1.6">'
                    f'<span style="font-weight:600;margin-right:8px;color:#000">{j+1}.</span>{item}</td></tr>'
                    for j, item in enumerate(items)
                ])
                content_html = f'<table style="width:100%;border-collapse:collapse">{list_html}</table>'
            else:
                content_html = f'<p style="font-size:15px;color:#1a1a1a;line-height:1.8;margin:0">{content}</p>'
        else:
            paragraphs   = [p.strip() for p in content.split("\n\n") if p.strip()]
            content_html = "".join([
                f'<p style="font-size:15px;color:#1a1a1a;line-height:1.8;margin:0 0 12px">{p}</p>'
                for p in paragraphs
            ])

        html_sections += f"""
        <tr>
          <td style="padding:28px 0 0">
            <p style="font-size:13px;font-weight:600;color:#000;margin:0 0 10px;font-family:Arial,sans-serif">{section}</p>
            <div style="border-top:2px solid #000;padding-top:14px">
              {content_html}
            </div>
          </td>
        </tr>
"""
    return html_sections


def build_email_html(analysis, briefing_date):
    sections_html = format_analysis_to_html(analysis)
    try:
        d              = datetime.strptime(briefing_date, "%Y-%m-%d")
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
    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT
    msg.attach(MIMEText(analysis, "plain"))
    msg.attach(MIMEText(build_email_html(analysis, briefing_date), "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
    print(f"Briefing sent to {EMAIL_RECIPIENT}")


# --- Run the pipeline ---

def run_morning_pipeline():
    # 1. Define dates
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    auth = FitbitAuth()
    client = FitbitClient(auth)
    
    # 2. Fetch Data with safety checks
    # We wrap these in safe_dict so if a call returns None, the script doesn't crash
    activity_data = safe_dict(client.fetch_activity(yesterday_str))
    sleep_data    = safe_dict(client.fetch_sleep(today_str))
    hrv_data      = safe_dict(client.fetch_hrv(today_str))
    spo2_data     = safe_dict(client.fetch_spo2(today_str))
    rhr_data      = safe_dict(client.fetch_heart_rate(today_str))

    # 3. Combine into a single record
    # This record is what is stored in Pinecone and sent to Claude
    combined_record = {
        "date": today_str,
        **activity_data,
        **sleep_data,
        **hrv_data,
        **spo2_data,
        **rhr_data
    }

    # 4. Handle Missing Data
    # If HRV or Sleep is missing (common if watch didn't sync), we don't want to store 0s
    if combined_record.get("hrv_rmssd") is None:
        print("!! WARNING: HRV data is missing. Ensure Fitbit app is synced.")

    # 5. Pinecone Sync
    try:
        index = init_pinecone()
        store_day(index, combined_record)
        # Pull 7 days of context so Claude can see the 5% RHR spikes
        history = get_rolling_summary(index, 7)
    except Exception as e:
        print(f"Pinecone Error: {e}")
        history = "Historical data unavailable due to database error."

    # 6. Analysis & Email
    # Pass the session notes from your GitHub Action environment variables
    session_info = f"{LAST_SESSION}: {SESSION_NOTES}" if LAST_SESSION else ""
    
    analysis = get_analysis(combined_record, history, session_info)
    
    subject = f"Cortex Briefing — {datetime.now().strftime('%A, %b %d')}"
    send_email(subject, analysis, today_str)

if __name__ == "__main__":
    run_morning_pipeline()
