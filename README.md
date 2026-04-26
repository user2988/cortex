# Cortex

Personal biometrics analytics platform built on Fitbit data. Cortex pulls your daily health metrics, scores your sleep and cardiovascular health, surfaces statistically significant patterns, and lets you run your own experiments — all through a private Streamlit dashboard backed by PostgreSQL.

---

## Who is this for?

Anyone with a Fitbit who wants to go beyond the default app and actually understand their data. Cortex is built for people who want to ask questions like:

- Does more active zone time the day before improve my HRV?
- What activity level correlates with my best deep sleep?
- Is my sleep efficiency trending up or down over the past month?
- What does my cardiovascular health look like relative to my own baseline?

You own your data, you run your own pipeline, and you see insights tailored to your body — not population averages.

---

## What it tracks

Every metric comes directly from your Fitbit via its API:

**Sleep**
- Duration, efficiency, time in bed
- Deep sleep, REM, light sleep, awake time

**Recovery**
- HRV (RMSSD and deep RMSSD)
- Resting heart rate
- SpO2 (avg, min, max)
- Respiratory rate

**Activity** *(prior day)*
- Steps, distance, calories burned
- Active zone minutes, very/fairly/lightly active minutes, sedentary minutes
- Heart rate zones (fat burn, cardio, peak)
- VO2 max

---

## How it works

### Data pipeline
Fitbit data is fetched automatically via GitHub Actions and stored in a Neon PostgreSQL database. The pipeline runs 4× daily to catch delayed Fitbit syncs, and a nightly backfill job covers any gaps over the past 14 days. Existing records are never overwritten — only null values get filled in.

### Scoring
An ML pipeline runs daily and computes two scores relative to **your own rolling 30-day baseline** — not a population average:

- **Sleep Score (0–100)** — weighted from duration (30%), deep sleep % (25%), REM % (25%), efficiency (20%)
- **Heart Score (0–100)** — weighted from HRV (40%), resting heart rate (40%), SpO2 (20%)

Scores only appear after 7 days of data and get more accurate over time.

### Recommendations
The same pipeline identifies which activity ranges correlate with your best sleep and heart scores, then generates plain-English recommendations specific to your data.

### Weekly findings
Every Sunday a job scans your biometric history for statistically significant correlations (p < 0.05) across common metric pairs — steps vs HRV, sleep efficiency vs HRV, sedentary time vs resting heart rate, etc. The top findings surface on your Insights page.

### Experiments
You can define your own hypothesis tests in the app — pick an input metric, an output metric, a lag, and a duration. Cortex tracks the experiment and computes the correlation once you have enough data.

---

## Dashboard

Three pages built with Streamlit:

**Insights**
Your daily scores, component breakdowns, 30-day trend charts, top biometric findings, and activity recommendations.

**Experiments**
Create, track, and analyse your own hypothesis tests. See R², p-value, and coefficient for each experiment.

**Explorer**
Ad-hoc statistical analysis across any of your metrics:
- Pearson / Spearman correlation
- Lagged correlation (up to 3-day lag)
- Rolling average
- 30-day OLS trend
- Multiple OLS regression
- Anomaly detection
- 7-day forecast (Prophet)
- Seasonal decomposition

---

## Pipeline schedule

| Job | Schedule |
|-----|----------|
| Fitbit sync (`cortex.py`) | 9am, 11am, 2pm, 5pm EDT |
| 14-day backfill (`backfill.py`) | 11pm EDT |
| ML scores + recommendations | 2pm EDT |
| Weekly findings | Sunday 10am EDT |

All jobs can also be triggered manually via GitHub Actions `workflow_dispatch`.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data source | Fitbit Web API (OAuth 2.0 PKCE) |
| Database | PostgreSQL (Neon) |
| Pipeline | Python + GitHub Actions |
| ML / scoring | scipy, statsmodels, scikit-learn |
| Forecasting | Prophet |
| Dashboard | Streamlit + Plotly |
| Interpretations | Claude API (Anthropic) |

---

## Setup

1. **Fitbit developer app** — Create an app at [dev.fitbit.com](https://dev.fitbit.com) with OAuth 2.0 and the following scopes: `sleep heartrate activity oxygen_saturation cardio_fitness respiratory_rate profile`

2. **Database** — Provision a PostgreSQL instance (Neon recommended). Run `database/schema.sql` to initialise the schema.

3. **Bootstrap tokens** — Run `bootstrap_tokens.ipynb` locally once to complete the OAuth flow and generate your initial `fitbit_tokens.json`.

4. **GitHub secrets** — Add the following to your repository secrets:

   | Secret | Description |
   |--------|-------------|
   | `FITBIT_CLIENT_ID` | Fitbit app client ID |
   | `FITBIT_CLIENT_SECRET` | Fitbit app client secret |
   | `FITBIT_TOKENS` | Contents of `fitbit_tokens.json` after bootstrap |
   | `DATABASE_URL` | PostgreSQL connection string |
   | `PAT_TOKEN` | GitHub personal access token (for updating `FITBIT_TOKENS` secret) |
   | `ANTHROPIC_API_KEY` | Claude API key (for experiment interpretations) |

5. **Deploy** — Push to `main`. GitHub Actions handles the rest. Connect your Streamlit app to the same repo and database.

---

## Database schema

| Table | Written by | Purpose |
|-------|-----------|---------|
| `biometrics` | Daily pipeline + backfill | One row per day of Fitbit metrics |
| `daily_scores` | ML pipeline | Sleep Score and Heart Score components |
| `score_recommendations` | ML pipeline | Activity ranges tied to best scores |
| `findings` | Weekly job + Explorer | Significant biometric correlations |
| `experiments` | Streamlit app | User-defined hypothesis tests |
| `ml_pipeline_log` | ML pipeline | Pipeline run history |
