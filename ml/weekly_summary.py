"""
Cortex ML — Weekly Coach Summary

Generates a personalised weekly summary by:
  1. Computing MAP and lifestyle stats for the completed week vs the prior week
  2. Identifying the top lifestyle driver (most important feature that changed)
  3. Calling Claude Haiku to write a coach-style narrative
  4. Storing the result in weekly_summaries

Called as Stage 6 of the daily ML pipeline. Generates only on Mondays
(or any day the current week's summary is missing).
Requires ANTHROPIC_API_KEY in the environment.
"""

import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import psycopg2

DATABASE_URL    = os.environ["DATABASE_URL"]
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"
SUMMARY_TOKENS  = 280

COL_LABELS: dict[str, str] = {
    "sleep_duration_min":       "sleep duration",
    "deep_sleep_min":           "deep sleep",
    "rem_sleep_min":            "REM sleep",
    "awake_min":                "time awake during sleep",
    "steps":                    "daily steps",
    "active_zone_min":          "active zone minutes",
    "calories_burned":          "calories burned",
    "sodium_mg":                "sodium intake",
    "protein_g":                "protein intake",
    "magnesium_mg":             "magnesium",
    "alcohol_units":            "alcohol",
    "caffeine_mg":              "caffeine",
    "hrv_ms":                   "heart rate variability",
    "rhr_bpm":                  "resting heart rate",
    "vo2_max":                  "VO2 max",
    # lag-1 versions
    "sleep_duration_min_lag1":  "sleep duration",
    "deep_sleep_min_lag1":      "deep sleep",
    "rem_sleep_min_lag1":       "REM sleep",
    "steps_lag1":               "daily steps",
    "active_zone_min_lag1":     "active zone minutes",
    "sodium_mg_lag1":           "sodium intake",
    "protein_g_lag1":           "protein intake",
    "magnesium_mg_lag1":        "magnesium",
    "alcohol_units_lag1":       "alcohol",
    "caffeine_mg_lag1":         "caffeine",
    "am_map_lag1":              "morning blood pressure",
    "pm_map_lag1":              "prior-day evening blood pressure",
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _week_stats(df: pd.DataFrame, map_scores: pd.Series, week_start: date) -> dict:
    week_end = week_start + timedelta(days=6)
    mask = (df.index >= pd.Timestamp(week_start)) & (df.index <= pd.Timestamp(week_end))
    week_map = map_scores[mask].dropna()
    return {
        "week_start":  week_start,
        "map_avg":     float(week_map.mean()) if not week_map.empty else None,
        "n_readings":  len(week_map),
        "df":          df[mask],
    }


def _load_latest_top_features() -> list[dict]:
    sql = "SELECT top_features FROM ml_model_runs ORDER BY run_at DESC LIMIT 1"
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
    except Exception:
        return []
    finally:
        conn.close()
    if not row or not row[0]:
        return []
    raw = row[0]
    return raw if isinstance(raw, list) else json.loads(raw)


def _top_driver(
    this_df: pd.DataFrame,
    last_df: pd.DataFrame,
    top_features: list[dict],
) -> dict | None:
    """
    Identify the variable most responsible for the MAP shift.
    Score = model_importance × |relative_change|.
    """
    if this_df.empty or last_df.empty or not top_features:
        return None

    best: dict | None = None
    best_score = 0.0

    for feat in top_features[:20]:
        fname     = feat["feature"]
        importance = float(feat["importance"])

        if fname not in this_df.columns or fname not in last_df.columns:
            continue

        this_vals = this_df[fname].dropna()
        last_vals = last_df[fname].dropna()
        if this_vals.empty or last_vals.empty:
            continue

        this_mean = float(this_vals.mean())
        last_mean = float(last_vals.mean())
        if abs(last_mean) < 1e-9:
            continue

        rel_change = abs(this_mean - last_mean) / abs(last_mean)
        score = importance * rel_change

        if score > best_score:
            best_score = score
            best = {
                "feature":  fname,
                "this_val": this_mean,
                "last_val": last_mean,
                "delta":    this_mean - last_mean,
            }

    return best


def _fmt(feature: str, value: float) -> str:
    if "min" in feature and "pct" not in feature:
        h, m = divmod(int(round(value)), 60)
        return f"{h}h {m:02d}m"
    if "_g" in feature:
        return f"{value:.1f}g"
    if "_mg" in feature:
        return f"{value:.0f}mg"
    if "_mcg" in feature:
        return f"{value:.0f}mcg"
    if "_pct" in feature:
        return f"{value:.1f}%"
    if "calories" in feature:
        return f"{int(value)} kcal"
    if "steps" in feature:
        return f"{int(value):,}"
    if "_km" in feature:
        return f"{value:.1f} km"
    if "map" in feature or "bpm" in feature:
        return f"{value:.1f} mmHg"
    return f"{value:.1f}"


# ─────────────────────────────────────────────────────────────
# NARRATIVE GENERATION
# ─────────────────────────────────────────────────────────────

def _build_prompt(
    readings_this: int,
    map_this: float,
    map_last: float | None,
    driver: dict | None,
) -> str:
    map_delta = (map_this - map_last) if map_last is not None else None

    if map_delta is None:
        map_line = f"Average MAP this week: {map_this:.1f} mmHg (first week of data)."
    elif map_delta < -0.5:
        map_line = f"Average MAP: {map_this:.1f} mmHg — down {abs(map_delta):.1f} from last week."
    elif map_delta > 0.5:
        map_line = f"Average MAP: {map_this:.1f} mmHg — up {map_delta:.1f} from last week."
    else:
        map_line = f"Average MAP: {map_this:.1f} mmHg — stable vs last week."

    driver_line = ""
    if driver:
        label     = COL_LABELS.get(driver["feature"], driver["feature"].replace("_lag1", "").replace("_", " "))
        this_fmt  = _fmt(driver["feature"], driver["this_val"])
        last_fmt  = _fmt(driver["feature"], driver["last_val"])
        driver_line = f"Top lifestyle driver: {label} — {this_fmt} this week vs {last_fmt} last week."

    return f"""You are a personal health coach for someone managing pre-hypertension.

Write a 3–4 sentence weekly summary as a direct message to them. Warm, specific, coach-like. No bullet points, no headers.

Data:
- PM readings logged: {readings_this}/7
- {map_line}
{driver_line}

Rules:
- Open with the MAP outcome in plain English (improved / worsened / stable)
- If there is a top driver, connect it specifically to the MAP result
- Close with one specific, achievable ask for the coming week
- Never repeat mmHg more than once
- Max 4 sentences"""


def _call_claude(
    readings_this: int,
    map_this: float,
    map_last: float | None,
    driver: dict | None,
) -> str:
    prompt = _build_prompt(readings_this, map_this, map_last, driver)
    try:
        import anthropic
        client  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=SUMMARY_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        print(f"[weekly_summary] Claude API failed ({exc}) — using template fallback")
        return _template_narrative(readings_this, map_this, map_last, driver)


def _template_narrative(
    readings_this: int,
    map_this: float,
    map_last: float | None,
    driver: dict | None,
) -> str:
    map_delta = (map_this - map_last) if map_last is not None else None

    if map_delta is None:
        outcome = f"First week of data in — your average MAP came in at {map_this:.1f} mmHg."
    elif map_delta < -0.5:
        outcome = f"Good week — MAP dropped {abs(map_delta):.1f} mmHg to {map_this:.1f}."
    elif map_delta > 0.5:
        outcome = f"MAP nudged up {map_delta:.1f} mmHg to {map_this:.1f} this week."
    else:
        outcome = f"MAP held steady at {map_this:.1f} mmHg — consistency counts."

    driver_sentence = ""
    ask = "Keep logging your PM reading every evening — the more data, the sharper the insights."

    if driver:
        label     = COL_LABELS.get(driver["feature"], driver["feature"].replace("_lag1", "").replace("_", " "))
        this_fmt  = _fmt(driver["feature"], driver["this_val"])
        last_fmt  = _fmt(driver["feature"], driver["last_val"])
        driver_sentence = f" The standout factor was {label}: {this_fmt} this week vs {last_fmt} the week before."

        fname = driver["feature"]
        went_up = driver["delta"] > 0
        if "sleep" in fname and not went_up:
            ask = "Aim to be in bed 30 minutes earlier every night this week."
        elif "sodium" in fname and went_up:
            ask = "Try keeping sodium under 2,000 mg daily this week."
        elif "alcohol" in fname and went_up:
            ask = "Limit alcohol to one unit or fewer on weeknights this week."
        elif "steps" in fname and not went_up:
            ask = "Hit 8,000 steps every day this week — even a short evening walk counts."
        elif "magnesium" in fname and not went_up:
            ask = "Boost magnesium this week — leafy greens, nuts, seeds, or a supplement."
        elif "caffeine" in fname and went_up:
            ask = "Cut off caffeine before 2 pm every day this week and see if it helps."

    compliance = (
        f" You logged {readings_this}/7 PM readings."
        if readings_this < 7
        else " Solid logging — 7/7 PM readings this week."
    )

    return f"{outcome}{driver_sentence}{compliance} This week: {ask}"


# ─────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────

def _summary_exists(week_start: date) -> bool:
    sql = "SELECT 1 FROM weekly_summaries WHERE week_start = %s LIMIT 1"
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (week_start,))
            return cur.fetchone() is not None
    except Exception:
        return False
    finally:
        conn.close()


def _save(
    week_start: date,
    map_this: float,
    map_last: float | None,
    readings_this: int,
    top_driver: str | None,
    top_driver_label: str | None,
    narrative: str,
) -> int:
    sql = """
        INSERT INTO weekly_summaries
            (week_start, map_avg_this, map_avg_last, map_delta,
             readings_this, top_driver, top_driver_label, narrative)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (week_start) DO UPDATE SET
            map_avg_this     = EXCLUDED.map_avg_this,
            map_avg_last     = EXCLUDED.map_avg_last,
            map_delta        = EXCLUDED.map_delta,
            readings_this    = EXCLUDED.readings_this,
            top_driver       = EXCLUDED.top_driver,
            top_driver_label = EXCLUDED.top_driver_label,
            narrative        = EXCLUDED.narrative,
            created_at       = NOW()
        RETURNING id
    """
    map_delta = round(float(map_this) - float(map_last), 2) if map_last is not None else None
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    week_start, round(map_this, 2), map_last,
                    map_delta, readings_this, top_driver, top_driver_label, narrative,
                ))
                return cur.fetchone()[0]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def generate(df: pd.DataFrame, map_scores: pd.Series) -> dict | None:
    """
    Generate the weekly coach summary for the most recently completed week.

    Only runs on Mondays, or when the summary for that week is missing.
    Returns the summary dict on success, None on skip or insufficient data.
    Never raises — pipeline failures here must not crash the orchestrator.
    """
    today       = date.today()
    last_monday = _monday_of(today) - timedelta(weeks=1)

    is_monday      = today.weekday() == 0
    already_exists = _summary_exists(last_monday)

    if not is_monday and already_exists:
        print(f"[weekly_summary] Skipping — not Monday and summary for {last_monday} already exists.")
        return None

    last_week  = _week_stats(df, map_scores, last_monday)
    prior_week = _week_stats(df, map_scores, last_monday - timedelta(weeks=1))

    if last_week["map_avg"] is None:
        print(f"[weekly_summary] No MAP readings for week of {last_monday} — skipping.")
        return None

    top_features  = _load_latest_top_features()
    driver        = _top_driver(last_week["df"], prior_week["df"], top_features)
    driver_label  = (
        COL_LABELS.get(driver["feature"], driver["feature"].replace("_lag1", "").replace("_", " "))
        if driver else None
    )

    narrative = _call_claude(
        readings_this=last_week["n_readings"],
        map_this=last_week["map_avg"],
        map_last=prior_week["map_avg"],
        driver=driver,
    )

    summary_id = _save(
        week_start=last_monday,
        map_this=last_week["map_avg"],
        map_last=prior_week["map_avg"],
        readings_this=last_week["n_readings"],
        top_driver=driver["feature"] if driver else None,
        top_driver_label=driver_label,
        narrative=narrative,
    )

    print(f"[weekly_summary] id={summary_id}  week={last_monday}  "
          f"MAP={last_week['map_avg']:.1f}  readings={last_week['n_readings']}/7")
    if driver_label:
        print(f"  Top driver: {driver_label}")

    return {
        "id":              summary_id,
        "week_start":      last_monday,
        "map_avg_this":    last_week["map_avg"],
        "map_avg_last":    prior_week["map_avg"],
        "readings_this":   last_week["n_readings"],
        "top_driver_label": driver_label,
        "narrative":       narrative,
    }
