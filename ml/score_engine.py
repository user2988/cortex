"""
Cortex ML — Score Engine

Computes Sleep Score (0–100) and Heart Score (0–100) for every date in the
biometrics table. Both scores are relative to the individual's own rolling
30-day baseline: a score of 75 means "better than 75% of your own recent days."

Sleep Score components  (weights: duration 30%, deep% 25%, rem% 25%, efficiency 20%)
    sleep_duration_min, deep_sleep_min/sleep_duration_min,
    rem_sleep_min/sleep_duration_min, sleep_efficiency_pct

Heart Score components  (weights: HRV 40%, RHR 40% inverted, SpO2 20%)
    hrv_ms, rhr_bpm (lower = better), spo2_avg_pct

Requires at least MIN_WINDOW days of prior data before a score is emitted.
Scores are upserted into daily_scores on every pipeline run.
"""

import os
import psycopg2
import numpy as np
import pandas as pd
from datetime import date

DATABASE_URL = os.environ["DATABASE_URL"]

MIN_WINDOW     = 7
ROLLING_WINDOW = 30

SLEEP_WEIGHTS = {"duration": 0.30, "deep_pct": 0.25, "rem_pct": 0.25, "efficiency": 0.20}
HEART_WEIGHTS = {"hrv": 0.40, "rhr": 0.40, "spo2": 0.20}


# ─────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────

def _load_biometrics() -> pd.DataFrame:
    sql = """
        SELECT date,
               sleep_duration_min, deep_sleep_min, rem_sleep_min,
               sleep_efficiency_pct,
               hrv_ms, rhr_bpm, spo2_avg_pct
        FROM biometrics
        ORDER BY date ASC
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Derived sleep percentage metrics
    dur = df["sleep_duration_min"].replace(0, np.nan)
    df["deep_pct"] = (df["deep_sleep_min"] / dur * 100).where(dur.notna())
    df["rem_pct"]  = (df["rem_sleep_min"]  / dur * 100).where(dur.notna())

    return df


# ─────────────────────────────────────────────────────────────
# PERCENTILE SCORING
# ─────────────────────────────────────────────────────────────

def _rolling_percentile(series: pd.Series, window: int, higher_is_better: bool) -> pd.Series:
    """
    For each date, compute what percentile the value falls in relative to
    the preceding `window` days (exclusive of the current date).
    Returns NaN when fewer than MIN_WINDOW prior values are available.
    """
    scores = pd.Series(index=series.index, dtype=float)
    values = series.values

    for i in range(len(series)):
        start = max(0, i - window)
        window_vals = values[start:i]
        window_vals = window_vals[~np.isnan(window_vals)]

        if len(window_vals) < MIN_WINDOW or np.isnan(values[i]):
            scores.iloc[i] = np.nan
            continue

        if higher_is_better:
            pct = float(np.mean(window_vals < values[i])) * 100
        else:
            pct = float(np.mean(window_vals > values[i])) * 100

        scores.iloc[i] = round(pct, 1)

    return scores


# ─────────────────────────────────────────────────────────────
# SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute sleep and heart scores for all available dates.

    Returns a DataFrame indexed by date with columns:
        sleep_score, heart_score,
        duration_score, deep_score, rem_score, efficiency_score,
        hrv_score, rhr_score, spo2_score,
        sleep_duration_min, deep_pct, rem_pct,
        hrv_ms, rhr_bpm, spo2_avg_pct
    """
    if df is None:
        df = _load_biometrics()
    if df.empty:
        return pd.DataFrame()

    W = ROLLING_WINDOW

    # ── Sleep components ─────────────────────────────────────
    dur_score  = _rolling_percentile(df["sleep_duration_min"],  W, higher_is_better=True)
    deep_score = _rolling_percentile(df["deep_pct"],            W, higher_is_better=True)
    rem_score  = _rolling_percentile(df["rem_pct"],             W, higher_is_better=True)
    eff_score  = _rolling_percentile(df["sleep_efficiency_pct"],W, higher_is_better=True)

    # Weighted sleep score — only if all components present
    sw = SLEEP_WEIGHTS
    sleep_components = pd.DataFrame({
        "duration":   dur_score  * sw["duration"],
        "deep_pct":   deep_score * sw["deep_pct"],
        "rem_pct":    rem_score  * sw["rem_pct"],
        "efficiency": eff_score  * sw["efficiency"],
    })
    sleep_score = sleep_components.sum(axis=1).where(sleep_components.notna().all(axis=1))

    # ── Heart components ─────────────────────────────────────
    hrv_score  = _rolling_percentile(df["hrv_ms"],       W, higher_is_better=True)
    rhr_score  = _rolling_percentile(df["rhr_bpm"],      W, higher_is_better=False)
    spo2_score = _rolling_percentile(df["spo2_avg_pct"], W, higher_is_better=True)

    hw = HEART_WEIGHTS
    heart_components = pd.DataFrame({
        "hrv":  hrv_score  * hw["hrv"],
        "rhr":  rhr_score  * hw["rhr"],
        "spo2": spo2_score * hw["spo2"],
    })
    heart_score = heart_components.sum(axis=1).where(heart_components.notna().all(axis=1))

    result = pd.DataFrame({
        "sleep_score":      sleep_score.round(1),
        "heart_score":      heart_score.round(1),
        "duration_score":   dur_score.round(1),
        "deep_score":       deep_score.round(1),
        "rem_score":        rem_score.round(1),
        "efficiency_score": eff_score.round(1),
        "hrv_score":        hrv_score.round(1),
        "rhr_score":        rhr_score.round(1),
        "spo2_score":       spo2_score.round(1),
        "sleep_duration_min": df["sleep_duration_min"],
        "deep_pct":           df["deep_pct"].round(1),
        "rem_pct":            df["rem_pct"].round(1),
        "hrv_ms":             df["hrv_ms"],
        "rhr_bpm":            df["rhr_bpm"],
        "spo2_avg_pct":       df["spo2_avg_pct"],
    }, index=df.index)

    return result.dropna(subset=["sleep_score", "heart_score"], how="all")


# ─────────────────────────────────────────────────────────────
# PERSIST
# ─────────────────────────────────────────────────────────────

def upsert(scores: pd.DataFrame) -> int:
    """Upsert score rows into daily_scores. Returns number of rows written."""
    if scores.empty:
        return 0

    sql = """
        INSERT INTO daily_scores (
            date,
            sleep_score, heart_score,
            duration_score, deep_score, rem_score, efficiency_score,
            hrv_score, rhr_score, spo2_score,
            sleep_duration_min, deep_pct, rem_pct,
            hrv_ms, rhr_bpm, spo2_avg_pct
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (date) DO UPDATE SET
            sleep_score      = COALESCE(EXCLUDED.sleep_score,      daily_scores.sleep_score),
            heart_score      = COALESCE(EXCLUDED.heart_score,      daily_scores.heart_score),
            duration_score   = COALESCE(EXCLUDED.duration_score,   daily_scores.duration_score),
            deep_score       = COALESCE(EXCLUDED.deep_score,       daily_scores.deep_score),
            rem_score        = COALESCE(EXCLUDED.rem_score,        daily_scores.rem_score),
            efficiency_score = COALESCE(EXCLUDED.efficiency_score, daily_scores.efficiency_score),
            hrv_score        = COALESCE(EXCLUDED.hrv_score,        daily_scores.hrv_score),
            rhr_score        = COALESCE(EXCLUDED.rhr_score,        daily_scores.rhr_score),
            spo2_score       = COALESCE(EXCLUDED.spo2_score,       daily_scores.spo2_score),
            sleep_duration_min = COALESCE(EXCLUDED.sleep_duration_min, daily_scores.sleep_duration_min),
            deep_pct         = COALESCE(EXCLUDED.deep_pct,         daily_scores.deep_pct),
            rem_pct          = COALESCE(EXCLUDED.rem_pct,          daily_scores.rem_pct),
            hrv_ms           = COALESCE(EXCLUDED.hrv_ms,           daily_scores.hrv_ms),
            rhr_bpm          = COALESCE(EXCLUDED.rhr_bpm,          daily_scores.rhr_bpm),
            spo2_avg_pct     = COALESCE(EXCLUDED.spo2_avg_pct,     daily_scores.spo2_avg_pct),
            computed_at      = NOW()
    """

    rows = []
    for dt, row in scores.iterrows():
        def v(col):
            val = row.get(col)
            return None if (val is None or (isinstance(val, float) and np.isnan(val))) else val

        rows.append((
            dt.date() if hasattr(dt, "date") else dt,
            v("sleep_score"), v("heart_score"),
            v("duration_score"), v("deep_score"), v("rem_score"), v("efficiency_score"),
            v("hrv_score"), v("rhr_score"), v("spo2_score"),
            v("sleep_duration_min"), v("deep_pct"), v("rem_pct"),
            v("hrv_ms"), v("rhr_bpm"), v("spo2_avg_pct"),
        ))

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
    finally:
        conn.close()

    return len(rows)
