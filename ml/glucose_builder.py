"""
Cortex ML — Component: Glucose Builder (v4)

Loads glucose_readings, meals, and medication regimens from PostgreSQL
and returns a date-indexed DataFrame of daily glucose aggregates plus
medication state flags. These become the target outputs for the
glucose-pivoted ML pipeline.

Degrades gracefully with sparse manual data:
  - fasting_glucose_mg_dl: one morning reading     -> populated
  - mean / TIR / CV      : >=3 readings/day needed -> NaN otherwise
  - post_meal_peak_avg   : >=1 meal-reading pair   -> NaN otherwise
  - post_meal_auc_avg    : CGM-dense only          -> NaN for Phase 0
  - dawn_phenomenon_delta: CGM-dense only          -> NaN for Phase 0

Medications are emitted as binary daily state flags (on_metformin,
on_glp1, ...). Cortex never recommends medication changes — these
are exogenous context for the model only.
"""

import pandas as pd
import numpy as np

from db import get_conn
from columns import GLUCOSE_OUTPUT_COLS, MEDICATION_CATEGORIES

# Clinical time-in-range window (non-diabetic target).
# Pre-diabetic / diabetic pipelines can widen to 70-180 later.
TIR_LOW  = 70.0
TIR_HIGH = 140.0

# Post-meal window for peak detection from CGM data.
POST_MEAL_WINDOW_MIN = 120

MED_FLAG_PREFIX = "on_"


# ─────────────────────────────────────────────────────────────
# DB LOADERS
# ─────────────────────────────────────────────────────────────

def _load_readings() -> pd.DataFrame:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts, mg_dl, source, meal_id
                FROM glucose_readings
                ORDER BY ts
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["ts"]    = pd.to_datetime(df["ts"], utc=True)
    df["mg_dl"] = pd.to_numeric(df["mg_dl"])
    df["date"]  = df["ts"].dt.date
    return df


def _load_meals() -> pd.DataFrame:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, ts, carbs_g, protein_g, fat_g, fibre_g, sugar_g, calories
                FROM meals
                ORDER BY ts
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["ts"]   = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    for c in ("carbs_g", "protein_g", "fat_g", "fibre_g", "sugar_g", "calories"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_medications() -> pd.DataFrame:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT category, start_date, end_date
                FROM medications
                ORDER BY start_date
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"]   = pd.to_datetime(df["end_date"])
    return df


# ─────────────────────────────────────────────────────────────
# DAILY AGGREGATION
# ─────────────────────────────────────────────────────────────

def _daily_readings_agg(readings: pd.DataFrame) -> pd.DataFrame:
    """Per-day fasting, mean, TIR, CV from raw readings."""
    empty_cols = [
        "fasting_glucose_mg_dl", "mean_glucose_mg_dl",
        "time_in_range_pct", "glucose_cv_pct",
    ]
    if readings.empty:
        return pd.DataFrame(columns=empty_cols)

    def agg(g: pd.DataFrame) -> pd.Series:
        mg = g["mg_dl"]
        n  = len(mg)

        # Fasting — prefer explicit manual_fasting tag, else earliest reading.
        fasting = g.loc[g["source"] == "manual_fasting", "mg_dl"]
        fasting_val = float(fasting.iloc[0]) if len(fasting) else float(mg.iloc[0])

        if n >= 3 and mg.mean():
            mean_val = float(mg.mean())
            tir_val  = float(((mg >= TIR_LOW) & (mg <= TIR_HIGH)).mean() * 100)
            cv_val   = float(mg.std(ddof=0) / mg.mean() * 100)
        else:
            mean_val = np.nan
            tir_val  = np.nan
            cv_val   = np.nan

        return pd.Series({
            "fasting_glucose_mg_dl": fasting_val,
            "mean_glucose_mg_dl"   : mean_val,
            "time_in_range_pct"    : tir_val,
            "glucose_cv_pct"       : cv_val,
        })

    return readings.groupby("date").apply(agg)


def _daily_meal_response(readings: pd.DataFrame,
                         meals: pd.DataFrame) -> pd.DataFrame:
    """
    Average post-meal peak per day. For each meal, use the linked
    manual_postmeal reading (by meal_id) or the max reading within
    the 2h post-meal window.
    """
    empty = pd.DataFrame(columns=["post_meal_peak_avg_mg_dl"])
    if meals.empty or readings.empty:
        return empty

    window = pd.Timedelta(minutes=POST_MEAL_WINDOW_MIN)
    peaks  = []
    for _, m in meals.iterrows():
        linked = readings[readings["meal_id"] == m["id"]]
        if not linked.empty:
            peak = float(linked["mg_dl"].max())
        else:
            t0, t1 = m["ts"], m["ts"] + window
            in_win = readings[(readings["ts"] >= t0) & (readings["ts"] <= t1)]
            if in_win.empty:
                continue
            peak = float(in_win["mg_dl"].max())
        peaks.append({"date": m["date"], "peak": peak})

    if not peaks:
        return empty
    df = pd.DataFrame(peaks)
    return df.groupby("date")["peak"].mean().to_frame("post_meal_peak_avg_mg_dl")


# ─────────────────────────────────────────────────────────────
# MEDICATION STATE FLAGS
# ─────────────────────────────────────────────────────────────

def _medication_flags(meds: pd.DataFrame,
                      dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Binary on_<category> flag per day for active regimens."""
    flags = pd.DataFrame(
        0,
        index=dates,
        columns=[f"{MED_FLAG_PREFIX}{c}" for c in MEDICATION_CATEGORIES],
        dtype=int,
    )
    if meds.empty or len(dates) == 0:
        return flags

    for _, row in meds.iterrows():
        col = f"{MED_FLAG_PREFIX}{row['category']}"
        if col not in flags.columns:
            continue  # unknown category — ignore rather than crash
        start = row["start_date"]
        end   = row["end_date"] if pd.notna(row["end_date"]) else dates.max()
        flags.loc[(dates >= start) & (dates <= end), col] = 1
    return flags


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def build() -> pd.DataFrame:
    """
    Build the daily glucose-outputs DataFrame.

    Returns a date-indexed DataFrame with every column in
    GLUCOSE_OUTPUT_COLS plus one on_<category> column per entry
    in MEDICATION_CATEGORIES. Empty if no readings exist.
    """
    print("[glucose_builder] Loading raw data...")
    readings = _load_readings()
    meals    = _load_meals()
    meds     = _load_medications()
    print(f"  Readings: {len(readings)}  Meals: {len(meals)}  Meds: {len(meds)}")

    if readings.empty:
        return pd.DataFrame(columns=GLUCOSE_OUTPUT_COLS)

    daily = _daily_readings_agg(readings)
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    peaks = _daily_meal_response(readings, meals)
    if not peaks.empty:
        peaks.index = pd.to_datetime(peaks.index)
        peaks.index.name = "date"

    out = daily.join(peaks, how="left")

    # Phase-2 metrics — need CGM-dense data.
    out["post_meal_auc_avg"]     = np.nan
    out["dawn_phenomenon_delta"] = np.nan

    out = out.reindex(columns=GLUCOSE_OUTPUT_COLS)

    flags = _medication_flags(meds, out.index)
    out   = out.join(flags)

    print(f"  Output days : {len(out)}")
    return out


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df.head())
