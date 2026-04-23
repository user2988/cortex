"""
Cortex ML — Component 1: Data Builder

Reads biometrics, nutrition, and blood-pressure session logs from
PostgreSQL, applies lag and smoothing transformations, and returns a
single clean modelling DataFrame.

Conventions
-----------
- Nutrition columns     : lagged 1 day  (yesterday's intake → today's BP)
- Sleep columns         : lagged 1 day  (last night's sleep → today's BP)
- Activity columns      : lagged 1 day  (yesterday's activity → today's BP)
- Prior AM / PM MAP     : lagged 1 day  (yesterday's BP → today's BP)
- Slow micronutrients   : 7-day rolling average applied *before* the 1-day lag
- Output (cardio) cols  : never lagged — kept for analysis but not used as
                          model features; target (PM MAP) comes from the
                          blood_pressure_logs table via bp_target.compute()
- Rows missing >50 % of feature columns are dropped
- Remaining nulls are imputed with each column's median
"""

import os
import psycopg2
import pandas as pd
import numpy as np

DATABASE_URL = os.environ["DATABASE_URL"]

# ─────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────

# Cardiovascular biometric outputs — kept in the DataFrame for Explorer /
# Insights but NEVER lagged and NOT used as model features.
# The actual ML target (PM MAP) comes from blood_pressure_logs.
OUTPUT_COLS = [
    "hrv_ms",
    "hrv_deep_rmssd",
    "rhr_bpm",
    "spo2_avg_pct",
    "spo2_min_pct",
    "spo2_max_pct",
    "respiratory_rate",
    "vo2_max",
]

# Sleep columns — lagged 1 day so last night's sleep quality is an input
# to today's blood-pressure prediction.
SLEEP_COLS = [
    "sleep_duration_min",
    "sleep_efficiency_pct",
    "deep_sleep_min",
    "rem_sleep_min",
    "light_sleep_min",
    "awake_min",
    "time_in_bed_min",
]

# Biometric inputs — activity metrics the user can influence.
# Lagged 1 day: yesterday's activity predicts today's recovery / BP.
ACTIVITY_COLS = [
    "steps",
    "active_zone_min",
    "very_active_min",
    "fairly_active_min",
    "lightly_active_min",
    "sedentary_min",
    "calories_burned",
    "distance_km",
    "time_in_fat_burn_min",
    "time_in_cardio_min",
    "time_in_peak_min",
]

# All nutrition input columns (excludes caffeine_last_time — non-numeric time string).
NUTRITION_COLS = [
    "calories_in", "protein_g", "carbs_g", "fat_g", "fibre_g",
    "sugar_g", "sodium_mg", "water_ml",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
    "trans_fat_g", "cholesterol_mg",
    "alcohol_units", "caffeine_mg",
    "omega3_mg", "omega6_mg", "ala_mg", "epa_mg", "dha_mg",
    "vitamin_a_mcg", "vitamin_d_iu", "vitamin_e_mg", "vitamin_k_mcg",
    "vitamin_c_mg", "thiamine_mg", "riboflavin_mg", "niacin_mg",
    "pantothenic_acid_mg", "vitamin_b6_mg", "biotin_mcg", "folate_mcg",
    "vitamin_b12_mcg",
    "calcium_mg", "iron_mg", "magnesium_mg", "phosphorus_mg",
    "potassium_mg", "zinc_mg",
    "selenium_mcg", "copper_mg", "manganese_mg", "chromium_mcg",
    "iodine_mcg", "molybdenum_mcg",
    "tryptophan_g", "threonine_g", "isoleucine_g", "leucine_g",
    "lysine_g", "methionine_g", "phenylalanine_g", "valine_g",
    "histidine_g", "alanine_g", "arginine_g", "aspartic_acid_g",
    "cystine_g", "glutamic_acid_g", "glycine_g", "proline_g",
    "serine_g", "tyrosine_g", "hydroxyproline_g",
]

# Micronutrients with slow-acting effects — stored in tissue, build up/deplete
# over days to weeks. Smoothed with a 7-day rolling average before lagging.
SLOW_MICRONUTRIENTS = [
    "vitamin_a_mcg", "vitamin_d_iu", "vitamin_e_mg", "vitamin_k_mcg",
    "omega3_mg", "omega6_mg", "ala_mg", "epa_mg", "dha_mg",
    "calcium_mg", "iron_mg", "magnesium_mg", "zinc_mg",
    "selenium_mcg", "copper_mg", "manganese_mg",
    "vitamin_b12_mcg", "folate_mcg", "biotin_mcg",
]

# BP feature column names (after merging from blood_pressure_logs)
BP_SESSION_COLS = ["am_map", "pm_map"]

MAX_NULL_FRACTION    = 0.50
SPARSE_COL_THRESHOLD = 0.70
CORRELATION_THRESHOLD = 0.95


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_raw() -> pd.DataFrame:
    """
    Load all biometrics and nutrition rows from PostgreSQL.

    Returns a DataFrame indexed by date. Rows where sleep_duration_min == 0
    are excluded (device failure / no-wear nights).
    """
    bio_cols   = OUTPUT_COLS + SLEEP_COLS + ACTIVITY_COLS
    bio_select = ", ".join(f"b.{c}" for c in bio_cols)
    nut_select = ", ".join(f"n.{c}" for c in NUTRITION_COLS)

    sql = f"""
        SELECT b.date, {bio_select}, {nut_select}
        FROM biometrics b
        LEFT JOIN nutrition n ON b.date = n.date
        ORDER BY b.date
    """

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[(df["sleep_duration_min"].isna()) | (df["sleep_duration_min"] > 0)]
    return df


def _load_bp_sessions() -> pd.DataFrame:
    """
    Load per-day AM and PM MAP from blood_pressure_logs.

    For each session, MAP is averaged across available readings.
    Returns a date-indexed DataFrame with columns am_map and pm_map.
    Returns an empty DataFrame if the table doesn't exist or has no data.
    """
    sql = """
        SELECT date, session,
               reading_1_systolic, reading_1_diastolic,
               reading_2_systolic, reading_2_diastolic
        FROM blood_pressure_logs
        WHERE reading_1_systolic IS NOT NULL AND reading_1_diastolic IS NOT NULL
        ORDER BY date, session
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])

    def _map(sys_val, dia_val):
        if sys_val is None or dia_val is None:
            return None
        return (float(sys_val) + 2.0 * float(dia_val)) / 3.0

    df["map_r1"] = df.apply(
        lambda r: _map(r["reading_1_systolic"], r["reading_1_diastolic"]), axis=1
    )
    df["map_r2"] = df.apply(
        lambda r: _map(r["reading_2_systolic"], r["reading_2_diastolic"]), axis=1
    )
    df["session_map"] = df[["map_r1", "map_r2"]].mean(axis=1, skipna=True)

    pivoted = df.pivot_table(index="date", columns="session", values="session_map", aggfunc="first")
    pivoted.columns.name = None

    result = pd.DataFrame(index=pivoted.index)
    result["am_map"] = pivoted.get("AM")
    result["pm_map"] = pivoted.get("PM")
    return result


# ─────────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────

def _apply_rolling(df: pd.DataFrame, cols: list[str], window: int = 7) -> pd.DataFrame:
    """Replace specified columns with their rolling mean (min_periods=3)."""
    present = [c for c in cols if c in df.columns]
    df[present] = df[present].rolling(window=window, min_periods=3).mean()
    return df


def _lag_cols(df: pd.DataFrame, cols: list[str], lag: int = 1) -> pd.DataFrame:
    """
    Shift specified columns forward by `lag` days and rename with a suffix.

    After shifting, col[t] contains the value from day t-lag, so the model
    can use yesterday's inputs to predict today's BP.
    """
    present = [c for c in cols if c in df.columns]
    lagged = df[present].shift(lag)
    lagged.columns = [f"{c}_lag{lag}" for c in present]
    return df.drop(columns=present).join(lagged)


# ─────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────

def _drop_sparse_rows(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Drop rows where more than MAX_NULL_FRACTION of feature columns are null."""
    present  = [c for c in feat_cols if c in df.columns]
    null_frac = df[present].isna().mean(axis=1)
    kept    = df[null_frac <= MAX_NULL_FRACTION]
    dropped = len(df) - len(kept)
    if dropped:
        print(f"  [data_builder] Dropped {dropped} sparse rows (>{MAX_NULL_FRACTION:.0%} nulls)")
    return kept


def _impute_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining nulls with each column's median."""
    medians = df.median(numeric_only=True)
    return df.fillna(medians)


def _drop_sparse_cols(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Drop feature columns missing in more than SPARSE_COL_THRESHOLD of rows."""
    present   = [c for c in feat_cols if c in df.columns]
    null_frac = df[present].isna().mean()
    to_drop   = null_frac[null_frac > SPARSE_COL_THRESHOLD].index.tolist()
    if to_drop:
        print(f"  [data_builder] Dropped {len(to_drop)} sparse columns "
              f"(>{SPARSE_COL_THRESHOLD:.0%} nulls): "
              f"{[c.replace('_lag1','') for c in to_drop]}")
        df = df.drop(columns=to_drop)
    return df


def _drop_correlated_cols(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Drop one of any feature pair where |r| > CORRELATION_THRESHOLD."""
    present = [c for c in feat_cols if c in df.columns]
    if len(present) < 2:
        return df

    corr  = df[present].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        partners = upper.index[upper[col] > CORRELATION_THRESHOLD].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            if df[col].notna().sum() >= df[partner].notna().sum():
                to_drop.add(partner)
            else:
                to_drop.add(col)
                break

    if to_drop:
        print(f"  [data_builder] Dropped {len(to_drop)} correlated columns "
              f"(|r|>{CORRELATION_THRESHOLD}): "
              f"{[c.replace('_lag1','') for c in sorted(to_drop)]}")
        df = df.drop(columns=list(to_drop))
    return df


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def build() -> pd.DataFrame:
    """
    Build the modelling DataFrame from PostgreSQL.

    Pipeline:
        1. Load raw biometrics + nutrition (joined on date)
        2. Apply 7-day rolling average to slow-acting micronutrients
        3. Lag all nutrition columns by 1 day
        4. Lag all sleep columns by 1 day
        5. Lag all activity columns by 1 day
        6. Load blood_pressure_logs; merge prior-day AM and PM MAP
        7. Drop rows with >50 % missing feature values
        8. Drop feature columns missing in >70 % of rows
        9. Drop one of any feature pair with |r| > 0.95
        10. Impute remaining nulls with column medians

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame. Feature columns are suffixed with ``_lag1``.
        OUTPUT_COLS remain unlagged (for analysis) but are excluded from
        the feature set used for model training.
        Returns an empty DataFrame if fewer than 2 rows are available.
    """
    print("[data_builder] Loading raw data...")
    df = _load_raw()
    print(f"  Raw rows: {len(df)}")

    if len(df) < 2:
        print("  Not enough data to build features.")
        return pd.DataFrame()

    # Smooth slow micronutrients before lagging
    df = _apply_rolling(df, SLOW_MICRONUTRIENTS, window=7)

    # Lag inputs: nutrition, sleep, activity
    df = _lag_cols(df, NUTRITION_COLS, lag=1)
    df = _lag_cols(df, SLEEP_COLS,     lag=1)
    df = _lag_cols(df, ACTIVITY_COLS,  lag=1)

    # Load prior-day AM and PM blood pressure MAPs
    bp_sessions = _load_bp_sessions()
    if not bp_sessions.empty:
        df = df.join(bp_sessions, how="left")
        bp_feat_present = [c for c in BP_SESSION_COLS if c in df.columns]
        if bp_feat_present:
            df = _lag_cols(df, bp_feat_present, lag=1)
            print(f"  BP session features merged: {[c + '_lag1' for c in bp_feat_present]}")

    # First row is always all-null for lagged columns — drop it
    df = df.iloc[1:]

    # Identify feature columns (everything except cardiovascular outputs)
    feat_cols = feature_cols(df)

    df = _drop_sparse_rows(df, feat_cols)
    df = _drop_sparse_cols(df, feat_cols)
    feat_cols = feature_cols(df)
    df = _drop_correlated_cols(df, feat_cols)
    feat_cols = feature_cols(df)
    df = _impute_medians(df)

    print(f"  Clean rows : {len(df)}")
    print(f"  Features   : {len(feat_cols)}")
    print(f"  Outputs    : {len([c for c in OUTPUT_COLS if c in df.columns])}")

    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the input feature column names from a built DataFrame."""
    return [c for c in df.columns if c not in OUTPUT_COLS]


def output_cols(df: pd.DataFrame) -> list[str]:
    """Return the cardiovascular output column names present in a built DataFrame."""
    return [c for c in OUTPUT_COLS if c in df.columns]


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df.head())
