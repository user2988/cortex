"""
Cortex ML — Component 1: Data Builder

Reads biometrics and nutrition from PostgreSQL, applies lag and smoothing
transformations, and returns a single clean modelling DataFrame.

Conventions
-----------
- Nutrition columns  : lagged 1 day  (yesterday's intake → today's biometrics)
- Slow micronutrients: 7-day rolling average applied *before* the 1-day lag
- Activity columns   : lagged 1 day  (yesterday's activity → today's recovery)
- Output columns     : never lagged  (these are what the wellness score predicts)
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

# Biometric outputs — the target variables for the wellness score.
# These are NEVER lagged; they are what the model learns to predict.
OUTPUT_COLS = [
    "sleep_duration_min",
    "sleep_efficiency_pct",
    "deep_sleep_min",
    "rem_sleep_min",
    "light_sleep_min",
    "awake_min",
    "time_in_bed_min",
    "hrv_ms",
    "hrv_deep_rmssd",
    "rhr_bpm",
    "spo2_avg_pct",
    "spo2_min_pct",
    "spo2_max_pct",
    "respiratory_rate",
    "vo2_max",
]

# Biometric inputs — activity metrics the user can influence.
# Lagged 1 day: yesterday's activity predicts today's recovery.
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
# over days to weeks. Smoothed with a 7-day rolling average before lagging so
# the model sees the accumulated exposure level rather than a single-day spike.
SLOW_MICRONUTRIENTS = [
    # Fat-soluble vitamins (stored in adipose/liver)
    "vitamin_a_mcg", "vitamin_d_iu", "vitamin_e_mg", "vitamin_k_mcg",
    # Omega fatty acids (incorporated into cell membranes over 1-4 weeks)
    "omega3_mg", "omega6_mg", "ala_mg", "epa_mg", "dha_mg",
    # Minerals with meaningful storage pools
    "calcium_mg", "iron_mg", "magnesium_mg", "zinc_mg",
    "selenium_mcg", "copper_mg", "manganese_mg",
    # B vitamins with hepatic storage
    "vitamin_b12_mcg", "folate_mcg", "biotin_mcg",
]

# Maximum fraction of feature columns that may be null before a row is dropped.
MAX_NULL_FRACTION    = 0.50
SPARSE_COL_THRESHOLD = 0.70   # drop feature columns missing in >70 % of rows
CORRELATION_THRESHOLD = 0.95  # drop one of any pair correlated above this


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_raw() -> pd.DataFrame:
    """
    Load all biometrics and nutrition rows from PostgreSQL.

    Returns a DataFrame indexed by date with all biometric and nutrition
    columns present. Rows where sleep_duration_min == 0 are excluded
    (device failure / no-wear nights).
    """
    bio_cols = OUTPUT_COLS + ACTIVITY_COLS
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

    # Drop device-failure rows (tracker worn but no sleep recorded)
    df = df[(df["sleep_duration_min"].isna()) | (df["sleep_duration_min"] > 0)]

    return df


# ─────────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────

def _apply_rolling(df: pd.DataFrame, cols: list[str], window: int = 7) -> pd.DataFrame:
    """
    Replace specified columns with their rolling mean (min_periods=3).

    Applied to slow-acting micronutrients before lagging so the model
    sees accumulated exposure rather than a single-day value.
    """
    present = [c for c in cols if c in df.columns]
    df[present] = df[present].rolling(window=window, min_periods=3).mean()
    return df


def _lag_cols(df: pd.DataFrame, cols: list[str], lag: int = 1) -> pd.DataFrame:
    """
    Shift specified columns forward by `lag` days and rename with a suffix.

    After shifting, col[t] contains the value from day t-lag, so the model
    can use yesterday's inputs to predict today's outputs.
    """
    present = [c for c in cols if c in df.columns]
    lagged = df[present].shift(lag)
    lagged.columns = [f"{c}_lag{lag}" for c in present]
    return df.drop(columns=present).join(lagged)


# ─────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────

def _drop_sparse_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Drop rows where more than MAX_NULL_FRACTION of feature columns are null.

    Output columns are excluded from this calculation — a row is kept as
    long as it has enough input signal, even if some outputs are missing.
    """
    present = [c for c in feature_cols if c in df.columns]
    null_frac = df[present].isna().mean(axis=1)
    kept = df[null_frac <= MAX_NULL_FRACTION]
    dropped = len(df) - len(kept)
    if dropped:
        print(f"  [data_builder] Dropped {dropped} sparse rows (>{MAX_NULL_FRACTION:.0%} nulls)")
    return kept


def _impute_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill remaining nulls with each column's median computed over the full series.

    Median is used rather than mean to reduce the influence of outlier days
    on the imputed values.
    """
    medians = df.median(numeric_only=True)
    return df.fillna(medians)


def _drop_sparse_cols(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Drop feature columns that are missing in more than SPARSE_COL_THRESHOLD
    of rows.

    Columns logged infrequently (many trace minerals, rare amino acids) add
    noise and consume feature slots without contributing signal. Output
    columns are never dropped.
    """
    present = [c for c in feature_cols if c in df.columns]
    null_frac = df[present].isna().mean()
    to_drop = null_frac[null_frac > SPARSE_COL_THRESHOLD].index.tolist()
    if to_drop:
        print(f"  [data_builder] Dropped {len(to_drop)} sparse columns (>{SPARSE_COL_THRESHOLD:.0%} nulls): "
              f"{[c.replace('_lag1','') for c in to_drop]}")
        df = df.drop(columns=to_drop)
    return df


def _drop_correlated_cols(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Drop redundant feature columns where |r| > CORRELATION_THRESHOLD with
    another feature column.

    Highly correlated features (e.g. amino acids that track protein 1:1)
    dilute feature importance scores and slow training without adding
    independent information. When two columns are correlated above the
    threshold, the one with more non-null values is kept.
    """
    present = [c for c in feature_cols if c in df.columns]
    if len(present) < 2:
        return df

    corr = df[present].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        partners = upper.index[upper[col] > CORRELATION_THRESHOLD].tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            # Keep whichever has more non-null values
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
        4. Lag all activity columns by 1 day
        5. Drop rows with >50 % missing feature values
        6. Drop feature columns missing in >70 % of rows
        7. Drop one of any feature pair with |r| > 0.95
        8. Impute remaining nulls with column medians

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with lagged feature columns and unlagged
        output columns. Feature columns are suffixed with ``_lag1``.
        Returns an empty DataFrame if fewer than 2 rows are available
        after cleaning.
    """
    print("[data_builder] Loading raw data...")
    df = _load_raw()
    print(f"  Raw rows: {len(df)}")

    if len(df) < 2:
        print("  Not enough data to build features.")
        return pd.DataFrame()

    # Smooth slow micronutrients before lagging
    df = _apply_rolling(df, SLOW_MICRONUTRIENTS, window=7)

    # Lag nutrition and activity inputs
    df = _lag_cols(df, NUTRITION_COLS, lag=1)
    df = _lag_cols(df, ACTIVITY_COLS, lag=1)

    # First row is always all-null for lagged columns — drop it
    df = df.iloc[1:]

    # Identify feature columns (everything except outputs)
    feature_cols = [c for c in df.columns if c not in OUTPUT_COLS]

    df = _drop_sparse_rows(df, feature_cols)
    df = _drop_sparse_cols(df, feature_cols)
    feature_cols = [c for c in df.columns if c not in OUTPUT_COLS]
    df = _drop_correlated_cols(df, feature_cols)
    feature_cols = [c for c in df.columns if c not in OUTPUT_COLS]
    df = _impute_medians(df)

    print(f"  Clean rows : {len(df)}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Outputs    : {len([c for c in OUTPUT_COLS if c in df.columns])}")

    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the input feature column names from a built DataFrame."""
    return [c for c in df.columns if c not in OUTPUT_COLS]


def output_cols(df: pd.DataFrame) -> list[str]:
    """Return the output column names present in a built DataFrame."""
    return [c for c in OUTPUT_COLS if c in df.columns]


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df.head())
