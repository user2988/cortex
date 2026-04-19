"""
Cortex ML — Glucose Data Builder (v4)

Builds the modelling DataFrame for the glucose-targeted pipeline by
joining daily glucose outputs (from ml.glucose_builder) with the
user's food, activity, sleep, recovery, and medication-state inputs.

Conventions
-----------
- **Lag is 0 for all inputs.** Unlike the wellness pipeline (which
  lags nutrition/activity by one day because the wellness *outputs*
  are recorded overnight), glucose aggregates on day d are caused by
  that same day's food and activity. Yesterday's dinner still moves
  today's fasting glucose, but for a daily-aggregate target the
  same-day signal dominates and keeps the model interpretable.
- Medication state flags (on_metformin, on_glp1, ...) are passed
  through as exogenous features. The stack optimiser locks them in
  as constants — Cortex never recommends medication changes.
- Sparse-row, sparse-column, and correlation filters reuse the same
  thresholds as the wellness data_builder. A shared helper is pulled
  from `ml.data_builder` so we don't duplicate the cleaning logic.
"""

import pandas as pd
import numpy as np

from db import get_conn
from columns import (
    ML_NUTRITION_PANEL,
    ML_ACTIVITY_PANEL,
    GLUCOSE_OUTPUT_COLS,
    MEDICATION_CATEGORIES,
)
from ml.data_builder import (
    _drop_sparse_rows,
    _drop_sparse_cols,
    _drop_correlated_cols,
    _impute_medians,
)
from ml import glucose_builder


# Biometric inputs used by the glucose model. Output-side biometrics
# from the wellness pipeline (sleep architecture, HRV) become *inputs*
# here because they causally precede glucose response.
GLUCOSE_INPUT_BIOMETRIC_COLS = [
    # Sleep
    "sleep_duration_min",
    "sleep_efficiency_pct",
    "deep_sleep_min",
    "rem_sleep_min",
    # Recovery
    "hrv_ms",
    "rhr_bpm",
]

MED_FLAG_COLS = [f"on_{c}" for c in MEDICATION_CATEGORIES]


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_inputs() -> pd.DataFrame:
    """
    Load the input feature panel — activity + sleep/recovery from
    `biometrics` joined with nutrition macros/micros from `nutrition`.
    Returns a date-indexed DataFrame.
    """
    bio_cols   = GLUCOSE_INPUT_BIOMETRIC_COLS + ML_ACTIVITY_PANEL
    bio_select = ", ".join(f"b.{c}" for c in bio_cols)
    nut_select = ", ".join(f"n.{c}" for c in ML_NUTRITION_PANEL)

    sql = f"""
        SELECT b.date, {bio_select}, {nut_select}
        FROM biometrics b
        LEFT JOIN nutrition n ON b.date = n.date
        ORDER BY b.date
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop device-failure rows (tracker worn but no sleep recorded).
    df = df[(df["sleep_duration_min"].isna()) | (df["sleep_duration_min"] > 0)]
    return df


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def build() -> pd.DataFrame:
    """
    Build the glucose modelling DataFrame.

    Pipeline:
        1. Load input features (activity, sleep, recovery, nutrition)
        2. Load glucose outputs + medication state flags
        3. Join on date (inner join — only days with both signals)
        4. Drop rows/cols that are too sparse
        5. Drop one of any feature pair with |r| > 0.95
        6. Impute remaining nulls with column medians

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with feature columns (same-day),
        glucose output columns (unlagged), and medication flags.
        Empty DataFrame if fewer than 2 rows survive the join.
    """
    print("[glucose_data_builder] Loading inputs...")
    inputs  = _load_inputs()
    print(f"  Input rows: {len(inputs)}")

    print("[glucose_data_builder] Loading glucose outputs...")
    outputs = glucose_builder.build()
    print(f"  Output rows: {len(outputs)}")

    if inputs.empty or outputs.empty:
        print("  Not enough data for glucose modelling yet.")
        return pd.DataFrame()

    # Inner join — we need both sides for a training row.
    df = inputs.join(outputs, how="inner")
    print(f"  Joined rows: {len(df)}")

    if len(df) < 2:
        print("  Not enough overlap to build features.")
        return pd.DataFrame()

    # Drop glucose output columns that are entirely NaN (Phase-2 metrics
    # in the sparse-manual regime). These would otherwise block the
    # imputer and clutter the target column list.
    all_null_outputs = [c for c in GLUCOSE_OUTPUT_COLS
                        if c in df.columns and df[c].isna().all()]
    if all_null_outputs:
        print(f"  Dropping all-NaN outputs: {all_null_outputs}")
        df = df.drop(columns=all_null_outputs)

    present_output_cols = [c for c in GLUCOSE_OUTPUT_COLS if c in df.columns]
    protected_cols = set(present_output_cols) | set(MED_FLAG_COLS)
    feature_cols = [c for c in df.columns if c not in protected_cols]

    df = _drop_sparse_rows(df, feature_cols)
    df = _drop_sparse_cols(df, feature_cols)
    feature_cols = [c for c in df.columns if c not in protected_cols]
    df = _drop_correlated_cols(df, feature_cols)
    feature_cols = [c for c in df.columns if c not in protected_cols]
    df = _impute_medians(df)

    print(f"  Clean rows : {len(df)}")
    print(f"  Features   : {len(feature_cols)}")
    print(f"  Outputs    : {len(present_output_cols)}")
    print(f"  Med flags  : {len([c for c in MED_FLAG_COLS if c in df.columns])}")

    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Input feature columns (excludes glucose outputs and med flags)."""
    protected = set(GLUCOSE_OUTPUT_COLS) | set(MED_FLAG_COLS)
    return [c for c in df.columns if c not in protected]


def output_cols(df: pd.DataFrame) -> list[str]:
    """Glucose output columns present in a built DataFrame."""
    return [c for c in GLUCOSE_OUTPUT_COLS if c in df.columns]


def med_flag_cols(df: pd.DataFrame) -> list[str]:
    """Medication state flags present in a built DataFrame."""
    return [c for c in MED_FLAG_COLS if c in df.columns]


if __name__ == "__main__":
    df = build()
    print(df.shape)
    print(df.head())
