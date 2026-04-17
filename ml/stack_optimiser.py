"""
Cortex ML — Component 4: Stack Optimiser

Uses the trained XGBoost model to find the nutrition and activity targets
that maximise the predicted wellness score for this specific user.

Optimisation principles
-----------------------
- Both nutrition and activity features are optimised.
- Supplement optimisation will be added once the supplements table is live.
- Only features the model considers important (above mean importance)
  are varied — the rest are held at the user's 30-day average.
- Activity recommendations are capped at 40 % above the user's 30-day
  average to prevent unrealistic targets for less active users.
- Nutrition recommendations are capped at 50 % above the user's 30-day
  average OR the absolute safe upper limit, whichever is lower.
- sedentary_min is treated as lower-is-better: its floor is capped at
  40 % below the user's average for the same reason.
- Values are rounded to clean, practical increments.

Optimisation method
-------------------
scipy.optimize.differential_evolution — a global, gradient-free
optimiser that works well with XGBoost's black-box predictions and
handles box constraints natively.
"""

import os
import json
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from scipy.optimize import differential_evolution

DATABASE_URL = os.environ["DATABASE_URL"]

MAX_ACTIVITY_DELTA  = 0.40   # activity: never exceed 40 % above 30d avg
MAX_NUTRITION_DELTA = 0.50   # nutrition: never exceed 50 % above 30d avg
LOOKBACK_DAYS       = 30     # window for computing "current" averages

# ─────────────────────────────────────────────────────────────
# ACTIVITY CONSTRAINTS
# ─────────────────────────────────────────────────────────────

# (absolute_min, absolute_max, round_to, lower_is_better)
ACTIVITY_BOUNDS: dict[str, tuple] = {
    "steps":                (2_000,  25_000, 500,  False),
    "active_zone_min":      (0,      180,    5,    False),
    "very_active_min":      (0,      120,    5,    False),
    "fairly_active_min":    (0,      120,    5,    False),
    "lightly_active_min":   (0,      300,    10,   False),
    "sedentary_min":        (240,    1_200,  15,   True),
    "calories_burned":      (1_500,  5_000,  50,   False),
    "distance_km":          (0.5,    30.0,   0.5,  False),
    "time_in_fat_burn_min": (0,      120,    5,    False),
    "time_in_cardio_min":   (0,      90,     5,    False),
    "time_in_peak_min":     (0,      60,     5,    False),
}

# ─────────────────────────────────────────────────────────────
# NUTRITION CONSTRAINTS
# ─────────────────────────────────────────────────────────────

# (absolute_min, absolute_max, round_to, lower_is_better)
# absolute_max is a hard safe upper limit — never exceeded regardless of
# the 50 % personal delta cap. Values sourced from NIH Tolerable Upper
# Intake Levels (UL) or conservative clinical practice where no UL exists.
NUTRITION_BOUNDS: dict[str, tuple] = {
    # Macros
    "calories_in":              (1_200, 4_000,   50,   False),
    "protein_g":                (20,    250,      5,    False),
    "carbs_g":                  (50,    500,      5,    False),
    "fat_g":                    (20,    200,      5,    False),
    "fibre_g":                  (5,     60,       1,    False),
    "sugar_g":                  (0,     100,      5,    True),   # lower is better
    "water_ml":                 (500,   5_000,    100,  False),
    "sodium_mg":                (500,   2_300,    50,   True),   # WHO upper limit
    # Fats
    "saturated_fat_g":          (0,     20,       1,    True),
    "omega3_mg":                (0,     3_000,    100,  False),
    "epa_mg":                   (0,     2_000,    100,  False),
    "dha_mg":                   (0,     2_000,    100,  False),
    # Vitamins — fat-soluble
    "vitamin_a_mcg":            (0,     3_000,    50,   False),  # NIH UL 3000
    "vitamin_d_iu":             (0,     4_000,    200,  False),  # NIH UL 4000
    "vitamin_e_mg":             (0,     1_000,    10,   False),  # NIH UL 1000
    "vitamin_k_mcg":            (0,     1_000,    10,   False),
    # Vitamins — water-soluble
    "vitamin_c_mg":             (0,     2_000,    50,   False),  # NIH UL 2000
    "vitamin_b6_mg":            (0,     100,      1,    False),  # NIH UL 100
    "vitamin_b12_mcg":          (0,     1_000,    10,   False),
    "folate_mcg":               (0,     1_000,    25,   False),  # NIH UL 1000
    "niacin_mg":                (0,     35,       1,    False),  # NIH UL 35
    "thiamine_mg":              (0,     100,      1,    False),
    "riboflavin_mg":            (0,     100,      1,    False),
    "pantothenic_acid_mg":      (0,     100,      1,    False),
    "biotin_mcg":               (0,     1_000,    10,   False),
    # Minerals
    "magnesium_mg":             (0,     420,      10,   False),  # NIH UL 420 dietary
    "zinc_mg":                  (0,     40,       1,    False),  # NIH UL 40
    "iron_mg":                  (0,     45,       1,    False),  # NIH UL 45
    "calcium_mg":               (0,     2_500,    50,   False),  # NIH UL 2500
    "potassium_mg":             (0,     4_700,    100,  False),
    "selenium_mcg":             (0,     400,      5,    False),  # NIH UL 400
    "copper_mg":                (0,     10,       0.5,  False),  # NIH UL 10
    # Stimulants
    "caffeine_mg":              (0,     400,      25,   True),   # FDA upper limit
    "alcohol_units":            (0,     14,       0.5,  True),   # lower is better
}


# ─────────────────────────────────────────────────────────────
# DATABASE — TABLE CREATION
# ─────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_recommendations (
    id                   SERIAL PRIMARY KEY,
    run_at               TIMESTAMPTZ NOT NULL,
    model_run_id         INTEGER REFERENCES ml_model_runs(id),
    confidence_tier      TEXT        NOT NULL,
    n_days_data          INTEGER     NOT NULL,
    current_wellness_avg NUMERIC(6, 2),
    predicted_wellness   NUMERIC(6, 2),
    recommendations      JSONB       NOT NULL,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
"""


def _ensure_table() -> None:
    """Create ml_recommendations if it does not already exist."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
    finally:
        conn.close()


def _write_recommendation(
    run_at: datetime,
    model_run_id: int | None,
    tier: str,
    n_days: int,
    current_avg: float,
    predicted: float,
    recommendations: dict,
) -> int:
    """
    Insert a recommendation record and return its generated id.

    Parameters
    ----------
    recommendations : structured dict with 'activity' and 'supplements' keys
    """
    sql = """
        INSERT INTO ml_recommendations
            (run_at, model_run_id, confidence_tier, n_days_data,
             current_wellness_avg, predicted_wellness, recommendations)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    run_at,
                    model_run_id,
                    tier,
                    int(n_days),
                    round(float(current_avg), 2),
                    round(float(predicted),   2),
                    json.dumps(recommendations),
                ))
                return cur.fetchone()[0]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _lag_name(col: str) -> str:
    """Return the lagged feature name as it appears in the model DataFrame."""
    return f"{col}_lag1"


def _round_to(value: float, increment: float) -> float:
    """Round value to the nearest increment."""
    if increment <= 0:
        return value
    return round(round(value / increment) * increment, 10)


def _compute_30d_averages(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute the user's trailing 30-day average for each optimisable column.

    Uses lagged column names (col_lag1) since those are what the model sees.
    Falls back to full-history median if fewer than 30 rows are available.
    Covers both activity and nutrition bounds.

    Experiment rows are excluded so recommendations reflect the user's true
    baseline, not values recorded during an intentional protocol change.
    """
    all_bounds = {**ACTIVITY_BOUNDS, **NUTRITION_BOUNDS}
    avgs = {}
    baseline = df[df["in_experiment"] == 0] if "in_experiment" in df.columns else df
    tail = baseline.tail(LOOKBACK_DAYS)
    for col in all_bounds:
        lag_col = _lag_name(col)
        if lag_col not in df.columns:
            continue
        series = tail[lag_col].dropna()
        if series.empty:
            series = baseline[lag_col].dropna()
        avgs[col] = float(series.mean()) if not series.empty else 0.0
    return avgs


def _user_bounds(col: str, avg: float) -> tuple[float, float]:
    """
    Compute the per-user optimisation bounds for an activity or nutrition column.

    Activity : hard limits tightened by 40 % personal delta cap.
    Nutrition: hard limits tightened by 50 % personal delta cap.
    For lower-is-better columns the floor is raised symmetrically.
    """
    if col in ACTIVITY_BOUNDS:
        hard_min, hard_max, _, lower_is_better = ACTIVITY_BOUNDS[col]
        delta = MAX_ACTIVITY_DELTA
    else:
        hard_min, hard_max, _, lower_is_better = NUTRITION_BOUNDS[col]
        delta = MAX_NUTRITION_DELTA

    if lower_is_better:
        soft_min = max(hard_min, avg * (1 - delta))
        return float(soft_min), float(hard_max)
    else:
        soft_max = min(hard_max, avg * (1 + delta))
        return float(hard_min), float(max(hard_min, soft_max))


def _select_optimisable(
    feature_cols: list[str],
    importances: dict[str, float],
    avgs: dict[str, float],
) -> tuple[list[str], list[str]]:
    """
    Return (activity_cols, nutrition_cols) that the model considers important.

    Only columns with importance above the mean across all features are
    included — this keeps recommendations focused on what actually moves
    the needle for this specific user.
    """
    all_importances = list(importances.values())
    mean_imp = float(np.mean(all_importances)) if all_importances else 0.0

    activity_cols  = []
    nutrition_cols = []

    for col in ACTIVITY_BOUNDS:
        lag_col = _lag_name(col)
        if lag_col not in feature_cols:
            continue
        if importances.get(lag_col, 0.0) > mean_imp and col in avgs:
            activity_cols.append(col)

    for col in NUTRITION_BOUNDS:
        lag_col = _lag_name(col)
        if lag_col not in feature_cols:
            continue
        if importances.get(lag_col, 0.0) > mean_imp and col in avgs:
            nutrition_cols.append(col)

    return activity_cols, nutrition_cols


# ─────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────

def _optimise(
    model,
    template: np.ndarray,
    feature_cols: list[str],
    opt_cols: list[str],
    opt_indices: list[int],
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """
    Run differential evolution to find the feature vector that maximises
    the predicted wellness score.

    Parameters
    ----------
    template    : baseline feature vector (30-day averages for all features)
    opt_cols    : activity column names being optimised (base names, not lagged)
    opt_indices : positions of opt_cols' lagged equivalents in feature_cols
    bounds      : (lower, upper) per optimised column

    Returns
    -------
    Optimised feature vector (same shape as template).
    """
    def objective(x: np.ndarray) -> float:
        vec = template.copy()
        for i, idx in enumerate(opt_indices):
            vec[idx] = x[i]
        return -float(model.predict(vec.reshape(1, -1))[0])

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=500,
        tol=1e-4,
        polish=True,
        workers=1,
    )
    optimised = template.copy()
    for i, idx in enumerate(opt_indices):
        optimised[idx] = result.x[i]
    return optimised


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def _build_rec(col, bounds_dict, avgs, optimised_vec, feature_cols, importances):
    """Build a single recommendation dict for one column."""
    _, _, round_inc, lower_is_better = bounds_dict[col]
    current     = avgs.get(col, 0.0)
    raw_rec     = float(optimised_vec[feature_cols.index(_lag_name(col))])
    recommended = _round_to(raw_rec, round_inc)
    delta_pct   = ((recommended - current) / current * 100) if current else 0.0

    if abs(delta_pct) < 2:
        direction = "maintain"
    elif lower_is_better:
        direction = "decrease" if recommended < current else "increase"
    else:
        direction = "increase" if recommended > current else "decrease"

    return {
        "metric":      col,
        "current_avg": round(current, 1),
        "recommended": round(recommended, 1),
        "direction":   direction,
        "change_pct":  round(delta_pct, 1),
        "importance":  round(importances.get(_lag_name(col), 0.0), 4),
    }


def optimise(
    df: pd.DataFrame,
    scores: pd.Series,
    train_result: dict,
) -> dict | None:
    """
    Find the nutrition and activity targets that maximise predicted wellness.

    Parameters
    ----------
    df           : feature DataFrame from data_builder.build()
    scores       : wellness scores from wellness_score.compute()
    train_result : dict returned by model_trainer.train()

    Returns
    -------
    dict with recommendation details, or None if optimisation cannot run.
    """
    _ensure_table()

    model        = train_result["model"]
    feature_cols = train_result["feature_cols"]
    tier         = train_result["tier"]
    n_rows       = train_result["n_rows"]
    model_run_id = train_result.get("run_id")

    importances = {
        entry["feature"]: entry["importance"]
        for entry in train_result["top_features"]
    }

    avgs = _compute_30d_averages(df)
    act_cols, nut_cols = _select_optimisable(feature_cols, importances, avgs)
    all_opt_cols = act_cols + nut_cols

    if not all_opt_cols:
        print("  No features met the importance threshold — skipping optimisation.")
        return None

    print(f"  Optimising {len(act_cols)} activity + {len(nut_cols)} nutrition features")

    # Template: full-history medians for all features (baseline days only)
    baseline_df = df[df["in_experiment"] == 0] if "in_experiment" in df.columns else df
    template = np.array([
        float(baseline_df[col].median()) if col in baseline_df.columns else 0.0
        for col in feature_cols
    ])
    # Ensure the model predicts for a normal (non-experiment) day
    if "in_experiment" in feature_cols:
        template[feature_cols.index("in_experiment")] = 0.0

    # Override all optimisable columns with 30-day averages
    for col in all_opt_cols:
        lag_col = _lag_name(col)
        if lag_col in feature_cols and col in avgs:
            template[feature_cols.index(lag_col)] = avgs[col]

    baseline_score = float(model.predict(template.reshape(1, -1))[0])
    print(f"  Baseline predicted wellness : {baseline_score:.2f}")

    # Optimisation bounds — one entry per optimisable column
    opt_indices = [feature_cols.index(_lag_name(c)) for c in all_opt_cols]
    opt_bounds  = [_user_bounds(c, avgs[c]) for c in all_opt_cols]

    optimised_vec   = _optimise(model, template, feature_cols, all_opt_cols, opt_indices, opt_bounds)
    optimised_score = float(model.predict(optimised_vec.reshape(1, -1))[0])
    print(f"  Optimised predicted wellness: {optimised_score:.2f}")

    activity_recs  = [_build_rec(c, ACTIVITY_BOUNDS,  avgs, optimised_vec, feature_cols, importances) for c in act_cols]
    nutrition_recs = [_build_rec(c, NUTRITION_BOUNDS, avgs, optimised_vec, feature_cols, importances) for c in nut_cols]

    activity_recs.sort( key=lambda x: abs(x["change_pct"]), reverse=True)
    nutrition_recs.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    recommendations = {
        "activity":    activity_recs,
        "nutrition":   nutrition_recs,
        "supplements": [],
    }

    # Current wellness average — baseline days only
    if "in_experiment" in df.columns:
        baseline_idx = df.index[df["in_experiment"] == 0]
        baseline_scores = scores[scores.index.isin(baseline_idx)]
    else:
        baseline_scores = scores
    current_avg_score = float(baseline_scores.dropna().tail(LOOKBACK_DAYS).mean())

    for label, recs in [("Activity", activity_recs), ("Nutrition", nutrition_recs)]:
        if recs:
            print(f"\n  {label} recommendations:")
            for rec in recs:
                print(f"    {rec['metric']:<30} {rec['direction']:<10} "
                      f"{rec['current_avg']:>9.1f} → {rec['recommended']:>9.1f}  "
                      f"({rec['change_pct']:+.1f}%)")

    run_at = datetime.now(timezone.utc)
    rec_id = _write_recommendation(
        run_at          = run_at,
        model_run_id    = model_run_id,
        tier            = tier,
        n_days          = n_rows,
        current_avg     = current_avg_score,
        predicted       = optimised_score,
        recommendations = recommendations,
    )
    print(f"\n  Recommendation written (id={rec_id}).")

    return {
        "rec_id":             rec_id,
        "tier":               tier,
        "n_days":             n_rows,
        "current_wellness":   round(current_avg_score, 2),
        "predicted_wellness": round(optimised_score, 2),
        "recommendations":    recommendations,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ml import data_builder, wellness_score, model_trainer

    print("[stack_optimiser] Building data...")
    df = data_builder.build()
    if df.empty:
        print("No data — exiting.")
        sys.exit(0)

    print("[stack_optimiser] Computing wellness scores...")
    scores = wellness_score.compute(df)

    print("[stack_optimiser] Training model...")
    result = model_trainer.train(df, scores)
    if result is None:
        print("Insufficient data for optimisation.")
        sys.exit(0)

    print("[stack_optimiser] Optimising stack...")
    rec = optimise(df, scores, result)
    if rec:
        print(f"\nDone. rec_id={rec['rec_id']}  "
              f"wellness {rec['current_wellness']:.1f} → {rec['predicted_wellness']:.1f}")
