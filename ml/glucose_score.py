"""
Cortex ML — Glucose Score (v4)

Computes a single composite glucose score (0–100) for each day.
This is the scalar target the XGBoost stack optimiser works against
for the glucose-pivoted pipeline.

Design principles
-----------------
- Scoring uses **absolute clinical bounds**, not personal min/max.
  Glucose has published reference ranges (ADA / AACE) that encode
  what we actually want the user moving toward — 70–140 mg/dL TIR,
  fasting < 100, peak < 140, CV < 36. Personal normalisation would
  hide the clinical reality (a user whose personal minimum fasting
  is 105 would score 100 on "their best day" even though they're
  still pre-diabetic).
- Higher score = better metabolic day.
- If a metric is NaN on a given day, weights are renormalised over
  the remaining metrics so sparse manual data still produces a score.
- Phase-2 CGM-only metrics (post_meal_auc_avg, dawn_phenomenon_delta)
  are ignored until Phase 2 wires them in.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CLINICAL BOUNDS (mg/dL unless noted)
# ─────────────────────────────────────────────────────────────
#
# For each metric: (best_value, worst_value).
# Values <= best_value score 100. Values >= worst_value score 0.
# Linear between. Direction (higher vs lower is better) is implicit
# in the ordering of the tuple: (best, worst).

CLINICAL_BOUNDS: dict[str, tuple[float, float]] = {
    # Time-in-range — already 0–100, higher is better.
    "time_in_range_pct":        (85.0, 50.0),
    # Fasting — <90 optimal, 100–125 pre-diabetic, >=126 diabetic.
    "fasting_glucose_mg_dl":    (90.0, 126.0),
    # Post-meal peak — <140 non-diabetic target, 180 ADA diabetic ceiling.
    "post_meal_peak_avg_mg_dl": (110.0, 180.0),
    # Glycaemic variability — CV% <36 considered stable (ADA/EASD 2023).
    "glucose_cv_pct":           (20.0, 50.0),
}

# Weights must sum to 1.0. TIR dominates because it's the most
# comprehensive single metric and the most responsive to lifestyle.
GLUCOSE_WEIGHTS: dict[str, float] = {
    "time_in_range_pct":        0.45,
    "fasting_glucose_mg_dl":    0.25,
    "post_meal_peak_avg_mg_dl": 0.25,
    "glucose_cv_pct":           0.05,
}


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _subscore(col: str, value: float) -> float:
    """Map a raw metric value to a 0-100 subscore using clinical bounds."""
    best, worst = CLINICAL_BOUNDS[col]
    if np.isnan(value):
        return np.nan
    if best == worst:
        return 50.0
    # Normalised distance from "worst" toward "best".
    span = best - worst
    frac = (value - worst) / span
    return float(np.clip(frac, 0.0, 1.0) * 100.0)


def _active_weights(df: pd.DataFrame) -> dict[str, float]:
    """Weights restricted to columns present in df, renormalised to 1.0."""
    active = {c: w for c, w in GLUCOSE_WEIGHTS.items() if c in df.columns}
    if not active:
        raise ValueError(
            "DataFrame contains none of the expected glucose metric columns."
        )
    total = sum(active.values())
    return {c: w / total for c, w in active.items()}


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.Series:
    """
    Compute the daily glucose score (0–100) for every row in df.

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame containing at least one glucose metric
        column. Typically the output of ml.glucose_builder.build().

    Returns
    -------
    pd.Series of float scores in [0, 100], indexed identically to df.
    NaN on rows where every scored metric is NaN.
    """
    weights = _active_weights(df)
    cols    = list(weights.keys())

    result = pd.Series(index=df.index, dtype=float, name="glucose_score")

    for idx in df.index:
        subs = {c: _subscore(c, float(df.at[idx, c])) for c in cols}
        present = {c: v for c, v in subs.items() if not np.isnan(v)}
        if not present:
            result[idx] = np.nan
            continue
        row_weights = {c: weights[c] for c in present}
        total       = sum(row_weights.values())
        score       = sum(v * row_weights[c] / total for c, v in present.items())
        result[idx] = round(score, 2)

    return result


def describe(df: pd.DataFrame) -> None:
    """Print a human-readable breakdown of active weights and averages."""
    weights = _active_weights(df)
    scores  = compute(df)

    print(f"\n{'-' * 64}")
    print(f"  Glucose Score — active metrics ({len(weights)})")
    print(f"{'-' * 64}")
    print(f"  {'Metric':<28} {'Weight':>7}  {'Best':>7}  {'Worst':>7}  {'Avg':>7}")
    print(f"  {'-'*28} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    for col in sorted(weights, key=lambda c: weights[c], reverse=True):
        best, worst = CLINICAL_BOUNDS[col]
        avg = df[col].mean()
        print(f"  {col:<28} {weights[col]:>6.1%}  {best:>7.1f}  {worst:>7.1f}  {avg:>7.2f}")

    valid = scores.dropna()
    print(f"{'-' * 64}")
    if len(valid):
        print(f"  Current avg glucose score  : {valid.mean():.1f}")
        print(f"  Min / Max                  : {valid.min():.1f} / {valid.max():.1f}")
    print(f"  Rows scored                : {len(valid)} / {len(scores)}")
    print(f"{'-' * 64}\n")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ml import glucose_builder

    df = glucose_builder.build()
    if df.empty:
        print("No glucose data available.")
    else:
        describe(df)
        print(compute(df).tail(10))
