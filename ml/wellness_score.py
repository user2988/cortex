"""
Cortex ML — Component 2: Wellness Score

Computes a single composite wellness score (0–100) for each day.
This is the target variable the XGBoost model learns to predict.

Design principles
-----------------
- Only sleep and cardiovascular output metrics contribute to the score.
- Each metric is normalised to the user's own personal min/max range so
  the score reflects *relative improvement for that individual*, not
  population averages.
- Metrics where lower is better are inverted before scoring.
- If a metric column is missing from the DataFrame entirely, its weight
  is redistributed proportionally across the remaining metrics.
- If a metric is present but NaN on a given day, that day's score is
  computed from whichever metrics are available, with weights renormalised
  on the fly for that row.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────────────────────

# Base weights — must sum to 1.0.
# Cardiovascular 54 % / Sleep 46 %.
# hrv_ms carries the most weight as the primary recovery signal.
WELLNESS_WEIGHTS: dict[str, float] = {
    # Cardiovascular
    "hrv_ms":               0.20,
    "hrv_deep_rmssd":       0.10,
    "rhr_bpm":              0.08,   # lower is better
    "spo2_avg_pct":         0.05,
    "spo2_min_pct":         0.05,
    "respiratory_rate":     0.03,   # lower is better
    "vo2_max":              0.03,
    # Sleep
    "sleep_efficiency_pct": 0.15,
    "deep_sleep_min":       0.12,
    "rem_sleep_min":        0.10,
    "sleep_duration_min":   0.06,
    "awake_min":            0.03,   # lower is better
}

# Metrics where a higher raw value means worse health — inverted before scoring.
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "rhr_bpm",
    "awake_min",
    "respiratory_rate",
})


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _active_weights(df: pd.DataFrame) -> dict[str, float]:
    """
    Return the weight map restricted to columns present in df, with weights
    renormalised to sum to 1.0.

    Raises ValueError if no wellness metric columns are found at all.
    """
    active = {col: w for col, w in WELLNESS_WEIGHTS.items() if col in df.columns}
    if not active:
        raise ValueError(
            "DataFrame contains none of the expected wellness metric columns. "
            "Ensure output columns are present and not renamed."
        )
    total = sum(active.values())
    return {col: w / total for col, w in active.items()}


def _normalise(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Min-max normalise each column to [0, 1] using the column's personal range.

    Returns
    -------
    normed : DataFrame of normalised (and direction-corrected) values
    col_min : Series of per-column minimums (for export / reuse)
    col_max : Series of per-column maximums (for export / reuse)
    """
    col_min = df[cols].min()
    col_max = df[cols].max()

    normed = pd.DataFrame(index=df.index)
    for col in cols:
        mn, mx = float(col_min[col]), float(col_max[col])
        if mx == mn:
            # Constant column — assign neutral mid-point so it contributes
            # zero variance without distorting the score.
            normed[col] = 0.5
        else:
            n = (df[col] - mn) / (mx - mn)
            if col in LOWER_IS_BETTER:
                n = 1.0 - n
            normed[col] = n.clip(0.0, 1.0)

    return normed, col_min, col_max


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def compute(df: pd.DataFrame) -> pd.Series:
    """
    Compute the daily wellness score for every row in df.

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame as produced by data_builder.build().
        Must contain at least one wellness metric column.

    Returns
    -------
    pd.Series
        Float scores in [0, 100], indexed identically to df.
        Rows where every wellness metric is NaN receive NaN.
    """
    weights = _active_weights(df)
    cols = list(weights.keys())
    normed, _, _ = _normalise(df, cols)

    result = pd.Series(index=df.index, dtype=float, name="wellness_score")

    for idx in df.index:
        row = normed.loc[idx]
        present = row.dropna()
        if present.empty:
            result[idx] = np.nan
            continue
        # Renormalise weights for this row in case some metrics are NaN
        row_weights = {col: weights[col] for col in present.index}
        total = sum(row_weights.values())
        score = sum(v * row_weights[col] / total for col, v in present.items())
        result[idx] = round(score * 100, 2)

    return result


def compute_with_bounds(
    df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, float], dict[str, float]]:
    """
    Compute wellness scores and return the normalisation bounds used.

    The bounds (personal min/max per metric) are needed by the Stack
    Optimiser to convert predicted feature values back into score space.

    Returns
    -------
    scores : pd.Series  — daily wellness scores (0–100)
    col_min : dict      — {column: personal minimum}
    col_max : dict      — {column: personal maximum}
    """
    weights = _active_weights(df)
    cols = list(weights.keys())
    normed, col_min, col_max = _normalise(df, cols)

    result = pd.Series(index=df.index, dtype=float, name="wellness_score")
    for idx in df.index:
        row = normed.loc[idx]
        present = row.dropna()
        if present.empty:
            result[idx] = np.nan
            continue
        row_weights = {col: weights[col] for col in present.index}
        total = sum(row_weights.values())
        score = sum(v * row_weights[col] / total for col, v in present.items())
        result[idx] = round(score * 100, 2)

    return result, col_min.to_dict(), col_max.to_dict()


def describe(df: pd.DataFrame) -> None:
    """
    Print a human-readable breakdown of the active weights and current
    average contribution of each metric to the wellness score.

    Useful for transparency and debugging.
    """
    weights = _active_weights(df)
    cols = list(weights.keys())
    normed, col_min, col_max = _normalise(df, cols)
    scores = compute(df)

    print(f"\n{'─' * 60}")
    print(f"  Wellness Score — active metrics ({len(cols)})")
    print(f"{'─' * 60}")
    print(f"  {'Metric':<25} {'Weight':>7}  {'Avg norm':>9}  {'Dir':>5}")
    print(f"  {'─'*25} {'─'*7}  {'─'*9}  {'─'*5}")

    for col in sorted(weights, key=lambda c: weights[c], reverse=True):
        avg_norm = normed[col].mean()
        direction = "↓ inv" if col in LOWER_IS_BETTER else "↑"
        print(f"  {col:<25} {weights[col]:>6.1%}  {avg_norm:>9.3f}  {direction:>5}")

    valid = scores.dropna()
    print(f"{'─' * 60}")
    print(f"  Current avg wellness score : {valid.mean():.1f}")
    print(f"  Min / Max                  : {valid.min():.1f} / {valid.max():.1f}")
    print(f"  Rows scored                : {len(valid)} / {len(scores)}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ml import data_builder

    df = data_builder.build()
    if df.empty:
        print("No data available.")
    else:
        describe(df)
        scores = compute(df)
        print(scores.tail(10))
