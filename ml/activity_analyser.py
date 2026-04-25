"""
Cortex ML — Activity Analyser

Finds the activity patterns that most reliably produce good sleep and heart
scores, using the individual's own data. No population benchmarks.

Approach
--------
Today's activity → tomorrow's sleep/heart score (lag-1 relationship, because
biometric data is stored on the wake-up date, one day after the activity).

For each activity metric the analyser:
  1. Computes Pearson correlation with sleep_score and heart_score
  2. Bins the metric into quantile-based ranges
  3. Identifies the optimal bin (highest average score)
  4. Generates a plain-English recommendation

Recommendations are ranked by impact (score difference: optimal bin vs rest)
and written to score_recommendations. Requires MIN_PAIRS matched activity→score
rows to produce meaningful output.
"""

import os
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timezone

DATABASE_URL = os.environ["DATABASE_URL"]
MIN_PAIRS    = 14   # minimum matched rows before we generate recommendations
N_BINS       = 5    # quantile bins per metric

ACTIVITY_METRICS = {
    "steps":                 "daily steps",
    "active_zone_min":       "active zone minutes",
    "very_active_min":       "very active minutes",
    "fairly_active_min":     "fairly active minutes",
    "lightly_active_min":    "lightly active minutes",
    "sedentary_min":         "sedentary time",
    "calories_burned":       "calories burned",
    "distance_km":           "distance walked/run",
}

METRIC_UNITS = {
    "steps":              lambda v: f"{int(v):,} steps",
    "active_zone_min":    lambda v: f"{int(v)} min",
    "very_active_min":    lambda v: f"{int(v)} min",
    "fairly_active_min":  lambda v: f"{int(v)} min",
    "lightly_active_min": lambda v: f"{int(v)} min",
    "sedentary_min":      lambda v: f"{int(v // 60)}h {int(v % 60)}m",
    "calories_burned":    lambda v: f"{int(v)} kcal",
    "distance_km":        lambda v: f"{v:.1f} km",
}


# ─────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────

def _load_paired() -> pd.DataFrame:
    """
    Join daily_scores (date D) with biometrics activity columns (date D-1).
    Returns one row per matched pair with activity inputs + score outputs.
    """
    sql = """
        SELECT
            s.date,
            s.sleep_score,
            s.heart_score,
            b.steps,
            b.active_zone_min,
            b.very_active_min,
            b.fairly_active_min,
            b.lightly_active_min,
            b.sedentary_min,
            b.calories_burned,
            b.distance_km
        FROM daily_scores s
        JOIN biometrics b ON b.date = s.date - INTERVAL '1 day'
        WHERE s.sleep_score IS NOT NULL
           OR s.heart_score IS NOT NULL
        ORDER BY s.date ASC
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
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────

def _fmt(metric: str, value: float) -> str:
    fmt = METRIC_UNITS.get(metric)
    return fmt(value) if fmt else f"{value:.1f}"


def _analyse_metric(
    metric: str,
    label: str,
    activity: pd.Series,
    score: pd.Series,
    score_name: str,
) -> dict | None:
    """Analyse one activity metric against one score. Returns recommendation dict or None."""
    combined = pd.DataFrame({"activity": activity, "score": score}).dropna()
    if len(combined) < MIN_PAIRS:
        return None

    corr = float(combined["activity"].corr(combined["score"]))
    if np.isnan(corr):
        return None

    # Bin into N_BINS quantile ranges
    try:
        combined["bin"] = pd.qcut(combined["activity"], q=N_BINS, duplicates="drop")
    except ValueError:
        return None

    bin_stats = (
        combined.groupby("bin", observed=True)["score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    bin_stats = bin_stats[bin_stats["count"] >= 3]
    if bin_stats.empty:
        return None

    best_idx  = bin_stats["mean"].idxmax()
    best_bin  = bin_stats.loc[best_idx, "bin"]
    best_avg  = float(bin_stats.loc[best_idx, "mean"])
    other_avg = float(combined[~combined["bin"].isin([best_bin])]["score"].mean())
    impact    = best_avg - other_avg

    if impact < 2.0:
        return None

    lo = _fmt(metric, best_bin.left  if not np.isinf(best_bin.left)  else combined["activity"].min())
    hi = _fmt(metric, best_bin.right if not np.isinf(best_bin.right) else combined["activity"].max())

    rec_text = (
        f"Your {score_name.lower()} is highest (avg {best_avg:.0f}/100) "
        f"on days when {label} is between {lo} and {hi} — "
        f"{impact:.0f} points above your average on other days."
    )

    return {
        "target_score":         score_name.lower(),
        "activity_metric":      metric,
        "activity_label":       label,
        "optimal_min":          float(best_bin.left)  if not np.isinf(best_bin.left)  else None,
        "optimal_max":          float(best_bin.right) if not np.isinf(best_bin.right) else None,
        "optimal_min_fmt":      lo,
        "optimal_max_fmt":      hi,
        "avg_score_in_range":   round(best_avg, 1),
        "avg_score_outside":    round(other_avg, 1),
        "score_delta":          round(impact, 1),
        "correlation":          round(corr, 4),
        "sample_size":          len(combined),
        "recommendation_text":  rec_text,
    }


def analyse(df: pd.DataFrame | None = None) -> list[dict]:
    """
    Run the full activity→score analysis. Returns list of recommendation dicts
    ranked by score_delta descending. Writes results to score_recommendations.
    """
    paired = _load_paired() if df is None else df
    if paired.empty or len(paired) < MIN_PAIRS:
        print(f"[activity_analyser] Only {len(paired)} paired rows — need {MIN_PAIRS}. Skipping.")
        return []

    recs = []
    for metric, label in ACTIVITY_METRICS.items():
        if metric not in paired.columns:
            continue
        activity = paired[metric].astype(float)

        for score_col, score_name in [("sleep_score", "Sleep"), ("heart_score", "Heart")]:
            if score_col not in paired.columns:
                continue
            rec = _analyse_metric(metric, label, activity, paired[score_col].astype(float), score_name)
            if rec:
                recs.append(rec)

    # Deduplicate: keep the higher-impact recommendation per metric
    seen: dict[str, dict] = {}
    for rec in recs:
        key = rec["activity_metric"]
        if key not in seen or rec["score_delta"] > seen[key]["score_delta"]:
            seen[key] = rec

    ranked = sorted(seen.values(), key=lambda r: r["score_delta"], reverse=True)

    if ranked:
        _save(ranked)
        print(f"[activity_analyser] Generated {len(ranked)} recommendations from {len(paired)} days of data.")
    else:
        print("[activity_analyser] No recommendations met the minimum impact threshold.")

    return ranked


# ─────────────────────────────────────────────────────────────
# PERSIST
# ─────────────────────────────────────────────────────────────

def _save(recs: list[dict]) -> None:
    """Replace all score_recommendations with the latest analysis results."""
    sql_delete = "DELETE FROM score_recommendations"
    sql_insert = """
        INSERT INTO score_recommendations (
            generated_at, target_score, activity_metric, activity_label,
            optimal_min, optimal_max, optimal_min_fmt, optimal_max_fmt,
            avg_score_in_range, avg_score_outside, score_delta,
            correlation, sample_size, recommendation_text
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    now = datetime.now(timezone.utc)
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql_delete)
                cur.executemany(sql_insert, [
                    (
                        now,
                        r["target_score"], r["activity_metric"], r["activity_label"],
                        r.get("optimal_min"), r.get("optimal_max"),
                        r["optimal_min_fmt"], r["optimal_max_fmt"],
                        r["avg_score_in_range"], r["avg_score_outside"],
                        r["score_delta"], r["correlation"],
                        r["sample_size"], r["recommendation_text"],
                    )
                    for r in recs
                ])
    finally:
        conn.close()
