"""
Cortex ML — Component 6: Outcome Evaluator

Closes the learning loop. For each mature recommendation (≥7 days old)
that has not yet been evaluated, this module compares the 7 days of
wellness before the recommendation to the 7 days after, and measures
how closely the user's actual behaviour matched the recommended targets.

Outputs
-------
One row per mature recommendation is written to `ml_recommendation_outcomes`
containing:
    - baseline_wellness_avg  : mean wellness in the 7 days before run_at
    - outcome_wellness_avg   : mean wellness in the 7 days after run_at
    - actual_delta           : outcome − baseline
    - predicted_delta        : predicted_wellness − current_wellness_avg
                               (from the rec itself)
    - adherence_overall      : importance-weighted mean of per-metric adherence
    - per_metric             : JSONB array, one entry per recommended metric

Adherence scoring
-----------------
For each metric in the recommendation:

    ratio = |actual_avg − recommended| / max(|recommended|, ε)
    adherence = max(0, 1 − min(1, ratio))

For metrics whose direction is "maintain", anything within ±5 % of the
recommended value counts as full adherence (1.0). The overall score is a
mean weighted by the importance value stored on each rec entry — metrics
the model cared about more count more.

Idempotency
-----------
`ml_recommendation_outcomes.rec_id` is UNIQUE and inserts use
ON CONFLICT DO NOTHING, so `evaluate()` is safe to re-run on every
pipeline invocation.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from db import get_conn
from columns import ACTIVITY_COLS, NUTRITION_COLS

WINDOW_DAYS       = 7      # baseline and outcome window length
MIN_OUTCOME_DAYS  = 4      # require at least this many scored days in the post-window
MAINTAIN_TOL      = 0.05   # ±5 % counts as full adherence for "maintain" metrics
EPS               = 1e-9

_METRIC_SOURCE: dict[str, str] = {
    **{c: "biometrics" for c in ACTIVITY_COLS},
    **{c: "nutrition"  for c in NUTRITION_COLS},
}

# ─────────────────────────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_recommendation_outcomes (
    id                     SERIAL PRIMARY KEY,
    rec_id                 INTEGER NOT NULL UNIQUE
                               REFERENCES ml_recommendations(id) ON DELETE CASCADE,
    evaluated_at           TIMESTAMPTZ DEFAULT NOW(),
    baseline_start         DATE,
    baseline_end           DATE,
    outcome_start          DATE,
    outcome_end            DATE,
    baseline_wellness_avg  NUMERIC(6, 2),
    outcome_wellness_avg   NUMERIC(6, 2),
    actual_delta           NUMERIC(6, 2),
    predicted_delta        NUMERIC(6, 2),
    adherence_overall      NUMERIC(4, 3),
    per_metric             JSONB NOT NULL
);
"""


def _ensure_table() -> None:
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────

def _fetch_unevaluated_recs(cutoff: datetime) -> list[dict]:
    """
    Return all recommendations older than `cutoff` that do not yet have an
    outcome row, ordered oldest first.
    """
    sql = """
        SELECT r.id, r.run_at, r.current_wellness_avg, r.predicted_wellness,
               r.recommendations
        FROM ml_recommendations r
        LEFT JOIN ml_recommendation_outcomes o ON o.rec_id = r.id
        WHERE o.rec_id IS NULL
          AND r.run_at <= %s
        ORDER BY r.run_at ASC
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (cutoff,))
            rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for rid, run_at, cur_w, pred_w, rec_json in rows:
        rec = rec_json if isinstance(rec_json, dict) else json.loads(rec_json)
        out.append({
            "id":                rid,
            "run_at":            run_at,
            "current_wellness":  float(cur_w)  if cur_w  is not None else None,
            "predicted_wellness": float(pred_w) if pred_w is not None else None,
            "recommendations":   rec,
        })
    return out


def _fetch_window_actuals(
    start_date,
    end_date,
    metrics: list[str],
) -> dict[str, float]:
    """
    Return {metric: mean_value_over_window} for the given inclusive date range.

    Skips metrics we don't know how to resolve. NaNs are ignored in the mean.
    Returns an empty dict if no rows are available.
    """
    bio_metrics = [m for m in metrics if _METRIC_SOURCE.get(m) == "biometrics"]
    nut_metrics = [m for m in metrics if _METRIC_SOURCE.get(m) == "nutrition"]

    if not bio_metrics and not nut_metrics:
        return {}

    selects = ["b.date"]
    selects += [f"b.{c}" for c in bio_metrics]
    selects += [f"n.{c}" for c in nut_metrics]

    sql = f"""
        SELECT {', '.join(selects)}
        FROM biometrics b
        LEFT JOIN nutrition n ON b.date = n.date
        WHERE b.date BETWEEN %s AND %s
        ORDER BY b.date
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (start_date, end_date))
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description]
    finally:
        conn.close()

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=col_names)
    df = df.drop(columns=["date"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    means = df.mean(numeric_only=True)
    return {m: float(means[m]) for m in means.index if not np.isnan(means[m])}


# ─────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────

def _score_metric(recommended: float, actual: float, direction: str) -> float:
    """Return per-metric adherence in [0, 1]."""
    denom = max(abs(recommended), EPS)
    ratio = abs(actual - recommended) / denom
    if direction == "maintain" and ratio <= MAINTAIN_TOL:
        return 1.0
    return float(max(0.0, 1.0 - min(1.0, ratio)))


def _window_avg(scores: pd.Series, start_date, end_date) -> tuple[float | None, int]:
    """
    Mean of `scores` (wellness, indexed by datetime) over the inclusive date
    range. Returns (mean_or_None, n_scored_days).
    """
    if scores is None or scores.empty:
        return None, 0
    # `scores` is indexed by DatetimeIndex (tz-naive from data_builder)
    idx_dates = scores.index.date
    mask = (idx_dates >= start_date) & (idx_dates <= end_date)
    window = scores[mask].dropna()
    if window.empty:
        return None, 0
    return float(window.mean()), int(len(window))


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def evaluate(scores: pd.Series) -> int:
    """
    Write outcome rows for every recommendation whose 7-day post-window has
    now elapsed. Safe to re-run on every pipeline invocation.

    Parameters
    ----------
    scores : pd.Series
        Wellness scores indexed by date, as returned by
        `wellness_score.compute()`. Used for baseline/outcome averages.

    Returns
    -------
    int
        The number of outcome rows written in this call.
    """
    _ensure_table()
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)

    recs = _fetch_unevaluated_recs(cutoff)
    if not recs:
        print("  No mature recommendations awaiting evaluation.")
        return 0

    print(f"  Evaluating {len(recs)} mature recommendation(s)...")

    written = 0
    for rec in recs:
        run_at      = rec["run_at"]
        run_date    = run_at.date() if hasattr(run_at, "date") else run_at
        baseline_s  = run_date - timedelta(days=WINDOW_DAYS)
        baseline_e  = run_date - timedelta(days=1)
        outcome_s   = run_date + timedelta(days=1)
        outcome_e   = run_date + timedelta(days=WINDOW_DAYS)

        baseline_avg, _             = _window_avg(scores, baseline_s, baseline_e)
        outcome_avg, outcome_n_days = _window_avg(scores, outcome_s,  outcome_e)

        if outcome_n_days < MIN_OUTCOME_DAYS:
            print(f"    rec {rec['id']}: only {outcome_n_days} scored day(s) "
                  f"in outcome window — skipping for now.")
            continue

        # Flatten activity + nutrition recs into one list for scoring
        activity  = rec["recommendations"].get("activity",  []) or []
        nutrition = rec["recommendations"].get("nutrition", []) or []
        all_recs  = activity + nutrition
        metrics   = [r["metric"] for r in all_recs]

        actuals = _fetch_window_actuals(outcome_s, outcome_e, metrics) if metrics else {}

        per_metric = []
        for entry in all_recs:
            metric      = entry["metric"]
            recommended = float(entry.get("recommended", 0.0))
            direction   = entry.get("direction", "maintain")
            importance  = float(entry.get("importance", 0.0))
            current_avg = float(entry.get("current_avg", 0.0))

            if metric not in actuals:
                per_metric.append({
                    "metric":      metric,
                    "recommended": recommended,
                    "actual":      None,
                    "adherence":   None,
                    "importance":  importance,
                    "direction":   direction,
                    "current_at_rec": current_avg,
                })
                continue

            actual    = actuals[metric]
            adherence = _score_metric(recommended, actual, direction)

            per_metric.append({
                "metric":      metric,
                "recommended": round(recommended, 3),
                "actual":      round(actual, 3),
                "adherence":   round(adherence, 3),
                "importance":  round(importance, 4),
                "direction":   direction,
                "current_at_rec": round(current_avg, 3),
            })

        # Importance-weighted overall adherence over metrics we could resolve
        scored = [m for m in per_metric if m["adherence"] is not None]
        if scored:
            total_w = sum(m["importance"] for m in scored)
            if total_w > 0:
                overall = sum(m["adherence"] * m["importance"] for m in scored) / total_w
            else:
                overall = float(np.mean([m["adherence"] for m in scored]))
            overall = round(float(overall), 3)
        else:
            overall = None

        actual_delta    = (outcome_avg - baseline_avg) if (outcome_avg is not None and baseline_avg is not None) else None
        predicted_delta = None
        if rec["predicted_wellness"] is not None and rec["current_wellness"] is not None:
            predicted_delta = rec["predicted_wellness"] - rec["current_wellness"]

        _insert_outcome(
            rec_id           = rec["id"],
            baseline_s       = baseline_s,
            baseline_e       = baseline_e,
            outcome_s        = outcome_s,
            outcome_e        = outcome_e,
            baseline_avg     = baseline_avg,
            outcome_avg      = outcome_avg,
            actual_delta     = actual_delta,
            predicted_delta  = predicted_delta,
            adherence_overall = overall,
            per_metric       = per_metric,
        )

        written += 1
        adh_str = f"{overall:.2f}" if overall is not None else "—"
        delta_str = f"{actual_delta:+.1f}" if actual_delta is not None else "—"
        print(f"    rec {rec['id']}: wellness "
              f"{baseline_avg:.1f} → {outcome_avg:.1f}  ({delta_str})  "
              f"adherence {adh_str}")

    print(f"  Wrote {written} outcome row(s).")
    return written


def _insert_outcome(
    rec_id,
    baseline_s, baseline_e,
    outcome_s, outcome_e,
    baseline_avg, outcome_avg,
    actual_delta, predicted_delta,
    adherence_overall,
    per_metric,
) -> None:
    sql = """
        INSERT INTO ml_recommendation_outcomes
            (rec_id, baseline_start, baseline_end, outcome_start, outcome_end,
             baseline_wellness_avg, outcome_wellness_avg,
             actual_delta, predicted_delta,
             adherence_overall, per_metric)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (rec_id) DO NOTHING
    """
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    rec_id,
                    baseline_s, baseline_e, outcome_s, outcome_e,
                    None if baseline_avg is None else round(baseline_avg, 2),
                    None if outcome_avg  is None else round(outcome_avg,  2),
                    None if actual_delta    is None else round(actual_delta,    2),
                    None if predicted_delta is None else round(predicted_delta, 2),
                    None if adherence_overall is None else round(adherence_overall, 3),
                    json.dumps(per_metric),
                ))
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ml import data_builder, wellness_score

    print("[outcome_evaluator] Building data...")
    df = data_builder.build()
    if df.empty:
        print("No data — exiting.")
        sys.exit(0)

    print("[outcome_evaluator] Computing wellness scores...")
    scores = wellness_score.compute(df)

    print("[outcome_evaluator] Evaluating mature recommendations...")
    evaluate(scores)
