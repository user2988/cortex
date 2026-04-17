"""
Cortex ML — Outcome Tracker

Evaluates past recommendations by:
  1. Comparing 7-day wellness before vs. after the recommendation (did it work?)
  2. Comparing actual nutrition/activity that week against recommended targets
     (did the user follow it?)

Both results are written to ml_recommendation_outcomes so the UI can show
not just whether wellness improved but whether the protocol was actually followed.

Adherence classification per metric
-------------------------------------
  followed : moved ≥ 80 % of the gap from baseline toward target
  partial  : moved 25–79 % in the right direction
  ignored  : moved < 25 % toward target (or in the wrong direction)

Overall adherence label
-----------------------
  high     : ≥ 60 % of metrics followed
  moderate : 25–59 % followed
  low      : < 25 % followed
"""

import os
import json
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DATABASE_URL = os.environ["DATABASE_URL"]

WINDOW_DAYS          = 7     # days before/after recommendation to compare
MIN_WINDOW_DAYS      = 3     # minimum valid days in a window
FOLLOWED_THRESHOLD   = 0.80  # ≥80 % of gap = followed
PARTIAL_THRESHOLD    = 0.25  # ≥25 % of gap = partial

# Metrics where lower is better — direction is inverted for adherence
LOWER_IS_BETTER = {
    "sedentary_min", "sugar_g", "sodium_mg", "saturated_fat_g",
    "caffeine_mg", "alcohol_units",
}

# ─────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────

CREATE_OUTCOMES_TABLE = """
CREATE TABLE IF NOT EXISTS ml_recommendation_outcomes (
    id                  SERIAL PRIMARY KEY,
    recommendation_id   INTEGER REFERENCES ml_recommendations(id),
    evaluated_at        TIMESTAMPTZ  NOT NULL,
    wellness_before_avg NUMERIC(6, 2),
    wellness_after_avg  NUMERIC(6, 2),
    wellness_delta      NUMERIC(6, 2),
    predicted_delta     NUMERIC(6, 2),
    n_days_before       INTEGER,
    n_days_after        INTEGER,
    adherence           JSONB,
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);
"""


def _ensure_tables() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_OUTCOMES_TABLE)
                # Add adherence column if table already exists without it
                cur.execute("""
                    ALTER TABLE ml_recommendation_outcomes
                    ADD COLUMN IF NOT EXISTS adherence JSONB
                """)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def _load_pending() -> list[dict]:
    """Return recommendations 7+ days old that have not yet been evaluated."""
    sql = """
        SELECT r.id, r.run_at, r.current_wellness_avg, r.predicted_wellness,
               r.recommendations
        FROM ml_recommendations r
        LEFT JOIN ml_recommendation_outcomes o ON o.recommendation_id = r.id
        WHERE o.id IS NULL
          AND r.run_at <= NOW() - INTERVAL '7 days'
        ORDER BY r.run_at
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        for r in rows:
            if isinstance(r["recommendations"], str):
                r["recommendations"] = json.loads(r["recommendations"])
        return rows
    finally:
        conn.close()


def _load_actuals(date_from: pd.Timestamp, date_to: pd.Timestamp) -> dict[str, float]:
    """
    Load actual 7-day averages for all nutrition and activity columns
    between date_from (inclusive) and date_to (exclusive).
    """
    sql = """
        SELECT b.date,
               b.steps, b.active_zone_min, b.very_active_min,
               b.fairly_active_min, b.lightly_active_min, b.sedentary_min,
               b.calories_burned, b.distance_km,
               b.time_in_fat_burn_min, b.time_in_cardio_min, b.time_in_peak_min,
               n.calories_in, n.protein_g, n.carbs_g, n.fat_g, n.fibre_g,
               n.sugar_g, n.sodium_mg, n.water_ml, n.saturated_fat_g,
               n.omega3_mg, n.epa_mg, n.dha_mg,
               n.vitamin_a_mcg, n.vitamin_d_iu, n.vitamin_e_mg, n.vitamin_k_mcg,
               n.vitamin_c_mg, n.vitamin_b6_mg, n.vitamin_b12_mcg,
               n.folate_mcg, n.niacin_mg, n.thiamine_mg, n.riboflavin_mg,
               n.pantothenic_acid_mg, n.biotin_mcg,
               n.magnesium_mg, n.zinc_mg, n.iron_mg, n.calcium_mg,
               n.potassium_mg, n.selenium_mcg, n.copper_mg,
               n.caffeine_mg, n.alcohol_units
        FROM biometrics b
        LEFT JOIN nutrition n ON b.date = n.date
        WHERE b.date >= %s AND b.date < %s
        ORDER BY b.date
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (date_from.date(), date_to.date()))
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=cols)
    df = df.drop(columns=["date"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return {col: float(df[col].mean()) for col in df.columns if df[col].notna().any()}


# ─────────────────────────────────────────────────────────────
# ADHERENCE COMPUTATION
# ─────────────────────────────────────────────────────────────

def _classify(actual: float, baseline: float, target: float, lower_is_better: bool) -> str:
    """
    Classify adherence for a single metric.

    The gap is the full recommended change. We measure how much of that
    gap the user actually covered in the right direction.
    """
    gap = target - baseline
    if abs(gap) < 1e-6:
        return "followed"   # target == baseline means maintain, trivially met

    actual_change = actual - baseline
    if lower_is_better:
        gap           = -gap
        actual_change = -actual_change

    ratio = actual_change / gap if gap > 0 else 0.0

    if ratio >= FOLLOWED_THRESHOLD:
        return "followed"
    if ratio >= PARTIAL_THRESHOLD:
        return "partial"
    return "ignored"


def _compute_adherence(recs_json: dict, actuals: dict) -> dict:
    """
    Compute per-metric adherence and an overall label.

    Parameters
    ----------
    recs_json : recommendations dict from ml_recommendations.recommendations
    actuals   : {metric: actual_7d_avg} from _load_actuals()

    Returns
    -------
    dict with 'metrics' list and 'overall' label
    """
    all_recs = recs_json.get("activity", []) + recs_json.get("nutrition", [])
    actionable = [r for r in all_recs if r.get("direction") != "maintain"]

    if not actionable:
        return {"overall": "n/a", "metrics": []}

    metric_results = []
    for r in actionable:
        col = r["metric"]
        if col not in actuals:
            continue

        actual   = actuals[col]
        baseline = r["current_avg"]
        target   = r["recommended"]
        lib      = col in LOWER_IS_BETTER

        status = _classify(actual, baseline, target, lib)
        metric_results.append({
            "metric":    col,
            "baseline":  round(baseline, 1),
            "target":    round(target, 1),
            "actual":    round(actual, 1),
            "direction": r["direction"],
            "status":    status,
        })

    if not metric_results:
        return {"overall": "insufficient_data", "metrics": []}

    n_followed = sum(1 for m in metric_results if m["status"] == "followed")
    pct = n_followed / len(metric_results)

    if pct >= 0.60:
        overall = "high"
    elif pct >= 0.25:
        overall = "moderate"
    else:
        overall = "low"

    return {
        "overall":     overall,
        "n_followed":  n_followed,
        "n_total":     len(metric_results),
        "metrics":     metric_results,
    }


# ─────────────────────────────────────────────────────────────
# DB WRITE
# ─────────────────────────────────────────────────────────────

def _write_outcome(
    recommendation_id: int,
    evaluated_at: datetime,
    before_avg: float | None,
    after_avg: float | None,
    predicted_delta: float | None,
    n_before: int,
    n_after: int,
    adherence: dict,
) -> None:
    delta = round(after_avg - before_avg, 2) if (after_avg and before_avg) else None
    sql = """
        INSERT INTO ml_recommendation_outcomes
            (recommendation_id, evaluated_at, wellness_before_avg,
             wellness_after_avg, wellness_delta, predicted_delta,
             n_days_before, n_days_after, adherence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    recommendation_id,
                    evaluated_at,
                    round(float(before_avg), 2) if before_avg is not None else None,
                    round(float(after_avg),  2) if after_avg  is not None else None,
                    delta,
                    round(float(predicted_delta), 2) if predicted_delta is not None else None,
                    n_before,
                    n_after,
                    json.dumps(adherence),
                ))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def evaluate(scores: pd.Series) -> int:
    """
    Evaluate all pending recommendations.

    For each unevaluated recommendation that is 7+ days old:
      - Compute 7-day wellness before and after
      - Load actual nutrition/activity for the post-recommendation week
      - Compute per-metric adherence vs. recommended targets
      - Write everything to ml_recommendation_outcomes

    Parameters
    ----------
    scores : full-history daily wellness scores (tz-naive index)

    Returns
    -------
    Number of recommendations evaluated.
    """
    _ensure_tables()
    pending = _load_pending()

    if not pending:
        print("  [outcome_tracker] No pending recommendations to evaluate.")
        return 0

    print(f"  [outcome_tracker] Evaluating {len(pending)} recommendation(s)...")

    for rec in pending:
        rec_date = pd.Timestamp(rec["run_at"]).tz_localize(None).normalize()

        # Wellness windows
        before = scores[
            (scores.index >= rec_date - pd.Timedelta(days=WINDOW_DAYS)) &
            (scores.index <  rec_date)
        ].dropna()
        after = scores[
            (scores.index >= rec_date) &
            (scores.index <  rec_date + pd.Timedelta(days=WINDOW_DAYS))
        ].dropna()

        before_avg = float(before.mean()) if len(before) >= MIN_WINDOW_DAYS else None
        after_avg  = float(after.mean())  if len(after)  >= MIN_WINDOW_DAYS else None

        predicted_delta = None
        if rec["current_wellness_avg"] and rec["predicted_wellness"]:
            predicted_delta = float(rec["predicted_wellness"]) - float(rec["current_wellness_avg"])

        # Adherence — compare actual inputs vs recommended targets
        actuals   = _load_actuals(rec_date, rec_date + pd.Timedelta(days=WINDOW_DAYS))
        adherence = _compute_adherence(rec["recommendations"], actuals)

        _write_outcome(
            recommendation_id = rec["id"],
            evaluated_at      = datetime.now(timezone.utc),
            before_avg        = before_avg,
            after_avg         = after_avg,
            predicted_delta   = predicted_delta,
            n_before          = len(before),
            n_after           = len(after),
            adherence         = adherence,
        )

        delta_str = f"{after_avg - before_avg:+.1f}" if before_avg and after_avg else "—"
        overall   = adherence.get("overall", "—")
        n_fol     = adherence.get("n_followed", 0)
        n_tot     = adherence.get("n_total", 0)
        print(f"    rec_id={rec['id']}  delta={delta_str}  "
              f"adherence={overall} ({n_fol}/{n_tot} metrics followed)")

    return len(pending)
