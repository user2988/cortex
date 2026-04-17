"""
Cortex ML — Outcome Tracker

Evaluates past recommendations by comparing the user's actual wellness
score in the 7 days before vs. the 7 days after each recommendation was
issued. Writes results to ml_recommendation_outcomes.

This closes the feedback loop: the pipeline can now report whether its
recommendations were followed by measurable improvement.

Design notes
------------
- Runs at the start of the weekly pipeline, before new training.
- Only evaluates recommendations that are 7+ days old and have not yet
  been evaluated (prevents double-counting).
- Wellness scores are recomputed fresh on the full history each run so
  normalisation bounds stay consistent with the user's current range.
- Outcome is descriptive — it shows what happened, not whether the user
  actually followed the recommendation (we don't track adherence yet).
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

DATABASE_URL = os.environ["DATABASE_URL"]

WINDOW_DAYS = 7   # days before and after recommendation to compare


# ─────────────────────────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
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
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);
"""


def _ensure_table() -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
    finally:
        conn.close()


def _load_pending() -> list[dict]:
    """
    Return recommendations that are 7+ days old and not yet evaluated.
    """
    sql = """
        SELECT r.id, r.run_at, r.current_wellness_avg, r.predicted_wellness
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
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _write_outcome(
    recommendation_id: int,
    evaluated_at: datetime,
    before_avg: float | None,
    after_avg: float | None,
    predicted_delta: float | None,
    n_before: int,
    n_after: int,
) -> None:
    delta = round(after_avg - before_avg, 2) if (after_avg is not None and before_avg is not None) else None
    sql = """
        INSERT INTO ml_recommendation_outcomes
            (recommendation_id, evaluated_at, wellness_before_avg,
             wellness_after_avg, wellness_delta, predicted_delta,
             n_days_before, n_days_after)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                ))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

def evaluate(scores: pd.Series) -> int:
    """
    Evaluate all pending recommendations against actual wellness scores.

    Parameters
    ----------
    scores : pd.Series
        Full-history daily wellness scores indexed by date (tz-naive).
        Produced by wellness_score.compute(df).

    Returns
    -------
    Number of recommendations evaluated.
    """
    _ensure_table()
    pending = _load_pending()

    if not pending:
        print("  [outcome_tracker] No pending recommendations to evaluate.")
        return 0

    print(f"  [outcome_tracker] Evaluating {len(pending)} recommendation(s)...")

    evaluated = 0
    for rec in pending:
        rec_date = pd.Timestamp(rec["run_at"]).tz_localize(None).normalize()

        before = scores[
            (scores.index >= rec_date - pd.Timedelta(days=WINDOW_DAYS)) &
            (scores.index <  rec_date)
        ].dropna()

        after = scores[
            (scores.index >= rec_date) &
            (scores.index <  rec_date + pd.Timedelta(days=WINDOW_DAYS))
        ].dropna()

        before_avg = float(before.mean()) if len(before) >= 3 else None
        after_avg  = float(after.mean())  if len(after)  >= 3 else None

        # Predicted delta: what the model said it would improve by
        predicted_delta = None
        if rec["current_wellness_avg"] is not None and rec["predicted_wellness"] is not None:
            predicted_delta = float(rec["predicted_wellness"]) - float(rec["current_wellness_avg"])

        _write_outcome(
            recommendation_id = rec["id"],
            evaluated_at      = datetime.now(timezone.utc),
            before_avg        = before_avg,
            after_avg         = after_avg,
            predicted_delta   = predicted_delta,
            n_before          = len(before),
            n_after           = len(after),
        )

        delta_str = (f"{after_avg - before_avg:+.1f}" if before_avg and after_avg else "insufficient data")
        print(f"    rec_id={rec['id']}  before={before_avg:.1f if before_avg else '—'}  "
              f"after={after_avg:.1f if after_avg else '—'}  delta={delta_str}")
        evaluated += 1

    return evaluated
