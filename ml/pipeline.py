"""
Cortex ML — Component 5: Pipeline Orchestrator

Runs the full ML pipeline end-to-end in the correct order:
    1. data_builder  — load and transform data
    2. bp_target     — compute daily MAP target from blood_pressure_logs
    3. outcome_tracker — evaluate last week's recommendations
    4. model_trainer  — train XGBoost model on MAP
    5. stack_optimiser — find optimal activity/nutrition targets

This is the entry point called by GitHub Actions each week.

Failure isolation
-----------------
Any exception inside the ML pipeline is caught, logged to the database,
and reported — but never re-raised. A crash here cannot affect the main
Fitbit / Cronometer data pipeline.

Exit codes
----------
0 — success or graceful skip (insufficient data)
1 — unexpected failure (logged to DB and stderr before exit)
"""

import os
import sys
import time
import traceback
import psycopg2
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on the path when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import data_builder, bp_target, model_trainer, stack_optimiser, outcome_tracker

DATABASE_URL = os.environ["DATABASE_URL"]

# ─────────────────────────────────────────────────────────────
# PIPELINE LOG TABLE
# ─────────────────────────────────────────────────────────────

CREATE_LOG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ml_pipeline_log (
    id            SERIAL PRIMARY KEY,
    run_at        TIMESTAMPTZ  NOT NULL,
    status        TEXT         NOT NULL,   -- 'success' | 'failed' | 'skipped'
    duration_sec  NUMERIC(8, 2),
    stage         TEXT,                    -- last stage reached before failure
    error_message TEXT,
    created_at    TIMESTAMPTZ  DEFAULT NOW()
);
"""


def _ensure_log_table() -> None:
    """Create ml_pipeline_log if it does not already exist."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_LOG_TABLE_SQL)
    finally:
        conn.close()


def _log(
    run_at: datetime,
    status: str,
    duration_sec: float,
    stage: str | None,
    error_message: str | None,
) -> None:
    """
    Write a pipeline run record to ml_pipeline_log.

    Failures here are printed but not raised — logging must never crash
    the process that called it.
    """
    sql = """
        INSERT INTO ml_pipeline_log
            (run_at, status, duration_sec, stage, error_message)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        run_at,
                        status,
                        round(float(duration_sec), 2),
                        stage,
                        error_message,
                    ))
        finally:
            conn.close()
    except Exception as log_err:
        print(f"[pipeline] WARNING: could not write to ml_pipeline_log: {log_err}",
              file=sys.stderr)


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def run() -> bool:  # noqa: C901
    """
    Execute the full ML pipeline and return True on success.

    Stages
    ------
    1. data_builder    — load and transform data
    2. bp_target       — compute daily MAP target from blood_pressure_logs
    3. outcome_tracker — evaluate last week's recommendations
    4. model_trainer   — train XGBoost model on MAP
    5. stack_optimiser — find optimal activity/nutrition targets

    Returns False on graceful skip or failure. Never raises.
    """
    run_at  = datetime.now(timezone.utc)
    t_start = time.perf_counter()
    stage   = "init"

    def elapsed() -> float:
        return time.perf_counter() - t_start

    print(f"[pipeline] Starting ML pipeline  {run_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print("─" * 60)

    try:
        _ensure_log_table()

        # ── Stage 1: Data ────────────────────────────────────
        stage = "data_builder"
        print(f"\n[pipeline] Stage 1 — {stage}")
        df = data_builder.build()

        if df.empty:
            print("[pipeline] No data available — skipping pipeline.")
            _log(run_at, "skipped", elapsed(), stage, "empty dataframe")
            return False

        # ── Stage 2: BP target (PM MAP) ──────────────────────
        stage = "bp_target"
        print(f"\n[pipeline] Stage 2 — {stage}")
        map_scores = bp_target.compute(df)
        valid = map_scores.dropna()

        if valid.empty:
            print("[pipeline] No PM blood pressure readings found — skipping pipeline.")
            print("           Log your PM reading daily; training starts after 7 readings.")
            _log(run_at, "skipped", elapsed(), stage, "no PM BP readings logged")
            return False

        # ── Stage 3: Outcome evaluation ──────────────────────
        stage = "outcome_tracker"
        print(f"\n[pipeline] Stage 3 — {stage}")
        outcome_tracker.evaluate(map_scores)

        # ── Stage 4: Model training ──────────────────────────
        stage = "model_trainer"
        print(f"\n[pipeline] Stage 4 — {stage}")
        train_result = model_trainer.train(df, map_scores)

        if train_result is None:
            print("[pipeline] Insufficient data — skipping optimisation.")
            _log(run_at, "skipped", elapsed(), stage, "insufficient data for training")
            return False

        # ── Stage 5: Stack optimisation ──────────────────────
        stage = "stack_optimiser"
        print(f"\n[pipeline] Stage 5 — {stage}")
        rec = stack_optimiser.optimise(df, map_scores, train_result)

        if rec is None:
            print("[pipeline] Optimisation produced no recommendations.")

        # ── Done ─────────────────────────────────────────────
        dur = elapsed()
        print(f"\n{'─' * 60}")
        print(f"[pipeline] Completed successfully in {dur:.1f}s")
        if rec:
            print(f"  PM MAP   : {rec['current_map']:.1f} → {rec['predicted_map']:.1f} mmHg (predicted)")
            print(f"  Tier     : {rec['tier']}")
            print(f"  Rec id   : {rec['rec_id']}")

        _log(run_at, "success", dur, stage, None)
        return True

    except Exception:
        dur       = elapsed()
        tb        = traceback.format_exc()
        short_err = tb.strip().splitlines()[-1]

        print(f"\n[pipeline] FAILED at stage '{stage}' after {dur:.1f}s",
              file=sys.stderr)
        print(tb, file=sys.stderr)

        _log(run_at, "failed", dur, stage, short_err)
        return False


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
