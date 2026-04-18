"""
Cortex ML — Component 5: Pipeline Orchestrator

Runs the full ML pipeline end-to-end in the correct order:
    1. data_builder      — load and transform data
    2. wellness_score    — compute daily target scores
    2.5 outcome_evaluator — score prior recs whose 7-day post-window has elapsed
    3. model_trainer     — train XGBoost model
    4. stack_optimiser   — find optimal activity targets

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

import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is on the path when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import data_builder, wellness_score, outcome_evaluator, model_trainer, stack_optimiser
from db import get_conn

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
    conn = get_conn()
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
        conn = get_conn()
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

def run() -> bool:
    """
    Execute the full ML pipeline and return True on success.

    Returns False on graceful skip (e.g. insufficient data) or failure.
    All exceptions are caught and logged — this function never raises.
    """
    run_at   = datetime.now(timezone.utc)
    t_start  = time.perf_counter()
    stage    = "init"

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

        # ── Stage 2: Wellness scores ─────────────────────────
        stage = "wellness_score"
        print(f"\n[pipeline] Stage 2 — {stage}")
        scores = wellness_score.compute(df)
        valid  = scores.dropna()
        print(f"  Scored rows : {len(valid)} / {len(scores)}")
        print(f"  Score range : {valid.min():.1f} – {valid.max():.1f}")
        print(f"  Score mean  : {valid.mean():.1f}")

        # ── Stage 2.5: Evaluate past recommendations ─────────
        # Close the learning loop: score how well prior recs played out.
        # Isolated try/except — a failure here must never block training.
        stage = "outcome_evaluator"
        print(f"\n[pipeline] Stage 2.5 — {stage}")
        try:
            outcome_evaluator.evaluate(scores)
        except Exception as eval_err:
            print(f"  [pipeline] outcome_evaluator failed (continuing): {eval_err}",
                  file=sys.stderr)

        # ── Stage 3: Model training ──────────────────────────
        stage = "model_trainer"
        print(f"\n[pipeline] Stage 3 — {stage}")
        train_result = model_trainer.train(df, scores)

        if train_result is None:
            print("[pipeline] Insufficient data — skipping optimisation.")
            _log(run_at, "skipped", elapsed(), stage, "insufficient data for training")
            return False

        # ── Stage 4: Stack optimisation ──────────────────────
        stage = "stack_optimiser"
        print(f"\n[pipeline] Stage 4 — {stage}")
        rec = stack_optimiser.optimise(df, scores, train_result)

        if rec is None:
            print("[pipeline] Optimisation produced no recommendations.")

        # ── Done ─────────────────────────────────────────────
        dur = elapsed()
        print(f"\n{'─' * 60}")
        print(f"[pipeline] Completed successfully in {dur:.1f}s")
        if rec:
            print(f"  Wellness : {rec['current_wellness']:.1f} → {rec['predicted_wellness']:.1f} (predicted)")
            print(f"  Tier     : {rec['tier']}")
            print(f"  Rec id   : {rec['rec_id']}")

        _log(run_at, "success", dur, stage, None)
        return True

    except Exception:
        dur       = elapsed()
        tb        = traceback.format_exc()
        short_err = tb.strip().splitlines()[-1]   # last line of traceback

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
