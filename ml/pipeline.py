"""
Cortex ML — Pipeline Orchestrator

Runs the ML pipeline end-to-end.

Stages
------
1. data_builder  — load and transform feature data
2. load_bp_targets — aggregate bp_readings into daily AM/PM averages
3. model_trainer — train one XGBoost regressor per AM target
                   (AM_systolic, AM_diastolic)

PM readings are aggregated and stored alongside AM readings but not
trained on: per the product spec, PM BP is secondary validation and
needs a different feature-alignment rule (same-day nutrition + caffeine
timing) that will be added in a follow-up.

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

from ml import data_builder, model_trainer

DATABASE_URL = os.environ["DATABASE_URL"]

# Targets trained each run, in order. Names double as model filenames
# (models/<name>_model.joblib) and the ml_model_runs.target_name value.
AM_TARGETS = ("am_systolic", "am_diastolic")

# Mapping from the short target name used in filenames/DB rows to the
# column name returned by data_builder.load_bp_targets().
TARGET_COLUMN = {
    "am_systolic":  "AM_systolic",
    "am_diastolic": "AM_diastolic",
}


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

def run() -> bool:
    """
    Execute the ML pipeline and return True on success.

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

        # ── Stage 1: Features ────────────────────────────────
        stage = "data_builder"
        print(f"\n[pipeline] Stage 1 — {stage}")
        df = data_builder.build()

        if df.empty:
            print("[pipeline] No feature data available — skipping pipeline.")
            _log(run_at, "skipped", elapsed(), stage, "empty feature dataframe")
            return False

        # ── Stage 2: BP targets ──────────────────────────────
        stage = "load_bp_targets"
        print(f"\n[pipeline] Stage 2 — {stage}")
        bp = data_builder.load_bp_targets()

        if bp.empty:
            print("[pipeline] bp_readings is empty — no BP logged yet.")
            print("           Start logging 2 AM + 2 PM readings per day to train.")
            _log(run_at, "skipped", elapsed(), stage, "no bp_readings rows")
            return False

        print(f"  Days with BP readings : {len(bp)}")
        for col in data_builder.BP_TARGET_COLS:
            n = int(bp[col].notna().sum())
            print(f"  {col:<13} : {n} day{'s' if n != 1 else ''}")

        # ── Stage 3: Train one model per AM target ───────────
        stage = "model_trainer"
        print(f"\n[pipeline] Stage 3 — {stage}")

        results: dict[str, dict | None] = {}
        for target_name in AM_TARGETS:
            col = TARGET_COLUMN[target_name]
            print(f"\n  ── Target: {target_name} ({col}) ──")
            # Reindex target to the feature frame so alignment is explicit;
            # days without a BP reading become NaN and are dropped inside train().
            target = bp[col].reindex(df.index)
            result = model_trainer.train(df, target, model_name=target_name)
            results[target_name] = result
            if result is None:
                print(f"  {target_name}: insufficient data — skipped.")

        trained = [name for name, r in results.items() if r is not None]

        if not trained:
            print("\n[pipeline] No models trained — insufficient paired data.")
            _log(run_at, "skipped", elapsed(), stage, "insufficient paired data")
            return False

        # ── Done ─────────────────────────────────────────────
        dur = elapsed()
        print(f"\n{'─' * 60}")
        print(f"[pipeline] Completed successfully in {dur:.1f}s")
        for name in trained:
            r = results[name]
            print(f"  {name:<13} tier={r['tier']:<10} "
                  f"rows={r['n_rows']:<4} "
                  f"test_mae={r['test_mae']:.2f} mmHg "
                  f"test_r2={r['test_r2']:.3f}")

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
