"""
Cortex ML — Pipeline Orchestrator

Runs daily via GitHub Actions. Two stages:
  1. score_engine   — compute Sleep Score + Heart Score for all dates
  2. activity_analyser — find optimal activity ranges, write recommendations

Exit codes: 0 = success or graceful skip, 1 = unexpected failure
"""

import os
import sys
import time
import traceback
import psycopg2
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import score_engine, activity_analyser

DATABASE_URL = os.environ["DATABASE_URL"]


def _log(run_at, status, duration_sec, stage, error_message):
    sql = """
        INSERT INTO ml_pipeline_log (run_at, status, duration_sec, stage, error_message)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (run_at, status, round(float(duration_sec), 2), stage, error_message))
        finally:
            conn.close()
    except Exception as e:
        print(f"[pipeline] WARNING: could not write to ml_pipeline_log: {e}", file=sys.stderr)


def run() -> bool:
    run_at  = datetime.now(timezone.utc)
    t_start = time.perf_counter()
    stage   = "init"

    def elapsed():
        return time.perf_counter() - t_start

    print(f"[pipeline] Starting  {run_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print("─" * 60)

    try:
        # ── Stage 1: Score computation ───────────────────────
        stage = "score_engine"
        print(f"\n[pipeline] Stage 1 — {stage}")
        scores = score_engine.compute()

        if scores.empty:
            print("[pipeline] No scores computed — insufficient biometric data.")
            _log(run_at, "skipped", elapsed(), stage, "no scores computed")
            return False

        n_written = score_engine.upsert(scores)
        print(f"  Scores written : {n_written} rows")
        print(f"  Sleep score    : {scores['sleep_score'].dropna().tail(1).values[0]:.1f}  (latest)")
        print(f"  Heart score    : {scores['heart_score'].dropna().tail(1).values[0]:.1f}  (latest)")

        # ── Stage 2: Activity analysis ───────────────────────
        stage = "activity_analyser"
        print(f"\n[pipeline] Stage 2 — {stage}")
        recs = activity_analyser.analyse()
        print(f"  Recommendations: {len(recs)}")

        # ── Done ─────────────────────────────────────────────
        dur = elapsed()
        print(f"\n{'─' * 60}")
        print(f"[pipeline] Completed in {dur:.1f}s")
        _log(run_at, "success", dur, stage, None)
        return True

    except Exception:
        dur       = elapsed()
        tb        = traceback.format_exc()
        short_err = tb.strip().splitlines()[-1]
        print(f"\n[pipeline] FAILED at '{stage}' after {dur:.1f}s", file=sys.stderr)
        print(tb, file=sys.stderr)
        _log(run_at, "failed", dur, stage, short_err)
        return False


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
