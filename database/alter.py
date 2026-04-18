"""
Cortex — Additive Migrations
Applies only the ALTER TABLE statements from schema.sql without dropping any tables.
Safe to run against a live database with existing data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_conn  # noqa: E402

MIGRATIONS = [
    "ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS hrv_deep_rmssd     NUMERIC(6, 2)",
    "ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS spo2_max_pct       NUMERIC(5, 2)",
    "ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS lightly_active_min INTEGER",
    "ALTER TABLE biometrics DROP COLUMN IF EXISTS   skin_temp_relative",
    "ALTER TABLE biometrics DROP COLUMN IF EXISTS   sleep_score",
    "ALTER TABLE biometrics DROP COLUMN IF EXISTS   bedtime_consistency_sd",
    # v2: findings table
    """CREATE TABLE IF NOT EXISTS findings (
        id            SERIAL PRIMARY KEY,
        variable_a    TEXT          NOT NULL,
        variable_b    TEXT,
        r_squared     NUMERIC(6, 4),
        p_value       NUMERIC(10, 8),
        coefficient   NUMERIC(10, 6),
        lag_days      INTEGER       DEFAULT 0,
        analysis_type TEXT          NOT NULL,
        sample_size   INTEGER,
        calculated_at TIMESTAMPTZ   DEFAULT NOW(),
        pinned        BOOLEAN       DEFAULT FALSE
    )""",
    "CREATE INDEX IF NOT EXISTS idx_findings_r2 ON findings(r_squared DESC)",
    # v2: experiments table
    """CREATE TABLE IF NOT EXISTS experiments (
        id            SERIAL PRIMARY KEY,
        name          TEXT          NOT NULL,
        variable_a    TEXT          NOT NULL,
        variable_b    TEXT          NOT NULL,
        lag_days      INTEGER       DEFAULT 0,
        method        TEXT          DEFAULT 'pearson',
        start_date    DATE          NOT NULL,
        duration_days INTEGER       NOT NULL,
        status        TEXT          DEFAULT 'active',
        interpretation TEXT,
        created_at    TIMESTAMPTZ   DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_experiments_date ON experiments(start_date DESC)",
    # v2: targets table
    """CREATE TABLE IF NOT EXISTS targets (
        variable      TEXT          PRIMARY KEY,
        target_value  NUMERIC(10, 2) NOT NULL,
        updated_at    TIMESTAMPTZ   DEFAULT NOW()
    )""",
    # v3: ML pipeline tables
    # Order matters — ml_recommendations FKs ml_model_runs, and
    # ml_recommendation_outcomes FKs ml_recommendations.
    """CREATE TABLE IF NOT EXISTS ml_model_runs (
        id              SERIAL PRIMARY KEY,
        run_at          TIMESTAMPTZ NOT NULL,
        confidence_tier TEXT        NOT NULL,
        n_rows          INTEGER     NOT NULL,
        n_features      INTEGER     NOT NULL,
        train_r2        NUMERIC(6, 4),
        test_r2         NUMERIC(6, 4),
        test_mae        NUMERIC(8, 4),
        test_rmse       NUMERIC(8, 4),
        top_features    JSONB,
        model_path      TEXT,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS ml_recommendations (
        id                   SERIAL PRIMARY KEY,
        run_at               TIMESTAMPTZ NOT NULL,
        model_run_id         INTEGER REFERENCES ml_model_runs(id),
        confidence_tier      TEXT        NOT NULL,
        n_days_data          INTEGER     NOT NULL,
        current_wellness_avg NUMERIC(6, 2),
        predicted_wellness   NUMERIC(6, 2),
        recommendations      JSONB       NOT NULL,
        created_at           TIMESTAMPTZ DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS ml_recommendation_outcomes (
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
    )""",
    """CREATE TABLE IF NOT EXISTS ml_pipeline_log (
        id            SERIAL PRIMARY KEY,
        run_at        TIMESTAMPTZ  NOT NULL,
        status        TEXT         NOT NULL,
        duration_sec  NUMERIC(8, 2),
        stage         TEXT,
        error_message TEXT,
        created_at    TIMESTAMPTZ  DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ml_model_runs_run_at  ON ml_model_runs(run_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ml_recs_run_at        ON ml_recommendations(run_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ml_outcomes_evaluated ON ml_recommendation_outcomes(evaluated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ml_pipeline_log       ON ml_pipeline_log(run_at DESC)",
    # v4: glucose pipeline tables
    # Order matters — glucose_readings.meal_id FKs meals(id).
    """CREATE TABLE IF NOT EXISTS meals (
        id          SERIAL PRIMARY KEY,
        ts          TIMESTAMPTZ   NOT NULL,
        name        TEXT,
        carbs_g     NUMERIC(8, 2),
        protein_g   NUMERIC(8, 2),
        fat_g       NUMERIC(8, 2),
        fibre_g     NUMERIC(8, 2),
        sugar_g     NUMERIC(8, 2),
        calories    NUMERIC(8, 2),
        notes       TEXT,
        created_at  TIMESTAMPTZ   DEFAULT NOW()
    )""",
    """CREATE TABLE IF NOT EXISTS glucose_readings (
        id          SERIAL PRIMARY KEY,
        ts          TIMESTAMPTZ   NOT NULL,
        mg_dl       NUMERIC(5, 1) NOT NULL,
        source      TEXT          NOT NULL,
        meal_id     INTEGER       REFERENCES meals(id) ON DELETE SET NULL,
        notes       TEXT,
        created_at  TIMESTAMPTZ   DEFAULT NOW(),
        UNIQUE (ts, source)
    )""",
    """CREATE TABLE IF NOT EXISTS medications (
        id          SERIAL PRIMARY KEY,
        name        TEXT          NOT NULL,
        category    TEXT          NOT NULL,
        dose_text   TEXT,
        start_date  DATE          NOT NULL,
        end_date    DATE,
        notes       TEXT,
        created_at  TIMESTAMPTZ   DEFAULT NOW()
    )""",
    "CREATE INDEX IF NOT EXISTS idx_glucose_ts        ON glucose_readings(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_meals_ts          ON meals(ts DESC)",
    "CREATE INDEX IF NOT EXISTS idx_medications_start ON medications(start_date DESC)",
]

def run():
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                for sql in MIGRATIONS:
                    cur.execute(sql)
                    print(f"  OK: {sql.strip()}")
        print("Migrations complete.")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
