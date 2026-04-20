"""
Cortex — Additive Migrations
Applies only the ALTER TABLE statements from schema.sql without dropping any tables.
Safe to run against a live database with existing data.
"""

import os
import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

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
    # Scrap the wellness-score stack — replaced by BP prediction (AM_systolic,
    # AM_diastolic). ml_model_runs and ml_pipeline_log are retained because
    # they are generic training/run metadata and will be reused for BP models.
    "DROP TABLE IF EXISTS ml_recommendation_outcomes",
    "DROP TABLE IF EXISTS ml_recommendations",
    # ml_model_runs kept, but its wellness-era rows have no target_name — add
    # the column so future BP runs can label themselves.
    "ALTER TABLE ml_model_runs ADD COLUMN IF NOT EXISTS target_name TEXT",
    # Blood pressure readings — four per day. AM averages are the primary ML
    # targets; PM readings are secondary validation.
    """CREATE TABLE IF NOT EXISTS bp_readings (
        date            DATE        NOT NULL,
        period          TEXT        NOT NULL CHECK (period IN ('AM', 'PM')),
        reading_index   SMALLINT    NOT NULL CHECK (reading_index IN (1, 2)),
        systolic        SMALLINT    NOT NULL CHECK (systolic  BETWEEN 60 AND 260),
        diastolic       SMALLINT    NOT NULL CHECK (diastolic BETWEEN 30 AND 160),
        pulse           SMALLINT,
        recorded_at     TIMESTAMPTZ,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (date, period, reading_index)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_bp_readings_date ON bp_readings(date DESC)",
]

def run():
    conn = psycopg2.connect(DATABASE_URL)
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
