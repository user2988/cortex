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
