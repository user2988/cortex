"""
Cortex — Database Cleanup
Drops tables that are no longer used by the application.
Safe to re-run: uses DROP TABLE IF EXISTS throughout.
"""

import os
import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]

# Order matters — drop dependents before parents
DROP_TABLES = [
    "nutrition",
    "ml_recommendation_outcomes",  # references ml_recommendations
    "ml_recommendations",          # references ml_model_runs
    "ml_model_runs",
    "weekly_summaries",
]

def run():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                for table in DROP_TABLES:
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    print(f"  Dropped: {table}")
        print("\nCleanup complete.")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
