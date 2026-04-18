"""
Cortex — Database Migration
Runs schema.sql against the Neon PostgreSQL instance.

While no production data exists, this drops and recreates all tables on each run
so schema changes are always applied cleanly. Once real data is accumulating,
this should be replaced with additive ALTER TABLE migrations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import get_conn  # noqa: E402

SCHEMA_FILE  = os.path.join(os.path.dirname(__file__), "schema.sql")

DROP_SQL = """
    DROP TABLE IF EXISTS nutrition CASCADE;
    DROP TABLE IF EXISTS weight    CASCADE;
    DROP TABLE IF EXISTS biometrics CASCADE;
"""

def run():
    with open(SCHEMA_FILE, "r") as f:
        schema_sql = f.read()

    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(DROP_SQL)
                cur.execute(schema_sql)
        print("Migration complete — all tables dropped and recreated.")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
