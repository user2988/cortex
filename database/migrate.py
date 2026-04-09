"""
Cortex — Database Migration
Runs schema.sql against the Neon PostgreSQL instance.
Execute once to initialise the database, safe to re-run (all statements use IF NOT EXISTS).
"""

import os
import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]
SCHEMA_FILE  = os.path.join(os.path.dirname(__file__), "schema.sql")

def run():
    with open(SCHEMA_FILE, "r") as f:
        sql = f.read()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print("Migration complete — all tables and indexes created.")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
