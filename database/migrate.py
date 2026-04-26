"""
Cortex — Database Migration
Drops and recreates biometrics from schema.sql.
Use alter.py for additive changes that preserve existing data.
"""

import os
import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]
SCHEMA_FILE  = os.path.join(os.path.dirname(__file__), "schema.sql")

DROP_SQL = """
    DROP TABLE IF EXISTS biometrics CASCADE;
"""

def run():
    with open(SCHEMA_FILE, "r") as f:
        schema_sql = f.read()

    conn = psycopg2.connect(DATABASE_URL)
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
