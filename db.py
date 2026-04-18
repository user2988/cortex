"""
Cortex — Shared database helpers.

Reads DATABASE_URL lazily at call time rather than at module import time,
so importing any Cortex module no longer requires the env var to be set.
This keeps IDE tooling, REPL exploration, and pytest imports working
without needing a live database connection.
"""

import os
import psycopg2


def get_conn():
    """
    Return a new psycopg2 connection to the Cortex PostgreSQL instance.

    Raises KeyError if DATABASE_URL is not set. Intentionally lazy — the
    error only surfaces when code actually needs the database, not at
    import time.
    """
    return psycopg2.connect(os.environ["DATABASE_URL"])
