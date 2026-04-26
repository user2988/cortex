"""
Cortex — Biometric Backfill
Fetches the past BACKFILL_DAYS of Fitbit data and writes any missing values
to PostgreSQL. Existing non-null values are never overwritten (COALESCE upsert).

Runs nightly via GitHub Actions after the daily pipeline.
Can also be triggered manually via workflow_dispatch.
"""

import time
from datetime import date, timedelta

from cortex import FitbitAuth, FitbitClient, store_biometrics

BACKFILL_DAYS = 14


def run_backfill():
    auth   = FitbitAuth()
    client = FitbitClient(auth)

    today = date.today()

    for i in range(1, BACKFILL_DAYS + 1):
        record_date = today - timedelta(days=i)
        sleep_date  = today - timedelta(days=i - 1)  # Fitbit keys sleep to wake-up date (D+1)

        d = record_date.isoformat()
        s = sleep_date.isoformat()

        print(f"\nBackfilling {d} (sleep date: {s})...")

        def safe_fetch(name, fn, *args):
            try:
                result = fn(*args)
                print(f"  {name} OK")
                return result
            except Exception as e:
                print(f"  {name} FAILED: {e}")
                return {}

        activity = safe_fetch("activity",       client.fetch_activity,       d)
        azm      = safe_fetch("azm",            client.fetch_azm,            d)
        hr_zones = safe_fetch("hr_zones",       client.fetch_hr_zones,       d)
        sleep    = safe_fetch("sleep",          client.fetch_sleep,          s)
        hrv      = safe_fetch("hrv",            client.fetch_hrv,            s)
        rhr      = safe_fetch("rhr",            client.fetch_heart_rate,     s)
        spo2     = safe_fetch("spo2",           client.fetch_spo2,           s)
        br       = safe_fetch("breathing_rate", client.fetch_breathing_rate, s)
        vo2max   = safe_fetch("vo2max",         client.fetch_vo2max,         s)

        record = {
            "date": d,
            **activity, **azm, **hr_zones, **sleep, **hrv, **rhr, **spo2, **br, **vo2max,
        }

        store_biometrics(record)
        time.sleep(1)

    print("\nBackfill complete.")


if __name__ == "__main__":
    run_backfill()
