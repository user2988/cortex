"""
Cortex ML — BP Target

Computes the daily PM Mean Arterial Pressure (MAP) from blood_pressure_logs
and aligns it to the feature DataFrame produced by data_builder.build().

Why PM specifically?
--------------------
The afternoon reading captures the cumulative effect of the day's activity,
nutrition, and sleep on blood pressure — making it the most informative
single reading for predicting lifestyle-driven change. The prior-day AM and
PM readings are both available as *input features* via data_builder; this
module provides only the *target* the model learns to predict.

MAP formula
-----------
    MAP = (systolic + 2 × diastolic) / 3

For the PM session, MAP is computed for each available reading and averaged
across the two readings to give one value per day.

Lower MAP is better. The stack optimiser minimises the predicted PM MAP
rather than maximising it.
"""

import os
import psycopg2
import numpy as np
import pandas as pd

DATABASE_URL = os.environ["DATABASE_URL"]


def _map_value(sys_val, dia_val) -> float | None:
    """Return MAP for a single reading, or None if either value is absent."""
    if sys_val is None or dia_val is None:
        return None
    try:
        return (float(sys_val) + 2.0 * float(dia_val)) / 3.0
    except (TypeError, ValueError):
        return None


def _load_pm_map() -> pd.Series:
    """
    Load PM session data from blood_pressure_logs and return daily PM MAP.

    Returns a pd.Series indexed by date (tz-naive), name='pm_map'.
    Only days with a logged PM reading are included; all others are NaN
    after reindex.
    """
    sql = """
        SELECT date,
               reading_1_systolic, reading_1_diastolic,
               reading_2_systolic, reading_2_diastolic
        FROM blood_pressure_logs
        WHERE session = 'PM'
          AND reading_1_systolic  IS NOT NULL
          AND reading_1_diastolic IS NOT NULL
        ORDER BY date
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    except Exception:
        return pd.Series(dtype=float, name="pm_map")
    finally:
        conn.close()

    if not rows:
        return pd.Series(dtype=float, name="pm_map")

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])

    df["map_r1"] = df.apply(
        lambda r: _map_value(r["reading_1_systolic"], r["reading_1_diastolic"]), axis=1
    )
    df["map_r2"] = df.apply(
        lambda r: _map_value(r["reading_2_systolic"], r["reading_2_diastolic"]), axis=1
    )
    df["pm_map"] = df[["map_r1", "map_r2"]].mean(axis=1, skipna=True)

    daily = df.set_index("date")["pm_map"]
    daily.name = "pm_map"
    return daily


def compute(df: pd.DataFrame) -> pd.Series:
    """
    Return daily PM MAP values aligned to the feature DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed DataFrame from data_builder.build().

    Returns
    -------
    pd.Series
        Float PM MAP values (mmHg), indexed identically to df.
        Dates without a logged PM reading receive NaN — those rows are
        excluded from model training automatically.
        Lower is better: the stack optimiser minimises predicted PM MAP.
    """
    pm_map = _load_pm_map()
    aligned = pm_map.reindex(df.index)
    aligned.name = "pm_map"

    valid = aligned.dropna()
    if not valid.empty:
        print(f"  PM MAP rows : {len(valid)} / {len(aligned)}")
        print(f"  PM MAP range: {valid.min():.1f} – {valid.max():.1f} mmHg")
        print(f"  PM MAP mean : {valid.mean():.1f} mmHg")

    return aligned


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
    from ml import data_builder

    df = data_builder.build()
    if df.empty:
        print("No data available.")
        sys.exit(0)

    scores = compute(df)
    print(scores.dropna().tail(10))
