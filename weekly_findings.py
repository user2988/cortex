"""
Cortex — Weekly Findings Job
Runs every Sunday via GitHub Actions.
Scans biometric self-pairs, filters p < 0.05, ranks by R²,
writes top results to the findings table.
Pinned findings are preserved and count toward the 10-row cap.
"""

import os
import sys
import psycopg2
import pandas as pd
import numpy as np
from scipy import stats

DATABASE_URL = os.environ["DATABASE_URL"]

FINDINGS_CAP = 10
MIN_SAMPLE   = 20
MAX_LAG      = 3

PRIORITY_BIOMETRICS = [
    "hrv_ms", "hrv_deep_rmssd",
    "rhr_bpm",
    "sleep_duration_min", "sleep_efficiency_pct",
    "deep_sleep_min", "rem_sleep_min",
    "spo2_avg_pct",
    "steps", "active_zone_min", "vo2_max",
    "respiratory_rate",
]

BIOMETRIC_PAIRS = [
    ("steps",             "hrv_ms"),
    ("steps",             "sleep_efficiency_pct"),
    ("steps",             "deep_sleep_min"),
    ("very_active_min",   "hrv_ms"),
    ("very_active_min",   "rem_sleep_min"),
    ("active_zone_min",   "hrv_ms"),
    ("distance_km",       "hrv_ms"),
    ("distance_km",       "sleep_efficiency_pct"),
    ("sedentary_min",     "hrv_ms"),
    ("sedentary_min",     "rhr_bpm"),
    ("rhr_bpm",           "sleep_efficiency_pct"),
    ("sleep_duration_min","hrv_ms"),
    ("sleep_efficiency_pct","hrv_ms"),
    ("deep_sleep_min",    "hrv_ms"),
    ("rem_sleep_min",     "hrv_ms"),
]


def load_data() -> pd.DataFrame:
    bio_cols = list({
        col for pair in BIOMETRIC_PAIRS for col in pair
    } | set(PRIORITY_BIOMETRICS))

    select = ", ".join(bio_cols)
    sql = f"SELECT date, {select} FROM biometrics ORDER BY date"

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["sleep_duration_min"].isna()) | (df["sleep_duration_min"] > 0)]
    return df


def pearson_lagged(x: pd.Series, y: pd.Series, lag: int):
    if lag > 0:
        x_shift = x.iloc[:-lag].values
        y_shift = y.iloc[lag:].values
    else:
        x_shift = x.values
        y_shift = y.values

    mask = ~(np.isnan(x_shift) | np.isnan(y_shift))
    if mask.sum() < MIN_SAMPLE:
        return None
    r, p = stats.pearsonr(x_shift[mask], y_shift[mask])
    return r, p, int(mask.sum())


def count_pinned() -> int:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM findings WHERE pinned = TRUE")
            return cur.fetchone()[0]
    finally:
        conn.close()


def replace_auto_findings(results: list[dict]) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM findings WHERE pinned = FALSE")
                for r in results:
                    cur.execute("""
                        INSERT INTO findings
                            (variable_a, variable_b, r_squared, p_value,
                             coefficient, lag_days, analysis_type, sample_size, pinned)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                    """, (
                        r["variable_a"], r["variable_b"],
                        round(r["r_squared"], 4), round(r["p_value"], 8),
                        round(r["coefficient"], 6), r["lag_days"],
                        "pearson", r["sample_size"],
                    ))
        print(f"  Wrote {len(results)} auto findings.")
    finally:
        conn.close()


def run():
    print("Loading data...")
    df = load_data()
    print(f"  Biometric rows: {len(df)}")

    pinned = count_pinned()
    slots  = max(0, FINDINGS_CAP - pinned)
    print(f"  Pinned findings: {pinned}  |  Auto slots: {slots}")

    if slots == 0:
        print("  All slots are pinned — nothing to update.")
        sys.exit(0)

    candidates = []

    print("Running biometric self-correlations...")
    for predictor, outcome in BIOMETRIC_PAIRS:
        if predictor not in df.columns or outcome not in df.columns:
            continue
        for lag in range(MAX_LAG + 1):
            result = pearson_lagged(df[predictor], df[outcome], lag)
            if result is None:
                continue
            r, p, n = result
            if p >= 0.05:
                continue
            candidates.append({
                "variable_a": predictor, "variable_b": outcome,
                "r_squared": r ** 2, "p_value": p,
                "coefficient": r, "lag_days": lag, "sample_size": n,
            })

    print(f"  Significant pairs found: {len(candidates)}")

    seen: dict[tuple, dict] = {}
    for c in candidates:
        key = (c["variable_a"], c["variable_b"])
        if key not in seen or c["r_squared"] > seen[key]["r_squared"]:
            seen[key] = c
    deduped = sorted(seen.values(), key=lambda x: x["r_squared"], reverse=True)
    top = deduped[:slots]

    print(f"  Top {len(top)} findings selected.")
    for i, f in enumerate(top, 1):
        lag_str = f"  lag={f['lag_days']}d" if f["lag_days"] > 0 else ""
        print(f"  {i:2}. {f['variable_a']} → {f['variable_b']}  "
              f"R²={f['r_squared']:.3f}  p={f['p_value']:.4f}  "
              f"n={f['sample_size']}{lag_str}")

    replace_auto_findings(top)
    print("Done.")


if __name__ == "__main__":
    run()
