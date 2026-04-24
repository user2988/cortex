"""Cortex Statistical Analysis Engine — v2"""

import os
import numpy as np
import pandas as pd
import psycopg2
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

DATABASE_URL = os.environ["DATABASE_URL"]

BIOMETRIC_COLS = [
    "sleep_duration_min", "sleep_efficiency_pct", "deep_sleep_min",
    "rem_sleep_min", "light_sleep_min", "awake_min", "time_in_bed_min",
    "hrv_ms", "hrv_deep_rmssd", "rhr_bpm",
    "spo2_avg_pct", "spo2_min_pct", "spo2_max_pct", "respiratory_rate",
    "steps", "active_zone_min", "very_active_min", "fairly_active_min",
    "lightly_active_min", "sedentary_min", "calories_burned",
    "distance_km", "vo2_max",
    "time_in_fat_burn_min", "time_in_cardio_min", "time_in_peak_min",
]

# Excludes alcohol_units, caffeine_mg, caffeine_last_time
NUTRITION_COLS = [
    "calories_in", "protein_g", "carbs_g", "fat_g", "fibre_g",
    "sugar_g", "sodium_mg", "water_ml",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
    "trans_fat_g", "cholesterol_mg",
    "omega3_mg", "omega6_mg", "ala_mg", "epa_mg", "dha_mg",
    "vitamin_a_mcg", "vitamin_d_iu", "vitamin_e_mg", "vitamin_k_mcg",
    "vitamin_c_mg", "thiamine_mg", "riboflavin_mg", "niacin_mg",
    "pantothenic_acid_mg", "vitamin_b6_mg", "biotin_mcg", "folate_mcg",
    "vitamin_b12_mcg",
    "calcium_mg", "iron_mg", "magnesium_mg", "phosphorus_mg",
    "potassium_mg", "zinc_mg",
    "selenium_mcg", "copper_mg", "manganese_mg", "chromium_mcg",
    "iodine_mcg", "molybdenum_mcg",
    "tryptophan_g", "threonine_g", "isoleucine_g", "leucine_g",
    "lysine_g", "methionine_g", "phenylalanine_g", "valine_g",
    "histidine_g", "alanine_g", "arginine_g", "aspartic_acid_g",
    "cystine_g", "glutamic_acid_g", "glycine_g", "proline_g",
    "serine_g", "tyrosine_g", "hydroxyproline_g",
]

COL_LABELS = {
    "sleep_duration_min":      "Sleep Duration (min)",
    "sleep_efficiency_pct":    "Sleep Efficiency (%)",
    "deep_sleep_min":          "Deep Sleep (min)",
    "rem_sleep_min":           "REM Sleep (min)",
    "light_sleep_min":         "Light Sleep (min)",
    "awake_min":               "Awake Time (min)",
    "time_in_bed_min":         "Time in Bed (min)",
    "hrv_ms":                  "HRV RMSSD (ms)",
    "hrv_deep_rmssd":          "HRV Deep RMSSD (ms)",
    "rhr_bpm":                 "Resting Heart Rate (bpm)",
    "spo2_avg_pct":            "SpO2 Average (%)",
    "spo2_min_pct":            "SpO2 Min (%)",
    "spo2_max_pct":            "SpO2 Max (%)",
    "respiratory_rate":        "Respiratory Rate (br/min)",
    "steps":                   "Steps",
    "active_zone_min":         "Active Zone Minutes",
    "very_active_min":         "Very Active (min)",
    "fairly_active_min":       "Fairly Active (min)",
    "lightly_active_min":      "Lightly Active (min)",
    "sedentary_min":           "Sedentary (min)",
    "calories_burned":         "Calories Burned",
    "distance_km":             "Distance (km)",
    "vo2_max":                 "VO2 Max",
    "time_in_fat_burn_min":    "Fat Burn Zone (min)",
    "time_in_cardio_min":      "Cardio Zone (min)",
    "time_in_peak_min":        "Peak Zone (min)",
    "calories_in":             "Calories In",
    "protein_g":               "Protein (g)",
    "carbs_g":                 "Carbohydrates (g)",
    "fat_g":                   "Total Fat (g)",
    "fibre_g":                 "Fibre (g)",
    "sugar_g":                 "Sugar (g)",
    "sodium_mg":               "Sodium (mg)",
    "water_ml":                "Water (ml)",
    "saturated_fat_g":         "Saturated Fat (g)",
    "monounsaturated_fat_g":   "Monounsaturated Fat (g)",
    "polyunsaturated_fat_g":   "Polyunsaturated Fat (g)",
    "trans_fat_g":             "Trans Fat (g)",
    "cholesterol_mg":          "Cholesterol (mg)",
    "omega3_mg":               "Omega-3 (mg)",
    "omega6_mg":               "Omega-6 (mg)",
    "ala_mg":                  "ALA (mg)",
    "epa_mg":                  "EPA (mg)",
    "dha_mg":                  "DHA (mg)",
    "vitamin_a_mcg":           "Vitamin A (mcg)",
    "vitamin_d_iu":            "Vitamin D (IU)",
    "vitamin_e_mg":            "Vitamin E (mg)",
    "vitamin_k_mcg":           "Vitamin K (mcg)",
    "vitamin_c_mg":            "Vitamin C (mg)",
    "thiamine_mg":             "Thiamine / B1 (mg)",
    "riboflavin_mg":           "Riboflavin / B2 (mg)",
    "niacin_mg":               "Niacin / B3 (mg)",
    "pantothenic_acid_mg":     "Pantothenic Acid / B5 (mg)",
    "vitamin_b6_mg":           "Vitamin B6 (mg)",
    "biotin_mcg":              "Biotin / B7 (mcg)",
    "folate_mcg":              "Folate / B9 (mcg)",
    "vitamin_b12_mcg":         "Vitamin B12 (mcg)",
    "calcium_mg":              "Calcium (mg)",
    "iron_mg":                 "Iron (mg)",
    "magnesium_mg":            "Magnesium (mg)",
    "phosphorus_mg":           "Phosphorus (mg)",
    "potassium_mg":            "Potassium (mg)",
    "zinc_mg":                 "Zinc (mg)",
    "selenium_mcg":            "Selenium (mcg)",
    "copper_mg":               "Copper (mg)",
    "manganese_mg":            "Manganese (mg)",
    "chromium_mcg":            "Chromium (mcg)",
    "iodine_mcg":              "Iodine (mcg)",
    "molybdenum_mcg":          "Molybdenum (mcg)",
    "tryptophan_g":            "Tryptophan (g)",
    "threonine_g":             "Threonine (g)",
    "isoleucine_g":            "Isoleucine (g)",
    "leucine_g":               "Leucine (g)",
    "lysine_g":                "Lysine (g)",
    "methionine_g":            "Methionine (g)",
    "phenylalanine_g":         "Phenylalanine (g)",
    "valine_g":                "Valine (g)",
    "histidine_g":             "Histidine (g)",
    "alanine_g":               "Alanine (g)",
    "arginine_g":              "Arginine (g)",
    "aspartic_acid_g":         "Aspartic Acid (g)",
    "cystine_g":               "Cystine (g)",
    "glutamic_acid_g":         "Glutamic Acid (g)",
    "glycine_g":               "Glycine (g)",
    "proline_g":               "Proline (g)",
    "serine_g":                "Serine (g)",
    "tyrosine_g":              "Tyrosine (g)",
    "hydroxyproline_g":        "Hydroxyproline (g)",
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(days: int = None) -> pd.DataFrame:
    """
    Load joined biometrics + nutrition from PostgreSQL.
    days=None returns all available data.
    Excludes device-failure rows (sleep_duration_min == 0).
    """
    bio = ", ".join(f"b.{c}" for c in BIOMETRIC_COLS)
    nut = ", ".join(f"n.{c}" for c in NUTRITION_COLS)
    where = f"WHERE b.date >= CURRENT_DATE - INTERVAL '{days} days'" if days else ""
    sql = f"""
        SELECT b.date, {bio}, {nut}
        FROM biometrics b
        LEFT JOIN nutrition n ON b.date = n.date
        {where}
        ORDER BY b.date
    """
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
    # Device failures: zero sleep when data should exist
    df = df[(df["sleep_duration_min"].isna()) | (df["sleep_duration_min"] > 0)]
    return df


# ─────────────────────────────────────────────────────────────
# TARGETS
# ─────────────────────────────────────────────────────────────

def load_targets() -> dict:
    """Returns {variable: target_value} for all user-defined targets."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT variable, target_value FROM targets")
            return {row[0]: float(row[1]) for row in cur.fetchall()}
    finally:
        conn.close()


def save_target(variable: str, value: float) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO targets (variable, target_value)
                    VALUES (%s, %s)
                    ON CONFLICT (variable) DO UPDATE
                        SET target_value = EXCLUDED.target_value,
                            updated_at   = NOW()
                """, (variable, value))
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# EXPERIMENTS
# ─────────────────────────────────────────────────────────────

def load_experiments() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, variable_a, variable_b, lag_days, method,
                       start_date, duration_days, status, interpretation, created_at
                FROM experiments
                ORDER BY created_at DESC
            """)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"]   = df["start_date"] + pd.to_timedelta(df["duration_days"], unit="d")
        df["is_complete"] = df["end_date"].dt.date <= pd.Timestamp.today().date()
    return df


def create_experiment(name: str, variable_a: str, variable_b: str, lag_days: int,
                      method: str, start_date, duration_days: int) -> int:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO experiments
                        (name, variable_a, variable_b, lag_days, method,
                         start_date, duration_days)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (name, variable_a, variable_b, lag_days, method,
                      start_date, duration_days))
                return cur.fetchone()[0]
    finally:
        conn.close()


def store_interpretation(experiment_id: int, interpretation: str) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE experiments
                    SET status = 'complete', interpretation = %s
                    WHERE id = %s
                """, (interpretation, experiment_id))
    finally:
        conn.close()


def delete_experiment(experiment_id: int) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
    finally:
        conn.close()


def run_experiment_analysis(df: pd.DataFrame, exp) -> dict:
    """
    Correlation over the experiment window.
    Returns stats + pre/during series for dual-color scatter.
    exp can be a dict or a pandas Series (DataFrame row).
    """
    var_a  = exp["variable_a"]
    var_b  = exp["variable_b"]
    lag    = int(exp["lag_days"])
    method = exp["method"]
    start  = pd.Timestamp(exp["start_date"])
    end    = start + pd.Timedelta(days=int(exp["duration_days"]))

    working = df[[var_a]].copy()
    working[var_b] = df[var_b].shift(-lag)
    working = working.dropna()

    pre    = working[working.index < start]
    during = working[(working.index >= start) & (working.index < end)]

    if len(during) < 3:
        return {"error": "Not enough data in experiment window yet — check back soon."}

    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = fn(during[var_a], during[var_b])
    r2 = r ** 2
    slope, intercept, *_ = stats.linregress(during[var_a].values, during[var_b].values)

    # OLS line fit across all available paired data for context
    all_paired = working.dropna()
    if len(all_paired) >= 3:
        full_slope, full_intercept, *_ = stats.linregress(
            all_paired[var_a].values, all_paired[var_b].values)
    else:
        full_slope, full_intercept = slope, intercept

    return dict(
        r2=round(r2, 4), r=round(r, 4), p_value=round(p, 6),
        coefficient=round(slope, 6), intercept=round(intercept, 6),
        n=len(during), label=summary_label(r2, p, slope),
        pre=pre, during=during, all_paired=all_paired,
        full_slope=full_slope, full_intercept=full_intercept,
        pre_avg_a=float(pre[var_a].mean()) if len(pre) else None,
        pre_avg_b=float(pre[var_b].mean()) if len(pre) else None,
        during_avg_a=float(during[var_a].mean()),
        during_avg_b=float(during[var_b].mean()),
    )


# ─────────────────────────────────────────────────────────────
# LLM INTERPRETATION
# ─────────────────────────────────────────────────────────────

def generate_interpretation(var_a: str, var_b: str, r2: float, p_value: float,
                             coefficient: float, lag: int, n: int,
                             pre_avg_a, pre_avg_b,
                             during_avg_a, during_avg_b) -> str:
    import anthropic
    a_lbl    = COL_LABELS.get(var_a, var_a)
    b_lbl    = COL_LABELS.get(var_b, var_b)
    lag_str  = f"{lag}-day lag" if lag else "same day"
    pre_ctx  = ""
    if pre_avg_a is not None:
        pre_ctx = (f"Before: {a_lbl} avg {pre_avg_a:.1f}, {b_lbl} avg {pre_avg_b:.1f}. "
                   f"During: {a_lbl} avg {during_avg_a:.1f}, {b_lbl} avg {during_avg_b:.1f}. ")

    prompt = (f"Variable A: {a_lbl} | Variable B: {b_lbl} | "
              f"R²: {r2} | p-value: {p_value} | Coefficient: {coefficient} | "
              f"Lag: {lag_str} | Sample: {n} days. {pre_ctx}")

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=(
            "You interpret statistical correlation results from a personal health tracking app. "
            "Never imply causation. Frame R² as explained variation. "
            "Always note correlation does not confirm causation. "
            "Do not give health advice or prescribe lifestyle changes. "
            "Describe what the data shows — nothing more. Maximum 3 sentences. "
            "If before/during averages differ noticeably, describe the visible shift."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def load_findings() -> pd.DataFrame:
    """Return all findings ordered by pinned first, then R² descending."""
    sql = """
        SELECT id, variable_a, variable_b, r_squared, p_value, coefficient,
               lag_days, analysis_type, sample_size, calculated_at, pinned
        FROM findings
        ORDER BY pinned DESC, r_squared DESC
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=cols)


def delete_finding(finding_id: int) -> None:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM findings WHERE id = %s", (finding_id,))
    finally:
        conn.close()


def save_finding(variable_a: str, variable_b: str | None, r_squared: float,
                 p_value: float, coefficient: float, lag_days: int,
                 analysis_type: str, sample_size: int, pinned: bool = True) -> None:
    sql = """
        INSERT INTO findings
            (variable_a, variable_b, r_squared, p_value, coefficient,
             lag_days, analysis_type, sample_size, pinned)
        VALUES
            (%(variable_a)s, %(variable_b)s, %(r_squared)s, %(p_value)s,
             %(coefficient)s, %(lag_days)s, %(analysis_type)s, %(sample_size)s, %(pinned)s)
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "variable_a": variable_a, "variable_b": variable_b,
                    "r_squared": r_squared, "p_value": p_value,
                    "coefficient": coefficient, "lag_days": lag_days,
                    "analysis_type": analysis_type, "sample_size": sample_size,
                    "pinned": pinned,
                })
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# LABEL HELPERS
# ─────────────────────────────────────────────────────────────

def r2_label(r2: float) -> str:
    if r2 < 0.10: return "No meaningful correlation"
    if r2 < 0.30: return "Weak"
    if r2 < 0.50: return "Moderate"
    if r2 < 0.70: return "Strong"
    return "Very strong"


def p_label(p: float) -> str:
    if p > 0.05:  return "not statistically significant"
    if p > 0.01:  return "statistically significant"
    return "highly statistically significant"


def summary_label(r2: float, p: float, coef: float) -> str:
    direction = "positive" if coef >= 0 else "negative"
    return f"{r2_label(r2)} {direction} — {p_label(p)}"


# ─────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _pair(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df[[a, b]].dropna()


def pearson_correlation(df: pd.DataFrame, var_a: str, var_b: str) -> dict:
    pair = _pair(df, var_a, var_b)
    if len(pair) < 3:
        return {"error": "Insufficient data (need ≥ 3 paired observations)"}
    r, p = stats.pearsonr(pair[var_a], pair[var_b])
    r2 = r ** 2
    slope, intercept, *_ = stats.linregress(pair[var_a], pair[var_b])
    return dict(r2=round(r2, 4), r=round(r, 4), p_value=round(p, 6),
                coefficient=round(slope, 6), intercept=round(intercept, 6),
                n=len(pair), label=summary_label(r2, p, slope),
                series_a=pair[var_a], series_b=pair[var_b])


def spearman_correlation(df: pd.DataFrame, var_a: str, var_b: str) -> dict:
    pair = _pair(df, var_a, var_b)
    if len(pair) < 3:
        return {"error": "Insufficient data (need ≥ 3 paired observations)"}
    r, p = stats.spearmanr(pair[var_a], pair[var_b])
    r2 = r ** 2
    slope, intercept, *_ = stats.linregress(pair[var_a], pair[var_b])
    return dict(r2=round(r2, 4), r=round(r, 4), p_value=round(p, 6),
                coefficient=round(slope, 6), intercept=round(intercept, 6),
                n=len(pair), label=summary_label(r2, p, slope),
                series_a=pair[var_a], series_b=pair[var_b])


def lagged_correlation(df: pd.DataFrame, var_a: str, var_b: str,
                       lag: int, method: str = "pearson") -> dict:
    """
    Pairs var_a[today] with var_b[today + lag].
    lag=1 → today's nutrition vs tomorrow's biometric.
    """
    shifted = df[[var_a]].copy()
    shifted[var_b] = df[var_b].shift(-lag)
    pair = shifted.dropna()
    if len(pair) < 3:
        return {"error": "Insufficient data after applying lag"}
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = fn(pair[var_a], pair[var_b])
    r2 = r ** 2
    slope, intercept, *_ = stats.linregress(pair[var_a].values, pair[var_b].values)
    return dict(r2=round(r2, 4), r=round(r, 4), p_value=round(p, 6),
                coefficient=round(slope, 6), intercept=round(intercept, 6),
                n=len(pair), lag=lag, label=summary_label(r2, p, slope),
                series_a=pair[var_a], series_b=pair[var_b])


def rolling_avg_correlation(df: pd.DataFrame, var_a: str, var_b: str,
                             window: int = 7, method: str = "pearson") -> dict:
    a = df[var_a].rolling(window, min_periods=max(3, window // 2)).mean()
    b = df[var_b].rolling(window, min_periods=max(3, window // 2)).mean()
    pair = pd.concat([a.rename(var_a), b.rename(var_b)], axis=1).dropna()
    if len(pair) < 3:
        return {"error": "Insufficient data for rolling window"}
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = fn(pair[var_a], pair[var_b])
    r2 = r ** 2
    slope, *_ = stats.linregress(pair[var_a].values, pair[var_b].values)
    return dict(r2=round(r2, 4), r=round(r, 4), p_value=round(p, 6),
                coefficient=round(slope, 6), n=len(pair), window=window,
                label=summary_label(r2, p, slope),
                series_a=pair[var_a], series_b=pair[var_b])


def ols_trend(df: pd.DataFrame, var_a: str) -> dict:
    """OLS of var_a vs time — slope is units/day."""
    series = df[var_a].dropna()
    if len(series) < 3:
        return {"error": "Insufficient data (need ≥ 3 observations)"}
    t = np.arange(len(series))
    model = sm.OLS(series.values, sm.add_constant(t)).fit()
    slope = model.params[1]
    fitted = pd.Series(model.fittedvalues, index=series.index)
    return dict(r2=round(model.rsquared, 4), p_value=round(model.pvalues[1], 6),
                coefficient=round(slope, 6), n=len(series),
                label=summary_label(model.rsquared, model.pvalues[1], slope),
                series=series, fitted=fitted)


def multiple_ols(df: pd.DataFrame, predictors: list[str], outcome: str) -> dict:
    data = df[predictors + [outcome]].dropna()
    if len(data) < len(predictors) + 2:
        return {"error": "Insufficient data for number of predictors"}
    X = sm.add_constant(data[predictors])
    model = sm.OLS(data[outcome], X).fit()
    return dict(r2=round(model.rsquared, 4), r2_adj=round(model.rsquared_adj, 4),
                p_value=round(model.f_pvalue, 6),
                coefficients={p: round(model.params[p], 6) for p in predictors},
                p_values={p: round(model.pvalues[p], 6) for p in predictors},
                n=len(data), fitted=model.fittedvalues, actual=data[outcome])


def anomaly_detection(df: pd.DataFrame, var_a: str,
                      window: int = 30, threshold: float = 1.5) -> dict:
    series = df[var_a].dropna()
    if len(series) < 7:
        return {"error": "Insufficient data (need ≥ 7 observations)"}
    rolling_mean = series.rolling(window, min_periods=7).mean()
    rolling_std  = series.rolling(window, min_periods=7).std()
    z_scores     = (series - rolling_mean) / rolling_std
    anomalies    = z_scores.abs() > threshold
    return dict(series=series, rolling_mean=rolling_mean, z_scores=z_scores,
                anomalies=anomalies, n_anomalies=int(anomalies.sum()),
                n=len(series), threshold=threshold, window=window)


def forecast(df: pd.DataFrame, var_a: str, horizon: int = 7) -> dict:
    series = df[var_a].dropna()
    if len(series) < 14:
        return {"error": "Need ≥ 14 days of data to forecast"}
    prophet_df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon)
    fc     = model.predict(future)
    return dict(series=series, forecast=fc, n=len(series))


def decompose(df: pd.DataFrame, var_a: str, period: int = 7) -> dict:
    series = df[var_a].dropna()
    if len(series) < period * 2:
        return {"error": f"Need ≥ {period * 2} days of data for decomposition"}
    result = seasonal_decompose(series, model="additive", period=period,
                                extrapolate_trend="freq")
    return dict(observed=result.observed, trend=result.trend,
                seasonal=result.seasonal, residual=result.resid, n=len(series))


# ─────────────────────────────────────────────────────────────
# BLOOD PRESSURE
# Canonical DDL lives in database/schema.sql — no duplicate DDL here.
# ─────────────────────────────────────────────────────────────

def save_bp_log(
    date,
    session: str,
    r1_sys: int | None,
    r1_dia: int | None,
    r2_sys: int | None,
    r2_dia: int | None,
) -> None:
    """Upsert a BP session log (AM or PM) for the given date."""
    sql = """
        INSERT INTO blood_pressure_logs
            (date, session, reading_1_systolic, reading_1_diastolic,
             reading_2_systolic, reading_2_diastolic)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (date, session) DO UPDATE SET
            reading_1_systolic  = EXCLUDED.reading_1_systolic,
            reading_1_diastolic = EXCLUDED.reading_1_diastolic,
            reading_2_systolic  = EXCLUDED.reading_2_systolic,
            reading_2_diastolic = EXCLUDED.reading_2_diastolic,
            logged_at           = NOW()
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (date, session, r1_sys, r1_dia, r2_sys, r2_dia))
    finally:
        conn.close()


def load_bp_logs(days: int = None) -> pd.DataFrame:
    """
    Load raw blood_pressure_logs rows.

    Returns a DataFrame with columns: date, session,
    reading_1_systolic, reading_1_diastolic, reading_2_systolic,
    reading_2_diastolic, logged_at.
    """
    where = f"WHERE date >= CURRENT_DATE - INTERVAL '{days} days'" if days else ""
    sql = f"""
        SELECT date, session,
               reading_1_systolic, reading_1_diastolic,
               reading_2_systolic, reading_2_diastolic,
               logged_at
        FROM blood_pressure_logs
        {where}
        ORDER BY date DESC, session
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _map_value(sys_val, dia_val):
    """MAP = (systolic + 2 * diastolic) / 3."""
    if sys_val is None or dia_val is None:
        return None
    try:
        return (float(sys_val) + 2.0 * float(dia_val)) / 3.0
    except (TypeError, ValueError):
        return None


def load_bp_daily_aggregates(days: int = None) -> pd.DataFrame:
    """
    Compute daily MAP aggregates from blood_pressure_logs.

    For each session the MAP of each reading is computed, then averaged
    across the two readings. The daily MAP is the mean across available
    sessions (AM and/or PM).

    Returns a DataFrame with columns: date, map_mmhg, n_sessions,
    am_map, pm_map.
    """
    raw = load_bp_logs(days)
    if raw.empty:
        return pd.DataFrame(columns=["date", "map_mmhg", "n_sessions", "am_map", "pm_map"])

    records = []
    for _, row in raw.iterrows():
        m1 = _map_value(row["reading_1_systolic"], row["reading_1_diastolic"])
        m2 = _map_value(row["reading_2_systolic"], row["reading_2_diastolic"])
        vals = [v for v in [m1, m2] if v is not None]
        session_map = float(np.mean(vals)) if vals else None
        records.append({"date": row["date"], "session": row["session"], "session_map": session_map})

    sess_df = pd.DataFrame(records)
    if sess_df.empty or sess_df["session_map"].isna().all():
        return pd.DataFrame(columns=["date", "map_mmhg", "n_sessions", "am_map", "pm_map"])

    pivoted = sess_df.pivot_table(index="date", columns="session", values="session_map", aggfunc="first")
    pivoted.columns.name = None

    result = pd.DataFrame(index=pivoted.index)
    result["am_map"] = pivoted.get("AM")
    result["pm_map"] = pivoted.get("PM")
    avail = result[["am_map", "pm_map"]].notna()
    result["n_sessions"] = avail.sum(axis=1)
    result["map_mmhg"] = result[["am_map", "pm_map"]].mean(axis=1, skipna=True)
    result = result.reset_index().rename(columns={"date": "date"})
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# ML RUN HISTORY LOADERS
# Both functions return (DataFrame, error_str | None).
# error_str is None on success, a short message on DB failure.
# Callers can distinguish "no rows yet" (empty df, error=None)
# from "query failed / schema mismatch" (empty df, error=<msg>).
# ─────────────────────────────────────────────────────────────

def load_pipeline_log(limit: int = 20) -> tuple[pd.DataFrame, str | None]:
    """Return (df, error) for recent ml_pipeline_log rows, newest first."""
    sql = f"""
        SELECT run_at, status, duration_sec, stage, error_message
        FROM ml_pipeline_log
        ORDER BY run_at DESC
        LIMIT {int(limit)}
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    except Exception as exc:
        return pd.DataFrame(), f"DB error: {exc}"
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["run_at"] = pd.to_datetime(df["run_at"])
    return df, None


def load_model_runs(limit: int = 20) -> tuple[pd.DataFrame, str | None]:
    """Return (df, error) for recent ml_model_runs rows, newest first."""
    sql = f"""
        SELECT id, run_at, confidence_tier, n_rows, n_features,
               train_r2, test_r2, test_mae, test_rmse, top_features, model_path
        FROM ml_model_runs
        ORDER BY run_at DESC
        LIMIT {int(limit)}
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    except Exception as exc:
        return pd.DataFrame(), f"DB error: {exc}"
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["run_at"] = pd.to_datetime(df["run_at"])
        for col in ["train_r2", "test_r2", "test_mae", "test_rmse"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, None
