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
