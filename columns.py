"""
Cortex — Shared column definitions.

Single source of truth for the biometric and nutrition column names that
appear across multiple modules (analysis, weekly findings, ML pipeline).

Keeping these here avoids the silent-drift failure mode where a column is
added in one place (e.g. a new micronutrient) and forgotten in another.

Pure data — no DB or env dependencies — so this module is safe to import
from anywhere without side effects.
"""

# ─────────────────────────────────────────────────────────────
# BIOMETRICS
# ─────────────────────────────────────────────────────────────

# Output metrics — model targets for the wellness score.
# Never lagged. These are what the model learns to predict.
OUTPUT_COLS = [
    "sleep_duration_min",
    "sleep_efficiency_pct",
    "deep_sleep_min",
    "rem_sleep_min",
    "light_sleep_min",
    "awake_min",
    "time_in_bed_min",
    "hrv_ms",
    "hrv_deep_rmssd",
    "rhr_bpm",
    "spo2_avg_pct",
    "spo2_min_pct",
    "spo2_max_pct",
    "respiratory_rate",
    "vo2_max",
]

# Activity inputs — metrics the user can influence.
# Lagged 1 day in the ML pipeline: yesterday's activity → today's recovery.
ACTIVITY_COLS = [
    "steps",
    "active_zone_min",
    "very_active_min",
    "fairly_active_min",
    "lightly_active_min",
    "sedentary_min",
    "calories_burned",
    "distance_km",
    "time_in_fat_burn_min",
    "time_in_cardio_min",
    "time_in_peak_min",
]

# Every biometric column — used by the analysis engine which treats all
# biometrics symmetrically for correlation work.
BIOMETRIC_COLS = OUTPUT_COLS + ACTIVITY_COLS

# ─────────────────────────────────────────────────────────────
# NUTRITION
# ─────────────────────────────────────────────────────────────

# All tracked nutrition columns in the `nutrition` table.
# Excludes `caffeine_last_time` — it's a TIME value, not numeric.
NUTRITION_COLS = [
    "calories_in", "protein_g", "carbs_g", "fat_g", "fibre_g",
    "sugar_g", "sodium_mg", "water_ml",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
    "trans_fat_g", "cholesterol_mg",
    "alcohol_units", "caffeine_mg",
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

# ─────────────────────────────────────────────────────────────
# ML FEATURE PANELS
# ─────────────────────────────────────────────────────────────
#
# Curated inputs for the wellness model. The full NUTRITION_COLS /
# ACTIVITY_COLS lists above remain the storage schema — we log everything
# Cronometer and Fitbit provide — but the model only sees the panel below.
#
# Rationale: with ~60–90 training rows, feeding 80 features to XGBoost is
# p ≈ N. Most of the dropped columns (trace minerals, individual amino
# acids, fat subtypes, most B vitamins) either lack mechanistic evidence
# for sleep/HRV/recovery on a 1–7 day horizon or are near-collinear with
# a macro already in the panel (e.g. tryptophan ↔ protein_g).
#
# Every output in OUTPUT_COLS has at least three inputs below with
# published mechanism:
#   - Sleep (duration/efficiency/deep/REM): magnesium, vit D, omega-3,
#     fibre, sugar, carbs, alcohol, caffeine, protein
#   - HRV / RHR: alcohol, caffeine, calories, protein, omega-3, water,
#     sodium, potassium, magnesium
#   - SpO2 / respiratory rate: iron, B12, folate, alcohol
#   - VO2 max: activity panel + iron, B12, protein, calories

ML_NUTRITION_PANEL = [
    "calories_in", "protein_g", "carbs_g", "fat_g",
    "fibre_g", "sugar_g", "water_ml",
    "sodium_mg", "potassium_mg", "magnesium_mg", "iron_mg",
    "vitamin_d_iu", "vitamin_b12_mcg", "folate_mcg",
    "omega3_mg",
    "alcohol_units", "caffeine_mg",
]

ML_ACTIVITY_PANEL = [
    "steps", "active_zone_min", "very_active_min",
    "sedentary_min", "distance_km",
]

# ─────────────────────────────────────────────────────────────
# GLUCOSE (v4 — pivot to glucose-targeted product)
# ─────────────────────────────────────────────────────────────
#
# Daily glucose aggregates derived from the `glucose_readings` table
# by ml/glucose_builder.py. These become the model's OUTPUT_COLS in
# the glucose-targeted pipeline — what we predict and optimise.
#
# Designed to degrade gracefully with sparse manual data:
#   - fasting_glucose_mg_dl: single morning reading -> populated
#   - mean / TIR / CV:      >=3 readings/day needed -> NaN otherwise
#   - post_meal_peak_avg:   needs meal+reading pair -> NaN otherwise
#   - post_meal_auc_avg:    needs CGM-dense data   -> Phase 2
#   - dawn_delta:           needs overnight curve  -> Phase 2
GLUCOSE_OUTPUT_COLS = [
    "fasting_glucose_mg_dl",      # first wake reading before any meal
    "mean_glucose_mg_dl",         # daily mean across all readings
    "time_in_range_pct",          # % readings in 70-140 mg/dL
    "glucose_cv_pct",             # SD/mean * 100 (variability)
    "post_meal_peak_avg_mg_dl",   # avg peak across meals (manual: single 1h reading)
    "post_meal_auc_avg",          # avg 2h iAUC across meals (CGM only)
    "dawn_phenomenon_delta",      # fasting - overnight nadir (CGM only)
]

# Medication regimens are stored as daily binary state features
# (on_metformin, on_glp1, ...) so the model can account for them
# without dose-response modelling. Cortex never recommends med
# changes — these are exogenous context only.
MEDICATION_CATEGORIES = [
    "metformin",
    "glp1",          # ozempic, wegovy, mounjaro, zepbound
    "insulin",
    "sglt2",         # jardiance, farxiga
    "sulfonylurea",  # glipizide, glyburide
    "supplement",    # berberine, magnesium, cinnamon, inositol, etc.
    "other",
]
