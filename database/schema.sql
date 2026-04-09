-- ─────────────────────────────────────────────────────────────
-- CORTEX — PostgreSQL Schema
-- v2: Biometrics + Nutrition + Weight
-- ─────────────────────────────────────────────────────────────

-- Biometrics — one row per day
-- Activity fields reflect yesterday's data (fetched the following morning)
-- Sleep/recovery fields reflect the overnight period ending on this date
CREATE TABLE IF NOT EXISTS biometrics (
    date                    DATE PRIMARY KEY,

    -- Sleep
    sleep_duration_min      INTEGER,
    sleep_efficiency_pct    INTEGER,
    deep_sleep_min          INTEGER,
    rem_sleep_min           INTEGER,
    light_sleep_min         INTEGER,
    awake_min               INTEGER,
    time_in_bed_min         INTEGER,
    sleep_onset_latency_min INTEGER,
    bedtime_consistency_sd  NUMERIC(6, 2),

    -- Recovery
    hrv_ms                  NUMERIC(6, 2),
    rhr_bpm                 INTEGER,
    spo2_avg_pct            NUMERIC(5, 2),
    spo2_min_pct            NUMERIC(5, 2),
    respiratory_rate        NUMERIC(5, 2),

    -- Activity (yesterday)
    steps                   INTEGER,
    active_zone_min         INTEGER,
    very_active_min         INTEGER,
    fairly_active_min       INTEGER,
    sedentary_min           INTEGER,
    calories_burned         INTEGER,
    distance_km             NUMERIC(6, 3),
    vo2_max                 NUMERIC(5, 2),
    time_in_fat_burn_min    INTEGER,
    time_in_cardio_min      INTEGER,
    time_in_peak_min        INTEGER,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);


-- Nutrition — one row per day
-- Populated from Cronometer CSV export (v2)
-- All 84 Cronometer nutrients stored; priority subset noted in PRD
CREATE TABLE IF NOT EXISTS nutrition (
    date                    DATE PRIMARY KEY REFERENCES biometrics(date),

    -- Energy & Macros
    calories_in             NUMERIC(8, 2),
    protein_g               NUMERIC(8, 2),
    carbs_g                 NUMERIC(8, 2),
    fat_g                   NUMERIC(8, 2),
    fibre_g                 NUMERIC(8, 2),
    sugar_g                 NUMERIC(8, 2),
    sodium_mg               NUMERIC(8, 2),
    water_ml                NUMERIC(8, 2),

    -- Fat subtypes
    saturated_fat_g         NUMERIC(8, 2),
    monounsaturated_fat_g   NUMERIC(8, 2),
    polyunsaturated_fat_g   NUMERIC(8, 2),
    trans_fat_g             NUMERIC(8, 2),
    cholesterol_mg          NUMERIC(8, 2),

    -- Stimulants / sleep impactors
    -- NOTE v3: alcohol and caffeine will migrate to a supplements/lifestyle table
    -- once the built-in logger replaces Cronometer
    alcohol_units           NUMERIC(5, 2),
    caffeine_mg             NUMERIC(8, 2),
    caffeine_last_time      TIME,

    -- Omega fatty acids
    omega3_mg               NUMERIC(8, 2),
    omega6_mg               NUMERIC(8, 2),
    ala_mg                  NUMERIC(8, 2),   -- alpha-linolenic acid
    epa_mg                  NUMERIC(8, 2),   -- eicosapentaenoic acid
    dha_mg                  NUMERIC(8, 2),   -- docosahexaenoic acid

    -- Fat-soluble vitamins
    vitamin_a_mcg           NUMERIC(8, 2),
    vitamin_d_iu            NUMERIC(8, 2),
    vitamin_e_mg            NUMERIC(8, 2),
    vitamin_k_mcg           NUMERIC(8, 2),

    -- Water-soluble vitamins
    vitamin_c_mg            NUMERIC(8, 2),
    thiamine_mg             NUMERIC(8, 2),   -- B1
    riboflavin_mg           NUMERIC(8, 2),   -- B2
    niacin_mg               NUMERIC(8, 2),   -- B3
    pantothenic_acid_mg     NUMERIC(8, 2),   -- B5
    vitamin_b6_mg           NUMERIC(8, 2),   -- B6
    biotin_mcg              NUMERIC(8, 2),   -- B7
    folate_mcg              NUMERIC(8, 2),   -- B9
    vitamin_b12_mcg         NUMERIC(8, 2),   -- B12

    -- Priority minerals
    calcium_mg              NUMERIC(8, 2),
    iron_mg                 NUMERIC(8, 2),
    magnesium_mg            NUMERIC(8, 2),
    phosphorus_mg           NUMERIC(8, 2),
    potassium_mg            NUMERIC(8, 2),
    zinc_mg                 NUMERIC(8, 2),

    -- Trace minerals
    selenium_mcg            NUMERIC(8, 2),
    copper_mg               NUMERIC(8, 2),
    manganese_mg            NUMERIC(8, 2),
    chromium_mcg            NUMERIC(8, 2),
    iodine_mcg              NUMERIC(8, 2),
    molybdenum_mcg          NUMERIC(8, 2),

    -- Amino acids (essential)
    tryptophan_g            NUMERIC(8, 4),
    threonine_g             NUMERIC(8, 4),
    isoleucine_g            NUMERIC(8, 4),
    leucine_g               NUMERIC(8, 4),
    lysine_g                NUMERIC(8, 4),
    methionine_g            NUMERIC(8, 4),
    phenylalanine_g         NUMERIC(8, 4),
    valine_g                NUMERIC(8, 4),
    histidine_g             NUMERIC(8, 4),

    -- Amino acids (non-essential)
    alanine_g               NUMERIC(8, 4),
    arginine_g              NUMERIC(8, 4),
    aspartic_acid_g         NUMERIC(8, 4),
    cystine_g               NUMERIC(8, 4),
    glutamic_acid_g         NUMERIC(8, 4),
    glycine_g               NUMERIC(8, 4),
    proline_g               NUMERIC(8, 4),
    serine_g                NUMERIC(8, 4),
    tyrosine_g              NUMERIC(8, 4),
    hydroxyproline_g        NUMERIC(8, 4),

    created_at              TIMESTAMPTZ DEFAULT NOW()
);


-- Weight — one row per week, logged manually every Monday morning
CREATE TABLE IF NOT EXISTS weight (
    date                    DATE PRIMARY KEY,
    weight_value            NUMERIC(5, 2) NOT NULL,
    weight_unit             CHAR(3)       NOT NULL DEFAULT 'kg' CHECK (weight_unit IN ('kg', 'lbs')),

    created_at              TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────
-- Indexes for common query patterns
-- ─────────────────────────────────────────────────────────────

-- Date range queries (rolling averages, trend windows)
CREATE INDEX IF NOT EXISTS idx_biometrics_date ON biometrics(date DESC);
CREATE INDEX IF NOT EXISTS idx_nutrition_date  ON nutrition(date DESC);
CREATE INDEX IF NOT EXISTS idx_weight_date     ON weight(date DESC);
