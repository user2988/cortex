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

    -- Recovery
    hrv_ms                  NUMERIC(6, 2),
    rhr_bpm                 INTEGER,
    spo2_avg_pct            NUMERIC(5, 2),

    -- Activity (yesterday)
    steps                   INTEGER,
    active_zone_min         INTEGER,
    very_active_min         INTEGER,
    fairly_active_min       INTEGER,
    sedentary_min           INTEGER,
    calories_burned         INTEGER,
    distance_km             NUMERIC(6, 3),
    vo2_max                 NUMERIC(5, 2),

    created_at              TIMESTAMPTZ DEFAULT NOW()
);


-- Nutrition — one row per day
-- Populated from Cronometer CSV export (v2)
CREATE TABLE IF NOT EXISTS nutrition (
    date                    DATE PRIMARY KEY REFERENCES biometrics(date),

    -- Macros
    calories_in             NUMERIC(8, 2),
    protein_g               NUMERIC(8, 2),
    carbs_g                 NUMERIC(8, 2),
    fat_g                   NUMERIC(8, 2),
    fibre_g                 NUMERIC(8, 2),
    sugar_g                 NUMERIC(8, 2),
    sodium_mg               NUMERIC(8, 2),
    water_ml                NUMERIC(8, 2),

    -- Stimulants / sleep impactors
    -- NOTE v3: alcohol and caffeine will migrate to a supplements/lifestyle table
    -- once the built-in logger replaces Cronometer
    alcohol_units           NUMERIC(5, 2),
    caffeine_mg             NUMERIC(8, 2),

    -- Priority micronutrients (highest analytical value per PRD)
    magnesium_mg            NUMERIC(8, 2),
    iron_mg                 NUMERIC(8, 2),
    zinc_mg                 NUMERIC(8, 2),
    calcium_mg              NUMERIC(8, 2),
    potassium_mg            NUMERIC(8, 2),
    vitamin_d_iu            NUMERIC(8, 2),
    omega3_mg               NUMERIC(8, 2),
    vitamin_b6_mg           NUMERIC(8, 2),
    vitamin_b12_mcg         NUMERIC(8, 2),
    folate_mcg              NUMERIC(8, 2),
    vitamin_c_mg            NUMERIC(8, 2),
    vitamin_e_mg            NUMERIC(8, 2),
    selenium_mcg            NUMERIC(8, 2),

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
