-- ─────────────────────────────────────────────────────────────
-- CORTEX — PostgreSQL Schema
-- v3: Biometrics + Nutrition + Weight
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
    hrv_deep_rmssd          NUMERIC(6, 2),
    rhr_bpm                 INTEGER,
    spo2_avg_pct            NUMERIC(5, 2),
    spo2_min_pct            NUMERIC(5, 2),
    spo2_max_pct            NUMERIC(5, 2),
    respiratory_rate        NUMERIC(5, 2),

    -- Activity (yesterday)
    steps                   INTEGER,
    active_zone_min         INTEGER,
    very_active_min         INTEGER,
    fairly_active_min       INTEGER,
    lightly_active_min      INTEGER,
    sedentary_min           INTEGER,
    calories_burned         INTEGER,
    distance_km             NUMERIC(6, 3),
    vo2_max                 NUMERIC(5, 2),
    time_in_fat_burn_min    INTEGER,
    time_in_cardio_min      INTEGER,
    time_in_peak_min        INTEGER,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);


-- v3 additive migrations (apply to existing databases without a full reset)
ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS hrv_deep_rmssd     NUMERIC(6, 2);
ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS spo2_max_pct       NUMERIC(5, 2);
ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS lightly_active_min INTEGER;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   skin_temp_relative;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   sleep_score;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   bedtime_consistency_sd;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   sleep_onset_latency_min;

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


-- Blood Pressure — up to 2 readings per session (AM / PM) per day
-- MAP (Mean Arterial Pressure) = (systolic + 2 * diastolic) / 3
-- Logged manually via the Streamlit UI; unique per date + session.
CREATE TABLE IF NOT EXISTS blood_pressure_logs (
    id                      SERIAL PRIMARY KEY,
    date                    DATE        NOT NULL,
    session                 TEXT        NOT NULL CHECK (session IN ('AM', 'PM')),

    reading_1_systolic      INTEGER,
    reading_1_diastolic     INTEGER,
    reading_2_systolic      INTEGER,
    reading_2_diastolic     INTEGER,

    logged_at               TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (date, session)
);


-- Findings — correlation/analysis results, updated weekly by the findings job
-- Also populated on demand via the Streamlit Explorer (pinned=true for manual saves)
CREATE TABLE IF NOT EXISTS findings (
    id              SERIAL PRIMARY KEY,
    variable_a      TEXT          NOT NULL,
    variable_b      TEXT,
    r_squared       NUMERIC(6, 4),
    p_value         NUMERIC(10, 8),
    coefficient     NUMERIC(10, 6),
    lag_days        INTEGER       DEFAULT 0,
    analysis_type   TEXT          NOT NULL,
    sample_size     INTEGER,
    calculated_at   TIMESTAMPTZ   DEFAULT NOW(),
    pinned          BOOLEAN       DEFAULT FALSE
);

-- Experiments — user-defined hypothesis tests with a fixed duration
CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    name            TEXT          NOT NULL,
    variable_a      TEXT          NOT NULL,
    variable_b      TEXT          NOT NULL,
    lag_days        INTEGER       DEFAULT 0,
    method          TEXT          DEFAULT 'pearson',
    start_date      DATE          NOT NULL,
    duration_days   INTEGER       NOT NULL,
    status          TEXT          DEFAULT 'active',
    interpretation  TEXT,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- Targets — user-defined daily targets for 30-Day Trends nutrition rows
CREATE TABLE IF NOT EXISTS targets (
    variable        TEXT          PRIMARY KEY,
    target_value    NUMERIC(10, 2) NOT NULL,
    updated_at      TIMESTAMPTZ   DEFAULT NOW()
);


-- Daily Scores — Sleep Score + Heart Score, computed by ml/score_engine.py
-- Both scores are 0–100, relative to the individual's own rolling 30-day baseline.
CREATE TABLE IF NOT EXISTS daily_scores (
    date                DATE        PRIMARY KEY,
    sleep_score         NUMERIC(5, 1),
    heart_score         NUMERIC(5, 1),
    -- Sleep components
    duration_score      NUMERIC(5, 1),
    deep_score          NUMERIC(5, 1),
    rem_score           NUMERIC(5, 1),
    efficiency_score    NUMERIC(5, 1),
    -- Heart components
    hrv_score           NUMERIC(5, 1),
    rhr_score           NUMERIC(5, 1),
    spo2_score          NUMERIC(5, 1),
    -- Raw values for display
    sleep_duration_min  INTEGER,
    deep_pct            NUMERIC(5, 2),
    rem_pct             NUMERIC(5, 2),
    hrv_ms              NUMERIC(6, 2),
    rhr_bpm             INTEGER,
    spo2_avg_pct        NUMERIC(5, 2),
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);


-- Score Recommendations — activity ranges that produce the best scores
-- Replaced entirely on each pipeline run; no historical retention needed.
CREATE TABLE IF NOT EXISTS score_recommendations (
    id                  SERIAL PRIMARY KEY,
    generated_at        TIMESTAMPTZ NOT NULL,
    target_score        TEXT        NOT NULL,   -- 'sleep' | 'heart'
    activity_metric     TEXT        NOT NULL,
    activity_label      TEXT        NOT NULL,
    optimal_min         NUMERIC(10, 2),
    optimal_max         NUMERIC(10, 2),
    optimal_min_fmt     TEXT,
    optimal_max_fmt     TEXT,
    avg_score_in_range  NUMERIC(5, 1),
    avg_score_outside   NUMERIC(5, 1),
    score_delta         NUMERIC(5, 1),
    correlation         NUMERIC(6, 4),
    sample_size         INTEGER,
    recommendation_text TEXT        NOT NULL
);


-- ─────────────────────────────────────────────────────────────
-- Indexes for common query patterns
-- ─────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_biometrics_date  ON biometrics(date DESC);
CREATE INDEX IF NOT EXISTS idx_nutrition_date   ON nutrition(date DESC);
CREATE INDEX IF NOT EXISTS idx_weight_date      ON weight(date DESC);
CREATE INDEX IF NOT EXISTS idx_findings_r2      ON findings(r_squared DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_date ON experiments(start_date DESC);
CREATE INDEX IF NOT EXISTS idx_bp_logs_date          ON blood_pressure_logs(date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_scores_date      ON daily_scores(date DESC);
CREATE INDEX IF NOT EXISTS idx_score_recs_generated   ON score_recommendations(generated_at DESC);


-- ─────────────────────────────────────────────────────────────
-- ML TABLES  (canonical DDL — ML modules self-create on first
-- run as a safety net, but this file is the authoritative source)
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_pipeline_log (
    id            SERIAL PRIMARY KEY,
    run_at        TIMESTAMPTZ  NOT NULL,
    status        TEXT         NOT NULL,   -- 'success' | 'failed' | 'skipped'
    duration_sec  NUMERIC(8, 2),
    stage         TEXT,
    error_message TEXT,
    created_at    TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_model_runs (
    id              SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL,
    confidence_tier TEXT        NOT NULL,
    n_rows          INTEGER     NOT NULL,
    n_features      INTEGER     NOT NULL,
    train_r2        NUMERIC(6, 4),
    test_r2         NUMERIC(6, 4),
    test_mae        NUMERIC(8, 4),
    test_rmse       NUMERIC(8, 4),
    top_features    JSONB,
    model_path      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_recommendations (
    id               SERIAL PRIMARY KEY,
    run_at           TIMESTAMPTZ NOT NULL,
    model_run_id     INTEGER REFERENCES ml_model_runs(id),
    confidence_tier  TEXT        NOT NULL,
    n_days_data      INTEGER     NOT NULL,
    current_map_avg  NUMERIC(6, 2),
    predicted_map    NUMERIC(6, 2),
    recommendations  JSONB       NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_recommendation_outcomes (
    id                  SERIAL PRIMARY KEY,
    recommendation_id   INTEGER REFERENCES ml_recommendations(id),
    evaluated_at        TIMESTAMPTZ  NOT NULL,
    map_before_avg      NUMERIC(6, 2),
    map_after_avg       NUMERIC(6, 2),
    map_delta           NUMERIC(6, 2),
    predicted_delta     NUMERIC(6, 2),
    n_days_before       INTEGER,
    n_days_after        INTEGER,
    created_at          TIMESTAMPTZ  DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────
-- Additive migrations — safe to re-run on any existing database
-- ─────────────────────────────────────────────────────────────

-- MAP-era columns (added when wellness score was replaced by BP target).
-- Old wellness_* columns are left in place; no data is deleted.
-- New inserts write only to the map_* columns.
ALTER TABLE IF EXISTS ml_recommendations
    ADD COLUMN IF NOT EXISTS current_map_avg NUMERIC(6, 2);
ALTER TABLE IF EXISTS ml_recommendations
    ADD COLUMN IF NOT EXISTS predicted_map   NUMERIC(6, 2);

ALTER TABLE IF EXISTS ml_recommendation_outcomes
    ADD COLUMN IF NOT EXISTS map_before_avg NUMERIC(6, 2);
ALTER TABLE IF EXISTS ml_recommendation_outcomes
    ADD COLUMN IF NOT EXISTS map_after_avg  NUMERIC(6, 2);
ALTER TABLE IF EXISTS ml_recommendation_outcomes
    ADD COLUMN IF NOT EXISTS map_delta      NUMERIC(6, 2);

-- BP cross-field safety constraints: systolic must exceed diastolic when
-- both values are present. NOT VALID skips retroactive row checks so
-- existing data is never blocked; all future inserts/updates are validated.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'bp_r1_systolic_gt_diastolic'
    ) THEN
        ALTER TABLE blood_pressure_logs
            ADD CONSTRAINT bp_r1_systolic_gt_diastolic
            CHECK (reading_1_systolic  IS NULL
                OR reading_1_diastolic IS NULL
                OR reading_1_systolic  > reading_1_diastolic)
            NOT VALID;
    END IF;
END$$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'bp_r2_systolic_gt_diastolic'
    ) THEN
        ALTER TABLE blood_pressure_logs
            ADD CONSTRAINT bp_r2_systolic_gt_diastolic
            CHECK (reading_2_systolic  IS NULL
                OR reading_2_diastolic IS NULL
                OR reading_2_systolic  > reading_2_diastolic)
            NOT VALID;
    END IF;
END$$;
