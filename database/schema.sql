-- ─────────────────────────────────────────────────────────────
-- CORTEX — PostgreSQL Schema
-- v4: Biometrics + Scores + Experiments + Findings
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

CREATE INDEX IF NOT EXISTS idx_biometrics_date        ON biometrics(date DESC);
CREATE INDEX IF NOT EXISTS idx_findings_r2            ON findings(r_squared DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_date       ON experiments(start_date DESC);
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


-- ─────────────────────────────────────────────────────────────
-- Additive migrations — safe to re-run on any existing database
-- ─────────────────────────────────────────────────────────────

ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS hrv_deep_rmssd     NUMERIC(6, 2);
ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS spo2_max_pct       NUMERIC(5, 2);
ALTER TABLE biometrics ADD COLUMN IF NOT EXISTS lightly_active_min INTEGER;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   skin_temp_relative;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   sleep_score;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   bedtime_consistency_sd;
ALTER TABLE biometrics DROP COLUMN IF EXISTS   sleep_onset_latency_min;
