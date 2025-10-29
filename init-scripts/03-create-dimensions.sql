-- =====================================================
-- DWH Dimensions (Kimball Model)
-- =====================================================

-- ===================
-- dim_symbol (SCD Type 1)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_symbol (
    symbol_id SERIAL PRIMARY KEY,
    symbol_code VARCHAR(20) UNIQUE NOT NULL,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    symbol_type VARCHAR(20) DEFAULT 'forex',
    exchange VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_symbol_code ON dw.dim_symbol(symbol_code);
CREATE INDEX idx_dim_symbol_active ON dw.dim_symbol(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE dw.dim_symbol IS 'Trading symbols dimension (SCD Type 1)';

-- ===================
-- dim_source (SCD Type 1)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_source (
    source_id SERIAL PRIMARY KEY,
    source_name VARCHAR(50) UNIQUE NOT NULL,
    source_type VARCHAR(20) NOT NULL,
    api_endpoint VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    cost_per_call DECIMAL(10,6),
    rate_limit_per_min INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_source_name ON dw.dim_source(source_name);

COMMENT ON TABLE dw.dim_source IS 'Data sources dimension (APIs, files, streams)';

-- ===================
-- dim_time_5m (Pre-populated)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_time_5m (
    time_id SERIAL PRIMARY KEY,
    ts_utc TIMESTAMPTZ UNIQUE NOT NULL,
    ts_cot TIMESTAMPTZ NOT NULL,
    date_cot DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    day INT NOT NULL,
    hour_utc INT NOT NULL,
    hour_cot INT NOT NULL,
    minute INT NOT NULL,
    day_of_week INT NOT NULL,
    is_trading_hour BOOLEAN NOT NULL,
    is_business_day BOOLEAN NOT NULL,
    trading_session VARCHAR(20),
    hhmm_cot VARCHAR(5) NOT NULL
);

CREATE INDEX idx_dim_time_ts_utc ON dw.dim_time_5m(ts_utc);
CREATE INDEX idx_dim_time_date_cot ON dw.dim_time_5m(date_cot);
CREATE INDEX idx_dim_time_trading ON dw.dim_time_5m(is_trading_hour) WHERE is_trading_hour = TRUE;
CREATE INDEX idx_dim_time_hhmm ON dw.dim_time_5m(hhmm_cot);

COMMENT ON TABLE dw.dim_time_5m IS 'Time dimension with 5-minute granularity (pre-populated 2020-2030)';

-- Function to populate dim_time_5m
CREATE OR REPLACE FUNCTION dw.populate_dim_time_5m(
    start_date DATE,
    end_date DATE
) RETURNS VOID AS $$
DECLARE
    current_ts TIMESTAMPTZ;
    cot_ts TIMESTAMPTZ;
    trading_hour BOOLEAN;
    business_day BOOLEAN;
    session VARCHAR(20);
BEGIN
    current_ts := start_date AT TIME ZONE 'UTC';

    WHILE current_ts <= end_date AT TIME ZONE 'UTC' LOOP
        cot_ts := current_ts AT TIME ZONE 'America/Bogota';
        business_day := EXTRACT(DOW FROM cot_ts) BETWEEN 1 AND 5;
        trading_hour := business_day AND
                       EXTRACT(HOUR FROM cot_ts) >= 8 AND
                       EXTRACT(HOUR FROM cot_ts) < 14;

        IF NOT trading_hour THEN
            session := 'closed';
        ELSIF EXTRACT(HOUR FROM cot_ts) < 10 THEN
            session := 'morning';
        ELSIF EXTRACT(HOUR FROM cot_ts) < 12 THEN
            session := 'midday';
        ELSE
            session := 'afternoon';
        END IF;

        INSERT INTO dw.dim_time_5m (
            ts_utc, ts_cot, date_cot, year, month, day,
            hour_utc, hour_cot, minute, day_of_week,
            is_trading_hour, is_business_day, trading_session, hhmm_cot
        ) VALUES (
            current_ts, cot_ts, cot_ts::DATE,
            EXTRACT(YEAR FROM cot_ts)::INT,
            EXTRACT(MONTH FROM cot_ts)::INT,
            EXTRACT(DAY FROM cot_ts)::INT,
            EXTRACT(HOUR FROM current_ts)::INT,
            EXTRACT(HOUR FROM cot_ts)::INT,
            EXTRACT(MINUTE FROM cot_ts)::INT,
            EXTRACT(DOW FROM cot_ts)::INT,
            trading_hour, business_day, session,
            TO_CHAR(cot_ts, 'HH24:MI')
        ) ON CONFLICT (ts_utc) DO NOTHING;

        current_ts := current_ts + INTERVAL '5 minutes';
    END LOOP;

    RAISE NOTICE 'Populated dim_time_5m from % to %', start_date, end_date;
END;
$$ LANGUAGE plpgsql;

-- ===================
-- dim_model (SCD Type 2)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_model (
    model_sk SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    architecture VARCHAR(100),
    framework VARCHAR(50),
    training_start_date DATE,
    training_end_date DATE,
    hyperparams JSONB,
    sha256_hash VARCHAR(64),
    version VARCHAR(20) NOT NULL,
    is_production BOOLEAN DEFAULT FALSE,

    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMPTZ DEFAULT '9999-12-31'::TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_model_id_current ON dw.dim_model(model_id, is_current) WHERE is_current = TRUE;
CREATE INDEX idx_dim_model_prod ON dw.dim_model(is_production) WHERE is_production = TRUE;
CREATE INDEX idx_dim_model_hash ON dw.dim_model(sha256_hash);

COMMENT ON TABLE dw.dim_model IS 'ML models dimension (SCD Type 2 for version history)';

-- ===================
-- dim_feature (SCD Type 1)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_feature (
    feature_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) UNIQUE NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    calculation_formula TEXT,
    tier INT,
    lag_bars INT DEFAULT 0,
    clip_threshold DECIMAL(5,4) DEFAULT 0.005,
    normalization_method VARCHAR(50),
    is_trainable BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_feature_name ON dw.dim_feature(feature_name);
CREATE INDEX idx_dim_feature_trainable ON dw.dim_feature(is_trainable) WHERE is_trainable = TRUE;

COMMENT ON TABLE dw.dim_feature IS 'RL features dimension';

-- ===================
-- dim_indicator (SCD Type 1)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_indicator (
    indicator_id SERIAL PRIMARY KEY,
    indicator_name VARCHAR(50) UNIQUE NOT NULL,
    indicator_family VARCHAR(50) NOT NULL,
    calculation_library VARCHAR(50),
    params JSONB,
    interpretation TEXT,
    signal_thresholds JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_indicator_name ON dw.dim_indicator(indicator_name);
CREATE INDEX idx_dim_indicator_family ON dw.dim_indicator(indicator_family);

COMMENT ON TABLE dw.dim_indicator IS 'Technical indicators dimension';

-- ===================
-- dim_reward_spec (SCD Type 2)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_reward_spec (
    reward_spec_sk SERIAL PRIMARY KEY,
    reward_spec_id VARCHAR(100) NOT NULL,
    reward_function VARCHAR(100) NOT NULL,
    reward_formula TEXT,
    params JSONB,

    pnl_weight DECIMAL(5,3) DEFAULT 1.0,
    cost_penalty DECIMAL(5,3) DEFAULT 1.0,
    hold_penalty DECIMAL(5,3) DEFAULT 0.0,

    sha256_hash VARCHAR(64),

    -- SCD2 fields
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMPTZ DEFAULT '9999-12-31'::TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_reward_spec_id ON dw.dim_reward_spec(reward_spec_id, is_current) WHERE is_current = TRUE;
CREATE INDEX idx_dim_reward_spec_hash ON dw.dim_reward_spec(sha256_hash);

COMMENT ON TABLE dw.dim_reward_spec IS 'RL reward specifications (SCD Type 2 for version tracking)';

-- ===================
-- dim_cost_model (SCD Type 2)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_cost_model (
    cost_model_sk SERIAL PRIMARY KEY,
    cost_model_id VARCHAR(100) NOT NULL,
    cost_model_name VARCHAR(200) NOT NULL,

    spread_p95_bps DECIMAL(8,4),
    spread_p99_bps DECIMAL(8,4),
    slippage_bps DECIMAL(8,4),
    commission_bps DECIMAL(8,4),
    peg_rate_pct DECIMAL(5,2),

    params JSONB,
    sha256_hash VARCHAR(64),

    -- SCD2 fields
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMPTZ DEFAULT '9999-12-31'::TIMESTAMPTZ,
    is_current BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_cost_model_id ON dw.dim_cost_model(cost_model_id, is_current) WHERE is_current = TRUE;
CREATE INDEX idx_dim_cost_model_hash ON dw.dim_cost_model(sha256_hash);

COMMENT ON TABLE dw.dim_cost_model IS 'Trading cost models (SCD Type 2 for version tracking)';

-- ===================
-- dim_episode
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_episode (
    episode_sk SERIAL PRIMARY KEY,
    episode_id VARCHAR(100) UNIQUE NOT NULL,
    symbol_id INT REFERENCES dw.dim_symbol(symbol_id),

    split VARCHAR(10) NOT NULL,  -- 'train', 'val', 'test'
    date_cot DATE NOT NULL,

    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    episode_length INT NOT NULL,

    is_terminal BOOLEAN DEFAULT FALSE,
    is_premium_window BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_episode_id ON dw.dim_episode(episode_id);
CREATE INDEX idx_dim_episode_split ON dw.dim_episode(split);
CREATE INDEX idx_dim_episode_date ON dw.dim_episode(date_cot);

COMMENT ON TABLE dw.dim_episode IS 'RL episodes dimension (60-bar windows for training)';

-- ===================
-- dim_backtest_run (Actually a dimension!)
-- ===================
CREATE TABLE IF NOT EXISTS dw.dim_backtest_run (
    run_sk SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    model_sk INT REFERENCES dw.dim_model(model_sk),
    symbol_id INT REFERENCES dw.dim_symbol(symbol_id),

    split VARCHAR(10) NOT NULL,
    date_range_start DATE NOT NULL,
    date_range_end DATE NOT NULL,

    initial_capital DECIMAL(15,2) DEFAULT 100000,

    features_sha256 VARCHAR(64),
    dataset_sha256 VARCHAR(64),
    model_sha256 VARCHAR(64),

    execution_date DATE NOT NULL,
    minio_manifest_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_dim_backtest_run_id ON dw.dim_backtest_run(run_id);
CREATE INDEX idx_dim_backtest_model ON dw.dim_backtest_run(model_sk);
CREATE INDEX idx_dim_backtest_split ON dw.dim_backtest_run(split);

COMMENT ON TABLE dw.dim_backtest_run IS 'Backtest executions dimension (one per run)';

-- Success message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '✅ DWH Dimensions Created Successfully';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Created 10 dimension tables:';
    RAISE NOTICE '  - dim_symbol (SCD1)';
    RAISE NOTICE '  - dim_source (SCD1)';
    RAISE NOTICE '  - dim_time_5m (pre-populated)';
    RAISE NOTICE '  - dim_model (SCD2)';
    RAISE NOTICE '  - dim_feature (SCD1)';
    RAISE NOTICE '  - dim_indicator (SCD1)';
    RAISE NOTICE '  - dim_reward_spec (SCD2)';
    RAISE NOTICE '  - dim_cost_model (SCD2)';
    RAISE NOTICE '  - dim_episode';
    RAISE NOTICE '  - dim_backtest_run';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;
