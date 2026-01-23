-- ============================================================================
-- Forecasting BI Schema
-- ============================================================================
-- Purpose: Store ML forecasting predictions, metrics, and consensus data
-- Version: 1.0.0
-- Created: 2026-01-22
-- ============================================================================

-- Schema for Forecasting BI
CREATE SCHEMA IF NOT EXISTS bi;

-- ============================================================================
-- DIMENSION TABLES
-- ============================================================================

-- Dimension: Models
CREATE TABLE IF NOT EXISTS bi.dim_models (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(20) NOT NULL CHECK (model_type IN ('linear', 'boosting', 'hybrid')),
    description TEXT,
    requires_scaling BOOLEAN DEFAULT FALSE,
    supports_early_stopping BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dimension: Horizons
CREATE TABLE IF NOT EXISTS bi.dim_horizons (
    horizon_id INTEGER PRIMARY KEY,
    horizon_label VARCHAR(20) NOT NULL,
    horizon_category VARCHAR(20) NOT NULL CHECK (horizon_category IN ('short', 'medium', 'long')),
    days INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- FACT TABLES
-- ============================================================================

-- Fact: Forecasts (main prediction table)
CREATE TABLE IF NOT EXISTS bi.fact_forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inference_date DATE NOT NULL,
    inference_week INTEGER NOT NULL CHECK (inference_week BETWEEN 1 AND 53),
    inference_year INTEGER NOT NULL CHECK (inference_year BETWEEN 2020 AND 2100),
    target_date DATE NOT NULL,
    model_id VARCHAR(50) NOT NULL REFERENCES bi.dim_models(model_id),
    horizon_id INTEGER NOT NULL REFERENCES bi.dim_horizons(horizon_id),
    base_price DECIMAL(12, 4) NOT NULL,
    predicted_price DECIMAL(12, 4) NOT NULL,
    predicted_return_pct DECIMAL(8, 4),
    price_change DECIMAL(12, 4),
    direction VARCHAR(4) CHECK (direction IN ('UP', 'DOWN')),
    signal INTEGER CHECK (signal IN (-1, 0, 1)),
    confidence DECIMAL(5, 4),
    actual_price DECIMAL(12, 4),
    direction_correct BOOLEAN,
    minio_week_path TEXT,
    mlflow_run_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_forecast_date_model_horizon UNIQUE(inference_date, model_id, horizon_id)
);

-- Fact: Consensus (aggregated predictions across all models)
CREATE TABLE IF NOT EXISTS bi.fact_consensus (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inference_date DATE NOT NULL,
    horizon_id INTEGER NOT NULL REFERENCES bi.dim_horizons(horizon_id),
    avg_predicted_price DECIMAL(12, 4),
    median_predicted_price DECIMAL(12, 4),
    std_predicted_price DECIMAL(12, 4),
    min_predicted_price DECIMAL(12, 4),
    max_predicted_price DECIMAL(12, 4),
    consensus_direction VARCHAR(4) CHECK (consensus_direction IN ('UP', 'DOWN')),
    bullish_count INTEGER DEFAULT 0,
    bearish_count INTEGER DEFAULT 0,
    total_models INTEGER DEFAULT 0,
    agreement_pct DECIMAL(5, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_consensus_date_horizon UNIQUE(inference_date, horizon_id)
);

-- Fact: Model Performance Metrics
CREATE TABLE IF NOT EXISTS bi.fact_model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_date DATE NOT NULL,
    evaluation_date DATE NOT NULL,
    model_id VARCHAR(50) NOT NULL REFERENCES bi.dim_models(model_id),
    horizon_id INTEGER NOT NULL REFERENCES bi.dim_horizons(horizon_id),
    direction_accuracy DECIMAL(6, 4),
    rmse DECIMAL(12, 4),
    mae DECIMAL(12, 4),
    mape DECIMAL(8, 4),
    r2 DECIMAL(6, 4),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    sample_count INTEGER,
    train_start_date DATE,
    train_end_date DATE,
    test_start_date DATE,
    test_end_date DATE,
    mlflow_run_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_metrics_train_model_horizon UNIQUE(training_date, model_id, horizon_id)
);

-- Fact: Weekly Inference Runs (audit trail)
CREATE TABLE IF NOT EXISTS bi.fact_inference_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    inference_week INTEGER NOT NULL,
    inference_year INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    models_count INTEGER DEFAULT 0,
    horizons_count INTEGER DEFAULT 0,
    forecasts_generated INTEGER DEFAULT 0,
    duration_seconds INTEGER,
    minio_artifacts_path TEXT,
    mlflow_experiment_id VARCHAR(100),
    error_message TEXT,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_forecasts_date ON bi.fact_forecasts(inference_date DESC);
CREATE INDEX IF NOT EXISTS idx_forecasts_model ON bi.fact_forecasts(model_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_horizon ON bi.fact_forecasts(horizon_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_week_year ON bi.fact_forecasts(inference_year, inference_week);
CREATE INDEX IF NOT EXISTS idx_forecasts_direction ON bi.fact_forecasts(direction);

CREATE INDEX IF NOT EXISTS idx_consensus_date ON bi.fact_consensus(inference_date DESC);
CREATE INDEX IF NOT EXISTS idx_consensus_horizon ON bi.fact_consensus(horizon_id);

CREATE INDEX IF NOT EXISTS idx_metrics_model ON bi.fact_model_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_metrics_date ON bi.fact_model_metrics(training_date DESC);

-- ============================================================================
-- SEED DATA: Dimensions
-- ============================================================================

-- Seed Horizons
INSERT INTO bi.dim_horizons (horizon_id, horizon_label, horizon_category, days) VALUES
(1, '1 day', 'short', 1),
(5, '5 days', 'short', 5),
(10, '10 days', 'medium', 10),
(15, '15 days', 'medium', 15),
(20, '20 days', 'medium', 20),
(25, '25 days', 'long', 25),
(30, '30 days', 'long', 30)
ON CONFLICT (horizon_id) DO NOTHING;

-- Seed Models
INSERT INTO bi.dim_models (model_id, model_name, model_type, description, requires_scaling, supports_early_stopping) VALUES
-- Linear Models
('ridge', 'Ridge Regression', 'linear', 'L2 regularized linear regression', TRUE, FALSE),
('bayesian_ridge', 'Bayesian Ridge', 'linear', 'Bayesian ridge regression with uncertainty estimation', TRUE, FALSE),
('ard', 'ARD Regression', 'linear', 'Automatic Relevance Determination regression', TRUE, FALSE),
-- Pure Boosting Models
('xgboost_pure', 'XGBoost', 'boosting', 'Extreme Gradient Boosting', FALSE, TRUE),
('lightgbm_pure', 'LightGBM', 'boosting', 'Light Gradient Boosting Machine', FALSE, TRUE),
('catboost_pure', 'CatBoost', 'boosting', 'Categorical Boosting', FALSE, TRUE),
-- Hybrid Models (Boosting + Ridge ensemble)
('hybrid_xgboost', 'XGBoost Hybrid', 'hybrid', 'XGBoost + Ridge ensemble', TRUE, TRUE),
('hybrid_lightgbm', 'LightGBM Hybrid', 'hybrid', 'LightGBM + Ridge ensemble', TRUE, TRUE),
('hybrid_catboost', 'CatBoost Hybrid', 'hybrid', 'CatBoost + Ridge ensemble', TRUE, TRUE)
ON CONFLICT (model_id) DO UPDATE SET
    model_name = EXCLUDED.model_name,
    model_type = EXCLUDED.model_type,
    description = EXCLUDED.description,
    requires_scaling = EXCLUDED.requires_scaling,
    supports_early_stopping = EXCLUDED.supports_early_stopping,
    updated_at = NOW();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Latest forecasts per model/horizon
CREATE OR REPLACE VIEW bi.v_latest_forecasts AS
SELECT DISTINCT ON (model_id, horizon_id)
    f.*,
    m.model_name,
    m.model_type,
    h.horizon_label,
    h.horizon_category
FROM bi.fact_forecasts f
JOIN bi.dim_models m ON f.model_id = m.model_id
JOIN bi.dim_horizons h ON f.horizon_id = h.horizon_id
ORDER BY model_id, horizon_id, inference_date DESC;

-- View: Latest consensus per horizon
CREATE OR REPLACE VIEW bi.v_latest_consensus AS
SELECT DISTINCT ON (horizon_id)
    c.*,
    h.horizon_label,
    h.horizon_category
FROM bi.fact_consensus c
JOIN bi.dim_horizons h ON c.horizon_id = h.horizon_id
ORDER BY horizon_id, inference_date DESC;

-- View: Model performance summary (latest metrics)
CREATE OR REPLACE VIEW bi.v_model_performance AS
SELECT DISTINCT ON (model_id, horizon_id)
    mm.*,
    m.model_name,
    m.model_type,
    h.horizon_label
FROM bi.fact_model_metrics mm
JOIN bi.dim_models m ON mm.model_id = m.model_id
JOIN bi.dim_horizons h ON mm.horizon_id = h.horizon_id
ORDER BY model_id, horizon_id, training_date DESC;

-- View: Dashboard summary (aggregated stats)
CREATE OR REPLACE VIEW bi.v_dashboard_summary AS
SELECT
    f.inference_date,
    f.inference_year,
    f.inference_week,
    COUNT(DISTINCT f.model_id) as models_count,
    COUNT(*) as forecasts_count,
    AVG(f.predicted_return_pct) as avg_return_pct,
    SUM(CASE WHEN f.direction = 'UP' THEN 1 ELSE 0 END) as bullish_count,
    SUM(CASE WHEN f.direction = 'DOWN' THEN 1 ELSE 0 END) as bearish_count
FROM bi.fact_forecasts f
GROUP BY f.inference_date, f.inference_year, f.inference_week
ORDER BY f.inference_date DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Calculate consensus for a given date and horizon
CREATE OR REPLACE FUNCTION bi.calculate_consensus(
    p_inference_date DATE,
    p_horizon_id INTEGER
) RETURNS void AS $$
DECLARE
    v_avg_price DECIMAL(12, 4);
    v_median_price DECIMAL(12, 4);
    v_std_price DECIMAL(12, 4);
    v_min_price DECIMAL(12, 4);
    v_max_price DECIMAL(12, 4);
    v_bullish INTEGER;
    v_bearish INTEGER;
    v_total INTEGER;
    v_direction VARCHAR(4);
BEGIN
    -- Calculate statistics
    SELECT
        AVG(predicted_price),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY predicted_price),
        STDDEV(predicted_price),
        MIN(predicted_price),
        MAX(predicted_price),
        SUM(CASE WHEN direction = 'UP' THEN 1 ELSE 0 END),
        SUM(CASE WHEN direction = 'DOWN' THEN 1 ELSE 0 END),
        COUNT(*)
    INTO v_avg_price, v_median_price, v_std_price, v_min_price, v_max_price,
         v_bullish, v_bearish, v_total
    FROM bi.fact_forecasts
    WHERE inference_date = p_inference_date
      AND horizon_id = p_horizon_id;

    -- Determine consensus direction
    IF v_bullish > v_bearish THEN
        v_direction := 'UP';
    ELSE
        v_direction := 'DOWN';
    END IF;

    -- Upsert consensus
    INSERT INTO bi.fact_consensus (
        inference_date, horizon_id, avg_predicted_price, median_predicted_price,
        std_predicted_price, min_predicted_price, max_predicted_price,
        consensus_direction, bullish_count, bearish_count, total_models,
        agreement_pct
    ) VALUES (
        p_inference_date, p_horizon_id, v_avg_price, v_median_price,
        v_std_price, v_min_price, v_max_price,
        v_direction, v_bullish, v_bearish, v_total,
        GREATEST(v_bullish, v_bearish)::DECIMAL / NULLIF(v_total, 0) * 100
    )
    ON CONFLICT (inference_date, horizon_id) DO UPDATE SET
        avg_predicted_price = EXCLUDED.avg_predicted_price,
        median_predicted_price = EXCLUDED.median_predicted_price,
        std_predicted_price = EXCLUDED.std_predicted_price,
        min_predicted_price = EXCLUDED.min_predicted_price,
        max_predicted_price = EXCLUDED.max_predicted_price,
        consensus_direction = EXCLUDED.consensus_direction,
        bullish_count = EXCLUDED.bullish_count,
        bearish_count = EXCLUDED.bearish_count,
        total_models = EXCLUDED.total_models,
        agreement_pct = EXCLUDED.agreement_pct,
        created_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-calculate consensus when forecasts are inserted
CREATE OR REPLACE FUNCTION bi.trigger_update_consensus()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM bi.calculate_consensus(NEW.inference_date, NEW.horizon_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_forecast_consensus ON bi.fact_forecasts;
CREATE TRIGGER trg_forecast_consensus
AFTER INSERT OR UPDATE ON bi.fact_forecasts
FOR EACH ROW
EXECUTE FUNCTION bi.trigger_update_consensus();

-- ============================================================================
-- GRANTS (adjust for your DB users)
-- ============================================================================

-- Grant usage to application user (adjust username as needed)
-- GRANT USAGE ON SCHEMA bi TO app_user;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA bi TO app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA bi TO app_user;
