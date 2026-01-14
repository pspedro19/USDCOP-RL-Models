-- Migration: 008_ffill_tracking_and_readiness.sql
-- Purpose: Add FFILL tracking columns and readiness logging
-- Contract: CTR-DB-008
-- Date: 2025-01-13

-- =============================================================================
-- 1. FFILL TRACKING METADATA TABLE
-- =============================================================================
-- Instead of adding columns per indicator (37+ columns), use a separate table
-- to track FFILL metadata per indicator per date.

CREATE TABLE IF NOT EXISTS macro_ffill_metadata (
    id SERIAL PRIMARY KEY,
    fecha DATE NOT NULL,
    column_name VARCHAR(100) NOT NULL,
    is_ffilled BOOLEAN DEFAULT FALSE,
    ffill_days INTEGER DEFAULT 0,
    ffill_source_date DATE,
    original_value NUMERIC(20, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint: one record per date per indicator
    UNIQUE(fecha, column_name)
);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_ffill_metadata_fecha ON macro_ffill_metadata(fecha);
CREATE INDEX IF NOT EXISTS idx_ffill_metadata_column ON macro_ffill_metadata(column_name);
CREATE INDEX IF NOT EXISTS idx_ffill_metadata_is_ffilled ON macro_ffill_metadata(is_ffilled) WHERE is_ffilled = TRUE;

COMMENT ON TABLE macro_ffill_metadata IS 'Tracks forward-fill metadata per macro indicator per date. Contract: CTR-DB-008-FFILL';


-- =============================================================================
-- 2. DAILY READINESS LOG TABLE
-- =============================================================================
-- Stores daily readiness reports for audit and monitoring

CREATE TABLE IF NOT EXISTS macro_readiness_log (
    id SERIAL PRIMARY KEY,
    report_date DATE NOT NULL,
    report_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    pipeline_execution_id VARCHAR(50),

    -- Overall status
    is_ready_for_inference BOOLEAN DEFAULT FALSE,
    readiness_score NUMERIC(5, 4) DEFAULT 0.0,

    -- Indicator counts
    total_indicators INTEGER DEFAULT 37,
    indicators_fresh INTEGER DEFAULT 0,
    indicators_ffilled INTEGER DEFAULT 0,
    indicators_stale INTEGER DEFAULT 0,
    indicators_missing INTEGER DEFAULT 0,
    indicators_error INTEGER DEFAULT 0,

    -- FFILL summary
    ffill_total_rows INTEGER DEFAULT 0,
    ffill_exceeded_limit INTEGER DEFAULT 0,

    -- Issues
    blocking_issues JSONB DEFAULT '[]'::JSONB,
    warnings JSONB DEFAULT '[]'::JSONB,

    -- Full indicator details (for debugging)
    indicator_details JSONB DEFAULT '[]'::JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- One report per date
    UNIQUE(report_date)
);

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_readiness_log_date ON macro_readiness_log(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_readiness_log_ready ON macro_readiness_log(is_ready_for_inference);

COMMENT ON TABLE macro_readiness_log IS 'Daily data readiness reports - single source of truth for inference. Contract: CTR-DB-008-READINESS';


-- =============================================================================
-- 3. EXTRACTION LOG TABLE
-- =============================================================================
-- Stores extraction attempt results for each source

CREATE TABLE IF NOT EXISTS macro_extraction_log (
    id SERIAL PRIMARY KEY,
    extraction_date DATE NOT NULL,
    source VARCHAR(50) NOT NULL,  -- fred, twelvedata, investing, banrep, bcrp

    -- Timing
    batch_start TIMESTAMP WITH TIME ZONE,
    batch_end TIMESTAMP WITH TIME ZONE,
    duration_seconds NUMERIC(10, 2),

    -- Counts
    total_indicators INTEGER DEFAULT 0,
    success_new INTEGER DEFAULT 0,
    success_same INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,

    -- Computed rates
    success_rate NUMERIC(5, 4) GENERATED ALWAYS AS (
        CASE WHEN total_indicators > 0
             THEN (success_new + success_same)::NUMERIC / total_indicators
             ELSE 0
        END
    ) STORED,
    new_data_rate NUMERIC(5, 4) GENERATED ALWAYS AS (
        CASE WHEN total_indicators > 0
             THEN success_new::NUMERIC / total_indicators
             ELSE 0
        END
    ) STORED,

    -- Details (individual attempts)
    attempts JSONB DEFAULT '[]'::JSONB,
    errors JSONB DEFAULT '[]'::JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for queries
CREATE INDEX IF NOT EXISTS idx_extraction_log_date ON macro_extraction_log(extraction_date DESC);
CREATE INDEX IF NOT EXISTS idx_extraction_log_source ON macro_extraction_log(source);

COMMENT ON TABLE macro_extraction_log IS 'Logs extraction attempts per source for monitoring and debugging. Contract: CTR-DB-008-EXTRACT';


-- =============================================================================
-- 4. VIEW: Latest Readiness Status
-- =============================================================================
-- Convenient view for checking current readiness status

CREATE OR REPLACE VIEW v_macro_readiness_current AS
SELECT
    report_date,
    is_ready_for_inference,
    readiness_score,
    indicators_fresh,
    indicators_ffilled,
    indicators_stale,
    indicators_missing,
    blocking_issues,
    report_time
FROM macro_readiness_log
WHERE report_date = CURRENT_DATE
ORDER BY report_time DESC
LIMIT 1;

COMMENT ON VIEW v_macro_readiness_current IS 'Current day macro data readiness status';


-- =============================================================================
-- 5. VIEW: Extraction Success Summary
-- =============================================================================

CREATE OR REPLACE VIEW v_extraction_summary AS
SELECT
    extraction_date,
    source,
    total_indicators,
    success_new,
    success_same,
    failures,
    success_rate,
    new_data_rate,
    duration_seconds
FROM macro_extraction_log
WHERE extraction_date >= CURRENT_DATE - 7
ORDER BY extraction_date DESC, source;

COMMENT ON VIEW v_extraction_summary IS 'Recent extraction success rates by source';


-- =============================================================================
-- 6. FUNCTION: Get Indicator Status
-- =============================================================================
-- Helper function to get current status of a specific indicator

CREATE OR REPLACE FUNCTION get_indicator_status(p_column_name VARCHAR)
RETURNS TABLE (
    column_name VARCHAR,
    fecha DATE,
    value NUMERIC,
    is_ffilled BOOLEAN,
    ffill_days INTEGER,
    status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p_column_name,
        m.fecha,
        CASE p_column_name
            WHEN 'fxrt_index_dxy_usa_d_dxy' THEN m.fxrt_index_dxy_usa_d_dxy
            WHEN 'volt_vix_usa_d_vix' THEN m.volt_vix_usa_d_vix
            WHEN 'finc_bond_yield10y_usa_d_ust10y' THEN m.finc_bond_yield10y_usa_d_ust10y
            -- Add other columns as needed
            ELSE NULL
        END,
        COALESCE(f.is_ffilled, FALSE),
        COALESCE(f.ffill_days, 0),
        CASE
            WHEN f.is_ffilled IS NULL OR NOT f.is_ffilled THEN 'fresh'
            WHEN f.ffill_days <= 5 THEN 'ffilled'
            ELSE 'stale'
        END
    FROM macro_indicators_daily m
    LEFT JOIN macro_ffill_metadata f ON m.fecha = f.fecha AND f.column_name = p_column_name
    WHERE m.fecha = CURRENT_DATE;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_indicator_status IS 'Get current status of a specific macro indicator';
