-- =====================================================
-- USDCOP Trading System - Database Extensions & Base Setup
-- This script runs FIRST (00-) to set up PostgreSQL prerequisites
-- =====================================================
-- File: init-scripts/00-init-extensions.sql
-- Purpose: Create extensions, set timezone, create base schemas
-- Compatible with: timescale/timescaledb:latest-pg15
-- =====================================================

-- =============================================================================
-- 1. POSTGRESQL EXTENSIONS
-- =============================================================================

-- UUID generation for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trigram support for text search and similarity matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- TimescaleDB for time-series optimization (required for hypertables)
-- Note: This is pre-installed in the timescaledb Docker image
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- PostGIS for geospatial data (optional, useful for location-based analysis)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- =============================================================================
-- 2. TIMEZONE CONFIGURATION
-- =============================================================================

-- Set default timezone to UTC for consistent timestamp handling
-- All application logic should convert to local time (America/Bogota) as needed
SET timezone = 'UTC';

-- Make UTC the default for new connections
ALTER DATABASE CURRENT_DATABASE SET timezone TO 'UTC';

-- =============================================================================
-- 3. BASE SCHEMAS
-- =============================================================================

-- Public schema (default) - Used for core trading data
-- Already exists by default, ensure proper permissions
GRANT ALL ON SCHEMA public TO PUBLIC;

-- Data Warehouse schema - Used for analytical/inference data
CREATE SCHEMA IF NOT EXISTS dw;
COMMENT ON SCHEMA dw IS 'Data Warehouse schema for analytics, inference results, and aggregated metrics';

-- Config schema - Used for model registry, feature configs
CREATE SCHEMA IF NOT EXISTS config;
COMMENT ON SCHEMA config IS 'Configuration schema for model registry, features, and system parameters';

-- Trading schema - Used for signals, trades, positions
CREATE SCHEMA IF NOT EXISTS trading;
COMMENT ON SCHEMA trading IS 'Trading schema for signals, trades, positions, and equity curves';

-- Staging schema - Used for temporary/ETL data
CREATE SCHEMA IF NOT EXISTS staging;
COMMENT ON SCHEMA staging IS 'Staging schema for ETL processes and temporary data';

-- Audit schema - Used for tracking changes and compliance
CREATE SCHEMA IF NOT EXISTS audit;
COMMENT ON SCHEMA audit IS 'Audit schema for change tracking, compliance logs, and system events';

-- =============================================================================
-- 4. UTILITY FUNCTIONS
-- =============================================================================

-- Function to check if a table exists
CREATE OR REPLACE FUNCTION table_exists(schema_name TEXT, tbl_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = schema_name
        AND table_name = tbl_name
    );
END;
$$ LANGUAGE plpgsql;

-- Function to get current database size
CREATE OR REPLACE FUNCTION get_db_size()
RETURNS TEXT AS $$
BEGIN
    RETURN pg_size_pretty(pg_database_size(current_database()));
END;
$$ LANGUAGE plpgsql;

-- Function to get table row counts (useful for health checks)
CREATE OR REPLACE FUNCTION get_table_stats()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    row_count BIGINT,
    table_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname::TEXT,
        relname::TEXT,
        n_live_tup::BIGINT,
        pg_size_pretty(pg_total_relation_size(schemaname || '.' || relname))::TEXT
    FROM pg_stat_user_tables
    ORDER BY n_live_tup DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 5. AUDIT LOGGING TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit.init_log (
    id SERIAL PRIMARY KEY,
    script_name TEXT NOT NULL,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN DEFAULT TRUE,
    duration_ms INTEGER,
    message TEXT,
    error_detail TEXT
);

COMMENT ON TABLE audit.init_log IS 'Tracks database initialization script execution';

-- =============================================================================
-- 6. VERIFICATION
-- =============================================================================

-- Log successful execution
INSERT INTO audit.init_log (script_name, message)
VALUES ('00-init-extensions.sql', 'Extensions and base schemas created successfully');

-- Display confirmation
DO $$
DECLARE
    ext_count INTEGER;
    schema_count INTEGER;
BEGIN
    -- Count installed extensions
    SELECT COUNT(*) INTO ext_count
    FROM pg_extension
    WHERE extname IN ('uuid-ossp', 'pg_trgm', 'timescaledb');

    -- Count created schemas
    SELECT COUNT(*) INTO schema_count
    FROM information_schema.schemata
    WHERE schema_name IN ('public', 'dw', 'staging', 'audit');

    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'DATABASE INITIALIZATION - PHASE 0 COMPLETE';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Extensions installed: %', ext_count;
    RAISE NOTICE 'Schemas created: %', schema_count;
    RAISE NOTICE 'Timezone set to: UTC';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;

-- Show installed extensions
SELECT extname AS extension, extversion AS version
FROM pg_extension
WHERE extname IN ('uuid-ossp', 'pg_trgm', 'timescaledb')
ORDER BY extname;
