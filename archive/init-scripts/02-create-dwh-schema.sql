-- ========================================================================
-- DWH SCHEMA CREATION - Professional Data Warehouse Architecture
-- Sistema USDCOP Trading - Kimball Model Implementation
-- ========================================================================
-- Version: 1.0
-- Date: 2025-10-22
-- Description: Creates staging, data warehouse, and data mart schemas
-- ========================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ========================================================================
-- STEP 1: Create Schemas
-- ========================================================================

-- Staging schema: Temporary tables for ETL processes
CREATE SCHEMA IF NOT EXISTS stg;
COMMENT ON SCHEMA stg IS 'Staging area for ETL operations - temporary tables';

-- Data Warehouse schema: Kimball star schema (dimensions + facts)
CREATE SCHEMA IF NOT EXISTS dw;
COMMENT ON SCHEMA dw IS 'Data Warehouse - Kimball dimensional model (dims + facts)';

-- Data Marts schema: Materialized views and aggregates for BI
CREATE SCHEMA IF NOT EXISTS dm;
COMMENT ON SCHEMA dm IS 'Data Marts - Materialized views and aggregates for BI dashboards';

-- ========================================================================
-- STEP 2: Grant Permissions
-- ========================================================================

-- Grant usage on schemas to admin user
GRANT USAGE ON SCHEMA stg TO admin;
GRANT USAGE ON SCHEMA dw TO admin;
GRANT USAGE ON SCHEMA dm TO admin;

-- Grant all privileges on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA stg TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dw TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dm TO admin;

-- Grant all privileges on all sequences
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA stg TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dw TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dm TO admin;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA stg GRANT ALL ON TABLES TO admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA dw GRANT ALL ON TABLES TO admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA dm GRANT ALL ON TABLES TO admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA stg GRANT ALL ON SEQUENCES TO admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA dw GRANT ALL ON SEQUENCES TO admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA dm GRANT ALL ON SEQUENCES TO admin;

-- ========================================================================
-- STEP 3: Create Utility Functions
-- ========================================================================

-- Function to generate SHA256 hash from text
CREATE OR REPLACE FUNCTION dw.generate_sha256(input_text TEXT)
RETURNS VARCHAR(64) AS $$
BEGIN
    RETURN encode(digest(input_text, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION dw.generate_sha256 IS 'Generate SHA256 hash for data lineage tracking';

-- Function to convert COT timestamp to UTC
CREATE OR REPLACE FUNCTION dw.cot_to_utc(ts_cot TIMESTAMPTZ)
RETURNS TIMESTAMPTZ AS $$
BEGIN
    RETURN ts_cot AT TIME ZONE 'America/Bogota' AT TIME ZONE 'UTC';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION dw.cot_to_utc IS 'Convert Colombian time (COT) to UTC';

-- Function to convert UTC to COT timestamp
CREATE OR REPLACE FUNCTION dw.utc_to_cot(ts_utc TIMESTAMPTZ)
RETURNS TIMESTAMPTZ AS $$
BEGIN
    RETURN ts_utc AT TIME ZONE 'UTC' AT TIME ZONE 'America/Bogota';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION dw.utc_to_cot IS 'Convert UTC to Colombian time (COT)';

-- Function to check if timestamp is in trading hours
CREATE OR REPLACE FUNCTION dw.is_trading_hour(ts_cot TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
DECLARE
    hour_cot INT;
    dow_cot INT;
BEGIN
    hour_cot := EXTRACT(HOUR FROM ts_cot);
    dow_cot := EXTRACT(DOW FROM ts_cot);

    -- Monday-Friday (1-5) and 8am-2pm COT
    RETURN dow_cot BETWEEN 1 AND 5 AND hour_cot >= 8 AND hour_cot < 14;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION dw.is_trading_hour IS 'Check if timestamp falls in Colombian trading hours (Mon-Fri 8am-2pm COT)';

-- ========================================================================
-- STEP 4: Create Audit Log Table
-- ========================================================================

CREATE TABLE IF NOT EXISTS dw.audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    schema_name VARCHAR(50) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(20) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    rows_affected INT,
    dag_id VARCHAR(200),
    run_id VARCHAR(200),
    task_id VARCHAR(200),
    execution_date TIMESTAMPTZ,
    user_name VARCHAR(100) DEFAULT CURRENT_USER,
    query_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON dw.audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_table ON dw.audit_log(schema_name, table_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_dag ON dw.audit_log(dag_id, run_id);

COMMENT ON TABLE dw.audit_log IS 'Audit log for all DWH operations - tracks data lineage';

-- Function to log operations
CREATE OR REPLACE FUNCTION dw.log_operation(
    p_schema_name VARCHAR(50),
    p_table_name VARCHAR(100),
    p_operation VARCHAR(20),
    p_rows_affected INT,
    p_dag_id VARCHAR(200) DEFAULT NULL,
    p_run_id VARCHAR(200) DEFAULT NULL,
    p_task_id VARCHAR(200) DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO dw.audit_log (
        schema_name, table_name, operation, rows_affected,
        dag_id, run_id, task_id, execution_date
    ) VALUES (
        p_schema_name, p_table_name, p_operation, p_rows_affected,
        p_dag_id, p_run_id, p_task_id, NOW()
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION dw.log_operation IS 'Log DWH operations for auditing and lineage tracking';

-- ========================================================================
-- STEP 5: Create Health Check View
-- ========================================================================

CREATE OR REPLACE VIEW dw.health_check AS
SELECT
    'dw' AS schema_name,
    COUNT(*) AS table_count,
    COALESCE(SUM(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))), 0)::BIGINT AS total_size_bytes,
    pg_size_pretty(COALESCE(SUM(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))), 0)) AS total_size_pretty,
    NOW() AS checked_at
FROM pg_tables
WHERE schemaname = 'dw'
UNION ALL
SELECT
    'dm' AS schema_name,
    COUNT(*) AS table_count,
    COALESCE(SUM(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))), 0)::BIGINT AS total_size_bytes,
    pg_size_pretty(COALESCE(SUM(pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))), 0)) AS total_size_pretty,
    NOW() AS checked_at
FROM pg_tables
WHERE schemaname = 'dm';

COMMENT ON VIEW dw.health_check IS 'Health check view for DWH schemas';

-- ========================================================================
-- COMPLETION MESSAGE
-- ========================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '✅ DWH Schema Creation Complete!';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE 'Created schemas:';
    RAISE NOTICE '  - stg: Staging area for ETL';
    RAISE NOTICE '  - dw:  Data Warehouse (Kimball dimensional model)';
    RAISE NOTICE '  - dm:  Data Marts (BI views and aggregates)';
    RAISE NOTICE '';
    RAISE NOTICE 'Created utility functions:';
    RAISE NOTICE '  - dw.generate_sha256()';
    RAISE NOTICE '  - dw.cot_to_utc()';
    RAISE NOTICE '  - dw.utc_to_cot()';
    RAISE NOTICE '  - dw.is_trading_hour()';
    RAISE NOTICE '  - dw.log_operation()';
    RAISE NOTICE '';
    RAISE NOTICE 'Created audit log table: dw.audit_log';
    RAISE NOTICE 'Created health check view: dw.health_check';
    RAISE NOTICE '';
    RAISE NOTICE 'Next step: Run 03-create-dimensions.sql';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '';
END $$;
