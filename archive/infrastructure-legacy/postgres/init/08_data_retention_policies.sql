-- ========================================
-- USDCOP Trading System - Data Retention Policies
-- Comprehensive data lifecycle management and cleanup procedures
-- ========================================

\echo '🗑️ Setting up data retention policies...'

-- ========================================
-- DATA RETENTION POLICY CONFIGURATION
-- ========================================

-- Create a configuration table for retention policies
CREATE TABLE IF NOT EXISTS data_retention_config (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL UNIQUE,
    retention_days INTEGER NOT NULL,
    retention_policy TEXT NOT NULL,
    cleanup_enabled BOOLEAN DEFAULT true,
    last_cleanup TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert retention policies
INSERT INTO data_retention_config (table_name, retention_days, retention_policy) VALUES
('market_data', -1, 'Permanent retention - core trading data'),
('trading_signals', -1, 'Permanent retention - performance analysis required'),
('trading_performance', -1, 'Permanent retention - historical performance tracking'),
('users', -1, 'Permanent retention - manual deletion only'),
('pipeline_runs', 90, 'Keep 90 days - operational audit trail'),
('api_usage', 30, 'Keep 30 days - rate limiting and monitoring'),
('backup_metadata', 30, 'Keep active backups + 30 days after deactivation'),
('ready_signals', 7, 'Keep 7 days after processing - coordination logs'),
('data_gaps', 30, 'Keep 30 days after resolution - quality tracking'),
('pipeline_health', 30, 'Keep 30 days - operational monitoring'),
('system_metrics', 30, 'Keep 30 days default, varies by category'),
('data_quality_checks', 60, 'Keep 60 days - quality history and trends'),
('user_sessions', 0, 'Delete immediately upon expiration - security'),
('websocket_connections', 1, 'Keep 1 day after disconnection - connection logs')
ON CONFLICT (table_name) DO UPDATE SET
    retention_days = EXCLUDED.retention_days,
    retention_policy = EXCLUDED.retention_policy,
    updated_at = NOW();

-- ========================================
-- ENHANCED CLEANUP FUNCTIONS
-- ========================================

-- Function to get retention policy for a table
CREATE OR REPLACE FUNCTION get_retention_policy(table_name_param VARCHAR)
RETURNS TABLE(
    table_name VARCHAR,
    retention_days INTEGER,
    retention_policy TEXT,
    cleanup_enabled BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        drc.table_name,
        drc.retention_days,
        drc.retention_policy,
        drc.cleanup_enabled
    FROM data_retention_config drc
    WHERE drc.table_name = table_name_param;
END;
$$ LANGUAGE plpgsql;

-- Enhanced cleanup function with detailed reporting
CREATE OR REPLACE FUNCTION comprehensive_data_cleanup()
RETURNS TABLE(
    table_name TEXT,
    retention_days INTEGER,
    cutoff_date TIMESTAMPTZ,
    records_deleted INTEGER,
    cleanup_status TEXT
) AS $$
DECLARE
    cleanup_record RECORD;
    deleted_count INTEGER;
    cutoff_timestamp TIMESTAMPTZ;
    result_record RECORD;
BEGIN
    -- Loop through each table with retention policy
    FOR cleanup_record IN
        SELECT drc.table_name, drc.retention_days, drc.cleanup_enabled
        FROM data_retention_config drc
        WHERE drc.cleanup_enabled = true AND drc.retention_days >= 0
        ORDER BY drc.table_name
    LOOP
        deleted_count := 0;
        cutoff_timestamp := NOW() - (cleanup_record.retention_days || ' days')::INTERVAL;

        -- Execute cleanup based on table type
        CASE cleanup_record.table_name
            WHEN 'api_usage' THEN
                EXECUTE format('DELETE FROM %I WHERE request_datetime < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'pipeline_runs' THEN
                EXECUTE format('DELETE FROM %I WHERE created_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'backup_metadata' THEN
                EXECUTE format('DELETE FROM %I WHERE is_active = false AND updated_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'ready_signals' THEN
                EXECUTE format('DELETE FROM %I WHERE status IN (''processed'', ''expired'') AND updated_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'data_gaps' THEN
                EXECUTE format('DELETE FROM %I WHERE resolution_status = ''resolved'' AND resolved_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'pipeline_health' THEN
                EXECUTE format('DELETE FROM %I WHERE measured_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'system_metrics' THEN
                -- Special handling for system_metrics with category-based retention
                DELETE FROM system_metrics
                WHERE (
                    (category = 'api' AND measured_at < NOW() - INTERVAL '7 days') OR
                    (category = 'pipeline' AND measured_at < NOW() - INTERVAL '30 days') OR
                    (category = 'trading' AND measured_at < NOW() - INTERVAL '90 days') OR
                    (category = 'system' AND measured_at < NOW() - INTERVAL '14 days') OR
                    (category NOT IN ('api', 'pipeline', 'trading', 'system') AND measured_at < cutoff_timestamp)
                );
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'data_quality_checks' THEN
                EXECUTE format('DELETE FROM %I WHERE created_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'user_sessions' THEN
                -- Always clean expired sessions regardless of retention days
                DELETE FROM user_sessions WHERE expires_at < NOW();
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            WHEN 'websocket_connections' THEN
                EXECUTE format('DELETE FROM %I WHERE status = ''disconnected'' AND disconnected_at < $1', cleanup_record.table_name)
                USING cutoff_timestamp;
                GET DIAGNOSTICS deleted_count = ROW_COUNT;

            ELSE
                -- Generic cleanup for other tables (assumes created_at column)
                BEGIN
                    EXECUTE format('DELETE FROM %I WHERE created_at < $1', cleanup_record.table_name)
                    USING cutoff_timestamp;
                    GET DIAGNOSTICS deleted_count = ROW_COUNT;
                EXCEPTION
                    WHEN undefined_column THEN
                        deleted_count := -1; -- Indicates column not found
                END;
        END CASE;

        -- Update last cleanup timestamp
        UPDATE data_retention_config
        SET last_cleanup = NOW()
        WHERE table_name = cleanup_record.table_name;

        -- Return result
        RETURN QUERY SELECT
            cleanup_record.table_name::TEXT,
            cleanup_record.retention_days,
            cutoff_timestamp,
            deleted_count,
            CASE
                WHEN deleted_count = -1 THEN 'ERROR: Column not found'
                WHEN deleted_count = 0 THEN 'SUCCESS: No records to delete'
                ELSE 'SUCCESS: ' || deleted_count || ' records deleted'
            END::TEXT;

    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Function to estimate cleanup impact before execution
CREATE OR REPLACE FUNCTION estimate_cleanup_impact()
RETURNS TABLE(
    table_name TEXT,
    retention_days INTEGER,
    cutoff_date TIMESTAMPTZ,
    estimated_deletions BIGINT,
    table_size_before TEXT,
    estimated_size_after TEXT
) AS $$
DECLARE
    cleanup_record RECORD;
    deletion_count BIGINT;
    cutoff_timestamp TIMESTAMPTZ;
    current_size BIGINT;
    estimated_remaining BIGINT;
BEGIN
    FOR cleanup_record IN
        SELECT drc.table_name, drc.retention_days
        FROM data_retention_config drc
        WHERE drc.cleanup_enabled = true AND drc.retention_days >= 0
        ORDER BY drc.table_name
    LOOP
        cutoff_timestamp := NOW() - (cleanup_record.retention_days || ' days')::INTERVAL;
        deletion_count := 0;

        -- Estimate deletions based on table type
        CASE cleanup_record.table_name
            WHEN 'api_usage' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM api_usage WHERE request_datetime < cutoff_timestamp;

            WHEN 'pipeline_runs' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM pipeline_runs WHERE created_at < cutoff_timestamp;

            WHEN 'backup_metadata' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM backup_metadata WHERE is_active = false AND updated_at < cutoff_timestamp;

            WHEN 'ready_signals' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM ready_signals WHERE status IN ('processed', 'expired') AND updated_at < cutoff_timestamp;

            WHEN 'data_gaps' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM data_gaps WHERE resolution_status = 'resolved' AND resolved_at < cutoff_timestamp;

            WHEN 'pipeline_health' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM pipeline_health WHERE measured_at < cutoff_timestamp;

            WHEN 'system_metrics' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM system_metrics
                WHERE (
                    (category = 'api' AND measured_at < NOW() - INTERVAL '7 days') OR
                    (category = 'pipeline' AND measured_at < NOW() - INTERVAL '30 days') OR
                    (category = 'trading' AND measured_at < NOW() - INTERVAL '90 days') OR
                    (category = 'system' AND measured_at < NOW() - INTERVAL '14 days')
                );

            WHEN 'user_sessions' THEN
                SELECT COUNT(*) INTO deletion_count
                FROM user_sessions WHERE expires_at < NOW();

            ELSE
                -- Generic count for other tables
                BEGIN
                    EXECUTE format('SELECT COUNT(*) FROM %I WHERE created_at < $1', cleanup_record.table_name)
                    INTO deletion_count
                    USING cutoff_timestamp;
                EXCEPTION
                    WHEN undefined_column THEN
                        deletion_count := 0;
                END;
        END CASE;

        -- Get current table size
        SELECT pg_total_relation_size(cleanup_record.table_name) INTO current_size;

        RETURN QUERY SELECT
            cleanup_record.table_name::TEXT,
            cleanup_record.retention_days,
            cutoff_timestamp,
            deletion_count,
            pg_size_pretty(current_size),
            pg_size_pretty(GREATEST(current_size * 0.1, current_size - (deletion_count * 1024)))::TEXT; -- Rough estimate

    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Function to safely execute cleanup with confirmation
CREATE OR REPLACE FUNCTION execute_data_cleanup(
    confirm_cleanup BOOLEAN DEFAULT false,
    table_filter VARCHAR DEFAULT NULL
)
RETURNS TABLE(
    operation TEXT,
    table_name TEXT,
    records_affected INTEGER,
    status TEXT,
    execution_time INTERVAL
) AS $$
DECLARE
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
    cleanup_result RECORD;
BEGIN
    start_time := NOW();

    IF NOT confirm_cleanup THEN
        RETURN QUERY SELECT
            'ESTIMATE'::TEXT,
            est.table_name,
            est.estimated_deletions::INTEGER,
            'Estimation only - set confirm_cleanup=true to execute'::TEXT,
            INTERVAL '0 seconds';
        FROM estimate_cleanup_impact() est
        WHERE table_filter IS NULL OR est.table_name = table_filter;
        RETURN;
    END IF;

    -- Execute actual cleanup
    FOR cleanup_result IN
        SELECT * FROM comprehensive_data_cleanup()
        WHERE table_filter IS NULL OR table_name = table_filter
    LOOP
        end_time := NOW();

        RETURN QUERY SELECT
            'CLEANUP'::TEXT,
            cleanup_result.table_name,
            cleanup_result.records_deleted,
            cleanup_result.cleanup_status,
            end_time - start_time;
    END LOOP;

    -- Log cleanup execution
    INSERT INTO system_metrics (
        metric_name,
        metric_value,
        metric_unit,
        category,
        subcategory,
        metadata,
        measured_at
    ) VALUES (
        'data_cleanup_executed',
        1,
        'execution',
        'system',
        'maintenance',
        jsonb_build_object(
            'table_filter', COALESCE(table_filter, 'all'),
            'execution_time', EXTRACT(EPOCH FROM (NOW() - start_time))
        ),
        NOW()
    );

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- MAINTENANCE PROCEDURES
-- ========================================

-- Function to analyze table bloat and recommend VACUUM
CREATE OR REPLACE FUNCTION analyze_table_maintenance()
RETURNS TABLE(
    table_name TEXT,
    live_tuples BIGINT,
    dead_tuples BIGINT,
    bloat_ratio DECIMAL(5,2),
    last_vacuum TIMESTAMPTZ,
    last_analyze TIMESTAMPTZ,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname || '.' || tablename as table_name,
        n_live_tup as live_tuples,
        n_dead_tup as dead_tuples,
        CASE
            WHEN n_live_tup + n_dead_tup = 0 THEN 0.00
            ELSE ROUND((n_dead_tup * 100.0) / (n_live_tup + n_dead_tup), 2)
        END as bloat_ratio,
        last_vacuum,
        last_analyze,
        CASE
            WHEN (n_dead_tup * 100.0) / GREATEST(n_live_tup + n_dead_tup, 1) > 20 THEN 'VACUUM recommended'
            WHEN last_analyze < NOW() - INTERVAL '7 days' THEN 'ANALYZE recommended'
            WHEN last_vacuum < NOW() - INTERVAL '30 days' AND n_dead_tup > 1000 THEN 'VACUUM recommended'
            ELSE 'No action needed'
        END as recommendation
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY (n_dead_tup * 100.0) / GREATEST(n_live_tup + n_dead_tup, 1) DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to execute recommended maintenance
CREATE OR REPLACE FUNCTION execute_table_maintenance(
    table_name_param VARCHAR DEFAULT NULL,
    force_vacuum BOOLEAN DEFAULT false
)
RETURNS TABLE(
    table_name TEXT,
    operation TEXT,
    duration INTERVAL,
    status TEXT
) AS $$
DECLARE
    maint_record RECORD;
    start_time TIMESTAMPTZ;
    end_time TIMESTAMPTZ;
BEGIN
    FOR maint_record IN
        SELECT
            atm.table_name,
            atm.bloat_ratio,
            atm.recommendation
        FROM analyze_table_maintenance() atm
        WHERE (table_name_param IS NULL OR atm.table_name LIKE '%' || table_name_param || '%')
        AND (atm.recommendation != 'No action needed' OR force_vacuum)
    LOOP
        start_time := NOW();

        IF maint_record.recommendation LIKE '%VACUUM%' OR force_vacuum THEN
            EXECUTE format('VACUUM ANALYZE %s', maint_record.table_name);
            end_time := NOW();

            RETURN QUERY SELECT
                maint_record.table_name,
                'VACUUM ANALYZE'::TEXT,
                end_time - start_time,
                'Completed successfully'::TEXT;

        ELSIF maint_record.recommendation LIKE '%ANALYZE%' THEN
            EXECUTE format('ANALYZE %s', maint_record.table_name);
            end_time := NOW();

            RETURN QUERY SELECT
                maint_record.table_name,
                'ANALYZE'::TEXT,
                end_time - start_time,
                'Completed successfully'::TEXT;
        END IF;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- AUTOMATED CLEANUP SCHEDULER SETUP
-- ========================================

-- Function to check if cleanup should run based on schedule
CREATE OR REPLACE FUNCTION should_run_cleanup()
RETURNS BOOLEAN AS $$
DECLARE
    last_cleanup TIMESTAMPTZ;
    hours_since_cleanup INTEGER;
BEGIN
    -- Get the most recent cleanup timestamp
    SELECT MAX(last_cleanup) INTO last_cleanup
    FROM data_retention_config
    WHERE cleanup_enabled = true;

    -- If never run, should run
    IF last_cleanup IS NULL THEN
        RETURN true;
    END IF;

    -- Calculate hours since last cleanup
    hours_since_cleanup := EXTRACT(EPOCH FROM (NOW() - last_cleanup)) / 3600;

    -- Run cleanup daily (24 hours)
    RETURN hours_since_cleanup >= 24;
END;
$$ LANGUAGE plpgsql;

-- Function for scheduled cleanup execution
CREATE OR REPLACE FUNCTION scheduled_cleanup()
RETURNS TEXT AS $$
DECLARE
    cleanup_needed BOOLEAN;
    cleanup_results TEXT := '';
    result_record RECORD;
BEGIN
    -- Check if cleanup is needed
    SELECT should_run_cleanup() INTO cleanup_needed;

    IF NOT cleanup_needed THEN
        RETURN 'Cleanup not needed - last run was less than 24 hours ago';
    END IF;

    -- Execute cleanup
    cleanup_results := 'Scheduled cleanup executed at ' || NOW()::TEXT || E'\n';

    FOR result_record IN
        SELECT * FROM execute_data_cleanup(confirm_cleanup := true)
    LOOP
        cleanup_results := cleanup_results ||
            result_record.table_name || ': ' ||
            result_record.records_affected || ' records, ' ||
            result_record.status || E'\n';
    END LOOP;

    -- Log the scheduled cleanup
    INSERT INTO system_metrics (
        metric_name,
        metric_value,
        metric_unit,
        category,
        subcategory,
        measured_at
    ) VALUES (
        'scheduled_cleanup_completed',
        1,
        'execution',
        'system',
        'maintenance',
        NOW()
    );

    RETURN cleanup_results;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- USAGE EXAMPLES AND DOCUMENTATION
-- ========================================

-- View current retention policies
CREATE OR REPLACE VIEW retention_policy_summary AS
SELECT
    table_name,
    retention_days,
    CASE
        WHEN retention_days = -1 THEN 'Permanent'
        WHEN retention_days = 0 THEN 'Immediate'
        ELSE retention_days || ' days'
    END as retention_period,
    retention_policy,
    cleanup_enabled,
    last_cleanup,
    CASE
        WHEN last_cleanup IS NULL THEN 'Never'
        ELSE EXTRACT(EPOCH FROM (NOW() - last_cleanup)) / 3600 || ' hours ago'
    END as time_since_cleanup
FROM data_retention_config
ORDER BY retention_days, table_name;

-- ========================================
-- CONFIRMAR CREACIÓN DE POLÍTICAS
-- ========================================

\echo '✅ Data retention policies configured successfully'
\echo '📋 Retention configuration table created'
\echo '🧹 Comprehensive cleanup functions available'
\echo '📊 Maintenance analysis functions ready'
\echo '⏰ Automated scheduling functions configured'
\echo ''
\echo 'Usage Examples:'
\echo '-- View retention policies:'
\echo 'SELECT * FROM retention_policy_summary;'
\echo ''
\echo '-- Estimate cleanup impact:'
\echo 'SELECT * FROM estimate_cleanup_impact();'
\echo ''
\echo '-- Execute cleanup (safe):'
\echo 'SELECT * FROM execute_data_cleanup(confirm_cleanup := true);'
\echo ''
\echo '-- Analyze table maintenance needs:'
\echo 'SELECT * FROM analyze_table_maintenance();'
\echo ''
\echo '-- Execute scheduled cleanup:'
\echo 'SELECT scheduled_cleanup();'