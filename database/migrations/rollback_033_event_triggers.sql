-- =============================================================================
-- ROLLBACK: 033_event_triggers.sql
-- Description: Rollback PostgreSQL NOTIFY triggers and V7.1 event tables
-- Author: Trading Team
-- Date: 2026-01-31
--
-- Usage:
--   psql -U postgres -d trading -f rollback_033_event_triggers.sql
--
-- WARNING: This will remove all event-driven infrastructure.
-- System will revert to polling-based sensors (V6 behavior).
-- =============================================================================

-- Confirm before execution
DO $$
BEGIN
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'ROLLBACK: V7.1 Event-Driven Infrastructure';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'This will:';
    RAISE NOTICE '  - Remove NOTIFY triggers from usdcop_m5_ohlcv';
    RAISE NOTICE '  - Remove NOTIFY triggers from inference_features_5m';
    RAISE NOTICE '  - Drop event_dead_letter_queue table';
    RAISE NOTICE '  - Drop event_processed_log table';
    RAISE NOTICE '  - Drop circuit_breaker_state table';
    RAISE NOTICE '  - Remove all event-related functions and views';
    RAISE NOTICE '-------------------------------------------------------------';
    RAISE NOTICE 'System will revert to V6 polling-based behavior.';
    RAISE NOTICE '=============================================================';
END $$;

-- =============================================================================
-- DROP TRIGGERS
-- =============================================================================

DROP TRIGGER IF EXISTS trg_notify_new_ohlcv_bar ON usdcop_m5_ohlcv;
DROP TRIGGER IF EXISTS trg_notify_features_ready ON inference_features_5m;

-- =============================================================================
-- DROP FUNCTIONS
-- =============================================================================

DROP FUNCTION IF EXISTS notify_new_ohlcv_bar();
DROP FUNCTION IF EXISTS notify_features_ready();
DROP FUNCTION IF EXISTS emit_heartbeat(TEXT);
DROP FUNCTION IF EXISTS dlq_enqueue(VARCHAR, VARCHAR, VARCHAR, JSONB, TEXT);
DROP FUNCTION IF EXISTS is_event_processed(VARCHAR);
DROP FUNCTION IF EXISTS mark_event_processed(VARCHAR, VARCHAR, VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS cleanup_old_processed_events();

-- =============================================================================
-- DROP VIEWS
-- =============================================================================

DROP VIEW IF EXISTS v_event_system_health;
DROP VIEW IF EXISTS v_dlq_summary;

-- =============================================================================
-- DROP TABLES
-- =============================================================================

DROP TABLE IF EXISTS event_dead_letter_queue;
DROP TABLE IF EXISTS event_processed_log;
DROP TABLE IF EXISTS circuit_breaker_state;

-- =============================================================================
-- VERIFICATION
-- =============================================================================

DO $$
DECLARE
    trigger_count INTEGER;
    table_count INTEGER;
BEGIN
    -- Check triggers removed
    SELECT COUNT(*) INTO trigger_count
    FROM pg_trigger t
    JOIN pg_class c ON t.tgrelid = c.oid
    WHERE t.tgname IN ('trg_notify_new_ohlcv_bar', 'trg_notify_features_ready');

    -- Check tables removed
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN ('event_dead_letter_queue', 'event_processed_log', 'circuit_breaker_state');

    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'ROLLBACK VERIFICATION';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'Remaining event triggers: %', trigger_count;
    RAISE NOTICE 'Remaining event tables: %', table_count;

    IF trigger_count = 0 AND table_count = 0 THEN
        RAISE NOTICE '✅ ROLLBACK SUCCESSFUL - V7.1 infrastructure removed';
    ELSE
        RAISE WARNING '⚠️  ROLLBACK INCOMPLETE - Some components remain';
    END IF;

    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'System is now in V6 mode (polling-based sensors)';
    RAISE NOTICE 'Restart Airflow to apply changes.';
    RAISE NOTICE '=============================================================';
END $$;
