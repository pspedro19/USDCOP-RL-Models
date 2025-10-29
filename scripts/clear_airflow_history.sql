-- ============================================================================
-- Airflow Execution History Cleanup Script
-- ============================================================================
-- Purpose: Clear ALL DAG execution history from Airflow metadata database
-- Result: Zero visible run history in Airflow UI (clean slate)
-- Safety: Preserves DAG definitions, variables, connections, and user accounts
-- ============================================================================

\echo '============================================================================'
\echo 'AIRFLOW HISTORY CLEANUP'
\echo '============================================================================'
\echo ''

-- Display current state
\echo 'BEFORE CLEANUP - Current Record Counts:'
\echo '----------------------------------------'
SELECT
    'dag_run' as table_name,
    COUNT(*) as records,
    pg_size_pretty(pg_total_relation_size('dag_run')) as size
FROM dag_run
UNION ALL
SELECT 'task_instance', COUNT(*), pg_size_pretty(pg_total_relation_size('task_instance'))
FROM task_instance
UNION ALL
SELECT 'task_fail', COUNT(*), pg_size_pretty(pg_total_relation_size('task_fail'))
FROM task_fail
UNION ALL
SELECT 'xcom', COUNT(*), pg_size_pretty(pg_total_relation_size('xcom'))
FROM xcom
UNION ALL
SELECT 'log', COUNT(*), pg_size_pretty(pg_total_relation_size('log'))
FROM log
UNION ALL
SELECT 'job', COUNT(*), pg_size_pretty(pg_total_relation_size('job'))
FROM job
UNION ALL
SELECT 'rendered_ti_fields', COUNT(*), pg_size_pretty(pg_total_relation_size('rendered_task_instance_fields'))
FROM rendered_task_instance_fields
ORDER BY records DESC;

\echo ''
\echo 'Calculating total history records...'
SELECT
    COALESCE(SUM(cnt), 0) as total_execution_records,
    'to be deleted' as status
FROM (
    SELECT COUNT(*) as cnt FROM dag_run
    UNION ALL SELECT COUNT(*) FROM task_instance
    UNION ALL SELECT COUNT(*) FROM task_fail
    UNION ALL SELECT COUNT(*) FROM log
    UNION ALL SELECT COUNT(*) FROM xcom
    UNION ALL SELECT COUNT(*) FROM job
    UNION ALL SELECT COUNT(*) FROM rendered_task_instance_fields
    UNION ALL SELECT COUNT(*) FROM task_reschedule
    UNION ALL SELECT COUNT(*) FROM sla_miss
    UNION ALL SELECT COUNT(*) FROM import_error
    UNION ALL SELECT COUNT(*) FROM dag_warning
    UNION ALL SELECT COUNT(*) FROM callback_request
    UNION ALL SELECT COUNT(*) FROM dataset_event
    UNION ALL SELECT COUNT(*) FROM dagrun_dataset_event
    UNION ALL SELECT COUNT(*) FROM trigger
    UNION ALL SELECT COUNT(*) FROM dag_run_note
    UNION ALL SELECT COUNT(*) FROM task_instance_note
) subq;

\echo ''
\echo '============================================================================'
\echo 'STARTING TRUNCATION...'
\echo '============================================================================'
\echo ''

-- ============================================================================
-- STEP 1: Clear main execution history (with CASCADE)
-- ============================================================================
\echo 'Step 1/5: Truncating dag_run (will cascade to task_instance, xcom, etc.)...'
TRUNCATE TABLE dag_run CASCADE;
\echo '  ✓ dag_run cleared (cascaded to child tables)'

-- ============================================================================
-- STEP 2: Clear standalone history tables
-- ============================================================================
\echo ''
\echo 'Step 2/5: Truncating standalone history tables...'
TRUNCATE TABLE log CASCADE;
\echo '  ✓ log cleared'

TRUNCATE TABLE job CASCADE;
\echo '  ✓ job cleared'

TRUNCATE TABLE import_error CASCADE;
\echo '  ✓ import_error cleared'

TRUNCATE TABLE dag_warning CASCADE;
\echo '  ✓ dag_warning cleared'

TRUNCATE TABLE callback_request CASCADE;
\echo '  ✓ callback_request cleared'

-- ============================================================================
-- STEP 3: Clear dataset event history
-- ============================================================================
\echo ''
\echo 'Step 3/5: Truncating dataset event tables...'
TRUNCATE TABLE dataset_event CASCADE;
\echo '  ✓ dataset_event cleared'

TRUNCATE TABLE dataset_dag_run_queue CASCADE;
\echo '  ✓ dataset_dag_run_queue cleared'

-- ============================================================================
-- STEP 4: Clear trigger history
-- ============================================================================
\echo ''
\echo 'Step 4/5: Truncating trigger table...'
TRUNCATE TABLE trigger CASCADE;
\echo '  ✓ trigger cleared'

-- ============================================================================
-- STEP 5: Clear SLA history
-- ============================================================================
\echo ''
\echo 'Step 5/5: Truncating SLA miss table...'
TRUNCATE TABLE sla_miss CASCADE;
\echo '  ✓ sla_miss cleared'

-- ============================================================================
-- VERIFICATION
-- ============================================================================
\echo ''
\echo '============================================================================'
\echo 'AFTER CLEANUP - Verification'
\echo '============================================================================'
\echo ''

\echo 'Record counts (all should be 0):'
\echo '--------------------------------'
SELECT
    'dag_run' as table_name,
    COUNT(*) as records,
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END as status
FROM dag_run
UNION ALL
SELECT 'task_instance', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM task_instance
UNION ALL
SELECT 'task_fail', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM task_fail
UNION ALL
SELECT 'xcom', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM xcom
UNION ALL
SELECT 'log', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM log
UNION ALL
SELECT 'job', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM job
UNION ALL
SELECT 'rendered_ti_fields', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM rendered_task_instance_fields
UNION ALL
SELECT 'task_reschedule', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM task_reschedule
UNION ALL
SELECT 'sla_miss', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM sla_miss
UNION ALL
SELECT 'import_error', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM import_error
UNION ALL
SELECT 'dag_warning', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM dag_warning
UNION ALL
SELECT 'callback_request', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM callback_request
UNION ALL
SELECT 'dataset_event', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM dataset_event
UNION ALL
SELECT 'dagrun_dataset_event', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM dagrun_dataset_event
UNION ALL
SELECT 'trigger', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM trigger
UNION ALL
SELECT 'dag_run_note', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM dag_run_note
UNION ALL
SELECT 'task_instance_note', COUNT(*),
    CASE WHEN COUNT(*) = 0 THEN '✓ CLEAN' ELSE '✗ FAILED' END
FROM task_instance_note
ORDER BY table_name;

\echo ''
\echo 'Total remaining history records:'
\echo '--------------------------------'
SELECT
    COALESCE(SUM(cnt), 0) as total_history_records,
    CASE
        WHEN COALESCE(SUM(cnt), 0) = 0 THEN '✓ SUCCESS - All history cleared'
        ELSE '✗ WARNING - Some records remain'
    END as cleanup_status
FROM (
    SELECT COUNT(*) as cnt FROM dag_run
    UNION ALL SELECT COUNT(*) FROM task_instance
    UNION ALL SELECT COUNT(*) FROM task_fail
    UNION ALL SELECT COUNT(*) FROM log
    UNION ALL SELECT COUNT(*) FROM xcom
    UNION ALL SELECT COUNT(*) FROM job
    UNION ALL SELECT COUNT(*) FROM rendered_task_instance_fields
    UNION ALL SELECT COUNT(*) FROM task_reschedule
    UNION ALL SELECT COUNT(*) FROM sla_miss
    UNION ALL SELECT COUNT(*) FROM import_error
    UNION ALL SELECT COUNT(*) FROM dag_warning
    UNION ALL SELECT COUNT(*) FROM callback_request
    UNION ALL SELECT COUNT(*) FROM dataset_event
    UNION ALL SELECT COUNT(*) FROM dagrun_dataset_event
    UNION ALL SELECT COUNT(*) FROM trigger
    UNION ALL SELECT COUNT(*) FROM dag_run_note
    UNION ALL SELECT COUNT(*) FROM task_instance_note
) subq;

\echo ''
\echo 'Configuration tables (preserved):'
\echo '----------------------------------'
SELECT
    'dag' as table_name,
    COUNT(*) as records,
    '✓ Preserved' as status
FROM dag
UNION ALL
SELECT 'connection', COUNT(*), '✓ Preserved' FROM connection
UNION ALL
SELECT 'variable', COUNT(*), '✓ Preserved' FROM variable
UNION ALL
SELECT 'ab_user', COUNT(*), '✓ Preserved' FROM ab_user
ORDER BY table_name;

\echo ''
\echo '============================================================================'
\echo 'CLEANUP COMPLETE'
\echo '============================================================================'
\echo ''
\echo 'Summary:'
\echo '  • All DAG execution history: CLEARED'
\echo '  • All task instance history: CLEARED'
\echo '  • All logs: CLEARED'
\echo '  • DAG definitions: PRESERVED'
\echo '  • Connections/Variables: PRESERVED'
\echo '  • User accounts: PRESERVED'
\echo ''
\echo 'Airflow UI will show ZERO run history.'
\echo 'DAGs can be manually triggered immediately.'
\echo '============================================================================'
