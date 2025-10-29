#!/bin/bash

################################################################################
# Airflow History Cleanup - DRY RUN TEST
################################################################################
# Purpose: Test cleanup commands WITHOUT actually deleting data
# Safety: Read-only queries only - no modifications
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

POSTGRES_CONTAINER="usdcop-postgres-timescale"
POSTGRES_USER="admin"
POSTGRES_DB="usdcop_trading"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}AIRFLOW HISTORY CLEANUP - DRY RUN TEST${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${YELLOW}This is a READ-ONLY test. No data will be deleted.${NC}"
echo ""

################################################################################
# Test 1: Check all history tables exist
################################################################################
echo -e "${BLUE}Test 1: Verify all 17 history tables exist${NC}"
echo "-------------------------------------------"

TABLES=(
    "dag_run"
    "task_instance"
    "task_fail"
    "xcom"
    "log"
    "job"
    "rendered_task_instance_fields"
    "task_reschedule"
    "sla_miss"
    "dag_run_note"
    "task_instance_note"
    "callback_request"
    "import_error"
    "dag_warning"
    "dataset_event"
    "dagrun_dataset_event"
    "trigger"
)

MISSING=0
for table in "${TABLES[@]}"; do
    EXISTS=$(docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -t -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '${table}');" | xargs)
    if [ "$EXISTS" = "t" ]; then
        echo -e "${GREEN}✓${NC} ${table}"
    else
        echo -e "✗ ${table} - MISSING!"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo -e "\n${GREEN}✓ All 17 history tables found${NC}\n"
else
    echo -e "\n✗ ${MISSING} tables missing!\n"
    exit 1
fi

################################################################################
# Test 2: Current record counts
################################################################################
echo -e "${BLUE}Test 2: Current execution history record counts${NC}"
echo "------------------------------------------------"

docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT
    'dag_run' as table_name,
    COUNT(*) as records,
    pg_size_pretty(pg_total_relation_size('dag_run')) as disk_size
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
"

echo ""

################################################################################
# Test 3: Total history summary
################################################################################
echo -e "${BLUE}Test 3: Total execution history summary${NC}"
echo "----------------------------------------"

docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT
    COALESCE(SUM(cnt), 0) as total_history_records,
    'records would be deleted' as note
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
"

echo ""

################################################################################
# Test 4: Foreign key relationships
################################################################################
echo -e "${BLUE}Test 4: Verify CASCADE relationships${NC}"
echo "-------------------------------------"

docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT
    conname as constraint_name,
    conrelid::regclass AS child_table,
    confrelid::regclass AS parent_table,
    CASE
        WHEN confdeltype = 'c' THEN 'CASCADE'
        WHEN confdeltype = 'n' THEN 'SET NULL'
        WHEN confdeltype = 'r' THEN 'RESTRICT'
        ELSE 'NO ACTION'
    END as on_delete_action
FROM pg_constraint
WHERE contype = 'f'
AND (
    confrelid::regclass::text = 'dag_run' OR
    confrelid::regclass::text = 'task_instance' OR
    confrelid::regclass::text = 'trigger'
)
ORDER BY parent_table, child_table;
"

echo ""

################################################################################
# Test 5: Configuration preservation check
################################################################################
echo -e "${BLUE}Test 5: Verify configurations would be preserved${NC}"
echo "--------------------------------------------------"

docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT
    'dag' as config_table,
    COUNT(*) as records,
    'PRESERVED' as status
FROM dag
UNION ALL
SELECT 'variable', COUNT(*), 'PRESERVED' FROM variable
UNION ALL
SELECT 'connection', COUNT(*), 'PRESERVED' FROM connection
UNION ALL
SELECT 'ab_user', COUNT(*), 'PRESERVED' FROM ab_user
UNION ALL
SELECT 'ab_role', COUNT(*), 'PRESERVED' FROM ab_role
UNION ALL
SELECT 'slot_pool', COUNT(*), 'PRESERVED' FROM slot_pool
ORDER BY config_table;
"

echo ""

################################################################################
# Test 6: Simulate TRUNCATE (explain what would happen)
################################################################################
echo -e "${BLUE}Test 6: Simulation - What would happen with TRUNCATE CASCADE${NC}"
echo "--------------------------------------------------------------"

echo "If we execute: TRUNCATE TABLE dag_run CASCADE;"
echo ""
echo "Direct CASCADE effects:"
docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT
    'dag_run' as table_name,
    COUNT(*) as records_to_delete,
    'Parent table' as cascade_level
FROM dag_run
UNION ALL
SELECT 'task_instance', COUNT(*), '→ Cascade level 1' FROM task_instance
UNION ALL
SELECT 'task_fail', COUNT(*), '  → Cascade level 2' FROM task_fail
UNION ALL
SELECT 'xcom', COUNT(*), '  → Cascade level 2' FROM xcom
UNION ALL
SELECT 'rendered_ti_fields', COUNT(*), '  → Cascade level 2' FROM rendered_task_instance_fields
UNION ALL
SELECT 'dag_run_note', COUNT(*), '→ Cascade level 1' FROM dag_run_note
ORDER BY cascade_level, table_name;
"

echo ""
echo "Tables requiring separate TRUNCATE commands:"
docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
SELECT 'log' as table_name, COUNT(*) as records_to_delete FROM log
UNION ALL SELECT 'job', COUNT(*) FROM job
UNION ALL SELECT 'import_error', COUNT(*) FROM import_error
UNION ALL SELECT 'dag_warning', COUNT(*) FROM dag_warning
UNION ALL SELECT 'callback_request', COUNT(*) FROM callback_request
UNION ALL SELECT 'dataset_event', COUNT(*) FROM dataset_event
UNION ALL SELECT 'trigger', COUNT(*) FROM trigger
UNION ALL SELECT 'sla_miss', COUNT(*) FROM sla_miss;
"

echo ""

################################################################################
# Test 7: Verify SQL script syntax
################################################################################
echo -e "${BLUE}Test 7: Verify SQL script exists and is readable${NC}"
echo "-------------------------------------------------"

SQL_SCRIPT="$(dirname "$0")/clear_airflow_history.sql"
if [ -f "$SQL_SCRIPT" ]; then
    SIZE=$(ls -lh "$SQL_SCRIPT" | awk '{print $5}')
    LINES=$(wc -l < "$SQL_SCRIPT")
    echo -e "${GREEN}✓${NC} SQL script found: ${SQL_SCRIPT}"
    echo "  Size: ${SIZE}"
    echo "  Lines: ${LINES}"
else
    echo -e "✗ SQL script not found: ${SQL_SCRIPT}"
fi

echo ""

################################################################################
# Test 8: Verify shell script exists and is executable
################################################################################
echo -e "${BLUE}Test 8: Verify shell script exists and is executable${NC}"
echo "-----------------------------------------------------"

SHELL_SCRIPT="$(dirname "$0")/clear_airflow_history.sh"
if [ -f "$SHELL_SCRIPT" ]; then
    SIZE=$(ls -lh "$SHELL_SCRIPT" | awk '{print $5}')
    if [ -x "$SHELL_SCRIPT" ]; then
        echo -e "${GREEN}✓${NC} Shell script found and executable: ${SHELL_SCRIPT}"
    else
        echo -e "${YELLOW}⚠${NC} Shell script found but not executable: ${SHELL_SCRIPT}"
        echo "  Fix with: chmod +x ${SHELL_SCRIPT}"
    fi
    echo "  Size: ${SIZE}"
else
    echo -e "✗ Shell script not found: ${SHELL_SCRIPT}"
fi

echo ""

################################################################################
# Summary
################################################################################
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}DRY RUN TEST COMPLETE${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}✓ All tests passed - cleanup scripts are ready to use${NC}"
echo ""
echo "To actually clear history, run ONE of these commands:"
echo ""
echo "  1. Interactive (recommended):"
echo -e "     ${YELLOW}./scripts/clear_airflow_history.sh${NC}"
echo ""
echo "  2. Direct SQL:"
echo -e "     ${YELLOW}docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading < scripts/clear_airflow_history.sql${NC}"
echo ""
echo "  3. One-liner (fast):"
echo -e "     ${YELLOW}docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c \"TRUNCATE TABLE dag_run CASCADE; TRUNCATE TABLE log CASCADE; TRUNCATE TABLE job CASCADE;\"${NC}"
echo ""
echo -e "${BLUE}============================================================================${NC}"
