#!/bin/bash

################################################################################
# Airflow History Cleanup Script
################################################################################
# Purpose: Clear all DAG execution history from Airflow metadata database
# Result: Zero visible run history in Airflow UI
# Safety: Preserves DAG definitions, variables, connections, users
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_CONTAINER="usdcop-postgres-timescale"
POSTGRES_USER="admin"
POSTGRES_DB="usdcop_trading"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_SCRIPT="${SCRIPT_DIR}/clear_airflow_history.sql"

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check if Docker is running
    if ! docker ps &> /dev/null; then
        print_error "Docker is not running"
        exit 1
    fi
    print_success "Docker is running"

    # Check if PostgreSQL container exists and is running
    if ! docker ps --filter "name=${POSTGRES_CONTAINER}" --filter "status=running" | grep -q "${POSTGRES_CONTAINER}"; then
        print_error "PostgreSQL container '${POSTGRES_CONTAINER}' is not running"
        exit 1
    fi
    print_success "PostgreSQL container is running"

    # Check if SQL script exists
    if [ ! -f "${SQL_SCRIPT}" ]; then
        print_error "SQL script not found: ${SQL_SCRIPT}"
        exit 1
    fi
    print_success "SQL script found"

    echo ""
}

get_current_counts() {
    print_header "Current Execution History"

    docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "
        SELECT
            COALESCE(SUM(cnt), 0) as total_records,
            pg_size_pretty(SUM(size_bytes)) as total_size
        FROM (
            SELECT COUNT(*) as cnt, pg_total_relation_size('dag_run') as size_bytes FROM dag_run
            UNION ALL SELECT COUNT(*), pg_total_relation_size('task_instance') FROM task_instance
            UNION ALL SELECT COUNT(*), pg_total_relation_size('task_fail') FROM task_fail
            UNION ALL SELECT COUNT(*), pg_total_relation_size('log') FROM log
            UNION ALL SELECT COUNT(*), pg_total_relation_size('xcom') FROM xcom
            UNION ALL SELECT COUNT(*), pg_total_relation_size('job') FROM job
        ) subq;
    "

    echo ""
}

confirm_cleanup() {
    print_header "Confirmation Required"

    echo -e "${YELLOW}WARNING: This operation will DELETE ALL Airflow execution history!${NC}"
    echo ""
    echo "This includes:"
    echo "  • All DAG runs"
    echo "  • All task instances"
    echo "  • All logs"
    echo "  • All XCom data"
    echo "  • All job history"
    echo ""
    echo "This will NOT delete:"
    echo "  • DAG definitions"
    echo "  • Variables"
    echo "  • Connections"
    echo "  • User accounts"
    echo ""

    read -p "Are you sure you want to proceed? (yes/no): " -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_warning "Operation cancelled by user"
        exit 0
    fi

    print_success "User confirmed - proceeding with cleanup"
    echo ""
}

create_backup() {
    print_header "Creating Backup (Optional)"

    read -p "Do you want to create a backup before cleanup? (yes/no): " -r
    echo ""

    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        BACKUP_FILE="/tmp/airflow_history_backup_$(date +%Y%m%d_%H%M%S).sql"
        print_warning "Creating backup to: ${BACKUP_FILE}"

        docker exec "${POSTGRES_CONTAINER}" pg_dump -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
            -t dag_run \
            -t task_instance \
            -t task_fail \
            -t log \
            -t xcom \
            -t job \
            -t rendered_task_instance_fields \
            > "${BACKUP_FILE}" 2>/dev/null

        if [ -f "${BACKUP_FILE}" ]; then
            print_success "Backup created: ${BACKUP_FILE}"
            print_warning "To restore: docker exec -i ${POSTGRES_CONTAINER} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} < ${BACKUP_FILE}"
        else
            print_warning "Backup creation skipped or failed"
        fi
    else
        print_warning "Skipping backup creation"
    fi

    echo ""
}

execute_cleanup() {
    print_header "Executing Cleanup"

    # Execute the SQL script
    docker exec -i "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" < "${SQL_SCRIPT}"

    echo ""
}

verify_cleanup() {
    print_header "Final Verification"

    # Quick count check
    REMAINING=$(docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -t -c "
        SELECT COALESCE(SUM(cnt), 0)
        FROM (
            SELECT COUNT(*) as cnt FROM dag_run
            UNION ALL SELECT COUNT(*) FROM task_instance
            UNION ALL SELECT COUNT(*) FROM task_fail
            UNION ALL SELECT COUNT(*) FROM log
            UNION ALL SELECT COUNT(*) FROM xcom
            UNION ALL SELECT COUNT(*) FROM job
        ) subq;
    " | xargs)

    if [ "$REMAINING" -eq 0 ]; then
        print_success "CLEANUP SUCCESSFUL - All execution history cleared (${REMAINING} records remaining)"
    else
        print_warning "CLEANUP INCOMPLETE - ${REMAINING} records still remain"
    fi

    echo ""
}

print_next_steps() {
    print_header "Next Steps"

    echo "1. Refresh the Airflow UI (Ctrl+R or Cmd+R)"
    echo "   URL: http://localhost:8080"
    echo ""
    echo "2. Verify DAG runs are empty:"
    echo "   • Browse > DAG Runs should show no records"
    echo "   • Each DAG's history should be empty"
    echo ""
    echo "3. Manually trigger DAGs as needed:"
    echo "   • Click the 'Play' button on any DAG"
    echo "   • Configure trigger options"
    echo "   • Monitor execution in real-time"
    echo ""
    echo "4. Check the following still work:"
    echo "   • Variables: Admin > Variables"
    echo "   • Connections: Admin > Connections"
    echo "   • Users: Security > List Users"
    echo ""

    print_success "Airflow is ready for fresh DAG executions!"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    print_header "AIRFLOW HISTORY CLEANUP TOOL"
    echo "Database: ${POSTGRES_DB}"
    echo "Container: ${POSTGRES_CONTAINER}"
    echo ""

    # Step 1: Check prerequisites
    check_prerequisites

    # Step 2: Show current state
    get_current_counts

    # Step 3: Confirm operation
    confirm_cleanup

    # Step 4: Optional backup
    create_backup

    # Step 5: Execute cleanup
    execute_cleanup

    # Step 6: Verify results
    verify_cleanup

    # Step 7: Show next steps
    print_next_steps

    print_header "CLEANUP COMPLETE"
}

# Run main function
main "$@"
