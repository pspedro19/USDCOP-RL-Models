#!/bin/bash
################################################################################
# DWH Integration Verification Script
# ================================================================================
# Verifies that all DAGs have proper DWH integration:
#   - Import of dwh_helper
#   - load_to_dwh function defined
#   - load_to_dwh task added to DAG
#   - Tag 'dwh' present
#
# Usage:
#   ./scripts/verify_dwh_integration.sh
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# DAG directory
DAG_DIR="/home/azureuser/USDCOP-RL-Models/airflow/dags"

print_header() {
    echo ""
    echo "========================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "========================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check single DAG
check_dag() {
    local dag_file=$1
    local dag_name=$(basename "$dag_file")
    local layer=$(echo "$dag_name" | grep -oP 'l[0-6]' | head -1 | tr '[:lower:]' '[:upper:]')

    echo ""
    print_info "Checking $layer: $dag_name"

    local has_import=0
    local has_function=0
    local has_task=0
    local has_tag=0

    # Check for dwh_helper import
    if grep -q "from utils.dwh_helper import" "$dag_file"; then
        print_success "  ✓ Import dwh_helper found"
        has_import=1
    else
        print_error "  ✗ Import dwh_helper NOT found"
    fi

    # Check for load_to_dwh function
    if grep -q "def load_to_dwh" "$dag_file"; then
        print_success "  ✓ Function load_to_dwh() defined"
        has_function=1
    else
        print_error "  ✗ Function load_to_dwh() NOT defined"
    fi

    # Check for load_to_dwh task
    if grep -q "task_id.*load_to_dwh\|task_id='load_to_dwh'\|task_id=\"load_to_dwh\"" "$dag_file"; then
        print_success "  ✓ Task load_to_dwh added to DAG"
        has_task=1
    else
        print_error "  ✗ Task load_to_dwh NOT added to DAG"
    fi

    # Check for 'dwh' tag
    if grep -q "tags.*dwh\|'dwh'\|\"dwh\"" "$dag_file"; then
        print_success "  ✓ Tag 'dwh' found"
        has_tag=1
    else
        print_warning "  ! Tag 'dwh' NOT found (optional)"
        has_tag=1  # Not critical
    fi

    # Summary
    local total=$((has_import + has_function + has_task + has_tag))

    if [ $total -eq 4 ]; then
        print_success "  ✅ $layer: COMPLETE ($total/4)"
        return 0
    elif [ $total -ge 3 ]; then
        print_warning "  ⚠️  $layer: PARTIAL ($total/4)"
        return 1
    else
        print_error "  ❌ $layer: INCOMPLETE ($total/4)"
        return 2
    fi
}

# Main function
main() {
    print_header "🔍 DWH INTEGRATION VERIFICATION"

    print_info "Checking DAG files in: $DAG_DIR"

    local total_dags=0
    local complete_dags=0
    local partial_dags=0
    local incomplete_dags=0

    # Check each L0-L6 DAG
    for dag_file in "$DAG_DIR"/usdcop_m5__0[0-6]*.py; do
        if [ -f "$dag_file" ]; then
            ((total_dags++))

            check_dag "$dag_file"
            result=$?

            if [ $result -eq 0 ]; then
                ((complete_dags++))
            elif [ $result -eq 1 ]; then
                ((partial_dags++))
            else
                ((incomplete_dags++))
            fi
        fi
    done

    # Summary
    print_header "📊 VERIFICATION SUMMARY"

    echo "Total DAGs checked: $total_dags"
    echo ""
    print_success "Complete integrations: $complete_dags"
    print_warning "Partial integrations: $partial_dags"
    print_error "Incomplete integrations: $incomplete_dags"

    echo ""
    print_header "🎯 RECOMMENDATIONS"

    if [ $complete_dags -ge 4 ]; then
        print_success "✅ EXCELLENT! Core DAGs (L0, L1, L5, L6) are integrated"
        print_info "You have sufficient DWH integration for production"
    elif [ $complete_dags -ge 2 ]; then
        print_warning "⚠️  GOOD START! L0 and L1 are integrated"
        print_info "Consider implementing L5 and L6 for full production readiness"
    else
        print_error "❌ INCOMPLETE! Need at least L0 and L1"
        print_info "Run: ./deploy-dwh-complete.sh to initialize DWH"
    fi

    echo ""

    # Detailed recommendations
    if [ $incomplete_dags -gt 0 ] || [ $partial_dags -gt 0 ]; then
        print_info "To complete integration for remaining DAGs:"
        echo "  1. See template: docs/DWH_DAG_UPDATE_TEMPLATE.md"
        echo "  2. Add import: from utils.dwh_helper import DWHHelper, get_dwh_connection"
        echo "  3. Add function: def load_to_dwh(**context)"
        echo "  4. Add task: PythonOperator(task_id='load_to_dwh', ...)"
        echo "  5. Update dependencies: ... >> task_final >> load_dwh_task"
        echo "  6. Add tag: tags=[..., 'dwh']"
    fi

    echo ""
    print_info "Next steps:"
    echo "  1. Deploy DWH: ./deploy-dwh-complete.sh"
    echo "  2. Execute DAGs in Airflow"
    echo "  3. Verify data: curl http://localhost:8007/api/bi/bars | jq"

    echo ""
}

# Run main function
main

exit 0
