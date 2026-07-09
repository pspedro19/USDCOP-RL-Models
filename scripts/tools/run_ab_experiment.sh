#!/bin/bash
# =============================================================================
# SCRIPT: run_ab_experiment.sh
# =============================================================================
# Ejecuta el pipeline completo para A/B testing de modelos
#
# Uso:
#   ./scripts/run_ab_experiment.sh [all|data|train|compare]
#
# Pasos:
#   1. data    - Cargar datos y generar datasets
#   2. train   - Entrenar ambos experimentos
#   3. compare - Comparar resultados y generar reporte
#   4. all     - Ejecutar todo el pipeline
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AIRFLOW_URL=${AIRFLOW_URL:-"http://localhost:8080"}
MLFLOW_URL=${MLFLOW_URL:-"http://localhost:5000"}
DASHBOARD_URL=${DASHBOARD_URL:-"http://localhost:3000"}

# Experiment configs
EXPERIMENT_A="baseline_full_macro"
EXPERIMENT_B="reduced_core_macro"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

wait_for_dag() {
    local dag_id=$1
    local timeout=${2:-600}  # 10 min default
    local start_time=$(date +%s)

    log_info "Waiting for DAG $dag_id to complete (timeout: ${timeout}s)..."

    while true; do
        # Check DAG status via Airflow API
        status=$(curl -s "$AIRFLOW_URL/api/v1/dags/$dag_id/dagRuns" \
            -H "Content-Type: application/json" \
            | jq -r '.dag_runs[-1].state' 2>/dev/null || echo "unknown")

        if [ "$status" = "success" ]; then
            log_success "DAG $dag_id completed successfully!"
            return 0
        elif [ "$status" = "failed" ]; then
            log_error "DAG $dag_id failed!"
            return 1
        fi

        # Check timeout
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -ge $timeout ]; then
            log_error "Timeout waiting for DAG $dag_id"
            return 1
        fi

        echo -n "."
        sleep 10
    done
}

trigger_dag() {
    local dag_id=$1
    local conf=${2:-"{}"}

    log_info "Triggering DAG: $dag_id"

    curl -X POST "$AIRFLOW_URL/api/v1/dags/$dag_id/dagRuns" \
        -H "Content-Type: application/json" \
        -d "{\"conf\": $conf}" \
        2>/dev/null

    echo ""
}

# =============================================================================
# Step 1: Data Pipeline
# =============================================================================

run_data_pipeline() {
    echo ""
    echo "============================================================"
    echo "  PASO 1: DATA PIPELINE (L0 + L1 + L2)"
    echo "============================================================"
    echo ""

    # Check if infrastructure is running
    log_info "Verificando infraestructura..."
    docker-compose ps | grep -q "postgres.*Up" || {
        log_error "PostgreSQL no está corriendo. Ejecuta: docker-compose up -d"
        exit 1
    }

    # Step 1a: Load OHLCV data (backfill if needed)
    log_info "Cargando datos OHLCV históricos..."
    trigger_dag "v3.l0_ohlcv_backfill" '{"start_date": "2023-01-01", "end_date": "2024-12-31"}'
    wait_for_dag "v3.l0_ohlcv_backfill" 1800  # 30 min timeout

    # Step 1b: Load macro indicators
    log_info "Cargando indicadores macro..."
    trigger_dag "v3.l0_macro_unified"
    wait_for_dag "v3.l0_macro_unified" 600

    # Step 1c: Generate features
    log_info "Generando features (L1)..."
    trigger_dag "v3.l1_feature_refresh"
    wait_for_dag "v3.l1_feature_refresh" 300

    # Step 1d: Generate RL datasets
    log_info "Generando datasets RL (L2)..."
    trigger_dag "v3.l2_preprocessing_pipeline" '{"generate_variants": true}'
    wait_for_dag "v3.l2_preprocessing_pipeline" 1200  # 20 min

    log_success "Data pipeline completado!"

    # Verify datasets exist
    echo ""
    log_info "Verificando datasets generados:"
    ls -la data/pipeline/07_output/datasets_5min/*.csv 2>/dev/null || {
        log_warn "No se encontraron datasets. Verificar L2 pipeline."
    }
}

# =============================================================================
# Step 2: Training Pipeline
# =============================================================================

run_training_pipeline() {
    echo ""
    echo "============================================================"
    echo "  PASO 2: TRAINING PIPELINE (L3)"
    echo "============================================================"
    echo ""

    # Train Experiment A: Full Macro
    log_info "Entrenando Experimento A: $EXPERIMENT_A"
    trigger_dag "v3.l3_model_training" "{
        \"experiment_config_path\": \"config/experiments/${EXPERIMENT_A}.yaml\",
        \"version\": \"exp_a_v1\",
        \"dvc_enabled\": true,
        \"mlflow_enabled\": true
    }"
    wait_for_dag "v3.l3_model_training" 7200  # 2 hours

    log_success "Experimento A completado!"

    # Train Experiment B: Reduced Macro
    log_info "Entrenando Experimento B: $EXPERIMENT_B"
    trigger_dag "v3.l3_model_training" "{
        \"experiment_config_path\": \"config/experiments/${EXPERIMENT_B}.yaml\",
        \"version\": \"exp_b_v1\",
        \"dvc_enabled\": true,
        \"mlflow_enabled\": true
    }"
    wait_for_dag "v3.l3_model_training" 7200  # 2 hours

    log_success "Experimento B completado!"

    # Show MLflow runs
    echo ""
    log_info "Experimentos registrados en MLflow:"
    echo "  URL: $MLFLOW_URL"
    echo "  Experiment: usdcop_ab_testing"
}

# =============================================================================
# Step 3: Validation & Comparison
# =============================================================================

run_validation_pipeline() {
    echo ""
    echo "============================================================"
    echo "  PASO 3: VALIDATION & A/B COMPARISON (L4)"
    echo "============================================================"
    echo ""

    # Run backtest for both experiments
    log_info "Ejecutando backtest para ambos experimentos..."
    trigger_dag "v3.l4_backtest_validation" '{"run_all_registered": true}'
    wait_for_dag "v3.l4_backtest_validation" 1800

    # Run A/B comparison
    log_info "Ejecutando comparación A/B..."
    trigger_dag "l4_experiment_runner" "{
        \"experiment_name\": \"${EXPERIMENT_B}\",
        \"compare_to\": \"${EXPERIMENT_A}\"
    }"
    wait_for_dag "l4_experiment_runner" 600

    log_success "Validación completada!"

    # Print comparison results
    echo ""
    echo "============================================================"
    echo "  RESULTADOS A/B TESTING"
    echo "============================================================"

    # Get results from MLflow
    python3 << 'EOF'
import mlflow
import json

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("usdcop_ab_testing")

# Get latest runs
runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=2)

if len(runs) >= 2:
    print("\n📊 Comparison Results:")
    print("-" * 60)

    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "Unknown")
        sharpe = run.get("metrics.sharpe_ratio", 0)
        drawdown = run.get("metrics.max_drawdown", 0)
        win_rate = run.get("metrics.win_rate", 0)

        print(f"\n🔹 {name}")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Max Drawdown: {drawdown:.2%}")
        print(f"   Win Rate: {win_rate:.2%}")

    print("\n" + "-" * 60)
else:
    print("Not enough runs found for comparison")
EOF

    echo ""
    log_info "Ver resultados completos en:"
    echo "  • MLflow: $MLFLOW_URL"
    echo "  • Dashboard: $DASHBOARD_URL"
}

# =============================================================================
# Quick Verification
# =============================================================================

verify_setup() {
    echo ""
    echo "============================================================"
    echo "  VERIFICACIÓN DE SETUP"
    echo "============================================================"
    echo ""

    # Check Docker containers
    log_info "Verificando containers Docker..."
    for svc in postgres redis minio mlflow airflow-webserver; do
        if docker-compose ps | grep -q "$svc.*Up"; then
            echo -e "  ${GREEN}✓${NC} $svc"
        else
            echo -e "  ${RED}✗${NC} $svc (not running)"
        fi
    done

    # Check experiment configs
    echo ""
    log_info "Verificando archivos de experimentos..."
    for exp in "$EXPERIMENT_A" "$EXPERIMENT_B"; do
        if [ -f "config/experiments/${exp}.yaml" ]; then
            echo -e "  ${GREEN}✓${NC} config/experiments/${exp}.yaml"
        else
            echo -e "  ${RED}✗${NC} config/experiments/${exp}.yaml (missing)"
        fi
    done

    # Check database tables
    echo ""
    log_info "Verificando tablas PostgreSQL..."
    docker-compose exec -T postgres psql -U postgres -d usdcop -c "
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema IN ('public', 'config', 'trading', 'dw')
        ORDER BY table_schema, table_name;
    " 2>/dev/null | head -20

    echo ""
    log_success "Verificación completada!"
}

# =============================================================================
# Main
# =============================================================================

print_usage() {
    echo "Uso: $0 [comando]"
    echo ""
    echo "Comandos:"
    echo "  verify  - Verificar setup (containers, archivos, tablas)"
    echo "  data    - Ejecutar pipeline de datos (L0 + L1 + L2)"
    echo "  train   - Entrenar ambos experimentos (L3)"
    echo "  compare - Validar y comparar experimentos (L4)"
    echo "  all     - Ejecutar pipeline completo"
    echo ""
    echo "Ejemplo:"
    echo "  $0 all     # Ejecutar todo"
    echo "  $0 train   # Solo entrenar (asume datos ya existen)"
}

case "${1:-all}" in
    verify)
        verify_setup
        ;;
    data)
        run_data_pipeline
        ;;
    train)
        run_training_pipeline
        ;;
    compare)
        run_validation_pipeline
        ;;
    all)
        verify_setup
        run_data_pipeline
        run_training_pipeline
        run_validation_pipeline
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETADO"
echo "============================================================"
echo ""
echo "URLs de acceso:"
echo "  • Airflow:   $AIRFLOW_URL"
echo "  • MLflow:    $MLFLOW_URL"
echo "  • Dashboard: $DASHBOARD_URL"
echo "  • MinIO:     http://localhost:9001"
echo ""
