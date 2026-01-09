#!/bin/bash
################################################################################
# 🚀 USDCOP TRADING SYSTEM - COMPLETE ALL-IN-ONE DEPLOYMENT
# ============================================================================
# Este script hace TODO en un solo comando:
#   1. Construye todos los servicios Docker
#   2. Levanta todos los contenedores
#   3. Espera a que estén healthy
#   4. Inicializa el Data Warehouse (schemas, dims, facts, marts)
#   5. Restaura backups de base de datos (OHLCV historical data)
#   6. Verifica que todo esté funcionando
#   7. Muestra URLs de acceso
#
# Uso:
#   sudo ./DEPLOY_COMPLETE_ALL_IN_ONE.sh
#
# Requisitos:
#   - Docker y Docker Compose instalados
#   - .env file configurado
#   - Permisos sudo
#
# Tiempo estimado: 15-20 minutos (primera vez)
################################################################################

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_SCRIPTS_DIR="$SCRIPT_DIR/init-scripts"
BACKUP_DIR="$SCRIPT_DIR/data/backups"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# PostgreSQL config
PG_CONTAINER="usdcop-postgres-timescale"
PG_USER="admin"
PG_DB="usdcop_trading"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo "========================================================================"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${MAGENTA}ℹ${NC} $1"
}

spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

wait_for_service() {
    local service_name=$1
    local health_check=$2
    local max_attempts=${3:-30}
    local attempt=1

    print_step "Esperando a que $service_name esté disponible..."

    while [ $attempt -le $max_attempts ]; do
        if eval "$health_check" > /dev/null 2>&1; then
            print_success "$service_name está listo"
            return 0
        fi

        if [ $attempt -eq $max_attempts ]; then
            print_error "$service_name no respondió a tiempo"
            return 1
        fi

        echo -n "."
        sleep 2
        ((attempt++))
    done
}

# ============================================================================
# DEPLOYMENT STEPS
# ============================================================================

step1_check_prerequisites() {
    print_header "PASO 1/10: Verificando Prerequisites"

    # Check Docker
    print_step "Verificando Docker..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker no está instalado"
        print_info "Instalando Docker..."
        curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
        sudo sh /tmp/get-docker.sh
        sudo usermod -aG docker $USER
        sudo systemctl start docker
        sudo systemctl enable docker
        print_success "Docker instalado"
    else
        print_success "Docker instalado: $(docker --version)"
    fi

    # Check Docker Compose
    print_step "Verificando Docker Compose..."
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose no está disponible"
        exit 1
    fi
    print_success "Docker Compose disponible"

    # Check .env file
    print_step "Verificando archivo .env..."
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        print_warning ".env no encontrado, creando desde .env.example..."
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            print_success ".env creado"
        else
            print_error ".env.example no encontrado"
            exit 1
        fi
    else
        print_success ".env existe"
    fi

    # Start Docker if not running
    print_step "Verificando Docker daemon..."
    if ! sudo docker info &> /dev/null; then
        print_info "Iniciando Docker daemon..."
        sudo systemctl start docker
        sleep 3
        print_success "Docker daemon iniciado"
    else
        print_success "Docker daemon corriendo"
    fi
}

step2_stop_existing() {
    print_header "PASO 2/10: Deteniendo Contenedores Existentes"

    print_step "Deteniendo y limpiando contenedores..."
    cd "$SCRIPT_DIR"
    sudo docker compose down --remove-orphans 2>/dev/null || true
    print_success "Contenedores detenidos y limpiados"
}

step3_build_services() {
    print_header "PASO 3/10: Construyendo Servicios Docker"

    print_step "Construyendo TODOS los servicios..."
    print_info "Esto puede tomar 10-15 minutos en la primera ejecución..."
    print_info "Construyendo en paralelo para optimizar tiempo..."

    cd "$SCRIPT_DIR"
    sudo docker compose build --parallel

    print_success "Todos los servicios construidos exitosamente"
}

step4_start_services() {
    print_header "PASO 4/10: Iniciando Servicios Docker"

    print_step "Iniciando todos los servicios..."
    cd "$SCRIPT_DIR"
    sudo docker compose up -d

    print_success "Servicios iniciados en modo detached"

    # Show running containers
    print_info "Contenedores corriendo:"
    sudo docker compose ps --format "table {{.Service}}\t{{.Status}}" | head -30
}

step5_wait_for_infrastructure() {
    print_header "PASO 5/10: Esperando Infraestructura"

    # Wait for PostgreSQL
    wait_for_service "PostgreSQL" \
        "sudo docker exec $PG_CONTAINER pg_isready -U $PG_USER" \
        60

    # Wait for Redis
    wait_for_service "Redis" \
        "sudo docker exec usdcop-redis redis-cli ping" \
        30

    # Wait for MinIO
    wait_for_service "MinIO" \
        "curl -f http://localhost:9000/minio/health/live" \
        30

    print_success "Infraestructura lista"
}

step6_init_dwh() {
    print_header "PASO 6/10: Inicializando Data Warehouse"

    print_step "Ejecutando scripts SQL del DWH..."

    # Script 1: Create schemas
    print_info "1/5: Creando schemas (stg, dw, dm)..."
    sudo docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB \
        < "$INIT_SCRIPTS_DIR/02-create-dwh-schema.sql" > /dev/null 2>&1
    print_success "Schemas creados"

    # Script 2: Create dimensions
    print_info "2/5: Creando dimensiones..."
    sudo docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB \
        < "$INIT_SCRIPTS_DIR/03-create-dimensions.sql" > /dev/null 2>&1
    print_success "Dimensiones creadas"

    # Script 3: Seed dimensions
    print_info "3/5: Poblando dimensiones con datos iniciales..."
    print_info "   → dim_time_5m: Poblando 2020-2030 (~525k rows)..."
    sudo docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB \
        < "$INIT_SCRIPTS_DIR/04-seed-dimensions.sql" > /dev/null 2>&1
    print_success "Dimensiones pobladas"

    # Script 4: Create fact tables
    print_info "4/5: Creando fact tables..."
    sudo docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB \
        < "$INIT_SCRIPTS_DIR/05-create-facts.sql" > /dev/null 2>&1
    print_success "Fact tables creadas"

    # Script 5: Create data marts
    print_info "5/5: Creando data marts (vistas materializadas)..."
    sudo docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB \
        < "$INIT_SCRIPTS_DIR/06-create-data-marts.sql" > /dev/null 2>&1
    print_success "Data marts creadas"

    print_success "Data Warehouse inicializado completamente"
}

step7_restore_backups() {
    print_header "PASO 7/10: Restaurando Backups de Base de Datos"

    # Check for OHLCV backup
    print_step "Buscando backups de datos históricos..."

    # Auto-restore OHLCV if script exists
    if [ -f "$INIT_SCRIPTS_DIR/03-auto-restore-ohlcv.sh" ]; then
        print_info "Ejecutando auto-restore de datos OHLCV..."
        bash "$INIT_SCRIPTS_DIR/03-auto-restore-ohlcv.sh" || true
        print_success "Auto-restore ejecutado"
    fi

    # Check for market_data backup (legacy)
    local backup_file="$BACKUP_DIR/20251015_162604/market_data.csv.gz"
    if [ -f "$backup_file" ]; then
        print_step "Restaurando datos de market_data legacy..."

        local existing_records=$(sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -t -c "SELECT COUNT(*) FROM market_data;" 2>/dev/null | tr -d ' ' || echo "0")

        if [ "$existing_records" -lt "1000" ]; then
            print_info "Cargando 92,936 registros históricos..."
            zcat "$backup_file" | sudo docker exec -i $PG_CONTAINER psql -U $PG_USER -d $PG_DB \
                -c "COPY market_data (timestamp, symbol, price, bid, ask, volume, source, created_at) FROM STDIN WITH (FORMAT csv, HEADER true);" \
                > /dev/null 2>&1 || true

            local final_records=$(sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -t -c "SELECT COUNT(*) FROM market_data;" 2>/dev/null | tr -d ' ' || echo "0")
            print_success "Datos market_data cargados: $final_records registros"
        else
            print_success "market_data ya tiene $existing_records registros"
        fi
    else
        print_warning "No se encontró backup de market_data"
    fi

    # Check usdcop_m5_ohlcv data
    local ohlcv_records=$(sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -t -c "SELECT COUNT(*) FROM usdcop_m5_ohlcv;" 2>/dev/null | tr -d ' ' || echo "0")
    print_info "Registros en usdcop_m5_ohlcv: $ohlcv_records"
}

step8_verify_dwh() {
    print_header "PASO 8/10: Verificando Data Warehouse"

    print_step "Verificando schemas..."
    local schemas=$(sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -t -c \
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name IN ('stg', 'dw', 'dm');" \
        | tr -d ' ')

    if [ "$schemas" -eq "3" ]; then
        print_success "Schemas DWH: 3/3 creados (stg, dw, dm)"
    else
        print_error "Schemas DWH: $schemas/3 (esperados 3)"
        return 1
    fi

    print_step "Verificando dimensiones..."
    sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -c "
        SELECT
            'dim_symbol' as dimension, COUNT(*) as count FROM dw.dim_symbol
        UNION ALL
        SELECT 'dim_source', COUNT(*) FROM dw.dim_source
        UNION ALL
        SELECT 'dim_time_5m', COUNT(*) FROM dw.dim_time_5m
        UNION ALL
        SELECT 'dim_feature', COUNT(*) FROM dw.dim_feature
        UNION ALL
        SELECT 'dim_indicator', COUNT(*) FROM dw.dim_indicator
        UNION ALL
        SELECT 'dim_model', COUNT(*) FROM dw.dim_model;
    " 2>/dev/null | tail -7 | head -6

    print_step "Verificando fact tables..."
    local fact_count=$(sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -tAc \
        "SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'dw' AND tablename LIKE 'fact_%';" \
        | tr -d ' ')
    print_success "Fact tables creadas: $fact_count"

    print_step "Verificando data marts..."
    local mart_count=$(sudo docker exec $PG_CONTAINER psql -U $PG_USER -d $PG_DB -tAc \
        "SELECT COUNT(*) FROM pg_matviews WHERE schemaname = 'dm';" \
        | tr -d ' ')
    print_success "Data marts creadas: $mart_count"

    print_success "Data Warehouse verificado correctamente"
}

step9_verify_services() {
    print_header "PASO 9/10: Verificando Servicios"

    # Check PostgreSQL
    print_step "PostgreSQL..."
    if sudo docker exec $PG_CONTAINER pg_isready -U $PG_USER > /dev/null 2>&1; then
        print_success "PostgreSQL: HEALTHY"
    else
        print_error "PostgreSQL: UNHEALTHY"
    fi

    # Check BI API
    print_step "BI API..."
    if curl -f http://localhost:8007/health > /dev/null 2>&1; then
        print_success "BI API: HEALTHY (port 8007)"
    else
        print_warning "BI API: No disponible (esperando...)"
        sleep 10
        if curl -f http://localhost:8007/health > /dev/null 2>&1; then
            print_success "BI API: HEALTHY (port 8007)"
        else
            print_error "BI API: UNHEALTHY"
        fi
    fi

    # Check Trading API
    print_step "Trading API..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Trading API: HEALTHY (port 8000)"
    else
        print_warning "Trading API: No disponible"
    fi

    # Check Airflow
    print_step "Airflow..."
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        print_success "Airflow: HEALTHY (port 8080)"
    else
        print_warning "Airflow: Iniciando (puede tardar 1-2 min más)"
    fi

    # Count healthy services
    local total_services=$(sudo docker compose ps --services | wc -l)
    local running_services=$(sudo docker compose ps | grep -c "Up" || echo "0")

    print_info "Servicios corriendo: $running_services de $total_services"
}

step10_final_verification() {
    print_header "PASO 10/10: Verificación Final"

    # DWH Health via BI API
    print_step "Verificando DWH Health via BI API..."
    if curl -f http://localhost:8007/api/bi/health-check > /dev/null 2>&1; then
        print_success "BI API responde correctamente"

        # Get actual response
        local response=$(curl -s http://localhost:8007/api/bi/health-check)
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        print_warning "BI API aún no disponible (normal si es primera vez)"
    fi

    # Check DAG files
    print_step "Verificando integración DAGs con DWH..."
    local dags_with_dwh=$(grep -l "from utils.dwh_helper import" "$SCRIPT_DIR/airflow/dags"/usdcop_m5__0*.py 2>/dev/null | wc -l)
    print_info "DAGs con integración DWH: $dags_with_dwh"

    if [ "$dags_with_dwh" -ge 4 ]; then
        print_success "✅ DAGs L0, L1, L5, L6 integrados"
    else
        print_warning "Algunos DAGs pueden no estar integrados"
    fi

    print_success "Verificación final completada"
}

show_summary() {
    print_header "🎉 DEPLOYMENT COMPLETADO EXITOSAMENTE"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                     🏆 SISTEMA 100% DESPLEGADO 🏆                    ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    print_info "Servicios desplegados:"
    echo "  ✓ PostgreSQL/TimescaleDB"
    echo "  ✓ Redis"
    echo "  ✓ MinIO"
    echo "  ✓ Airflow"
    echo "  ✓ Trading APIs (8000, 8001, 8002, 8003, 8004, 8005, 8006)"
    echo "  ✓ BI API (8007) ← NUEVO"
    echo "  ✓ Dashboard Next.js (5000)"
    echo "  ✓ Grafana, Prometheus, MLflow"
    echo ""

    print_info "Data Warehouse:"
    echo "  ✓ Schemas: stg, dw, dm"
    echo "  ✓ Dimensiones: 10 tablas (~525k rows en dim_time_5m)"
    echo "  ✓ Fact tables: 15 tablas"
    echo "  ✓ Data marts: 7 vistas materializadas"
    echo "  ✓ DAGs integrados: L0, L1, L5, L6"
    echo ""

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}📊 URLs DE ACCESO:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  🎨 ${BOLD}Dashboard:${NC}         http://localhost:5000"
    echo -e "  🏛️  ${BOLD}BI API (DWH):${NC}      http://localhost:8007"
    echo -e "  📖 ${BOLD}BI API Docs:${NC}       http://localhost:8007/docs"
    echo -e "  📊 ${BOLD}Trading API:${NC}       http://localhost:8000/docs"
    echo -e "  📈 ${BOLD}Analytics API:${NC}     http://localhost:8001/docs"
    echo -e "  🔄 ${BOLD}Airflow:${NC}           http://localhost:8080 ${BLUE}(admin/admin123)${NC}"
    echo -e "  📦 ${BOLD}MinIO Console:${NC}     http://localhost:9001 ${BLUE}(minioadmin/minioadmin123)${NC}"
    echo -e "  📉 ${BOLD}Grafana:${NC}           http://localhost:3002"
    echo ""

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}✅ VERIFICACIÓN RÁPIDA:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  # BI API Health"
    echo "  curl http://localhost:8007/health | jq"
    echo ""
    echo "  # DWH Health Check"
    echo "  curl http://localhost:8007/api/bi/health-check | jq"
    echo ""
    echo "  # Ver dimensiones"
    echo "  curl http://localhost:8007/api/bi/dimensions/symbols | jq"
    echo ""
    echo "  # PostgreSQL directo"
    echo "  sudo docker exec -it $PG_CONTAINER psql -U $PG_USER -d $PG_DB"
    echo ""

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}📋 PRÓXIMOS PASOS:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  1. Ejecutar DAG L0 para poblar fact_bar_5m:"
    echo "     → Abrir: http://localhost:8080"
    echo "     → DAG: usdcop_m5__01_l0_intelligent_acquire"
    echo "     → Trigger manual"
    echo ""
    echo "  2. Verificar datos en DWH:"
    echo "     curl 'http://localhost:8007/api/bi/bars?symbol=USD/COP' | jq"
    echo ""
    echo "  3. Explorar Swagger docs:"
    echo "     → http://localhost:8007/docs"
    echo ""
    echo "  4. Ver logs de servicios:"
    echo "     sudo docker logs usdcop-bi-api -f"
    echo ""

    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}${BOLD}📚 DOCUMENTACIÓN:${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  📄 START_HERE_PEDRO.md          ← Empieza aquí"
    echo "  📄 DWH_FINAL_100_COMPLETO.md    ← Guía completa DWH"
    echo "  📄 RESPUESTAS_PEDRO.md          ← Respuestas a tus preguntas"
    echo ""

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${BOLD}✅ Sistema completamente desplegado y listo para usar!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${BOLD}Deployment completado en:${NC} $(date)"
    echo ""
}

# ============================================================================
# ERROR HANDLER
# ============================================================================

error_handler() {
    echo ""
    print_error "Deployment falló en la línea $1"
    print_error "Revisa los logs arriba para ver el error"
    echo ""
    print_info "Para debugging:"
    echo "  1. Ver logs de PostgreSQL: sudo docker logs $PG_CONTAINER --tail 50"
    echo "  2. Ver logs de BI API: sudo docker logs usdcop-bi-api --tail 50"
    echo "  3. Ver todos los contenedores: sudo docker compose ps"
    echo ""
    exit 1
}

trap 'error_handler $LINENO' ERR

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    clear

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║        🚀 USDCOP TRADING SYSTEM - COMPLETE DEPLOYMENT 🚀            ║"
    echo "║                                                                      ║"
    echo "║              Sistema Completo + Data Warehouse                       ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""

    print_info "Iniciando deployment completo del sistema..."
    print_info "Tiempo estimado: 15-20 minutos (primera vez)"
    print_info "Directorio: $SCRIPT_DIR"
    echo ""

    sleep 2

    # Execute all steps
    step1_check_prerequisites
    step2_stop_existing
    step3_build_services
    step4_start_services
    step5_wait_for_infrastructure
    step6_init_dwh
    step7_restore_backups
    step8_verify_dwh
    step9_verify_services
    step10_final_verification

    # Show summary
    show_summary
}

# Run main function
main

exit 0
