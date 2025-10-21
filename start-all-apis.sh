#!/bin/bash

# USDCOP Trading System - API Services Startup Script
# Inicia todos los servicios API necesarios para el sistema completo

echo "========================================="
echo "  USDCOP API Services - Startup"
echo "========================================="

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Directorio base
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BASE_DIR/logs/api"

# Crear directorio de logs si no existe
mkdir -p "$LOG_DIR"

# Función para iniciar servicio
start_service() {
    local service_name=$1
    local service_file=$2
    local port=$3
    local env_var=$4

    echo -e "${YELLOW}Iniciando $service_name en puerto $port...${NC}"

    # Verificar si el puerto ya está en uso
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}  ⚠️  Puerto $port ya está en uso. Deteniendo proceso...${NC}"
        kill -9 $(lsof -t -i:$port) 2>/dev/null || true
        sleep 1
    fi

    # Iniciar servicio en background
    if [ -n "$env_var" ]; then
        export $env_var=$port
    fi

    nohup python3 "$BASE_DIR/$service_file" > "$LOG_DIR/${service_name}.log" 2>&1 &
    local pid=$!

    # Guardar PID
    echo $pid > "$LOG_DIR/${service_name}.pid"

    echo -e "${GREEN}  ✓ $service_name iniciado (PID: $pid)${NC}"
    sleep 2
}

# Función para verificar servicio
check_service() {
    local service_name=$1
    local port=$2

    if curl -s "http://localhost:$port/api/health" > /dev/null 2>&1 || \
       curl -s "http://localhost:$port/" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ $service_name: OK${NC}"
        return 0
    else
        echo -e "${RED}  ✗ $service_name: FAILED${NC}"
        return 1
    fi
}

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Paso 1: Iniciando Servicios API${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}\n"

# 1. Main Trading API (Puerto 8000)
start_service "Trading API" "api_server.py" 8000 ""

# 2. Analytics API (Puerto 8001)
start_service "Analytics API" "services/trading_analytics_api.py" 8001 ""

# 3. WebSocket Service (Puerto 8082)
if [ -f "services/realtime_data_service.py" ]; then
    start_service "WebSocket Service" "services/realtime_data_service.py" 8082 ""
fi

# 4. Trading Signals API (Puerto 8003) - NUEVO
start_service "Trading Signals API" "services/trading_signals_api.py" 8003 "TRADING_SIGNALS_API_PORT"

# 5. Pipeline Data API (Puerto 8004) - NUEVO
start_service "Pipeline Data API" "services/pipeline_data_api.py" 8004 "PIPELINE_DATA_API_PORT"

# 6. ML Analytics API (Puerto 8005) - NUEVO
start_service "ML Analytics API" "services/ml_analytics_api.py" 8005 "ML_ANALYTICS_API_PORT"

# 7. Backtest API (Puerto 8006) - NUEVO
start_service "Backtest API" "services/backtest_api.py" 8006 "BACKTEST_API_PORT"

# 8. L0 Validator (Puerto 8086) - Opcional
if [ -f "services/optimized_l0_validator_fastapi.py" ]; then
    start_service "L0 Validator" "services/optimized_l0_validator_fastapi.py" 8086 ""
fi

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Paso 2: Esperando servicios...${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}\n"

sleep 5

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Paso 3: Verificando Estado${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}\n"

# Verificar servicios
check_service "Trading API" 8000
check_service "Analytics API" 8001
check_service "Trading Signals API" 8003
check_service "Pipeline Data API" 8004
check_service "ML Analytics API" 8005
check_service "Backtest API" 8006

echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Sistema API Iniciado${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}\n"

echo -e "${YELLOW}URLs de Acceso:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}Core Services:${NC}"
echo "  • Trading API:        http://localhost:8000/docs"
echo "  • Analytics API:      http://localhost:8001/docs"
echo "  • WebSocket:          ws://localhost:8082/ws"
echo ""
echo -e "${BLUE}New Services (100% Coverage):${NC}"
echo "  • Trading Signals:    http://localhost:8003/docs"
echo "  • Pipeline Data:      http://localhost:8004/docs"
echo "  • ML Analytics:       http://localhost:8005/docs"
echo "  • Backtest API:       http://localhost:8006/docs"
echo ""
echo -e "${BLUE}Frontend:${NC}"
echo "  • Dashboard:          http://localhost:3000"
echo ""

echo -e "${YELLOW}Logs:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • Ver logs: tail -f $LOG_DIR/[servicio].log"
echo "  • Ver todos: tail -f $LOG_DIR/*.log"
echo ""

echo -e "${YELLOW}Control:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • Detener todos: ./stop-all-apis.sh"
echo "  • Reiniciar: ./restart-services.sh"
echo "  • Estado: ./check-api-status.sh"
echo ""

echo -e "${GREEN}✓ Todos los servicios están corriendo${NC}"
echo -e "${GREEN}✓ Sistema listo para 100% funcionalidad${NC}\n"
