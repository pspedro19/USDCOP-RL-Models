#!/bin/bash

# USDCOP Trading System - Stop All API Services

echo "========================================="
echo "  Deteniendo Servicios API"
echo "========================================="

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$BASE_DIR/logs/api"

# Puertos de servicios
declare -A SERVICES=(
    ["Trading API"]=8000
    ["Analytics API"]=8001
    ["WebSocket Service"]=8082
    ["Trading Signals API"]=8003
    ["Pipeline Data API"]=8004
    ["ML Analytics API"]=8005
    ["Backtest API"]=8006
    ["L0 Validator"]=8086
)

echo -e "\n${YELLOW}Deteniendo servicios...${NC}\n"

for service in "${!SERVICES[@]}"; do
    port=${SERVICES[$service]}

    # Buscar proceso por puerto
    pid=$(lsof -t -i:$port 2>/dev/null)

    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Deteniendo $service (PID: $pid, Port: $port)...${NC}"
        kill -9 $pid 2>/dev/null

        # Verificar si se detuvo
        sleep 1
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
            echo -e "${GREEN}  ✓ $service detenido${NC}"
        else
            echo -e "${RED}  ✗ Error al detener $service${NC}"
        fi
    else
        echo -e "${YELLOW}$service no está corriendo${NC}"
    fi

    # Eliminar archivo PID si existe
    pid_file="$LOG_DIR/$(echo $service | tr ' ' '-').pid"
    [ -f "$pid_file" ] && rm "$pid_file"
done

echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Todos los servicios detenidos${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}\n"
