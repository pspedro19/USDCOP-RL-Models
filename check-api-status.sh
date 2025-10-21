#!/bin/bash

# USDCOP Trading System - Check API Services Status

echo "========================================="
echo "  Estado de Servicios API"
echo "========================================="

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Función para verificar servicio
check_service() {
    local service_name=$1
    local port=$2
    local endpoint=$3

    echo -e "\n${BLUE}$service_name (Port $port):${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Verificar si el puerto está en uso
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        local pid=$(lsof -t -i:$port)
        echo -e "  Estado:    ${GREEN}● Running${NC}"
        echo -e "  PID:       $pid"

        # Verificar endpoint HTTP
        if curl -s "http://localhost:$port$endpoint" > /dev/null 2>&1; then
            echo -e "  HTTP:      ${GREEN}✓ Respondiendo${NC}"

            # Obtener versión/info si está disponible
            local response=$(curl -s "http://localhost:$port/" 2>/dev/null)
            if [ -n "$response" ]; then
                local version=$(echo $response | jq -r '.version // "N/A"' 2>/dev/null)
                if [ "$version" != "null" ] && [ "$version" != "N/A" ]; then
                    echo -e "  Version:   $version"
                fi
            fi
        else
            echo -e "  HTTP:      ${YELLOW}⚠ No responde${NC}"
        fi

        echo -e "  URL:       ${BLUE}http://localhost:$port/docs${NC}"
    else
        echo -e "  Estado:    ${RED}○ Stopped${NC}"
        echo -e "  HTTP:      ${RED}✗ No disponible${NC}"
    fi
}

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Servicios Core${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

check_service "Trading API" 8000 "/api/latest/USDCOP"
check_service "Analytics API" 8001 "/api/health"
check_service "WebSocket Service" 8082 "/health"

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Nuevos Servicios (100% Coverage)${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

check_service "Trading Signals API" 8003 "/api/health"
check_service "Pipeline Data API" 8004 "/api/health"
check_service "ML Analytics API" 8005 "/api/health"
check_service "Backtest API" 8006 "/api/health"

echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Servicios Opcionales${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"

check_service "L0 Validator" 8086 "/health"

# Resumen
echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  Resumen${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}\n"

total=0
running=0

for port in 8000 8001 8003 8004 8005 8006 8082; do
    total=$((total + 1))
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        running=$((running + 1))
    fi
done

echo -e "  Total servicios:    $total"
echo -e "  Corriendo:          ${GREEN}$running${NC}"
echo -e "  Detenidos:          ${RED}$((total - running))${NC}"

if [ $running -eq $total ]; then
    echo -e "\n  ${GREEN}✓ Todos los servicios están operacionales${NC}"
elif [ $running -gt 0 ]; then
    echo -e "\n  ${YELLOW}⚠ Algunos servicios no están corriendo${NC}"
    echo -e "  ${YELLOW}  Ejecuta: ./start-all-apis.sh${NC}"
else
    echo -e "\n  ${RED}✗ Ningún servicio está corriendo${NC}"
    echo -e "  ${RED}  Ejecuta: ./start-all-apis.sh${NC}"
fi

echo ""
