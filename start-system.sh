#!/bin/bash

# USDCOP Trading System - Quick Start Script
# Inicia todos los servicios necesarios para el sistema de trading

echo "========================================="
echo "  USDCOP Trading System - Quick Start"
echo "========================================="

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Verificar Docker
echo -e "\n${YELLOW}Verificando Docker...${NC}"
if ! docker info &> /dev/null; then
    echo "Iniciando Docker..."
    sudo systemctl start docker
    sleep 3
fi

# 2. Iniciar servicios principales
echo -e "\n${YELLOW}Iniciando servicios...${NC}"

# PostgreSQL
echo "• PostgreSQL/TimescaleDB..."
sudo docker compose up -d postgres 2>/dev/null

# Redis
echo "• Redis..."
sudo docker compose up -d redis 2>/dev/null

# MinIO
echo "• MinIO..."
sudo docker compose up -d minio minio-createbuckets 2>/dev/null

# Trading API
echo "• Trading API..."
sudo docker compose up -d trading-api 2>/dev/null

# Dashboard
echo "• Dashboard..."
sudo docker compose up -d dashboard 2>/dev/null

sleep 5

# 3. Verificar estado
echo -e "\n${YELLOW}Estado de servicios:${NC}"
sudo docker compose ps --format "table {{.Names}}\t{{.Status}}"

# 4. URLs de acceso
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Sistema iniciado!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "\n${YELLOW}Acceso:${NC}"
echo "• Dashboard: http://localhost:3000"
echo "• API: http://localhost:8000/docs"
echo "• MinIO: http://localhost:9001"
echo ""
echo -e "${YELLOW}Comandos útiles:${NC}"
echo "• Ver logs: sudo docker compose logs -f [servicio]"
echo "• Detener: sudo docker compose down"
echo "• Reiniciar: sudo docker compose restart [servicio]"