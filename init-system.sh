#!/bin/bash
# =============================================================================
# USDCOP Trading System - Complete Initialization Script
# =============================================================================
# Este script maneja la inicialización completa del sistema incluyendo:
# 1. Levantar servicios base
# 2. Auto-inicialización de datos con backup/restore
# 3. Verificación de estado final

set -e

echo "🚀 USDCOP TRADING SYSTEM - COMPLETE INITIALIZATION"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if services are healthy
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=0

    log_info "Checking health of $service..."

    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps $service | grep -q "healthy"; then
            log_success "$service is healthy"
            return 0
        fi

        attempt=$((attempt + 1))
        log_info "Waiting for $service to be healthy... ($attempt/$max_attempts)"
        sleep 10
    done

    log_error "$service failed to become healthy after $max_attempts attempts"
    return 1
}

# Step 1: Start infrastructure services
log_info "Step 1: Starting infrastructure services..."
docker-compose up -d postgres redis minio

# Step 2: Wait for services to be healthy
log_info "Step 2: Waiting for services to be healthy..."
check_service_health postgres
check_service_health redis
check_service_health minio

# Step 3: Initialize MinIO buckets
log_info "Step 3: Initializing MinIO buckets..."
docker-compose up minio-init

# Step 4: Create backups directory
log_info "Step 4: Creating backups directory..."
mkdir -p ./backups
chmod 755 ./backups

# Step 5: Run backup/restore initialization
log_info "Step 5: Running backup/restore auto-initialization..."
if docker-compose -f docker-compose.backup-init.yml up --build; then
    log_success "Backup/restore initialization completed"
else
    log_warning "Backup/restore initialization had issues, continuing..."
fi

# Step 6: Start remaining services
log_info "Step 6: Starting remaining services (Airflow, etc.)..."
docker-compose up -d

# Step 7: Final verification
log_info "Step 7: Final system verification..."
sleep 30  # Wait for all services to settle

# Check if we can connect to database and verify data
log_info "Verifying database connectivity and data..."
if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "SELECT COUNT(*) FROM market_data WHERE symbol = 'USDCOP';" >/dev/null 2>&1; then
    RECORD_COUNT=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM market_data WHERE symbol = 'USDCOP';" | tr -d ' ')
    if [ "$RECORD_COUNT" -gt 0 ]; then
        log_success "Database verified: $RECORD_COUNT USDCOP records found"
    else
        log_warning "Database connected but no USDCOP data found"
    fi
else
    log_error "Could not verify database"
fi

# Show final status
echo ""
echo "🎉 INITIALIZATION COMPLETE!"
echo "========================="
echo ""
log_info "System Status:"
docker-compose ps

echo ""
log_info "Access URLs:"
echo "   📊 Airflow:  http://localhost:8080 (admin/admin123)"
echo "   📈 Grafana:  http://localhost:3001 (admin/admin123)"
echo "   💾 MinIO:    http://localhost:9001 (minioadmin/minioadmin123)"
echo "   🗄️  Postgres: localhost:5432 (admin/admin123)"

echo ""
log_info "Backup Commands:"
echo "   📦 Create backup:     python3 scripts/backup_restore_system.py backup"
echo "   🔄 Restore backup:    python3 scripts/backup_restore_system.py restore"
echo "   📋 List backups:      python3 scripts/backup_restore_system.py list"
echo "   📊 Show stats:        python3 scripts/backup_restore_system.py stats"

echo ""
log_success "USDCOP Trading System is ready! 🚀"