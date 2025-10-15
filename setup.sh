#!/bin/bash
# =============================================================================
# USDCOP Trading System - Complete Setup Script
# =============================================================================
# Este script configura e inicializa completamente el sistema de trading USDCOP

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

echo "🚀 USDCOP TRADING SYSTEM - COMPLETE SETUP"
echo "=========================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    log_error "docker-compose is not installed. Please install it first."
    exit 1
fi

# Step 1: Stop any existing services
log_info "Step 1: Stopping any existing services..."
docker-compose down --remove-orphans 2>/dev/null || true

# Step 2: Create necessary directories
log_info "Step 2: Creating directories..."
mkdir -p ./backups
mkdir -p ./init-scripts
chmod 755 ./backups

# Step 3: Start infrastructure services
log_info "Step 3: Starting infrastructure services..."
docker-compose up -d postgres redis minio

# Step 4: Wait for services to be healthy
log_info "Step 4: Waiting for services to be healthy..."

wait_for_service() {
    local service=$1
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps $service | grep -q "healthy"; then
            log_success "$service is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        log_info "Waiting for $service... ($attempt/$max_attempts)"
        sleep 5
    done

    log_error "$service failed to become healthy"
    return 1
}

wait_for_service postgres
wait_for_service redis
wait_for_service minio

# Step 5: Initialize MinIO buckets
log_info "Step 5: Initializing MinIO buckets..."
docker-compose up minio-init

# Step 6: Run auto-initialization (backup/restore if needed)
log_info "Step 6: Running auto-initialization..."
if python3 scripts/simple_backup_restore.py auto-init; then
    log_success "Auto-initialization completed"
else
    log_warning "Auto-initialization had issues, continuing..."
fi

# Step 7: Start all remaining services
log_info "Step 7: Starting all services..."
docker-compose up -d

# Step 8: Wait for all services to settle
log_info "Step 8: Waiting for all services to be ready..."
sleep 30

# Step 9: Final verification
log_info "Step 9: Final verification..."

# Check database connection and data
if python3 scripts/simple_backup_restore.py stats >/dev/null 2>&1; then
    RECORD_COUNT=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM market_data WHERE symbol = 'USDCOP';" 2>/dev/null | tr -d ' ' || echo "0")

    if [ "$RECORD_COUNT" -gt 0 ]; then
        log_success "Database verified: $RECORD_COUNT USDCOP records"
    else
        log_warning "Database connected but no USDCOP data found"
        log_info "You can run the L0 pipeline to fetch data"
    fi
else
    log_warning "Could not verify database connection"
fi

# Show service status
echo ""
log_info "Current service status:"
docker-compose ps

# Show access information
echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
log_info "Access URLs:"
echo "   📊 Airflow:  http://localhost:8080 (admin/admin123)"
echo "   📈 Grafana:  http://localhost:3001 (admin/admin123)"
echo "   💾 MinIO:    http://localhost:9001 (minioadmin/minioadmin123)"
echo "   🗄️  Postgres: localhost:5432 (admin/admin123)"

echo ""
log_info "Backup/Restore Commands:"
echo "   📦 Create backup:     python3 scripts/simple_backup_restore.py backup"
echo "   🔄 Restore backup:    python3 scripts/simple_backup_restore.py restore"
echo "   📋 List backups:      python3 scripts/simple_backup_restore.py list"
echo "   📊 Show stats:        python3 scripts/simple_backup_restore.py stats"

echo ""
log_info "Data Pipeline Commands:"
echo "   🚀 Run L0 pipeline:   docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__01_l0_intelligent_acquire"
echo "   📊 Check L0 status:   docker exec usdcop-airflow-webserver airflow dags state usdcop_m5__01_l0_intelligent_acquire"

echo ""
log_success "USDCOP Trading System is ready! 🚀"

# Optional: Create first backup if we have data
if [ "$RECORD_COUNT" -gt 0 ]; then
    log_info "Creating initial backup of current data..."
    python3 scripts/simple_backup_restore.py backup
fi
