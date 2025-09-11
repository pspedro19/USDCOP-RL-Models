#!/bin/bash

# USDCOP Trading System - Complete Service Startup Script
# This script starts all services including the professional dashboard

set -e

echo "🚀 Starting USDCOP Trading System with Professional Dashboard..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Docker is running"

# Stop any existing services
print_info "Stopping existing services..."
docker-compose down --remove-orphans 2>/dev/null || true

# Clean up networks
docker network prune -f > /dev/null 2>&1 || true

print_status "Cleaned up existing services"

# Start core services first (database, cache, storage)
print_info "Starting core services (PostgreSQL, Redis, MinIO)..."
docker-compose up -d postgres redis minio

# Wait for core services to be healthy
print_info "Waiting for core services to be ready..."
sleep 30

# Check core services health
print_info "Checking core services health..."
if docker-compose ps postgres | grep -q "healthy"; then
    print_status "PostgreSQL is healthy"
else
    print_warning "PostgreSQL might still be starting..."
fi

if docker-compose ps redis | grep -q "healthy"; then
    print_status "Redis is healthy"
else
    print_warning "Redis might still be starting..."
fi

# Start monitoring services
print_info "Starting monitoring services (Prometheus, Grafana)..."
docker-compose up -d prometheus grafana

# Start Airflow services
print_info "Starting Airflow services..."
docker-compose up -d airflow-init
sleep 10
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker

# Start the professional dashboard
print_info "Building and starting Professional Trading Dashboard..."
docker-compose build trading-dashboard
docker-compose up -d trading-dashboard

# Wait for dashboard to be ready
print_info "Waiting for dashboard to be ready..."
sleep 45

# Final status check
echo ""
echo "🔍 Service Status Check:"
echo "========================"

# Check each service
services=("postgres" "redis" "minio" "prometheus" "grafana" "trading-dashboard")
for service in "${services[@]}"; do
    if docker-compose ps "$service" | grep -q "Up"; then
        print_status "$service is running"
    else
        print_error "$service is not running"
    fi
done

echo ""
echo "🌐 Access URLs:"
echo "==============="
echo -e "${GREEN}📊 Professional Trading Dashboard: ${BLUE}http://138.68.252.54:3000${NC}"
echo -e "${GREEN}🔐 Login Credentials: ${BLUE}admin / admin${NC}"
echo -e "${GREEN}📈 Grafana Monitoring: ${BLUE}http://138.68.252.54:3100${NC} (admin/grafana123)"
echo -e "${GREEN}💾 MinIO Console: ${BLUE}http://138.68.252.54:9001${NC} (minioadmin/minioadmin123)"
echo -e "${GREEN}📊 Prometheus: ${BLUE}http://138.68.252.54:9090${NC}"
echo -e "${GREEN}⚙️  Airflow: ${BLUE}http://138.68.252.54:8080${NC} (admin/admin123)"

echo ""
print_status "All services started! Professional Dashboard is ready at http://138.68.252.54:3000"
echo ""
print_info "To stop all services: docker-compose down"
print_info "To view logs: docker-compose logs -f trading-dashboard"