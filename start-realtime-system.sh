#!/bin/bash

# 🚀 USDCOP Real-time Trading System Startup Script
# Automated deployment with proper dependency management

set -e

echo "🚀 Starting USDCOP Real-time Trading System"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your API keys before continuing"
    echo "Press Enter when ready to continue..."
    read
fi

# Check required environment variables
print_status "Checking required environment variables..."

if [ -z "${TWELVEDATA_API_KEY_1}" ]; then
    print_warning "TWELVEDATA_API_KEY_1 not set in environment"
    echo "Please set your TwelveData API key in .env file"
fi

# Start the system with proper sequencing
print_status "Starting infrastructure services (PostgreSQL, Redis, MinIO)..."

# Phase 1: Core infrastructure
docker-compose up -d postgres redis minio

print_status "Waiting for infrastructure services to be ready..."
sleep 30

# Initialize MinIO buckets
print_status "Initializing MinIO buckets..."
docker-compose up minio-init

# Phase 2: Airflow services
print_status "Starting Airflow services..."
docker-compose up -d airflow-init

print_status "Waiting for Airflow initialization..."
sleep 60

docker-compose up -d airflow-scheduler airflow-webserver airflow-worker

print_status "Waiting for Airflow services to be ready..."
sleep 30

# Phase 3: Real-time orchestrator (main service)
print_status "Starting USDCOP Real-time Orchestrator..."
docker-compose up -d usdcop-realtime-orchestrator

# Phase 4: Supporting services
print_status "Starting WebSocket and monitoring services..."
docker-compose up -d websocket-service health-monitor

# Phase 5: Web services
print_status "Starting dashboard and monitoring services..."
docker-compose up -d dashboard prometheus grafana

# Phase 6: Reverse proxy
print_status "Starting reverse proxy..."
docker-compose up -d nginx

print_status "Waiting for all services to be ready..."
sleep 45

# Health checks
print_status "Performing health checks..."

echo ""
echo "🔍 Service Health Status:"
echo "========================"

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U admin > /dev/null 2>&1; then
    print_success "PostgreSQL: Ready"
else
    print_error "PostgreSQL: Not ready"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis: Ready"
else
    print_error "Redis: Not ready"
fi

# Check Airflow
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    print_success "Airflow: Ready"
else
    print_warning "Airflow: Starting up (may take a few more minutes)"
fi

# Check Real-time Orchestrator
if curl -s http://localhost:8085/health > /dev/null 2>&1; then
    print_success "Real-time Orchestrator: Ready"
else
    print_warning "Real-time Orchestrator: Starting up"
fi

# Check Dashboard
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    print_success "Dashboard: Ready"
else
    print_warning "Dashboard: Starting up"
fi

echo ""
print_success "USDCOP Real-time Trading System is starting up!"
echo ""

echo "📊 Access Points:"
echo "=================="
echo "🎛️  Main Dashboard:      http://localhost:3000"
echo "⚙️  Airflow UI:          http://localhost:8080 (admin/admin123)"
echo "🔄 Real-time Status:     http://localhost:8085/health"
echo "📡 WebSocket Service:    ws://localhost:8082/ws"
echo "🗄️  PostgreSQL:          localhost:5432 (admin/admin123)"
echo "🔗 MinIO Console:        http://localhost:9001 (minioadmin/minioadmin123)"
echo "📈 Grafana:              http://localhost:3002 (admin/admin123)"
echo "📊 Prometheus:           http://localhost:9090"
echo "🌐 Nginx Gateway:        http://localhost:80"

echo ""
echo "🕒 Market Hours: Monday-Friday, 8:00 AM - 12:55 PM (COT)"
echo ""

echo "📋 System Status Commands:"
echo "=========================="
echo "📊 Check all services:    docker-compose ps"
echo "📜 View orchestrator logs: docker-compose logs -f usdcop-realtime-orchestrator"
echo "🔍 Check database schema:  docker-compose exec postgres psql -U admin -d usdcop_trading -c '\\dt'"
echo "🎯 Market status:          curl http://localhost:8085/status"
echo "💾 Latest price:           curl http://localhost:8085/market/latest"

echo ""
print_status "Monitoring system startup..."
print_status "The real-time orchestrator will wait for L0 pipeline completion before starting data collection"
print_status "During market hours (L-V 8:00-12:55 COT), real-time data collection will begin automatically"

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Monitor Airflow DAGs at http://localhost:8080"
echo "2. Wait for L0 pipeline to complete (loads historical data)"
echo "3. Real-time collection starts automatically during market hours"
echo "4. Monitor real-time status at http://localhost:8085/status"
echo "5. View live dashboard at http://localhost:3000"

echo ""
print_success "System startup complete! 🚀"

# Optional: Show logs
read -p "Would you like to see real-time logs? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Showing real-time orchestrator logs (Ctrl+C to exit)..."
    docker-compose logs -f usdcop-realtime-orchestrator
fi