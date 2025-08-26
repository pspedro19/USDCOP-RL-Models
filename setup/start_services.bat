@echo off
echo ==========================================
echo USDCOP Trading Platform - Service Launcher
echo ==========================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [1/5] Stopping existing services...
docker-compose -f docker-compose.complete.yml down

echo.
echo [2/5] Building custom images...
docker-compose -f docker-compose.complete.yml build

echo.
echo [3/5] Starting services...
docker-compose -f docker-compose.complete.yml up -d

echo.
echo [4/5] Waiting for services to be healthy...
timeout /t 30 /nobreak >nul

echo.
echo [5/5] Checking service status...
docker-compose -f docker-compose.complete.yml ps

echo.
echo ==========================================
echo Services URLs:
echo ==========================================
echo.
echo AIRFLOW:          http://localhost:8081 (admin/admin123)
echo KAFKA UI:         http://localhost:8080
echo MINIO:            http://localhost:9001 (minioadmin/minioadmin123)
echo MLFLOW:           http://localhost:5000
echo GRAFANA:          http://localhost:3000 (admin/admin123)
echo PGADMIN:          http://localhost:5050 (admin@trading.local/admin123)
echo PREMIUM DASHBOARD: http://localhost:8090
echo PROMETHEUS:       http://localhost:9090
echo.
echo ==========================================
echo.
echo To view logs: docker-compose -f docker-compose.complete.yml logs -f [service_name]
echo To stop all: docker-compose -f docker-compose.complete.yml down
echo.
pause