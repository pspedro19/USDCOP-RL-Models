@echo off
REM API Monitoring Server Startup Script
REM =====================================

echo Starting API Monitoring System...
echo.

REM Set environment variables
set PYTHONPATH=%CD%
set API_MONITOR_URL=http://localhost:8001
set FLASK_ENV=development

REM Create data cache directory if it doesn't exist
if not exist "data_cache" mkdir data_cache

echo [INFO] Starting API Monitoring Server on port 8001...
echo [INFO] Dashboard will be available at http://localhost:3000
echo [INFO] API endpoints will be available at http://localhost:8001
echo.

REM Start the API monitoring server
python api_monitoring_server.py

pause