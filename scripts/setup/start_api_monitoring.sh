#!/bin/bash

# API Monitoring Server Startup Script (Linux/MacOS)
# ===================================================

echo "Starting API Monitoring System..."
echo

# Set environment variables
export PYTHONPATH=$(pwd)
export API_MONITOR_URL=http://localhost:8001
export FLASK_ENV=development

# Create data cache directory if it doesn't exist
mkdir -p data_cache

echo "[INFO] Starting API Monitoring Server on port 8001..."
echo "[INFO] Dashboard will be available at http://localhost:3000"
echo "[INFO] API endpoints will be available at http://localhost:8001"
echo

# Start the API monitoring server
python api_monitoring_server.py