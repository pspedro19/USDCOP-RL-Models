#!/bin/bash
# Trading Signals Service - Local Development Launcher
# =====================================================

echo "========================================"
echo "Trading Signals Service - Local Launch"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration"
fi

# Create models directory
mkdir -p models logs

echo ""
echo "========================================"
echo "Starting Trading Signals Service..."
echo "========================================"
echo ""
echo "Service will be available at: http://localhost:8003"
echo "API Documentation: http://localhost:8003/docs"
echo "WebSocket endpoint: ws://localhost:8003/ws/signals"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the service
python main.py
