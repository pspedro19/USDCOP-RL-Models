@echo off
REM Trading Signals Service - Local Development Launcher (Windows)
REM ===============================================================

echo ========================================
echo Trading Signals Service - Local Launch
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt

REM Create .env if not exists
if not exist ".env" (
    echo Creating .env from template...
    copy .env.example .env
    echo WARNING: Please edit .env with your configuration
)

REM Create directories
if not exist "models\" mkdir models
if not exist "logs\" mkdir logs

echo.
echo ========================================
echo Starting Trading Signals Service...
echo ========================================
echo.
echo Service will be available at: http://localhost:8003
echo API Documentation: http://localhost:8003/docs
echo WebSocket endpoint: ws://localhost:8003/ws/signals
echo.
echo Press Ctrl+C to stop
echo.

REM Run the service
python main.py

pause
