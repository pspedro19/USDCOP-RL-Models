@echo off
echo ========================================
echo    USDCOP Trading System Launcher
echo ========================================
echo.
echo Starting the main dashboard server...
echo.
echo Features:
echo   ✓ Professional Trading UI
echo   ✓ Clean Architecture
echo   ✓ Real-time Data
echo   ✓ Stable Performance
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python main_server.py

pause
