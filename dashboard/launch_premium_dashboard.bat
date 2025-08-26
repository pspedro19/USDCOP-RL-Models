@echo off
echo ==========================================
echo USDCOP Premium Session Dashboard Launcher
echo Dataset: Silver Premium (Lun-Vie 08:00-14:00 COT)
echo ==========================================
echo.

REM Kill any existing Python processes on port 8090
echo Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8090') do taskkill /F /PID %%a 2>nul

REM Navigate to project directory
cd /d "C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL"

REM Start Premium Data Server
echo Starting Premium Data Server on port 8090...
start /min cmd /c "python dashboard\premium_data_server.py"

REM Wait for server to start
echo Waiting for server to initialize...
timeout /t 3 /nobreak >nul

REM Open dashboard in browser
echo Opening Premium Dashboard in browser...
start http://localhost:8090/premium_dashboard.html

echo.
echo ==========================================
echo Dashboard is running!
echo.
echo Server: http://localhost:8090
echo Session: Premium (Lun-Vie 08:00-14:00 COT)
echo Dataset: 86,272 records (90.9%% completeness)
echo Quality: OPTIMAL for trading and ML
echo.
echo Press Ctrl+C to stop the server
echo ==========================================

REM Keep window open
pause