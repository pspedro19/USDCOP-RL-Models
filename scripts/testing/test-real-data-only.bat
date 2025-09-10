@echo off
echo ================================
echo TEST: Verificando solo datos reales
echo ================================
echo.

echo 1. Probando API historica...
curl -s http://localhost:3000/api/data/historical | findstr "success error"
echo.

echo 2. Probando deteccion de gaps...
curl -s http://localhost:3000/api/data/gaps?action=detect | findstr "totalGaps"
echo.

echo 3. Intentando llenar gaps (debe fallar o usar solo datos reales)...
curl -s http://localhost:3000/api/data/gaps?action=fill | findstr "synthetic"
echo.

echo 4. Probando datos en tiempo real...
curl -s http://localhost:3000/api/market/realtime?action=fetch | findstr "success error"
echo.

echo ================================
echo IMPORTANTE: No debe aparecer "synthetic" ni "mock" en los resultados
echo ================================
pause