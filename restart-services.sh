#!/bin/bash

echo "🚀 Reiniciando servicios USDCOP Trading System..."
echo "=============================================="

# 1. Detener todos los procesos
echo "⏹️  Deteniendo procesos existentes..."
pkill -f "node" 2>/dev/null
pkill -f "python3" 2>/dev/null
pkill -f "uvicorn" 2>/dev/null
sleep 2

# 2. Iniciar API Backend
echo "🔧 Iniciando API Backend en puerto 8000..."
cd /home/GlobalForex/USDCOP-RL-Models
nohup python3 api_server.py > api.log 2>&1 &
sleep 3

# 3. Verificar API
echo "✅ Verificando API..."
curl -s http://localhost:8000/api/market/health | jq

# 4. Iniciar Dashboard
echo "🎨 Iniciando Dashboard en puerto 5000..."
cd usdcop-trading-dashboard
export TRADING_API_URL=http://localhost:8000
nohup npm start > dashboard.log 2>&1 &
sleep 5

# 5. Estado final
echo ""
echo "✅ Servicios reiniciados:"
echo "- API: http://localhost:8000/docs"
echo "- Dashboard: http://localhost:5000"
echo ""
echo "📊 Para ver logs:"
echo "- tail -f api.log"
echo "- tail -f usdcop-trading-dashboard/dashboard.log"
echo ""
echo "🔍 Para verificar conectividad:"
echo "python3 test_api_connectivity.py"