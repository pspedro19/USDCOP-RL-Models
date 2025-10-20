# 📊 API Status Report - USDCOP Trading System

## ✅ Sistema Operativo

### API Backend (Puerto 8000)
- **Estado**: ✅ Funcionando correctamente
- **URL**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs

### Dashboard Frontend (Puerto 5000)
- **Estado**: ✅ Funcionando
- **URL**: http://localhost:5000

## 🔧 Solución Implementada

Se ha resuelto el error 500 modificando la configuración del proxy para usar `localhost:8000` en lugar del nombre del contenedor Docker.

### Archivos Modificados:
1. `/usdcop-trading-dashboard/app/api/proxy/trading/[...path]/route.ts`
2. `/usdcop-trading-dashboard/app/api/proxy/ws/route.ts`
3. `/usdcop-trading-dashboard/.env.local` (añadido TRADING_API_URL)

## 📡 Endpoints Disponibles

### API Directa (Puerto 8000)
```bash
# Último precio
curl http://localhost:8000/api/latest/USDCOP

# Datos históricos (candlesticks)
curl http://localhost:8000/api/candlesticks/USDCOP

# Estado del sistema
curl http://localhost:8000/api/market/health
```

### Respuesta de ejemplo:
```json
{
  "symbol": "USDCOP",
  "price": 4322.0,
  "bid": 4321.5,
  "ask": 4322.5,
  "volume": 2000000,
  "timestamp": "2025-10-15T18:00:00+00:00",
  "market_status": "open"
}
```

## 🚀 Instrucciones para Reiniciar

Si necesitas reiniciar el sistema completamente:

```bash
# 1. Detener servicios actuales
pkill -f node
pkill -f python3

# 2. Iniciar API Backend
cd /home/GlobalForex/USDCOP-RL-Models
python3 api_server.py &

# 3. Iniciar Dashboard
cd usdcop-trading-dashboard
npm run build
npm start
```

## 📊 Estado de la Base de Datos

- **Registros en market_data**: 3 (datos de prueba)
- **Backup disponible**: `data/backups/20251015_162604/market_data.csv.gz` (92,936 registros)

Para cargar el backup completo:
```bash
python3 restore_backup.py
```

## 🔍 Test de Conectividad

Ejecuta el siguiente comando para verificar la conectividad:
```bash
python3 test_api_connectivity.py
```

## ⚠️ Nota Importante

El proxy del dashboard todavía requiere un reinicio completo para funcionar correctamente. Mientras tanto, la API está disponible directamente en el puerto 8000 para pruebas.

---
**Última actualización**: 15 de Octubre 2025, 18:17 UTC