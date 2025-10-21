# 📊 USDCOP Trading Dashboard - Documentación Completa de las 13 Vistas Profesionales

**Fecha:** 2025-10-21
**Estado:** ✅ Verificación Completa
**Cobertura:** 13/13 Vistas (100%)

---

## 📋 Índice de Navegación

### 🎯 TRADING (5 vistas)
1. [Dashboard Home](#1-dashboard-home) - Terminal unificado profesional
2. [Professional Terminal](#2-professional-terminal) - Terminal profesional avanzado
3. [Live Trading](#3-live-trading) - Trading en tiempo real
4. [Executive Overview](#4-executive-overview) - Vista ejecutiva
5. [Trading Signals](#5-trading-signals) - Señales de trading con IA

### ⚠️ RISK (2 vistas)
6. [Risk Monitor](#6-risk-monitor) - Monitor de riesgo en tiempo real
7. [Risk Alerts](#7-risk-alerts) - Centro de alertas de riesgo

### 🔄 PIPELINE (5 vistas)
8. [L0 - Raw Data](#8-l0---raw-data) - Datos crudos de mercado
9. [L1 - Features](#9-l1---features) - Estadísticas de features
10. [L3 - Correlations](#10-l3---correlations) - Matriz de correlaciones
11. [L4 - RL Ready](#11-l4---rl-ready) - Datos listos para RL
12. [L5 - Model](#12-l5---model) - Rendimiento de modelos

### ⚙️ SYSTEM (1 vista)
13. [Backtest Results](#13-backtest-results) - Análisis de backtest L6

---

## 🎯 TRADING SECTION

### 1. Dashboard Home

**ID de Vista:** `dashboard-home`
**Ruta:** `/` (página principal)
**Componente:** `/components/views/UnifiedTradingTerminal.tsx`
**Frecuencia de Actualización:** Tiempo real (WebSocket) + 30s (polling)

#### 📊 Valores Numéricos Mostrados

| Métrica | Formato | Fuente de Datos | Ejemplo |
|---------|---------|-----------------|---------|
| **Precio Actual** | `$X,XXX.XX COP` | WebSocket + API latest | `$4,285.50 COP` |
| **Cambio 24h** | `±X.XX%` (color dinámico) | API stats | `+0.45%` ↗️ |
| **Cambio Absoluto 24h** | `±XXX.XX` | Calculado | `+19.25` |
| **Volumen 24h** | `X.XXM` | API stats | `1.2M` |
| **High 24h** | `$X,XXX.XX` | API stats | `$4,295.75` |
| **Low 24h** | `$X,XXX.XX` | API stats | `$4,270.25` |
| **Spread** | `X.XX bps` | bid-ask | `2.50 bps` |
| **Volatilidad** | `X.XX%` | Calculado | `1.25%` |
| **P&L Session** | `±$XXX.XX` | API session-pnl | `+$250.50` |
| **Market Status** | `OPEN/CLOSED` | Horario COT | `OPEN` 🟢 |

#### 📈 Visualizaciones

1. **Gráfico Principal**
   - Tipo: Candlestick (TradingView-style)
   - Timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`
   - Indicadores técnicos:
     - EMA 20 (azul)
     - EMA 50 (naranja)
     - EMA 200 (rojo)
     - Bollinger Bands
     - RSI (panel inferior)
     - MACD (panel inferior)
     - Volume (panel inferior)

2. **Mini Charts (4 paneles)**
   - Spread histórico
   - Volatilidad histórica
   - Volume profile
   - P&L acumulado

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/proxy/trading/latest/USDCOP` | GET | Precio actual | api_server.py:8000 | ✅ |
| `/api/proxy/trading/stats/USDCOP` | GET | Estadísticas 24h | api_server.py:8000 | ✅ |
| `/api/proxy/trading/candlesticks/USDCOP` | GET | Datos OHLCV | api_server.py:8000 | ✅ |
| `/api/analytics/session-pnl?symbol=USDCOP` | GET | P&L sesión | trading_analytics_api.py:8001 | ✅ |
| `ws://localhost:8082/ws` | WebSocket | Precio tiempo real | realtime_data_service.py:8082 | ✅ |

#### ✅ Verificación de Funcionamiento

- ✅ Precio se actualiza en tiempo real
- ✅ Gráfico muestra candlesticks correctamente
- ✅ Indicadores técnicos calculados
- ✅ Colores dinámicos (verde/rojo) según cambio
- ✅ Market status detecta horario COT
- ✅ Fallback a datos históricos si WebSocket falla

---

### 2. Professional Terminal

**ID de Vista:** `professional-terminal`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/ProfessionalTradingTerminal.tsx`
**Frecuencia de Actualización:** 5 segundos (configurable)

#### 📊 Valores Numéricos Mostrados

| Métrica | Formato | Fuente de Datos | Ejemplo |
|---------|---------|-----------------|---------|
| **Precio BID** | `X,XXX.XX` | API latest | `4,285.25` |
| **Precio ASK** | `X,XXX.XX` | API latest | `4,285.75` |
| **Precio MID** | `X,XXX.XX` | Calculado | `4,285.50` |
| **Spread (bps)** | `X.XX` | (ask-bid)/mid * 10000 | `1.17` |
| **Spread (%)** | `X.XX%` | (ask-bid)/mid * 100 | `0.012%` |
| **ATR (14)** | `XX.XX` | Indicador técnico | `25.50` |
| **Volatilidad Intraday** | `X.XX%` | StdDev returns | `0.85%` |
| **Drawdown Actual** | `-X.XX%` | Max price - current | `-2.15%` |
| **Progreso Sesión** | `XX%` | (time elapsed / total) * 100 | `65%` |
| **Tiempo Restante** | `Xh XXm` | Cierre - now | `1h 45m` |

#### 📊 Posiciones del Portafolio

Tabla con columnas:
- **Symbol** | **Quantity** | **Avg Price** | **Current Price** | **P&L** | **P&L %** | **Market Value** | **Weight %**

Ejemplo de datos:
```
USDCOP_SPOT | 2,000,000 | 4,280.50 | 4,285.50 | +$10,000 | +0.12% | $8,571,000 | 85%
COP_BONDS   | 1,000,000 | 1.20     | 1.20     | +$24,000 | +2.00% | $1,200,000  | 12%
OIL_HEDGE   | 100,000   | 3.00     | 3.00     | -$9,000  | -3.00% | $300,000    | 3%
```

#### 📈 Visualizaciones

1. **Gráfico Principal**: Multi-timeframe candlestick
2. **Panel de Spreads**: Línea temporal del spread
3. **Panel de Métricas**: Cards con ATR, Volatilidad, Drawdown
4. **Barra de Progreso**: Sesión de trading (8:00 AM - 12:55 PM COT)

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/trading/positions?symbol=USDCOP` | GET | Posiciones portafolio | api_server.py:8000 | ✅ |
| `/api/proxy/trading/latest/USDCOP` | GET | Precios bid/ask | api_server.py:8000 | ✅ |
| `/api/proxy/trading/candlesticks/USDCOP` | GET | Datos históricos | api_server.py:8000 | ✅ |
| `ws://localhost:8082/ws` | WebSocket | Updates en tiempo real | realtime_data_service.py:8082 | ✅ |

#### ✅ Verificación de Funcionamiento

- ✅ Muestra bid/ask correctamente
- ✅ Calcula spread en bps
- ✅ ATR calculado correctamente
- ✅ Tabla de posiciones poblada
- ✅ Barra de progreso actualiza cada minuto
- ✅ Colores según profit/loss

---

### 3. Live Trading

**ID de Vista:** `live-terminal`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/LiveTradingTerminal.tsx`
**Frecuencia de Actualización:** Tiempo real (WebSocket)

#### 📊 Valores Numéricos Mostrados (Métricas RL)

| Métrica RL | Target/Threshold | Formato | Fuente de Datos | Ejemplo |
|------------|------------------|---------|-----------------|---------|
| **Spread Captured** | < 21.5 bps | `XX.X bps` | API rl-metrics | `18.5 bps` ✅ |
| **Peg Rate** | < 5% | `X.X%` | API rl-metrics | `3.2%` ✅ |
| **Trades per Episode** | 2-10 | `X` | API rl-metrics | `6` ✅ |
| **Action Balance (Buy)** | ~33% | `XX%` | API rl-metrics | `35%` |
| **Action Balance (Sell)** | ~33% | `XX%` | API rl-metrics | `32%` |
| **Action Balance (Hold)** | ~33% | `XX%` | API rl-metrics | `33%` |
| **Episodes Completados** | - | `XXX` | API rl-metrics | `247` |
| **Total Steps** | - | `XXX,XXX` | API rl-metrics | `125,430` |
| **Reward Promedio** | - | `XXX.XX` | API rl-metrics | `1,250.50` |

#### 📊 Indicadores de Estado

Cada métrica tiene indicador visual:
- 🟢 **Verde**: Dentro del target
- 🟡 **Amarillo**: Cerca del límite
- 🔴 **Rojo**: Fuera del target

#### 📈 Visualizaciones

1. **Gráfico de Reward**: Line chart con reward por episodio
2. **Action Distribution**: Pie chart con % de cada acción
3. **Spread Captured Timeline**: Line chart histórico
4. **Peg Rate Timeline**: Line chart con threshold line

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/analytics/rl-metrics?symbol=USDCOP&days=30` | GET | Métricas RL | trading_analytics_api.py:8001 | ✅ |
| `/api/proxy/trading/latest/USDCOP` | GET | Precio actual | api_server.py:8000 | ✅ |
| `ws://localhost:8082/ws` | WebSocket | Updates tiempo real | realtime_data_service.py:8082 | ✅ |

#### ✅ Verificación de Funcionamiento

- ✅ Métricas RL actualizadas cada 60s
- ✅ Indicadores de estado (verde/amarillo/rojo) funcionan
- ✅ Action balance suma 100%
- ✅ Gráficos se renderizan correctamente
- ✅ Threshold lines visibles

---

### 4. Executive Overview

**ID de Vista:** `executive-overview`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/ExecutiveOverview.tsx`
**Frecuencia de Actualización:** 120 segundos (2 minutos)

#### 📊 Valores Numéricos Mostrados (KPIs de Performance)

| KPI | Target | Formato | Fuente de Datos | Ejemplo |
|-----|--------|---------|-----------------|---------|
| **Sortino Ratio** | ≥ 1.3-1.5 | `X.XX` | API performance-kpis | `1.87` ✅ |
| **Calmar Ratio** | ≥ 0.8 | `X.XX` | API performance-kpis | `1.05` ✅ |
| **Sharpe Ratio** | ≥ 1.0 | `X.XX` | API performance-kpis | `1.45` ✅ |
| **Max Drawdown** | ≤ 15% | `-XX.X%` | API performance-kpis | `-8.5%` ✅ |
| **Current Drawdown** | - | `-X.X%` | API performance-kpis | `-2.1%` |
| **Profit Factor** | ≥ 1.5 | `X.XX` | API performance-kpis | `2.34` ✅ |
| **CAGR** | - | `XX.X%` | API performance-kpis | `12.5%` |
| **Volatilidad** | - | `XX.X%` | API performance-kpis | `15.2%` |
| **Benchmark Spread** | - | `±X.XX%` | API performance-kpis | `+3.25%` |

#### 📊 Production Gates (Puertas de Producción)

Tabla con gates:

| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| Sortino > 1.3 | 1.30 | 1.87 | ✅ PASS |
| Calmar > 0.8 | 0.80 | 1.05 | ✅ PASS |
| Max DD < 15% | 15% | 8.5% | ✅ PASS |
| Profit Factor > 1.5 | 1.50 | 2.34 | ✅ PASS |
| Sharpe > 1.0 | 1.00 | 1.45 | ✅ PASS |

**Passing Gates:** `X/Y` (ej: `5/5` ✅)
**Production Ready:** `YES/NO`

#### 📈 Visualizaciones

1. **KPI Cards**: 9 cards grandes con valores y tendencias
2. **Production Gates Table**: Tabla con status pass/fail
3. **Drawdown Chart**: Area chart mostrando drawdown histórico
4. **Performance Trend**: Line chart con métricas en el tiempo

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/analytics/performance-kpis?symbol=USDCOP&days=90` | GET | KPIs de performance | trading_analytics_api.py:8001 | ✅ |
| `/api/analytics/production-gates?symbol=USDCOP&days=90` | GET | Gates de producción | trading_analytics_api.py:8001 | ✅ |

#### ✅ Verificación de Funcionamiento

- ✅ Todos los KPIs muestran valores reales
- ✅ Production gates evalúan correctamente
- ✅ Status pass/fail con colores correctos
- ✅ Gráficos renderizan con datos históricos
- ✅ Auto-refresh cada 2 minutos

---

### 5. Trading Signals

**ID de Vista:** `trading-signals`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/TradingSignals.tsx`
**Frecuencia de Actualización:** 30 segundos

#### 📊 Valores Numéricos Mostrados (Señales)

| Campo | Formato | Fuente de Datos | Ejemplo |
|-------|---------|-----------------|---------|
| **Signal Type** | `BUY/SELL/HOLD` | API trading/signals | `BUY` 🟢 |
| **Confidence** | `XX.X%` | API trading/signals | `87.5%` |
| **Price** | `$X,XXX.XX` | API trading/signals | `$4,285.50` |
| **Stop Loss** | `$X,XXX.XX` | API trading/signals | `$4,270.00` |
| **Take Profit** | `$X,XXX.XX` | API trading/signals | `$4,320.00` |
| **Expected Return** | `±X.XX%` | API trading/signals | `+0.81%` |
| **Risk Score** | `X.X/10` | API trading/signals | `3.2/10` |
| **Time Horizon** | `Xm-Ym` | API trading/signals | `15-30 min` |
| **Model Source** | String | API trading/signals | `PPO_LSTM_v2.1` |

#### 📊 Indicadores Técnicos (en señal)

| Indicador | Formato | Ejemplo |
|-----------|---------|---------|
| **RSI** | `XX.X` | `28.5` (oversold) |
| **MACD** | `XX.X` | `15.2` |
| **MACD Signal** | `XX.X` | `12.1` |
| **MACD Histogram** | `±X.X` | `+3.1` |
| **Bollinger Upper** | `$X,XXX.XX` | `$4,310.00` |
| **Bollinger Middle** | `$X,XXX.XX` | `$4,285.00` |
| **Bollinger Lower** | `$X,XXX.XX` | `$4,260.00` |
| **EMA 20** | `$X,XXX.XX` | `$4,282.50` |
| **EMA 50** | `$X,XXX.XX` | `$4,275.30` |
| **Volume Ratio** | `X.XX` | `1.45` |

#### 📊 Performance Metrics (del sistema de señales)

| Métrica | Formato | Ejemplo |
|---------|---------|---------|
| **Win Rate** | `XX.X%` | `68.5%` |
| **Avg Win** | `$XXX.XX` | `$125.50` |
| **Avg Loss** | `$XXX.XX` | `$75.25` |
| **Profit Factor** | `X.XX` | `2.34` |
| **Sharpe Ratio** | `X.XX` | `1.87` |
| **Total Signals** | `XXX` | `150` |
| **Successful Signals** | `XXX` | `103` |

#### 📊 Reasoning (razones de la señal)

Lista de strings como:
- "RSI oversold (28.5)"
- "MACD bullish crossover"
- "High ML confidence (87.5%)"
- "Support level at 4280"

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/trading/signals?symbol=USDCOP&limit=5` | GET | Señales reales | trading_signals_api.py:8003 | ✅ **NEW** |
| `/api/trading/signals-test?limit=5` | GET | Señales mock (fallback) | trading_signals_api.py:8003 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Señal principal muestra tipo (BUY/SELL/HOLD)
- ✅ Confidence % con barra de progreso
- ✅ Stop loss y take profit calculados
- ✅ Risk score en escala 1-10
- ✅ Reasoning list poblada
- ✅ Indicadores técnicos todos visibles
- ✅ Performance metrics actualizadas
- ✅ Fallback a signals-test funciona

---

## ⚠️ RISK SECTION

### 6. Risk Monitor

**ID de Vista:** `risk-monitor`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/RealTimeRiskMonitor.tsx`
**Frecuencia de Actualización:** 5/10/30 segundos (configurable)

#### 📊 Valores Numéricos Mostrados

| Métrica de Riesgo | Formato | Fuente de Datos | Ejemplo |
|-------------------|---------|-----------------|---------|
| **Portfolio Value** | `$XX,XXX,XXX` | API trading/positions | `$10,071,000` |
| **Total P&L** | `±$XXX,XXX` | Calculado | `+$25,000` |
| **Total P&L %** | `±X.XX%` | Calculado | `+0.25%` |
| **VaR 95% (Portfolio)** | `$XXX,XXX` | API risk-metrics | `$150,000` |
| **VaR 95% (%)** | `X.XX%` | API risk-metrics | `1.49%` |
| **VaR 99%** | `$XXX,XXX` | API risk-metrics | `$225,000` |
| **Expected Shortfall 95%** | `$XXX,XXX` | API risk-metrics | `$187,500` |
| **Leverage** | `X.XX` | API risk-metrics | `1.25` |
| **Gross Exposure** | `$XX,XXX,XXX` | API risk-metrics | `$12,500,000` |
| **Net Exposure** | `$XX,XXX,XXX` | API risk-metrics | `$10,000,000` |
| **Current Drawdown** | `-X.XX%` | API risk-metrics | `-2.15%` |
| **Max Drawdown** | `-XX.X%` | API risk-metrics | `-8.5%` |
| **Portfolio Volatility** | `XX.X%` | API risk-metrics | `15.2%` |
| **Liquidity Score** | `XX/100` | API risk-metrics | `85/100` |
| **Time to Liquidate** | `X.X hours` | API risk-metrics | `2.5 hours` |

#### 📊 Posiciones con Risk Scores

Tabla extendida con:
- **Symbol** | **Market Value** | **Weight** | **P&L** | **VaR Score** | **Leverage Score** | **Liquidity Score** | **Concentration Score**

Ejemplo:
```
USDCOP_SPOT | $8,571,000 | 85% | +$10,000 | 45/100 | 85/100 | 90/100 | 85/100
COP_BONDS   | $1,200,000 | 12% | +$24,000 | 44/100 | 12/100 | 80/100 | 12/100
OIL_HEDGE   | $300,000   | 3%  | -$9,000  | 70/100 | 3/100  | 70/100 | 3/100
```

#### 📊 Market Conditions

| Condición | Formato | Fuente | Ejemplo |
|-----------|---------|--------|---------|
| **VIX** | `XX.XX` | API market-conditions | `18.50` |
| **Market Volatility** | `XX.X%` | API market-conditions | `15.2%` |
| **Spreads Avg** | `XX.X bps` | API market-conditions | `2.5 bps` |
| **Oil Price** | `$XXX.XX` | API market-conditions | `$82.50` |
| **EM Sentiment** | `Positive/Neutral/Negative` | API market-conditions | `Neutral` |

#### 📊 Stress Test Scenarios

| Escenario | Impact | Status |
|-----------|--------|--------|
| **Market Crash -20%** | `-$XXX,XXX` | 🔴 High Risk |
| **Volatility Spike +50%** | `-$XXX,XXX` | 🟡 Medium Risk |
| **Liquidity Crunch** | `-$XXX,XXX` | 🟡 Medium Risk |
| **Best Case +10%** | `+$XXX,XXX` | 🟢 Opportunity |
| **Worst Case -30%** | `-$XXX,XXX` | 🔴 Critical |

#### 📈 Visualizaciones

1. **Risk Heatmap**: Mapa de calor con scores de cada posición
2. **VaR Distribution**: Histogram con distribución de VaR
3. **Drawdown Timeline**: Area chart con drawdown histórico
4. **Exposure Breakdown**: Pie chart con exposición por asset
5. **Volatility Trend**: Line chart con volatilidad en el tiempo

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/trading/positions?symbol=USDCOP` | GET | Posiciones | api_server.py:8000 | ✅ |
| `/api/analytics/risk-metrics?symbol=USDCOP&portfolio_value=10000000&days=30` | GET | Métricas de riesgo | trading_analytics_api.py:8001 | ✅ |
| `/api/analytics/market-conditions?symbol=USDCOP&days=30` | GET | Condiciones mercado | trading_analytics_api.py:8001 | ✅ |

#### ✅ Verificación de Funcionamiento

- ✅ Portfolio value calcula correctamente
- ✅ VaR en $ y % coherentes
- ✅ Risk scores para cada posición
- ✅ Stress tests muestran impactos
- ✅ Heatmap renderiza con colores
- ✅ Auto-refresh configurable (5/10/30s)
- ✅ Market conditions actualizadas

---

### 7. Risk Alerts

**ID de Vista:** `risk-alerts`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/RiskAlertsCenter.tsx`
**Frecuencia de Actualización:** Tiempo real (event-driven)

#### 📊 Valores Numéricos Mostrados

| Estadística de Alertas | Formato | Ejemplo |
|------------------------|---------|---------|
| **Total Alerts** | `XXX` | `45` |
| **Critical Alerts** | `XX` | `3` 🔴 |
| **High Priority** | `XX` | `12` 🟠 |
| **Medium Priority** | `XX` | `18` 🟡 |
| **Low Priority** | `XX` | `12` 🔵 |
| **Active Alerts** | `XX` | `30` |
| **Acknowledged** | `XX` | `15` |
| **Resolved** | `XX` | `0` |

#### 📊 Estructura de cada Alerta

| Campo | Formato | Ejemplo |
|-------|---------|---------|
| **Alert ID** | `alert_XXXXXXX` | `alert_1729500` |
| **Type** | String | `limit_breach` |
| **Severity** | `critical/high/medium/low` | `high` 🟠 |
| **Title** | String | "VaR 95% Limit Exceeded" |
| **Message** | String | "Portfolio VaR has exceeded the 2% limit" |
| **Current Value** | Number | `2.15%` |
| **Threshold** | Number | `2.00%` |
| **Timestamp** | ISO string | `2025-10-21T10:30:00Z` |
| **Status** | `active/acknowledged/resolved` | `active` |
| **Affected Asset** | String | `USDCOP_SPOT` |

#### 📊 Tipos de Alertas

1. **limit_breach**: Límite de riesgo excedido
2. **volatility_spike**: Pico de volatilidad
3. **correlation_anomaly**: Anomalía en correlaciones
4. **drawdown_warning**: Advertencia de drawdown
5. **liquidity_alert**: Alerta de liquidez
6. **position_size**: Tamaño de posición excesivo
7. **spread_widening**: Spreads ampliándose
8. **market_hours**: Fuera de horario de mercado

#### 📊 Configuración de Thresholds

| Threshold | Valor por Defecto | Configurable | Ejemplo |
|-----------|-------------------|--------------|---------|
| **VaR 95% Limit** | `2.0%` | ✅ Sí | `2.0%` |
| **Max Drawdown** | `15.0%` | ✅ Sí | `15.0%` |
| **Position Size Max** | `90.0%` | ✅ Sí | `90.0%` |
| **Volatility Threshold** | `20.0%` | ✅ Sí | `20.0%` |
| **Spread Warning** | `5.0 bps` | ✅ Sí | `5.0 bps` |
| **Leverage Max** | `2.0` | ✅ Sí | `2.0` |

#### 📊 Canales de Notificación

| Canal | Estado | Configuración |
|-------|--------|---------------|
| **In-App Sound** | ✅ Enabled | Volume slider |
| **Browser Notification** | ✅ Enabled | Toggle on/off |
| **Email** | ⚪ Disabled | Email address input |
| **SMS** | ⚪ Disabled | Phone number input |
| **Push Notification** | ⚪ Disabled | Device registration |
| **Webhook** | ⚪ Disabled | URL input |

#### 📈 Visualizaciones

1. **Alert Timeline**: Chart mostrando alertas en el tiempo
2. **Severity Distribution**: Pie chart con distribución por severidad
3. **Alert Type Breakdown**: Bar chart con tipos de alertas
4. **Threshold Monitor**: Gauges mostrando valores vs thresholds

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/alerts/system` | GET | Obtener alertas | Next.js route | ✅ |
| `/api/alerts/acknowledge` | POST | Acknowledge alerta | Next.js route | ✅ |
| `/api/alerts/configure` | POST | Configurar thresholds | Next.js route | ✅ |
| `/api/alerts/notifications` | POST | Configurar notificaciones | Next.js route | ✅ |

#### ✅ Verificación de Funcionamiento

- ✅ Alertas se muestran en tiempo real
- ✅ Filtros por severidad funcionan
- ✅ Búsqueda de alertas operativa
- ✅ Acknowledge workflow completo
- ✅ Thresholds configurables
- ✅ Notificaciones de sonido activas
- ✅ Colores según severidad

---

## 🔄 PIPELINE SECTION

### 8. L0 - Raw Data

**ID de Vista:** `l0-raw-data`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/L0RawDataDashboard.tsx`
**Frecuencia de Actualización:** 60 segundos

#### 📊 Valores Numéricos Mostrados (Estadísticas Globales)

| Estadística | Formato | Fuente de Datos | Ejemplo |
|-------------|---------|-----------------|---------|
| **Total Records** | `XX,XXX` | API l0/statistics | `92,936` |
| **Date Range (Days)** | `XXX days` | Calculado | `294 days` |
| **Earliest Date** | `YYYY-MM-DD` | API l0/statistics | `2024-01-01` |
| **Latest Date** | `YYYY-MM-DD` | API l0/statistics | `2025-10-21` |
| **Symbols Count** | `X` | API l0/statistics | `1` |
| **Min Price** | `$X,XXX.XX` | API l0/statistics | `$3,850.25` |
| **Max Price** | `$X,XXX.XX` | API l0/statistics | `$4,520.75` |
| **Avg Price** | `$X,XXX.XX` | API l0/statistics | `$4,185.50` |
| **Std Dev Price** | `$XXX.XX` | Calculado | `$125.75` |
| **Avg Volume** | `X.XXM` | API l0/statistics | `1.25M` |

#### 📊 Distribución de Fuentes

| Fuente | Records | Porcentaje | Estado |
|--------|---------|------------|--------|
| **PostgreSQL** | `XX,XXX` | `XX%` | ✅ Primary |
| **MinIO** | `X,XXX` | `X%` | ✅ Archive |
| **TwelveData** | `XXX` | `X%` | ✅ Real-time |

#### 📊 Tabla de Datos Crudos (últimos 1000 registros)

Columnas:
- **Timestamp** | **Symbol** | **Close** | **Bid** | **Ask** | **Volume** | **Source**

Ejemplo de fila:
```
2025-10-21 10:30:00 | USDCOP | 4,285.50 | 4,285.25 | 4,285.75 | 1,200,000 | postgres
```

Funcionalidades:
- 🔍 Búsqueda por timestamp
- 📊 Ordenamiento por columna
- 📄 Paginación (10/25/50/100 por página)
- 📥 Export a CSV

#### 📈 Visualizaciones

1. **Price Distribution**: Histogram mostrando distribución de precios
2. **Volume Over Time**: Bar chart con volumen diario
3. **Source Distribution**: Pie chart con % por fuente
4. **Data Availability**: Calendar heatmap mostrando días con datos

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/pipeline/l0/raw-data?limit=1000&offset=0` | GET | Datos crudos | pipeline_data_api.py:8004 | ✅ **NEW** |
| `/api/pipeline/l0/statistics` | GET | Estadísticas L0 | pipeline_data_api.py:8004 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Total records muestra valor real (92K+)
- ✅ Date range calculado correctamente
- ✅ Price stats (min/max/avg) coherentes
- ✅ Tabla poblada con datos reales
- ✅ Paginación funciona
- ✅ Source distribution suma 100%
- ✅ Gráficos renderizan correctamente
- ✅ Auto-refresh cada 60s

---

### 9. L1 - Features

**ID de Vista:** `l1-features`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/L1FeatureStats.tsx`
**Frecuencia de Actualización:** 60 segundos

#### 📊 Valores Numéricos Mostrados (Estadísticas Globales)

| Estadística | Formato | Fuente de Datos | Ejemplo |
|-------------|---------|-----------------|---------|
| **Total Episodes** | `XXX` | API l1/episodes | `247` |
| **Total Data Points** | `XXX,XXX` | Sumatoria | `125,430` |
| **Avg Episode Length** | `XXX` | Calculado | `508` |
| **Quality Score** | `XX.X%` | API l1/quality-report | `95.5%` |
| **Completeness** | `XX.X%` | API l1/quality-report | `98.2%` |
| **Episodes with Gaps** | `XX` | API l1/quality-report | `12` |
| **Avg Session Duration** | `Xh XXm` | Calculado | `4h 55m` |

#### 📊 Tabla de Episodios

Columnas:
- **Episode ID** | **Timestamp** | **Duration** | **Data Points** | **Quality %** | **Completeness %** | **Session Type**

Ejemplo de fila:
```
ep_0_2025-10-21 | 2025-10-21 08:00:00 | 4h 55m | 590 | 97.5% | 99.0% | TRADING
```

Session Types:
- `TRADING`: Sesión de trading normal
- `VALIDATION`: Sesión de validación
- `BACKTEST`: Sesión de backtest

#### 📊 Quality Metrics por Episodio

| Métrica | Threshold | Formato | Ejemplo |
|---------|-----------|---------|---------|
| **Quality Score** | ≥ 90% | `XX.X%` 🟢/🟡/🔴 | `97.5%` 🟢 |
| **Completeness** | ≥ 95% | `XX.X%` 🟢/🟡/🔴 | `99.0%` 🟢 |
| **Data Gaps** | = 0 | `X gaps` | `0 gaps` 🟢 |

Indicadores de color:
- 🟢 Verde: Dentro del threshold
- 🟡 Amarillo: Cerca del threshold
- 🔴 Rojo: Por debajo del threshold

#### 📈 Visualizaciones

1. **Episode Length Distribution**: Histogram con distribución de duración
2. **Quality Trend**: Line chart con quality score en el tiempo
3. **Completeness Trend**: Line chart con completeness en el tiempo
4. **Session Type Distribution**: Pie chart con % por tipo

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/pipeline/l1/episodes?limit=100` | GET | Lista de episodios | pipeline_data_api.py:8004 | ✅ **NEW** |
| `/api/pipeline/l1/quality-report` | GET | Reporte de calidad | pipeline_data_api.py:8004 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Total episodes muestra valor real
- ✅ Quality score calculado correctamente
- ✅ Tabla poblada con episodios
- ✅ Indicadores de color funcionan
- ✅ Session types categorizados
- ✅ Gráficos muestran tendencias
- ✅ Auto-refresh cada 60s

---

### 10. L3 - Correlations

**ID de Vista:** `l3-correlations`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/L3Correlations.tsx`
**Frecuencia de Actualización:** 60 segundos

#### 📊 Valores Numéricos Mostrados (Estadísticas Globales)

| Estadística | Formato | Fuente de Datos | Ejemplo |
|-------------|---------|-----------------|---------|
| **Total Features** | `XX` | API l3/features | `17` |
| **Average IC** | `X.XXX` | Calculado | `0.125` |
| **High IC Features** | `XX` | Filtrado (IC > 0.1) | `8` |
| **Low IC Features** | `XX` | Filtrado (IC < 0.05) | `3` |
| **Samples Used** | `XXX` | API metadata | `1000` |

#### 📊 Tabla de Features con IC (Information Coefficient)

Columnas:
- **Feature Name** | **IC Mean** | **IC Std** | **Rank IC** | **Correlation** | **Mean** | **Std Dev** | **Range**

Ejemplo de filas:
```
price_change      | 0.145 | 0.023 | 0.132 | 0.85  | 0.0012 | 0.0145 | [-0.05, 0.05]
volume_change     | 0.089 | 0.018 | 0.076 | 0.42  | 0.0008 | 0.1250 | [-0.50, 0.50]
spread            | -0.032| 0.012 | -0.028| -0.15 | 2.50   | 0.75   | [1.00, 5.00]
rsi               | 0.112 | 0.020 | 0.105 | 0.65  | 50.0   | 15.2   | [0, 100]
macd              | 0.098 | 0.015 | 0.090 | 0.58  | 0.0    | 5.5    | [-20, 20]
bollinger_position| 0.067 | 0.011 | 0.062 | 0.38  | 0.5    | 0.25   | [0, 1]
ema_20_50_cross   | 0.134 | 0.022 | 0.125 | 0.72  | 0.0    | 0.5    | [-1, 1]
time_of_day       | 0.045 | 0.008 | 0.041 | 0.22  | 12.0   | 2.5    | [0, 24]
day_of_week       | 0.012 | 0.005 | 0.010 | 0.08  | 3.0    | 1.4    | [1, 5]
volatility        | 0.156 | 0.025 | 0.142 | 0.88  | 0.015  | 0.005  | [0.005, 0.030]
```

#### 📊 Matriz de Correlación

Heatmap 17x17 mostrando correlaciones entre features:
- Escala de color: -1.0 (rojo) a +1.0 (verde)
- Diagonal siempre 1.0 (auto-correlación)
- Valores mostrados en cada celda

#### 📊 Ranking de Features por IC

Tabla ordenada por IC Mean descendente:

| Rank | Feature | IC Mean | Status |
|------|---------|---------|--------|
| 1 | volatility | 0.156 | 🟢 Excellent |
| 2 | price_change | 0.145 | 🟢 Excellent |
| 3 | ema_20_50_cross | 0.134 | 🟢 Good |
| 4 | rsi | 0.112 | 🟢 Good |
| ... | ... | ... | ... |

#### 📈 Visualizaciones

1. **Correlation Heatmap**: Matriz de correlación 17x17
2. **IC Distribution**: Histogram con distribución de IC
3. **Top Features**: Bar chart con top 10 features por IC
4. **Feature Importance Timeline**: Line chart con IC en el tiempo

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/pipeline/l3/features?limit=1000` | GET | Features y correlaciones | pipeline_data_api.py:8004 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Total features muestra 17
- ✅ IC calculado para cada feature
- ✅ Heatmap renderiza correctamente
- ✅ Colores según valor de correlación
- ✅ Ranking ordenado por IC
- ✅ Gráficos muestran distribuciones
- ✅ Auto-refresh cada 60s

---

### 11. L4 - RL Ready

**ID de Vista:** `l4-rl-ready`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/L4RLReadyData.tsx`
**Frecuencia de Actualización:** 60 segundos

#### 📊 Valores Numéricos Mostrados (Estadísticas Globales)

| Estadística | Formato | Fuente de Datos | Ejemplo |
|-------------|---------|-----------------|---------|
| **Total Episodes** | `XXX` | API l4/dataset | `247` |
| **Total Timesteps** | `XXX,XXX` | API l4/dataset | `125,430` |
| **Feature Count** | `XX` | API l4/dataset | `17` |
| **Action Space Size** | `X` | API l4/dataset | `3` (Buy/Sell/Hold) |
| **Observation Space Dim** | `XX` | Feature count | `17` |
| **Avg Episode Length** | `XXX` | Total timesteps / episodes | `508` |

#### 📊 Split Distribution

Tabla con distribución Train/Validation/Test:

| Split | Episodes | Timesteps | Avg Length | Reward Mean | Reward Std | Percentage |
|-------|----------|-----------|------------|-------------|------------|------------|
| **TRAIN** | 173 | 87,801 | 507 | 1,250.5 | 325.2 | 70% |
| **VALIDATION** | 50 | 25,086 | 502 | 1,180.3 | 310.5 | 20% |
| **TEST** | 24 | 12,543 | 523 | 1,220.8 | 318.7 | 10% |
| **TOTAL** | 247 | 125,430 | 508 | - | - | 100% |

#### 📊 Feature Space Descripción

Lista de 17 features con sus especificaciones:

| Feature ID | Name | Type | Range | Normalization |
|------------|------|------|-------|---------------|
| 0 | price | float | [0, ∞) | MinMax |
| 1 | volume | float | [0, ∞) | Log + MinMax |
| 2 | bid | float | [0, ∞) | MinMax |
| 3 | ask | float | [0, ∞) | MinMax |
| 4 | spread | float | [0, ∞) | MinMax |
| 5 | rsi | float | [0, 100] | None |
| 6 | macd | float | (-∞, ∞) | Standardize |
| 7 | bollinger_upper | float | [0, ∞) | MinMax |
| 8 | bollinger_lower | float | [0, ∞) | MinMax |
| 9 | ema_20 | float | [0, ∞) | MinMax |
| 10 | ema_50 | float | [0, ∞) | MinMax |
| ... | ... | ... | ... | ... |

#### 📊 Action Space Descripción

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | HOLD | Mantener posición actual |
| 1 | BUY | Comprar USDCOP |
| 2 | SELL | Vender USDCOP |

#### 📊 Dataset Quality Metrics

| Métrica | Valor | Status |
|---------|-------|--------|
| **Missing Values** | `0.0%` | ✅ Perfect |
| **Outliers Detected** | `0.5%` | ✅ Acceptable |
| **Data Consistency** | `99.8%` | ✅ Excellent |
| **Feature Correlation** | `<0.9` | ✅ Good |
| **Class Balance** | `33/33/33%` | ✅ Balanced |

#### 📈 Visualizaciones

1. **Split Distribution**: Pie chart con % de cada split
2. **Episode Length Distribution**: Histogram por split
3. **Reward Distribution**: Box plot por split
4. **Action Distribution**: Stacked bar chart con % de acciones
5. **Feature Correlation Matrix**: Heatmap de correlaciones

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/pipeline/l4/dataset?split=train` | GET | Dataset TRAIN | pipeline_data_api.py:8004 | ✅ **NEW** |
| `/api/pipeline/l4/dataset?split=test` | GET | Dataset TEST | pipeline_data_api.py:8004 | ✅ **NEW** |
| `/api/pipeline/l4/dataset?split=val` | GET | Dataset VAL | pipeline_data_api.py:8004 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Total episodes suma correctamente
- ✅ Split distribution suma 100%
- ✅ Feature count correcto (17)
- ✅ Action space size correcto (3)
- ✅ Reward statistics calculadas
- ✅ Gráficos renderizan por split
- ✅ Quality metrics visibles
- ✅ Auto-refresh cada 60s

---

### 12. L5 - Model

**ID de Vista:** `l5-model`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/L5ModelDashboard.tsx`
**Frecuencia de Actualización:** 60 segundos

#### 📊 Valores Numéricos Mostrados (Estadísticas Globales)

| Estadística | Formato | Fuente de Datos | Ejemplo |
|-------------|---------|-----------------|---------|
| **Total Models** | `X` | API l5/models | `3` |
| **Average Size** | `XXX.X MB` | Calculado | `125.5 MB` |
| **ONNX Models** | `X` | Filtrado | `3` |
| **Latest Version** | `X.X` | Max version | `3.0` |
| **Total Training Episodes** | `X,XXX` | Sumatoria | `4,500` |
| **Avg Validation Reward** | `XXX.X` | Promedio | `1,217.2` |

#### 📊 Tabla de Modelos

Columnas:
- **Model ID** | **Name** | **Version** | **Algorithm** | **Architecture** | **Format** | **Size (MB)** | **Created** | **Training Episodes** | **Validation Reward** | **Status** | **Checkpoint Path**

Ejemplo de filas:
```
ppo_lstm_v2_1 | PPO with LSTM | 2.1 | PPO | LSTM | ONNX | 115.2 MB | 2025-10-19 15:30 | 1,500 | 1,250.5 | active | models/ppo_lstm_v2_1.pkl
a2c_gru_v1_5 | A2C with GRU | 1.5 | A2C | GRU | ONNX | 95.8 MB | 2025-10-14 10:15 | 1,200 | 1,150.2 | inactive | models/a2c_gru_v1_5.pkl
sac_transformer_v3_0 | SAC Transformer | 3.0 | SAC | Transformer | ONNX | 165.5 MB | 2025-10-20 18:45 | 1,800 | 1,320.8 | testing | models/sac_transformer_v3_0.pkl
```

#### 📊 Detalles del Modelo Activo (Latest)

Sección destacada con borde verde para el modelo más reciente:

| Campo | Valor |
|-------|-------|
| **Model ID** | `sac_transformer_v3_0` |
| **Full Name** | `SAC with Transformer` |
| **Version** | `3.0` |
| **Algorithm** | `SAC (Soft Actor-Critic)` |
| **Architecture** | `Transformer` |
| **Format** | `ONNX` |
| **File Size** | `165.5 MB` |
| **Created Date** | `2025-10-20 18:45:00` |
| **Training Episodes** | `1,800` |
| **Validation Reward** | `1,320.8` |
| **Status** | `testing` 🟡 |
| **Checkpoint Path** | `/models/sac_transformer_v3_0.pkl` |

#### 📊 Métricas de Entrenamiento (por modelo)

| Métrica | Formato | Ejemplo |
|---------|---------|---------|
| **Training Episodes** | `X,XXX` | `1,800` |
| **Total Steps** | `XXX,XXX` | `250,000` |
| **Avg Reward** | `XXX.X` | `1,250.5` |
| **Best Reward** | `XXX.X` | `1,580.2` |
| **Final Reward** | `XXX.X` | `1,320.8` |
| **Training Time** | `XXh XXm` | `12h 35m` |
| **Convergence** | `Episode XXX` | `Episode 1,200` |

#### 📊 Status de Modelos

| Status | Count | Descripción |
|--------|-------|-------------|
| **active** 🟢 | 1 | Modelo en uso productivo |
| **testing** 🟡 | 1 | Modelo en fase de prueba |
| **inactive** 🔴 | 1 | Modelo archivado |
| **training** 🔵 | 0 | Modelo en entrenamiento |

#### 📈 Visualizaciones

1. **Model Comparison**: Bar chart comparando validation rewards
2. **Size Distribution**: Pie chart con tamaño por modelo
3. **Training Timeline**: Timeline mostrando fechas de creación
4. **Algorithm Distribution**: Pie chart con % por algoritmo
5. **Reward Trend**: Line chart con rewards por versión

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/pipeline/l5/models` | GET | Lista de modelos | pipeline_data_api.py:8004 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Total models muestra valor correcto
- ✅ Latest model destacado visualmente
- ✅ Tabla poblada con todos los modelos
- ✅ Métricas de entrenamiento visibles
- ✅ Status icons correctos
- ✅ Gráficos comparativos renderizan
- ✅ File sizes calculados
- ✅ Auto-refresh cada 60s

---

## ⚙️ SYSTEM SECTION

### 13. Backtest Results

**ID de Vista:** `backtest-results`
**Ruta:** `/` con navegación sidebar
**Componente:** `/components/views/L6BacktestResults.tsx`
**Frecuencia de Actualización:** 60 segundos

#### 📊 Valores Numéricos Mostrados (Performance Metrics)

**8 Cards Principales:**

| Card | Métrica | Formato | Fuente de Datos | Ejemplo | Target |
|------|---------|---------|-----------------|---------|--------|
| 1 | **Sharpe Ratio** | `X.XX` | API l6/backtest-results | `1.87` | ≥ 1.0 ✅ |
| 2 | **Sortino Ratio** | `X.XX` | API l6/backtest-results | `2.15` | ≥ 1.3 ✅ |
| 3 | **Calmar Ratio** | `X.XX` | API l6/backtest-results | `1.05` | ≥ 0.8 ✅ |
| 4 | **Max Drawdown** | `-XX.X%` | API l6/backtest-results | `-8.5%` | ≤ -15% ✅ |
| 5 | **Total Return** | `±XX.X%` | API l6/backtest-results | `+12.5%` | > 0% ✅ |
| 6 | **Volatility** | `XX.X%` | API l6/backtest-results | `15.2%` | - |
| 7 | **Profit Factor** | `X.XX` | API l6/backtest-results | `2.34` | ≥ 1.5 ✅ |
| 8 | **Win Rate** | `XX.X%` | API l6/backtest-results | `68.5%` | ≥ 55% ✅ |

#### 📊 Trade Statistics

| Métrica | Formato | Ejemplo |
|---------|---------|---------|
| **Total Trades** | `XXX` | `247` |
| **Winning Trades** | `XXX` | `169` |
| **Losing Trades** | `XXX` | `78` |
| **Win Rate** | `XX.X%` | `68.5%` |
| **Avg Trade P&L** | `±$XXX.XX` | `+$145.30` |
| **Avg Win** | `$XXX.XX` | `$325.50` |
| **Avg Loss** | `$XXX.XX` | `-$185.25` |
| **Largest Win** | `$XXX.XX` | `$1,250.00` |
| **Largest Loss** | `$XXX.XX` | `-$750.50` |
| **Avg Trade Duration** | `XXm` | `45m` |

#### 📊 Risk Metrics

| Métrica | Formato | Ejemplo |
|---------|---------|---------|
| **VaR 99%** | `XXX bps` | `215 bps` |
| **Expected Shortfall (CVaR)** | `XXX bps` | `287 bps` |
| **Max Drawdown** | `-XX.X%` | `-8.5%` |
| **Avg Drawdown** | `-X.X%` | `-2.3%` |
| **Drawdown Duration** | `X.X days` | `5.2 days` |
| **Recovery Time** | `X.X days` | `3.5 days` |
| **Downside Deviation** | `XX.X%` | `9.8%` |

#### 📊 Execution & Capacity

| Métrica | Formato | Ejemplo |
|---------|---------|---------|
| **Beta** | `X.XX` | `0.85` |
| **Alpha** | `X.XX%` | `3.25%` |
| **Cost to Alpha Ratio** | `X.X%` | `2.5%` |
| **Avg Slippage** | `X.X bps` | `1.2 bps` |
| **Avg Commission** | `$X.XX` | `$0.50` |

#### 📊 Top Bar KPIs (Hedge Fund Style)

Barra superior con 6 métricas principales:

| Métrica | Valor | Color |
|---------|-------|-------|
| **CAGR** | `12.5%` | 🟢 Verde |
| **Sharpe** | `1.87` | 🟢 Verde |
| **Sortino** | `2.15` | 🟢 Verde |
| **Calmar** | `1.05` | 🟢 Verde |
| **Max DD** | `-8.5%` | 🟢 Verde |
| **Vol** | `15.2%` | 🟡 Amarillo |

#### 📊 Daily Returns Table (últimos 30 días)

Columnas:
- **Date** | **Return %** | **Cumulative Return %** | **Portfolio Value** | **Drawdown %** | **Volume**

Ejemplo de fila:
```
2025-10-21 | +0.12% | +12.50% | $112,500 | -0.50% | 1,200,000
```

#### 📊 Trade Ledger (últimos 50 trades)

Columnas:
- **Trade ID** | **Timestamp** | **Symbol** | **Side** | **Quantity** | **Price** | **P&L** | **Commission** | **Duration**

Ejemplo de fila:
```
trade_247 | 2025-10-21 10:30 | USDCOP | BUY | 1,000 | 4,285.50 | +$250.50 | $0.50 | 45m
```

#### 📈 Visualizaciones

1. **Equity Curve**: Line chart con valor del portafolio en el tiempo
2. **Drawdown Chart**: Area chart con drawdown histórico
3. **Daily Returns Distribution**: Histogram de returns
4. **P&L by Trade**: Waterfall chart de P&L acumulado
5. **Win/Loss Distribution**: Box plot de wins vs losses
6. **Rolling Sharpe**: Line chart con Sharpe rolling 30d

#### 🔌 Endpoints Utilizados

| Endpoint | Método | Propósito | Backend | Estado |
|----------|--------|-----------|---------|--------|
| `/api/pipeline/l6/backtest-results?split=test` | GET | Resultados backtest | pipeline_data_api.py:8004 | ✅ **NEW** |
| `/api/backtest/results` | GET | Resultados completos | backtest_api.py:8006 | ✅ **NEW** |

#### ✅ Verificación de Funcionamiento

- ✅ Top bar KPIs con valores reales
- ✅ 8 metric cards pobladas
- ✅ Trade statistics calculadas
- ✅ Risk metrics visibles
- ✅ Daily returns table con 30 días
- ✅ Trade ledger con 50 trades
- ✅ Equity curve renderiza
- ✅ Drawdown chart muestra historia
- ✅ Colores según performance
- ✅ Auto-refresh cada 60s

---

## 📊 RESUMEN FINAL DE VERIFICACIÓN

### ✅ Estado de Implementación

| Vista | Endpoints | Valores | Gráficos | Funcionalidad | Estado |
|-------|-----------|---------|----------|---------------|--------|
| 1. Dashboard Home | ✅ 5/5 | ✅ 10+ | ✅ 5 | ✅ 100% | ✅ |
| 2. Professional Terminal | ✅ 4/4 | ✅ 15+ | ✅ 4 | ✅ 100% | ✅ |
| 3. Live Trading | ✅ 3/3 | ✅ 12+ | ✅ 4 | ✅ 100% | ✅ |
| 4. Executive Overview | ✅ 2/2 | ✅ 14+ | ✅ 4 | ✅ 100% | ✅ |
| 5. Trading Signals | ✅ 2/2 | ✅ 20+ | ✅ 2 | ✅ 100% | ✅ |
| 6. Risk Monitor | ✅ 3/3 | ✅ 25+ | ✅ 5 | ✅ 100% | ✅ |
| 7. Risk Alerts | ✅ 4/4 | ✅ 15+ | ✅ 4 | ✅ 100% | ✅ |
| 8. L0 Raw Data | ✅ 2/2 | ✅ 12+ | ✅ 4 | ✅ 100% | ✅ |
| 9. L1 Features | ✅ 2/2 | ✅ 10+ | ✅ 4 | ✅ 100% | ✅ |
| 10. L3 Correlations | ✅ 1/1 | ✅ 20+ | ✅ 4 | ✅ 100% | ✅ |
| 11. L4 RL Ready | ✅ 3/3 | ✅ 18+ | ✅ 5 | ✅ 100% | ✅ |
| 12. L5 Model | ✅ 1/1 | ✅ 15+ | ✅ 5 | ✅ 100% | ✅ |
| 13. Backtest Results | ✅ 2/2 | ✅ 30+ | ✅ 6 | ✅ 100% | ✅ |
| **TOTAL** | **✅ 34/34** | **✅ 200+** | **✅ 54** | **✅ 100%** | **✅** |

### 📊 Totales por Categoría

| Categoría | Valores Numéricos | Gráficos | Endpoints |
|-----------|-------------------|----------|-----------|
| **Trading** | 77+ valores | 19 gráficos | 16 endpoints |
| **Risk** | 40+ valores | 9 gráficos | 7 endpoints |
| **Pipeline** | 85+ valores | 22 gráficos | 10 endpoints |
| **System** | 30+ valores | 6 gráficos | 2 endpoints |
| **TOTAL** | **200+ valores** | **54 gráficos** | **34 endpoints** |

### ✅ Verificación Final

```
┌─────────────────────────────────────────────┐
│  ✅ 13/13 VISTAS DOCUMENTADAS               │
│  ✅ 34/34 ENDPOINTS VERIFICADOS             │
│  ✅ 200+ VALORES NUMÉRICOS IDENTIFICADOS    │
│  ✅ 54 GRÁFICOS Y VISUALIZACIONES           │
│  ✅ 100% FUNCIONALIDAD COMPLETA             │
│  ✅ TODOS LOS BACKENDS IMPLEMENTADOS        │
└─────────────────────────────────────────────┘
```

---

**Documento Generado:** 2025-10-21
**Estado:** ✅ COMPLETO - Verificación 100%
**Cobertura:** 13/13 Vistas Profesionales
**Endpoints:** 34/34 Implementados y Funcionando
