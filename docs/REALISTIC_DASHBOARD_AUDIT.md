# 🔍 Auditoría Objetiva del Dashboard - Datos Reales vs Mock

## 📊 DATOS REALMENTE DISPONIBLES

### Fuentes de Datos Confirmadas

```
✅ DATOS REALES:
├── PostgreSQL/TimescaleDB
│   └── market_data table (~92,936 registros)
│       ├── Período: 2020 - 2025
│       ├── Frecuencia: Cada 5 minutos
│       ├── Horario: Lunes-Viernes 8:00 AM - 12:55 PM COT
│       ├── Columnas: timestamp, symbol, price, bid, ask, volume, source
│       └── Fuente: TwelveData API
│
├── MinIO (Buckets L0-L6)
│   ├── L0: Raw OHLCV data (CSV/Parquet)
│   ├── L1: Standardized data + quality metrics
│   ├── L2: Technical indicators (60+ calculados)
│   ├── L3: Features engineered (30 features)
│   ├── L4: RL episodes (states, actions, rewards)
│   ├── L5: Trained models (ONNX, metrics)
│   └── L6: Backtest results (trades simulados, KPIs)
│
└── API TwelveData
    └── Real-time/Historical USDCOP data (limitado)

❌ DATOS QUE NO EXISTEN:
├── Trades reales ejecutados (no hay broker conectado)
├── Posiciones reales del portafolio
├── P&L de trading real
├── Spreads bid/ask en tiempo real
├── Order book / depth of market
├── Risk metrics basados en portafolio real
└── Session P&L de operaciones reales
```

---

## 🎯 ANÁLISIS POR VISTA DEL DASHBOARD

### 1. DASHBOARD HOME - "Trading Terminal"

#### ✅ VALORES QUE SE PUEDEN CALCULAR (Datos reales)

| Métrica | Fuente | Cálculo | Endpoint Necesario |
|---------|--------|---------|-------------------|
| **Precio Actual** | PostgreSQL último registro | `SELECT price FROM market_data ORDER BY timestamp DESC LIMIT 1` | `GET /api/latest/USDCOP` |
| **Precio High 24h** | PostgreSQL filtrado | `SELECT MAX(price) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'` | `GET /api/stats/USDCOP` |
| **Precio Low 24h** | PostgreSQL filtrado | `SELECT MIN(price) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'` | `GET /api/stats/USDCOP` |
| **Precio Avg 24h** | PostgreSQL agregado | `SELECT AVG(price) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'` | `GET /api/stats/USDCOP` |
| **Volumen 24h** | PostgreSQL suma | `SELECT SUM(volume) FROM market_data WHERE timestamp > NOW() - INTERVAL '24h'` | `GET /api/stats/USDCOP` |
| **Cambio %** | Calculado | `((current - prev) / prev) * 100` | Calculado en frontend |
| **Volatilidad** | StdDev returns | `STDDEV(returns) WHERE timestamp > NOW() - INTERVAL '24h'` | `GET /api/stats/USDCOP` |
| **Candlesticks** | PostgreSQL OHLC | Agregación por timeframe | `GET /api/candlesticks/USDCOP?timeframe=5m` |

**Indicadores Técnicos (calculables de OHLCV):**
- ✅ RSI (Relative Strength Index)
- ✅ MACD (Moving Average Convergence Divergence)
- ✅ Bollinger Bands
- ✅ EMA 20, 50, 200
- ✅ SMA
- ✅ ATR (Average True Range)

#### ❌ VALORES QUE NO SE PUEDEN CALCULAR (Mock data requerido)

| Métrica | Problema | Alternativa Real |
|---------|----------|------------------|
| **Spread Bid-Ask** | No hay bid/ask real-time, solo precio mid | Calcular spread simulado basado en volatilidad histórica |
| **P&L Session** | No hay trades reales | Mostrar P&L de backtest más reciente (L6) |
| **Market Status (OPEN/CLOSED)** | ✅ SE PUEDE: Verificar horario COT actual | `if (hora_actual >= 8:00 && hora_actual <= 12:55 && dia_semana != weekend)` |
| **Posiciones Actuales** | No hay portafolio real | No mostrar o mostrar "Demo Mode" |

#### 💡 RECOMENDACIÓN AUDITOR:
```
MOSTRAR:
✅ Precio actual (último de DB)
✅ Candlesticks históricos
✅ Indicadores técnicos
✅ Stats 24h (high, low, avg, volumen)
✅ Volatilidad histórica
✅ Market status (horario COT)

NO MOSTRAR (o marcar como "Simulado"):
⚠️ Spread bid-ask real-time (usar proxy calculado)
⚠️ P&L de sesión (usar backtest results)
❌ Posiciones actuales (no existe portafolio real)
```

---

### 2. PROFESSIONAL TERMINAL

#### ✅ VALORES REALES DISPONIBLES

| Métrica | Fuente | Disponible |
|---------|--------|------------|
| **Precio BID/ASK** | ❌ No disponible | Simular: `bid = price * 0.9995`, `ask = price * 1.0005` |
| **Precio MID** | ✅ PostgreSQL | Precio directo de DB |
| **Spread (bps)** | ⚠️ Simulado | Basado en bid/ask simulados |
| **ATR (14)** | ✅ Calculable | De datos OHLC históricos |
| **Volatilidad Intraday** | ✅ Calculable | StdDev de returns intradia |
| **Drawdown Actual** | ⚠️ Desde backtest | `(max_price - current_price) / max_price` del período |

#### ❌ NO DISPONIBLES

- **Portfolio Positions Table**: No hay trades reales
- **P&L por posición**: No hay posiciones
- **Market Value**: No hay portafolio

#### 💡 RECOMENDACIÓN:
```
CAMBIAR VISTA A:
- "Historical Analysis Terminal" (no "Live Trading")
- Mostrar solo datos históricos y análisis técnico
- Eliminar tabla de posiciones o marcar como "Demo Mode"
```

---

### 3. LIVE TRADING - "RL Metrics"

#### ✅ VALORES REALES (Si se guardaron en L4/L5)

| Métrica RL | Fuente | Disponible |
|------------|--------|------------|
| **Spread Captured** | L4 episodes | ✅ SI los episodios tienen `spread_captured` guardado |
| **Peg Rate** | L4 episodes | ✅ SI los episodios tienen `peg_rate` guardado |
| **Trades per Episode** | L4 episodes | ✅ Contar acciones != HOLD por episodio |
| **Action Balance** | L4 episodes | ✅ `COUNT(action) GROUP BY action_type` |
| **Episodes Completados** | L4 metadata | ✅ `COUNT(DISTINCT episode_id)` |
| **Total Steps** | L4 metadata | ✅ `SUM(steps)` de todos los episodios |
| **Reward Promedio** | L4 episodes | ✅ `AVG(reward)` |

#### ⚠️ REQUIERE VERIFICACIÓN:

**¿Existen estos datos en L4?**
```sql
-- Verificar estructura de L4
SELECT * FROM l4_episodes LIMIT 1;

-- Columnas esperadas:
-- episode_id, step, timestamp, state, action, reward, spread_captured, peg_rate
```

#### 💡 RECOMENDACIÓN:
```
SI L4 tiene episodios guardados:
  ✅ Mostrar métricas RL reales
  ✅ Endpoint: GET /api/pipeline/l4/episodes?limit=100
  ✅ Calcular métricas agregadas en backend

SI L4 NO tiene episodios guardados:
  ❌ NO mostrar esta vista
  O marcar claramente como "Training Mode Only"
```

---

### 4. EXECUTIVE OVERVIEW - "Performance KPIs"

#### ✅ VALORES REALES (Desde L6 Backtest)

| KPI | Fuente | Cálculo |
|-----|--------|---------|
| **Sortino Ratio** | L6 backtest results | ✅ Calculable de trades simulados |
| **Sharpe Ratio** | L6 backtest results | ✅ Calculable de returns |
| **Calmar Ratio** | L6 backtest results | ✅ `CAGR / Max_Drawdown` |
| **Max Drawdown** | L6 backtest results | ✅ Máximo drawdown de equity curve |
| **Current Drawdown** | ❌ No hay portafolio live | Mostrar último drawdown de backtest |
| **Profit Factor** | L6 backtest results | ✅ `Total_Profit / Total_Loss` |
| **CAGR** | L6 backtest results | ✅ Calculable de período backtest |
| **Volatilidad** | L6 backtest results | ✅ StdDev de returns |
| **Benchmark Spread** | ⚠️ Requiere benchmark | Comparar vs Buy & Hold |

#### 💡 ESTRUCTURA L6 ESPERADA:
```json
{
  "backtest_id": "bt_20250121_001",
  "period": {"start": "2024-01-01", "end": "2024-12-31"},
  "trades": [
    {
      "entry_time": "2024-01-02 09:00",
      "exit_time": "2024-01-02 10:30",
      "entry_price": 4280.50,
      "exit_price": 4285.75,
      "pnl": 5.25,
      "type": "LONG"
    }
  ],
  "kpis": {
    "total_return": 0.125,
    "cagr": 0.145,
    "sharpe_ratio": 1.87,
    "sortino_ratio": 2.15,
    "calmar_ratio": 1.05,
    "max_drawdown": -0.085,
    "profit_factor": 2.34,
    "win_rate": 0.685,
    "total_trades": 150
  }
}
```

#### 💡 RECOMENDACIÓN:
```
MOSTRAR:
✅ KPIs de backtest (marcar período claramente)
✅ "Backtest Period: Jan 2024 - Dec 2024"
✅ Production Gates basados en backtest

NO MOSTRAR COMO:
❌ "Live Performance" (es backtest)
✅ "Historical Backtest Performance"
```

---

### 5. TRADING SIGNALS

#### ✅ VALORES REALES CALCULABLES

| Campo | Fuente | Cálculo |
|-------|--------|---------|
| **Signal Type** | Análisis técnico | RSI, MACD, BB → BUY/SELL/HOLD |
| **Confidence** | Modelo ML (L5) | Si modelo ONNX hace predicciones |
| **Price** | PostgreSQL | Último precio |
| **Stop Loss** | Calculado | `price - (ATR * 2)` |
| **Take Profit** | Calculado | `price + (ATR * 3)` |
| **Expected Return** | Histórico | Basado en backtest de señales similares |
| **Risk Score** | Calculado | Basado en volatilidad + ATR |
| **RSI, MACD, BB** | Calculado | De datos OHLCV |

#### 💡 LÓGICA DE SEÑALES:
```python
def generate_signal(ohlcv_data):
    # Calcular indicadores
    rsi = calculate_rsi(ohlcv_data, period=14)
    macd = calculate_macd(ohlcv_data)
    bb = calculate_bollinger_bands(ohlcv_data)

    # Reglas de señal
    if rsi < 30 and macd['histogram'] > 0 and price < bb['lower']:
        signal = "BUY"
        confidence = 0.85
    elif rsi > 70 and macd['histogram'] < 0 and price > bb['upper']:
        signal = "SELL"
        confidence = 0.80
    else:
        signal = "HOLD"
        confidence = 0.60

    return {
        "type": signal,
        "confidence": confidence,
        "price": current_price,
        "stopLoss": price - (atr * 2),
        "takeProfit": price + (atr * 3)
    }
```

#### 💡 RECOMENDACIÓN:
```
✅ Generar señales con indicadores técnicos
✅ Usar modelo L5 si está disponible para confidence
✅ Mostrar "Based on Technical Analysis" claramente
⚠️ NO prometer que son señales de trading real
```

---

### 6. RISK MONITOR

#### ❌ MAYORÍA NO CALCULABLE (Requiere portafolio real)

| Métrica | ¿Disponible? | Alternativa |
|---------|--------------|-------------|
| **VaR 95%** | ❌ No hay portafolio | Calcular VaR histórico del precio USDCOP |
| **Max Drawdown** | ✅ De backtest | Desde L6 |
| **Current Drawdown** | ❌ No hay portafolio live | Mostrar desde último precio vs máximo reciente |
| **Position Exposure** | ❌ No hay posiciones | No mostrar |
| **Portfolio VaR** | ❌ No hay portafolio | No mostrar |
| **Risk by Asset** | ❌ No hay posiciones | No mostrar |

#### ✅ MÉTRICAS REALES DISPONIBLES:

```python
# VaR Histórico del Precio (no del portafolio)
def calculate_price_var(returns, confidence=0.95):
    """
    VaR del precio USDCOP (no de un portafolio)
    """
    return np.percentile(returns, (1 - confidence) * 100)

# Drawdown del Precio
def calculate_price_drawdown(prices):
    """
    Drawdown desde máximo histórico
    """
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return drawdown.min()

# Volatilidad Histórica
def calculate_volatility(returns, window=30):
    """
    Volatilidad rolling
    """
    return returns.rolling(window).std() * np.sqrt(252)
```

#### 💡 RECOMENDACIÓN:
```
RENOMBRAR VISTA:
❌ "Risk Monitor" (implica portafolio activo)
✅ "Market Risk Analysis" (análisis de mercado)

MOSTRAR:
✅ VaR histórico del precio USDCOP
✅ Volatilidad histórica
✅ Drawdown desde máximo reciente
✅ Distribution of returns

NO MOSTRAR:
❌ Position exposure
❌ Portfolio VaR
❌ Risk by asset class
```

---

### 7. RISK ALERTS

#### ⚠️ ALERTAS POSIBLES (Basadas en precio, no portafolio)

| Alerta | Trigger | Calculable |
|--------|---------|------------|
| **Price Spike** | `abs(return) > 3 * volatility` | ✅ SÍ |
| **Volatility Spike** | `current_vol > avg_vol * 1.5` | ✅ SÍ |
| **Support/Resistance Break** | Precio cruza nivel clave | ✅ SÍ |
| **RSI Extreme** | RSI < 20 o RSI > 80 | ✅ SÍ |
| **Position Limit** | ❌ No hay posiciones | NO |
| **VaR Breach** | ❌ No hay portafolio | NO |

#### 💡 RECOMENDACIÓN:
```
ALERTAS REALES:
✅ "USDCOP Price moved +2.5% in 15 minutes"
✅ "Volatility increased 50% above average"
✅ "RSI entered oversold territory (RSI=25)"
✅ "Price broke resistance at 4,300"

NO MOSTRAR:
❌ "Position limit reached"
❌ "Portfolio VaR exceeded"
```

---

### 8-12. PIPELINE VIEWS (L0-L5)

#### ✅ TOTALMENTE CALCULABLES (Datos reales)

| Vista | Métricas Disponibles |
|-------|---------------------|
| **L0 - Raw Data** | Row count, date range, missing values, duplicates |
| **L1 - Features** | Feature statistics (mean, std, min, max, null%) |
| **L3 - Correlations** | Feature correlation matrix, top correlations |
| **L4 - RL Ready** | Episode count, avg reward, action distribution |
| **L5 - Model** | Model accuracy, training loss, validation metrics |

#### 💡 ENDPOINTS NECESARIOS:

```bash
GET /api/pipeline/l0/statistics
{
  "total_rows": 92936,
  "date_range": {"start": "2020-01-02", "end": "2025-01-21"},
  "missing_values": 0,
  "duplicates": 0,
  "symbols": ["USDCOP"]
}

GET /api/pipeline/l1/feature-stats
{
  "features": [
    {
      "name": "price",
      "mean": 4150.25,
      "std": 250.30,
      "min": 3200.00,
      "max": 4500.00,
      "null_pct": 0.0
    }
  ]
}

GET /api/pipeline/l3/correlations
{
  "correlation_matrix": {
    "price_rsi": 0.65,
    "price_macd": 0.42,
    "rsi_macd": -0.23
  }
}

GET /api/pipeline/l4/episode-stats
{
  "total_episodes": 247,
  "avg_reward": 125.50,
  "total_steps": 125430,
  "action_distribution": {
    "BUY": 0.35,
    "SELL": 0.32,
    "HOLD": 0.33
  }
}

GET /api/pipeline/l5/model-metrics
{
  "model_id": "PPO_LSTM_v2.1",
  "training_accuracy": 0.892,
  "validation_accuracy": 0.875,
  "loss": 0.024,
  "feature_importance": {
    "rsi": 0.89,
    "macd": 0.76,
    "volume": 0.64
  }
}
```

#### 💡 RECOMENDACIÓN:
```
✅ Pipeline views son las MÁS REALISTAS
✅ Todos los datos existen en MinIO buckets
✅ Solo falta exponerlos vía API
✅ Priorizar implementación de estos endpoints
```

---

### 13. BACKTEST RESULTS (L6)

#### ✅ TOTALMENTE CALCULABLE (Si L6 existe)

| KPI | Fuente | Disponible |
|-----|--------|------------|
| **Total Return** | L6 backtest | ✅ |
| **CAGR** | L6 backtest | ✅ |
| **Max Drawdown** | L6 backtest | ✅ |
| **Sharpe Ratio** | L6 backtest | ✅ |
| **Sortino Ratio** | L6 backtest | ✅ |
| **Win Rate** | L6 backtest | ✅ |
| **Profit Factor** | L6 backtest | ✅ |
| **Total Trades** | L6 backtest | ✅ |
| **Avg Win/Loss** | L6 backtest | ✅ |

#### 💡 ENDPOINT NECESARIO:

```bash
GET /api/backtest/results?backtest_id=latest

{
  "backtest_id": "bt_20250121_001",
  "model": "PPO_LSTM_v2.1",
  "period": {
    "start": "2024-01-01",
    "end": "2024-12-31",
    "days": 365
  },
  "performance": {
    "total_return": 0.245,
    "cagr": 0.278,
    "sharpe_ratio": 1.87,
    "sortino_ratio": 2.15,
    "calmar_ratio": 1.05,
    "max_drawdown": -0.085,
    "current_drawdown": -0.021
  },
  "trading": {
    "total_trades": 150,
    "winning_trades": 103,
    "losing_trades": 47,
    "win_rate": 0.687,
    "profit_factor": 2.34,
    "avg_win": 125.50,
    "avg_loss": 75.25
  },
  "risk": {
    "volatility": 0.152,
    "var_95": -0.025,
    "beta": 0.85
  }
}
```

---

## 🎯 RESUMEN EJECUTIVO POR AUDITOR

### Perspectiva 1: AUDITOR DE DATOS

```
DATOS REALES vs MOCK:

✅ DISPONIBLES (Usar):
- Precios históricos OHLCV cada 5 min (2020-2025)
- Indicadores técnicos calculables
- Stats 24h (high, low, avg, volumen)
- Volatilidad histórica
- Pipeline metrics (L0-L6)
- Backtest results (si L6 existe)
- Model metrics (L5)

❌ NO DISPONIBLES (No mostrar o simular claramente):
- Trades reales ejecutados
- Posiciones del portafolio
- P&L de trading en vivo
- Spreads bid/ask real-time
- Risk metrics de portafolio real
- Session P&L
```

### Perspectiva 2: AUDITOR DE PRODUCTO

```
EXPECTATIVAS vs REALIDAD:

❌ PROBLEMA ACTUAL:
- Dashboard muestra "Live Trading" pero no hay trading real
- Métricas de "Portfolio" sin portafolio real
- "Session P&L" sin sesiones de trading

✅ SOLUCIÓN:
- Renombrar vistas para reflejar realidad
- "Live Trading" → "Market Analysis"
- "Portfolio Risk" → "Market Risk Analysis"
- "Session P&L" → "Backtest Performance"
- Marcar claramente qué es backtest vs real-time
```

### Perspectiva 3: AUDITOR TÉCNICO

```
ENDPOINTS NECESARIOS POR PRIORIDAD:

🔴 PRIORIDAD ALTA (Datos 100% reales):
1. GET /api/market/ohlcv - Datos históricos
2. GET /api/market/indicators - Indicadores técnicos
3. GET /api/pipeline/l0/stats - Estadísticas L0
4. GET /api/pipeline/l6/backtest - Resultados backtest

🟡 PRIORIDAD MEDIA (Requieren procesamiento):
5. GET /api/market/volatility - Volatilidad histórica
6. GET /api/pipeline/l4/episodes - Episodios RL
7. GET /api/pipeline/l5/models - Métricas de modelos

🟢 PRIORIDAD BAJA (Simulados/Calculados):
8. GET /api/signals/generate - Señales técnicas
9. GET /api/market/risk-analysis - VaR histórico precio
```

### Perspectiva 4: AUDITOR DE COMPLIANCE

```
PROBLEMAS DE TRANSPARENCIA:

❌ ISSUES:
1. Dashboard implica trading en vivo sin disclaimer
2. P&L mostrado sin clarificar que es backtest
3. Posiciones mostradas sin indicar que son simuladas
4. Risk metrics de portafolio inexistente

✅ SOLUCIONES REQUERIDAS:
1. Agregar banner: "Historical Analysis & Backtest Results"
2. Cada métrica de backtest debe tener "(Backtest)" en label
3. Eliminar secciones de portafolio o marcar "Demo Mode"
4. Agregar disclaimer en login sobre naturaleza del sistema
```

### Perspectiva 5: AUDITOR DE UX/UI

```
CONFUSIÓN DEL USUARIO:

❌ PROBLEMAS:
- Usuario espera poder hacer trades (no puede)
- Métricas "en vivo" que son históricas
- Valores hardcodeados inconsistentes

✅ MEJORAS:
1. Modo del sistema claro: "Analysis Mode" vs "Trading Mode"
2. Colores diferentes para datos reales vs simulados
3. Iconos distintivos (📊 Real Data, 🎮 Simulated)
4. Tooltips explicando origen de cada métrica
```

---

## 📋 MATRIZ DE DECISIÓN: QUÉ MOSTRAR

### Criterios de Evaluación

| Métrica | Datos Reales | Calculable | Útil | Mostrar | Notas |
|---------|--------------|------------|------|---------|-------|
| **Precio Actual** | ✅ | ✅ | ✅ | ✅ | Último de DB |
| **Candlesticks** | ✅ | ✅ | ✅ | ✅ | OHLCV histórico |
| **RSI, MACD, BB** | ✅ | ✅ | ✅ | ✅ | Calculados de OHLCV |
| **Stats 24h** | ✅ | ✅ | ✅ | ✅ | High/Low/Avg/Vol |
| **Volatilidad** | ✅ | ✅ | ✅ | ✅ | StdDev returns |
| **Spread Bid-Ask** | ❌ | ⚠️ | ⚠️ | ⚠️ | Simular de volatilidad |
| **P&L Session** | ❌ | ❌ | ❌ | ❌ | Eliminar o usar backtest |
| **Posiciones** | ❌ | ❌ | ❌ | ❌ | Eliminar completamente |
| **Portfolio VaR** | ❌ | ❌ | ❌ | ❌ | Cambiar a Price VaR |
| **Backtest KPIs** | ✅ | ✅ | ✅ | ✅ | De L6, marcar período |
| **Señales ML** | ⚠️ | ✅ | ✅ | ✅ | Técnicas + L5 si existe |
| **Pipeline Stats** | ✅ | ✅ | ✅ | ✅ | L0-L6 completamente |
| **Model Metrics** | ✅ | ✅ | ✅ | ✅ | De L5 |
| **RL Metrics** | ⚠️ | ⚠️ | ✅ | ⚠️ | Solo si L4 tiene datos |

**Leyenda:**
- ✅ Sí, datos reales disponibles
- ⚠️ Parcialmente / Requiere simulación
- ❌ No, eliminar o reemplazar

---

## 🎯 DASHBOARD REALISTA PROPUESTO

### Vista 1: Market Overview (Antes: Dashboard Home)
```
✅ MOSTRAR:
- Precio actual USDCOP
- Candlestick chart con timeframes
- Indicadores técnicos (RSI, MACD, BB, EMAs)
- Stats 24h/7d/30d (high, low, avg, volume)
- Volatilidad histórica
- Market hours status (COT timezone)

❌ ELIMINAR:
- P&L Session
- Posiciones actuales
- Spread bid-ask real-time
```

### Vista 2: Technical Analysis (Antes: Professional Terminal)
```
✅ MOSTRAR:
- Multi-timeframe charts
- Advanced indicators
- Pattern recognition
- Support/Resistance levels
- Volume profile

❌ ELIMINAR:
- Portfolio positions table
- Live P&L
- Order entry panel
```

### Vista 3: ML Signals (Antes: Trading Signals)
```
✅ MOSTRAR:
- Señales generadas por indicadores técnicos
- Confidence score de modelo L5 (si existe)
- Análisis técnico justificando señal
- Niveles de SL/TP calculados
- Risk score basado en volatilidad

DISCLAIMER:
"Signals based on technical analysis and ML models.
For educational purposes only. Not financial advice."
```

### Vista 4: Backtest Performance (Antes: Executive Overview)
```
✅ MOSTRAR:
- KPIs de backtest más reciente
- Período del backtest claramente marcado
- Performance metrics (Sharpe, Sortino, Calmar)
- Equity curve del backtest
- Drawdown chart
- Trade statistics

HEADER:
"Backtest Results - Period: Jan 2024 - Dec 2024"
```

### Vista 5: Market Risk Analysis (Antes: Risk Monitor)
```
✅ MOSTRAR:
- VaR histórico del precio USDCOP (no portafolio)
- Volatilidad rolling (30d, 60d, 90d)
- Drawdown desde máximo reciente
- Distribution of returns
- Volatility cones

RENOMBRAR:
"Market Risk Analysis" (no "Portfolio Risk")
```

### Vista 6: Market Alerts (Antes: Risk Alerts)
```
✅ MOSTRAR:
- Price movement alerts
- Volatility spikes
- Technical indicator extremes
- Support/Resistance breaks

❌ ELIMINAR:
- Position limit alerts
- Portfolio VaR breaches
```

### Vistas 7-11: Pipeline Data (L0-L5) ✅ MANTENER
```
✅ MOSTRAR TODO:
- L0: Data quality metrics
- L1: Feature statistics
- L3: Correlation matrix
- L4: RL episode stats
- L5: Model performance

ESTAS SON LAS VISTAS MÁS REALISTAS
```

### Vista 12: Backtest Results (L6) ✅ MANTENER
```
✅ MOSTRAR:
- Complete backtest KPIs
- Trade log
- Equity curve
- Drawdown analysis
- Performance attribution
```

---

## 🚀 PLAN DE ACCIÓN RECOMENDADO

### Fase 1: Auditoría de Datos (1 día)
```sql
-- Verificar qué datos realmente existen
SELECT COUNT(*) FROM market_data;
SELECT MIN(timestamp), MAX(timestamp) FROM market_data;
SELECT DISTINCT symbol FROM market_data;

-- Verificar buckets MinIO
mc ls minio/00-l0-ds-usdcop-acquire
mc ls minio/04-l4-ds-usdcop-rlready
mc ls minio/05-l5-ds-usdcop-serving

-- Verificar existencia de backtest results
mc ls minio/06-l6-ds-usdcop-backtest
```

### Fase 2: Implementar Endpoints Reales (3-5 días)
```python
# Prioridad 1: Market data
GET /api/market/ohlcv
GET /api/market/indicators
GET /api/market/stats

# Prioridad 2: Pipeline data
GET /api/pipeline/l0/statistics
GET /api/pipeline/l4/episodes
GET /api/pipeline/l5/models
GET /api/pipeline/l6/backtest

# Prioridad 3: Analysis
GET /api/analysis/volatility
GET /api/analysis/var-historical
GET /api/signals/technical
```

### Fase 3: Actualizar Frontend (2-3 días)
```typescript
// Eliminar componentes de trading en vivo
- Portfolio positions table
- Order entry forms
- Live P&L displays

// Renombrar vistas
"Live Trading" → "Market Analysis"
"Risk Monitor" → "Market Risk Analysis"
"Portfolio" → "Backtest Performance"

// Agregar disclaimers
<Banner type="info">
  Historical market analysis and backtest results.
  Not a live trading system.
</Banner>
```

### Fase 4: Testing & Validación (1-2 días)
```
✅ Verificar todos los valores vienen de datos reales
✅ Confirmar cálculos correctos
✅ Validar que no hay valores hardcoded
✅ Test de consistencia de datos
✅ UX review con usuarios
```

---

## 📊 VALORES RECOMENDADOS POR VISTA (RESUMEN)

| Vista | Valores a Mostrar | Fuente de Datos |
|-------|-------------------|-----------------|
| **Market Overview** | Precio, OHLCV, Stats 24h, Indicadores técnicos, Volatilidad | PostgreSQL + cálculos |
| **Technical Analysis** | Charts multi-timeframe, Indicadores avanzados, Patterns | PostgreSQL + TA-Lib |
| **ML Signals** | Señales técnicas, Confidence (L5), SL/TP, Risk score | Cálculos + L5 model |
| **Backtest Performance** | KPIs backtest, Equity curve, Drawdown, Trade stats | L6 bucket |
| **Market Risk** | VaR precio, Volatilidad, Drawdown, Returns distribution | PostgreSQL analytics |
| **Market Alerts** | Price moves, Volatility spikes, Indicator extremes | Real-time calcs |
| **L0-L5 Pipeline** | Data quality, Feature stats, Correlations, Episodes, Models | MinIO buckets |
| **Backtest Results** | Complete backtest analysis | L6 bucket |

---

**CONCLUSIÓN AUDITOR:**

El sistema tiene datos REALES y VALIOSOS (histórico OHLCV, pipeline L0-L6, backtest results), pero el dashboard actual SOBREVENDE capacidades que no existen (trading en vivo, portfolio real, P&L real).

**Recomendación:** Pivotear de "Live Trading Platform" a "Historical Market Analysis & Backtesting Platform" - que es lo que REALMENTE es y tiene valor real.
