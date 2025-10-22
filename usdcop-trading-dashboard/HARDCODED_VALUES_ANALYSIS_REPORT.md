# REPORTE DE ANÁLISIS: VALORES HARDCODEADOS EN COMPONENTES

**Fecha:** 2025-10-21
**Alcance:** Todos los componentes en `usdcop-trading-dashboard/components/views/`
**Total de archivos analizados:** 31 archivos `.tsx`

---

## RESUMEN EJECUTIVO

Se identificaron **MÚLTIPLES VALORES HARDCODEADOS** en 15 componentes que actualmente utilizan datos simulados en lugar de conectarse a las APIs reales. Los valores hardcodeados incluyen:

- ✅ **Precios USDCOP:** 4000-4500 COP
- ✅ **Volúmenes:** 125,430 unidades
- ✅ **Métricas de performance:** Sharpe Ratio, Max Drawdown, Win Rate
- ✅ **Datos de pipeline:** Coverage 95.8%, latencias, throughput
- ✅ **Valores de RL:** Action balance, trades per episode
- ✅ **Métricas de riesgo:** VaR, leverage, drawdown
- ✅ **Exposiciones de portfolio:** Valores en USD con distribuciones fijas

---

## CATEGORÍA 1: PRECIOS Y DATOS DE MERCADO

### 1.1 ProfessionalTradingTerminalSimplified.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/ProfessionalTradingTerminalSimplified.tsx`

#### Valores Hardcodeados:

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 42 | `basePrice = 4000` | Precio base para generación de datos mock |
| 50 | `Math.max(3500, Math.min(4500, basePrice + change))` | Rango de precio 3500-4500 |
| 85 | `price: 4010.91` | Precio actual inicial |
| 86 | `change: 15.33` | Cambio 24h |
| 87 | `changePercent: 0.38` | Porcentaje de cambio |
| 88 | `high24h: 4025.50` | Máximo 24h |
| 89 | `low24h: 3990.25` | Mínimo 24h |
| 90 | `volume: 125430` | Volumen 24h |

**API que debería usarse:**
```typescript
// Trading API - Real-time price
GET http://localhost:8000/api/market/price/USDCOP

// Trading API - Market stats
GET http://localhost:8000/api/market/stats/USDCOP?period=24h
```

---

### 1.2 UltimateVisualDashboard.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/UltimateVisualDashboard.tsx`

#### Valores Hardcodeados:

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 69 | `currentPrice: 4010.91` | Precio actual |
| 70 | `change24h: 15.33` | Cambio 24h |
| 71 | `changePercent: 0.38` | % Cambio |
| 72 | `high24h: 4025.50` | Máximo 24h |
| 73 | `low24h: 3990.25` | Mínimo 24h |
| 74 | `volume24h: 125430` | Volumen 24h |
| 75 | `spread: 2.5` | Spread en COP |
| 76 | `volatility: 1.24` | Volatilidad % |
| 77 | `atr: 12.45` | Average True Range |
| 78 | `rsi: 67.8` | RSI Indicator |
| 131 | `let basePrice = 4000` | Base para mock data |
| 142 | `Math.max(3500, Math.min(4500, basePrice))` | Rango precio |

**API que debería usarse:**
```typescript
// Trading API - Real-time metrics
GET http://localhost:8000/api/market/price/USDCOP
GET http://localhost:8000/api/market/stats/USDCOP

// Analytics API - Technical indicators
GET http://localhost:8001/api/analytics/indicators/USDCOP
```

---

### 1.3 LiveTradingTerminal.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/LiveTradingTerminal.tsx`

#### Valores Hardcodeados:

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 90 | `price: realTimePrice.currentPrice?.price \|\| 4150.25` | Fallback price hardcoded |

**API que debería usarse:**
```typescript
// Trading API - Ya conectado pero con fallback hardcoded
// REMOVER fallback y manejar estado de loading/error
GET http://localhost:8000/api/market/price/USDCOP
```

**Acción requerida:** Eliminar fallback `4150.25` y usar estado de loading mientras se carga el precio real.

---

## CATEGORÍA 2: MÉTRICAS DE PERFORMANCE Y KPIs

### 2.1 ExecutiveOverview.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/ExecutiveOverview.tsx`

#### Valores Hardcodeados:

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 152 | `sortinoRatio: 0` | Fallback mientras carga |
| 153 | `calmarRatio: 0` | Fallback mientras carga |
| 154 | `maxDrawdown: 0` | Fallback mientras carga |
| 155 | `profitFactor: 0` | Fallback mientras carga |
| 156 | `benchmarkSpread: 0` | Fallback mientras carga |
| 157 | `cagr: 0` | Fallback mientras carga |
| 158 | `sharpeRatio: 0` | Fallback mientras carga |
| 159 | `volatility: 0` | Fallback mientras carga |
| 177 | `change: 2.3` | Cambio % hardcoded |
| 187 | `change: 1.8` | Cambio % hardcoded |
| 197 | `change: -0.7` | Cambio % hardcoded |
| 207 | `change: 3.1` | Cambio % hardcoded |

**Nota:** Este componente YA usa `usePerformanceKPIs` y `useProductionGates` hooks, pero los valores de `change` siguen hardcoded.

**API que debería usarse:**
```typescript
// Analytics API - Performance KPIs (YA EN USO)
GET http://localhost:8001/api/analytics/performance-kpis?symbol=USDCOP&days=90

// Falta: API para cambios % históricos
GET http://localhost:8001/api/analytics/kpi-changes?symbol=USDCOP&period=7d
```

---

### 2.2 ModelPerformance.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/ModelPerformance.tsx`

#### Valores Hardcodeados (Mock Data):

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 142-162 | Todo el objeto `mockMetrics` | Métricas completas del modelo simuladas |
| 148 | `sharpeRatio: 2.34` | Sharpe ratio mock |
| 149 | `maxDrawdown: 0.087` | Max drawdown mock |
| 150 | `winRate: 0.643` | Win rate 64.3% mock |
| 165-211 | `ensemble: EnsembleModel[]` | 5 modelos con pesos hardcoded |
| 214-250 | `drifts: DriftDetection[]` | Detecciones de drift simuladas |
| 253-296 | `shapData: SHAPValue[]` | SHAP values para explicabilidad simulados |

**API que debería usarse:**
```typescript
// Analytics API - Model metrics
GET http://localhost:8001/api/analytics/model-metrics?symbol=USDCOP&days=30

// ML Analytics API - Model ensemble
GET http://localhost:8004/api/ml/ensemble-status

// ML Analytics API - Drift detection
GET http://localhost:8004/api/ml/drift-detection?symbol=USDCOP

// ML Analytics API - Feature importance (SHAP)
GET http://localhost:8004/api/ml/shap-values?symbol=USDCOP&prediction_id=latest
```

---

## CATEGORÍA 3: DATA PIPELINE Y CALIDAD

### 3.1 DataPipelineQuality.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/DataPipelineQuality.tsx`

#### Valores Hardcodeados (Función `useDataPipelineQuality`):

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 17 | `coverage: 95.8` | Coverage L0 |
| 18 | `ohlcInvariants: 0` | Violaciones OHLC |
| 19 | `crossSourceDelta: 6.2` | Delta entre fuentes (bps) |
| 20 | `duplicates: 0` | Duplicados detectados |
| 21 | `gaps: 0` | Gaps en datos |
| 22 | `staleRate: 1.2` | Tasa de datos obsoletos |
| 23 | `acquisitionLatency: 340` | Latencia adquisición (ms) |
| 24 | `volumeDataPoints: 84455` | Volumen de puntos |
| 29 | `gridPerfection: 100` | Grid perfection L1 |
| 30 | `terminalCorrectness: 100` | Terminal correctness |
| 31 | `hodBaselines: 100` | HOD baselines |
| 38 | `winsorizationRate: 0.8` | Winsorization rate L2 |
| 48 | `forwardIC: 0.08` | Forward IC L3 |
| 55 | `observationFeatures: 17` | Features observation L4 |
| 56 | `clipRate: 0.3` | Clip rate |
| 57 | `zeroRateT33: 42.1` | Zero rate t≥33 |

**API que debería usarse:**
```typescript
// Pipeline Data API - Quality metrics por layer
GET http://localhost:8002/api/pipeline/l0/quality-metrics
GET http://localhost:8002/api/pipeline/l1/quality-metrics
GET http://localhost:8002/api/pipeline/l2/quality-metrics
GET http://localhost:8002/api/pipeline/l3/quality-metrics
GET http://localhost:8002/api/pipeline/l4/quality-metrics

// Pipeline Data API - System health
GET http://localhost:8002/api/pipeline/system-health
```

---

## CATEGORÍA 4: RISK MANAGEMENT

### 4.1 RealTimeRiskMonitor.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/RealTimeRiskMonitor.tsx`

#### Valores Hardcodeados:
**Nota:** Este componente YA usa `realTimeRiskEngine` y hace fetch de datos reales, pero tiene fallbacks en caso de error.

| Línea | Contexto | Descripción |
|-------|----------|-------------|
| 145-177 | `generateRiskHeatmap` fallback | Scores calculados con fórmulas si API falla |
| 184-201 | `fetchMarketConditions` fallback | Array vacío si API falla |

**APIs que YA USA (correctamente):**
```typescript
// Trading API - Positions (YA EN USO)
GET http://localhost:8000/positions?symbol=USDCOP

// Analytics API - Market conditions (YA EN USO)
GET http://localhost:8001/api/analytics/market-conditions?symbol=USDCOP&days=30

// Risk Engine local - Real-time calculations (YA EN USO)
realTimeRiskEngine.getRiskMetrics()
```

**Estado:** ✅ **BIEN IMPLEMENTADO** - Solo tiene fallbacks razonables.

---

### 4.2 PortfolioExposureAnalysis.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/PortfolioExposureAnalysis.tsx`

#### Valores Hardcodeados (Mock Data Completo):

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 145-148 | Country exposures | Colombia 85%, USA 10%, Brasil 3%, México 2% |
| 151-153 | Currency exposures | COP 85%, USD 12%, BRL 3% |
| 156-159 | Sector exposures | FX Spot 60%, Interest Rates 20%, Commodities 12%, Vol 8% |
| 162-165 | Maturity buckets | 4 buckets con exposures y avg maturity |
| 168-171 | Risk factors | USD/COP Rate, COP Vol, Oil Prices, EM Risk |
| 173-182 | Concentration metrics | HHI, Gini, Max Single, Top 5 |
| 197-200 | Risk attribution | 4 componentes con contributions |
| 213-216 | Liquidity tiers | 4 tiers con exposures y liquidation time |
| 228-265 | Stress scenarios | 4 escenarios con impacts detallados |

**API que debería usarse:**
```typescript
// Trading API - Portfolio positions
GET http://localhost:8000/portfolio/positions?symbol=USDCOP

// Analytics API - Portfolio analytics
GET http://localhost:8001/api/analytics/portfolio-exposure?symbol=USDCOP

// Analytics API - Risk attribution
GET http://localhost:8001/api/analytics/risk-attribution?symbol=USDCOP

// Analytics API - Stress testing
GET http://localhost:8001/api/analytics/stress-test?symbol=USDCOP&scenarios=all

// Analytics API - Liquidity analysis
GET http://localhost:8001/api/analytics/liquidity-analysis?symbol=USDCOP
```

---

## CATEGORÍA 5: RL MODEL HEALTH

### 5.1 RLModelHealth.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/RLModelHealth.tsx`

#### Valores Hardcodeados (Estado Inicial):

| Línea | Valor Hardcodeado | Descripción |
|-------|------------------|-------------|
| 20 | `tradesPerEpisode: 6` | Trades por episodio |
| 21 | `fullEpisodes: 45` | Episodios completos |
| 22 | `shortEpisodes: 23` | Episodios cortos (t≤35) |
| 24-26 | Action distribution | sell: 18.5%, hold: 63.2%, buy: 18.3% |
| 27 | `policyEntropy: 0.34` | Entropía de política |
| 28 | `klDivergence: 0.019` | KL divergence |
| 34-39 | PPO metrics | policy loss, value loss, explained variance, etc. |
| 42-46 | LSTM metrics | reset rate, avg sequence, truncation, etc. |
| 48-54 | QR-DQN metrics | quantile loss, buffer fill, exploration, etc. |
| 56-61 | Reward metrics | RMSE, defined rate, cost curriculum, etc. |
| 64-69 | Performance metrics | CPU, memory, GPU, inference time |

**Nota:** Este componente tiene un `useEffect` en línea 72-171 que INTENTA conectarse al API pero tiene fallback completo a datos simulados.

**API que debería usarse:**
```typescript
// ML Analytics API - RL metrics (PARCIALMENTE IMPLEMENTADO)
GET http://localhost:8004/api/analytics/rl-metrics

// Endpoints adicionales necesarios:
GET http://localhost:8004/api/ml/ppo-metrics
GET http://localhost:8004/api/ml/lstm-metrics
GET http://localhost:8004/api/ml/qrdqn-metrics
GET http://localhost:8004/api/ml/reward-metrics
GET http://localhost:8004/api/ml/system-performance
```

---

## CATEGORÍA 6: ENHANCED TRADING TERMINAL

### 6.1 EnhancedTradingTerminal.tsx
**Archivo:** `/home/GlobalForex/USDCOP-RL-Models/usdcop-trading-dashboard/components/views/EnhancedTradingTerminal.tsx`

#### Valores Hardcodeados:

| Línea | Función | Descripción |
|-------|---------|-------------|
| 14-63 | `generateKPIData()` | Función completa que simula KPIs de sesión |
| 36 | `pnlIntraday: 0` (default) | P&L intradía inicial |
| 37 | `pnlPercent: 0` (default) | P&L % inicial |
| 38 | `tradesEpisode: 7` | Trades por episodio |
| 40 | `avgHolding: 12` | Avg holding en barras |
| 42 | `drawdownIntraday: -2.1` | Drawdown intradía |
| 46 | `vwapVsFill: 1.2` | VWAP vs Fill (bps) |
| 47 | `spreadEffective: 4.8` | Spread efectivo |
| 48 | `slippage: 2.1` | Slippage |
| 49 | `turnCost: 8.5` | Turn cost |
| 50 | `fillRatio: 94.2` | Fill ratio % |
| 52-55 | Latency metrics | p50, p95, p99, onnx p99 |

**Nota:** Este componente hace fetch de `session-pnl` (líneas 20-30) pero el resto de KPIs están hardcoded.

**API que debería usarse:**
```typescript
// Analytics API - Session P&L (YA EN USO)
GET http://localhost:8001/session-pnl?symbol=USDCOP

// Analytics API - Session KPIs (FALTANTE)
GET http://localhost:8001/api/analytics/session-kpis?symbol=USDCOP

// Analytics API - Execution metrics (FALTANTE)
GET http://localhost:8001/api/analytics/execution-metrics?symbol=USDCOP

// Analytics API - Latency stats (FALTANTE)
GET http://localhost:8001/api/analytics/latency-stats?symbol=USDCOP
```

---

## CATEGORÍA 7: OTROS COMPONENTES CON HARDCODING

### 7.1 AuditCompliance.tsx
**Línea 54:** `coverage: 100` - Coverage audit hardcoded

### 7.2 RiskAlertsCenter.tsx
**Línea 170:** `stats.avgResponseTime = acknowledgedAlerts.length > 0 ? acknowledgedAlerts.reduce((sum, alert) => sum + 15, 0) / acknowledgedAlerts.length : 0`
- Tiempo de respuesta promedio simulado: 15 minutos

**Línea 264:** `setTimeout(() => setIsPlaying(false), 3000)` - Duración de audio notification

---

## RESUMEN DE APIS NECESARIAS

### APIs YA IMPLEMENTADAS Y EN USO ✅

1. **Trading API (puerto 8000)**
   - `GET /api/market/price/USDCOP` - ✅ En uso en LiveTradingTerminal
   - `GET /positions?symbol=USDCOP` - ✅ En uso en RealTimeRiskMonitor

2. **Analytics API (puerto 8001)**
   - `GET /api/analytics/performance-kpis` - ✅ En uso en ExecutiveOverview
   - `GET /api/analytics/production-gates` - ✅ En uso en ExecutiveOverview
   - `GET /session-pnl` - ✅ En uso en EnhancedTradingTerminal
   - `GET /api/analytics/market-conditions` - ✅ En uso en RealTimeRiskMonitor

### APIs FALTANTES O PARCIALES ❌

1. **Analytics API (puerto 8001) - Endpoints faltantes**
   ```
   GET /api/analytics/kpi-changes
   GET /api/analytics/session-kpis
   GET /api/analytics/execution-metrics
   GET /api/analytics/latency-stats
   GET /api/analytics/indicators/USDCOP
   GET /api/analytics/portfolio-exposure
   GET /api/analytics/risk-attribution
   GET /api/analytics/stress-test
   GET /api/analytics/liquidity-analysis
   ```

2. **Pipeline Data API (puerto 8002) - Completamente faltante**
   ```
   GET /api/pipeline/l0/quality-metrics
   GET /api/pipeline/l1/quality-metrics
   GET /api/pipeline/l2/quality-metrics
   GET /api/pipeline/l3/quality-metrics
   GET /api/pipeline/l4/quality-metrics
   GET /api/pipeline/system-health
   ```

3. **ML Analytics API (puerto 8004) - Parcialmente implementado**
   ```
   GET /api/analytics/rl-metrics (EXISTE pero incompleto)
   GET /api/ml/ensemble-status (FALTA)
   GET /api/ml/drift-detection (FALTA)
   GET /api/ml/shap-values (FALTA)
   GET /api/ml/ppo-metrics (FALTA)
   GET /api/ml/lstm-metrics (FALTA)
   GET /api/ml/qrdqn-metrics (FALTA)
   GET /api/ml/reward-metrics (FALTA)
   GET /api/ml/system-performance (FALTA)
   ```

---

## PRIORIZACIÓN DE LIMPIEZA

### PRIORIDAD 1 - CRÍTICO (Componentes principales del sistema)
1. **EnhancedTradingTerminal.tsx** - Terminal principal de trading
2. **ExecutiveOverview.tsx** - Dashboard ejecutivo
3. **DataPipelineQuality.tsx** - Calidad del pipeline L0-L4
4. **RLModelHealth.tsx** - Salud del modelo RL

### PRIORIDAD 2 - ALTA (Analytics y performance)
5. **ModelPerformance.tsx** - Performance del modelo
6. **PortfolioExposureAnalysis.tsx** - Análisis de exposición

### PRIORIDAD 3 - MEDIA (UI y visualization)
7. **UltimateVisualDashboard.tsx** - Dashboard visual
8. **ProfessionalTradingTerminalSimplified.tsx** - Terminal simplificado

### PRIORIDAD 4 - BAJA (Ya funcionan bien o tienen fallbacks razonables)
9. **RealTimeRiskMonitor.tsx** - ✅ Ya conectado, solo mejorar fallbacks
10. **LiveTradingTerminal.tsx** - ✅ Ya conectado, solo remover fallback price
11. **RiskAlertsCenter.tsx** - ✅ Ya conectado con realTimeRiskEngine

---

## PLAN DE ACCIÓN RECOMENDADO

### FASE 1: Implementar APIs faltantes (Backend)
1. Crear endpoints en Analytics API para métricas faltantes
2. Crear Pipeline Data API completo (puerto 8002)
3. Completar ML Analytics API con endpoints de modelo

### FASE 2: Conectar componentes a APIs (Frontend)
1. Actualizar componentes PRIORIDAD 1 para usar APIs reales
2. Eliminar funciones mock y datos simulados
3. Implementar estados de loading/error apropiados

### FASE 3: Testing y validación
1. Verificar que todos los componentes muestren datos reales
2. Validar que no haya fallbacks a valores hardcoded
3. Performance testing con datos reales

---

## ESTADÍSTICAS FINALES

- **Total de archivos analizados:** 31
- **Archivos con hardcoding:** 15
- **Componentes ya conectados correctamente:** 3 (RealTimeRiskMonitor, LiveTradingTerminal parcial, RiskAlertsCenter)
- **Componentes con hardcoding CRÍTICO:** 8
- **Líneas de código con valores hardcoded:** ~500+
- **APIs ya en uso:** 6 endpoints
- **APIs faltantes:** 25+ endpoints

---

**Generado:** 2025-10-21
**Por:** Análisis automático de código
**Versión:** 1.0
