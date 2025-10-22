# QUICK REFERENCE: VALORES HARDCODEADOS POR COMPONENTE

## ÍNDICE DE COMPONENTES

| # | Componente | Estado | Hardcoding | Prioridad |
|---|------------|--------|------------|-----------|
| 1 | EnhancedTradingTerminal | ⚠️ PARCIAL | KPIs sesión, ejecución, latencia | 🔴 P1 |
| 2 | ExecutiveOverview | ⚠️ PARCIAL | Cambios % históricos | 🔴 P1 |
| 3 | DataPipelineQuality | ❌ MOCK | Todas las métricas L0-L4 | 🔴 P1 |
| 4 | RLModelHealth | ⚠️ PARCIAL | Métricas PPO/LSTM/QR-DQN | 🔴 P1 |
| 5 | ModelPerformance | ❌ MOCK | Sharpe, drawdown, win rate, SHAP | 🟠 P2 |
| 6 | PortfolioExposureAnalysis | ❌ MOCK | Exposures completos, stress tests | 🟠 P2 |
| 7 | UltimateVisualDashboard | ❌ MOCK | Precios, volatilidad, RSI, ATR | 🟡 P3 |
| 8 | ProfessionalTradingTerminalSimplified | ❌ MOCK | Precios completos, stats 24h | 🟡 P3 |
| 9 | RealTimeRiskMonitor | ✅ CONECTADO | Solo fallbacks razonables | ✅ OK |
| 10 | LiveTradingTerminal | ✅ CONECTADO | Solo fallback price 4150.25 | ✅ OK |
| 11 | RiskAlertsCenter | ✅ CONECTADO | Solo duración audio | ✅ OK |
| 12 | AuditCompliance | ⚠️ PARCIAL | coverage: 100 | 🟢 P4 |

**Leyenda:**
- ✅ CONECTADO = Ya usa APIs reales, mínimo hardcoding
- ⚠️ PARCIAL = Algunas métricas reales, otras hardcoded
- ❌ MOCK = Completamente simulado con datos mock

---

## VALORES HARDCODEADOS MÁS COMUNES

### Precios USDCOP
```typescript
4000, 4010.91, 4025.50, 3990.25, 4150.25, 4165
Rango típico: 3500-4500
```

### Volúmenes
```typescript
125430, 125000, 84455
```

### Métricas de Performance
```typescript
sharpeRatio: 2.34
maxDrawdown: 0.087
winRate: 0.643 (64.3%)
```

### Data Pipeline
```typescript
coverage: 95.8, 100
gridPerfection: 100
terminalCorrectness: 100
```

### RL Model
```typescript
tradesPerEpisode: 6, 7
policyEntropy: 0.34
klDivergence: 0.019
action distribution: { sell: 18.5%, hold: 63.2%, buy: 18.3% }
```

### Spreads y Costos
```typescript
spread: 2.5 COP
vwapVsFill: 1.2 bps
slippage: 2.1 bps
turnCost: 8.5 bps
```

### Latencias
```typescript
p50: 45ms, p95: 78ms, p99: 95ms
onnxP99: 12ms
acquisitionLatency: 340ms
```

---

## MAPA DE COMPONENTES → APIs

### Trading API (puerto 8000)
```
✅ EN USO:
- LiveTradingTerminal → /api/market/price/USDCOP
- RealTimeRiskMonitor → /positions?symbol=USDCOP

❌ NECESARIAS:
- UltimateVisualDashboard → /api/market/stats/USDCOP
- ProfessionalTradingTerminalSimplified → /api/market/stats/USDCOP?period=24h
```

### Analytics API (puerto 8001)
```
✅ EN USO:
- ExecutiveOverview → /api/analytics/performance-kpis
- ExecutiveOverview → /api/analytics/production-gates
- EnhancedTradingTerminal → /session-pnl
- RealTimeRiskMonitor → /api/analytics/market-conditions

❌ NECESARIAS:
- ExecutiveOverview → /api/analytics/kpi-changes
- EnhancedTradingTerminal → /api/analytics/session-kpis
- EnhancedTradingTerminal → /api/analytics/execution-metrics
- EnhancedTradingTerminal → /api/analytics/latency-stats
- UltimateVisualDashboard → /api/analytics/indicators/USDCOP
- ModelPerformance → /api/analytics/model-metrics
- PortfolioExposureAnalysis → /api/analytics/portfolio-exposure
- PortfolioExposureAnalysis → /api/analytics/risk-attribution
- PortfolioExposureAnalysis → /api/analytics/stress-test
- PortfolioExposureAnalysis → /api/analytics/liquidity-analysis
```

### Pipeline Data API (puerto 8002) - ⚠️ TODO FALTA
```
❌ NECESARIAS:
- DataPipelineQuality → /api/pipeline/l0/quality-metrics
- DataPipelineQuality → /api/pipeline/l1/quality-metrics
- DataPipelineQuality → /api/pipeline/l2/quality-metrics
- DataPipelineQuality → /api/pipeline/l3/quality-metrics
- DataPipelineQuality → /api/pipeline/l4/quality-metrics
- DataPipelineQuality → /api/pipeline/system-health
```

### ML Analytics API (puerto 8004)
```
⚠️ PARCIAL:
- RLModelHealth → /api/analytics/rl-metrics (existe pero incompleto)

❌ NECESARIAS:
- ModelPerformance → /api/ml/ensemble-status
- ModelPerformance → /api/ml/drift-detection
- ModelPerformance → /api/ml/shap-values
- RLModelHealth → /api/ml/ppo-metrics
- RLModelHealth → /api/ml/lstm-metrics
- RLModelHealth → /api/ml/qrdqn-metrics
- RLModelHealth → /api/ml/reward-metrics
- RLModelHealth → /api/ml/system-performance
```

---

## CHECKLIST DE LIMPIEZA POR COMPONENTE

### 🔴 PRIORIDAD 1

#### [ ] EnhancedTradingTerminal.tsx
- [ ] Conectar `tradesEpisode`, `avgHolding`, `drawdownIntraday` a API
- [ ] Conectar métricas de ejecución: `vwapVsFill`, `spreadEffective`, `slippage`, `turnCost`, `fillRatio`
- [ ] Conectar métricas de latencia: `p50`, `p95`, `p99`, `onnxP99`
- [ ] Eliminar función `generateKPIData` y usar datos reales

#### [ ] ExecutiveOverview.tsx
- [ ] Conectar valores de `change` % para cada KPI card
- [ ] Remover fallbacks de `sortinoRatio`, `calmarRatio`, etc.

#### [ ] DataPipelineQuality.tsx
- [ ] Implementar API completo en puerto 8002
- [ ] Conectar todas las métricas L0-L4
- [ ] Eliminar hook `useDataPipelineQuality` con datos mock

#### [ ] RLModelHealth.tsx
- [ ] Completar endpoint `/api/analytics/rl-metrics`
- [ ] Crear endpoints específicos para PPO, LSTM, QR-DQN
- [ ] Eliminar estado inicial con valores hardcoded

### 🟠 PRIORIDAD 2

#### [ ] ModelPerformance.tsx
- [ ] Crear endpoints para ensemble, drift detection, SHAP
- [ ] Eliminar objeto completo `mockMetrics`
- [ ] Conectar gráficos a datos reales

#### [ ] PortfolioExposureAnalysis.tsx
- [ ] Crear endpoints de exposure, stress test, liquidity
- [ ] Eliminar funciones mock completas
- [ ] Conectar visualizaciones a datos reales

### 🟡 PRIORIDAD 3

#### [ ] UltimateVisualDashboard.tsx
- [ ] Conectar a Trading API para precios
- [ ] Conectar a Analytics API para indicadores técnicos
- [ ] Eliminar función `generateSpectacularMockData`

#### [ ] ProfessionalTradingTerminalSimplified.tsx
- [ ] Conectar a Trading API
- [ ] Eliminar función `generateMockData`

### 🟢 PRIORIDAD 4

#### [ ] LiveTradingTerminal.tsx
- [ ] Remover fallback `4150.25`
- [ ] Mejorar manejo de loading state

#### [ ] RealTimeRiskMonitor.tsx
- [ ] Validar fallbacks actuales
- [ ] No requiere cambios mayores

#### [ ] AuditCompliance.tsx
- [ ] Conectar `coverage: 100` a API real
- [ ] Baja prioridad

---

## RESUMEN EJECUTIVO

### Estado Actual
- **31 componentes** analizados
- **15 componentes** con hardcoding
- **~500+ líneas** con valores hardcoded
- **6 endpoints** funcionando correctamente
- **25+ endpoints** faltantes

### Trabajo Pendiente
1. **Backend:** Implementar ~25 endpoints nuevos
2. **Frontend:** Limpiar 15 componentes
3. **Testing:** Validar integración completa

### Tiempo Estimado
- **Backend APIs:** 2-3 días
- **Frontend cleanup:** 2-3 días
- **Testing:** 1 día
- **Total:** ~1 semana trabajo completo

---

**Reporte generado:** 2025-10-21
**Ver reporte detallado:** `HARDCODED_VALUES_ANALYSIS_REPORT.md`
