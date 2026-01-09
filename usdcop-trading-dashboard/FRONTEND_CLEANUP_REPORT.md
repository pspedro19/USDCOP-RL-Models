# FRONTEND CLEANUP DIAGNOSTIC REPORT
## USDCOP Trading Dashboard - Análisis de Dependencias

**Fecha:** 2026-01-08
**Objetivo:** Identificar código usado vs. código obsoleto para mantener solo Landing, Login y Dashboard.

---

## RESUMEN EJECUTIVO

| Categoría | Archivos | Estado |
|-----------|----------|--------|
| **Páginas a Mantener** | 3 | Landing, Login, Dashboard |
| **Páginas a Eliminar** | 6 | trading, ml-analytics, agent-trading, risk, trade-history, test |
| **Componentes Esenciales** | ~25 | En uso por las 3 páginas principales |
| **Componentes Eliminables** | ~80 | No usados por las páginas principales |
| **Reducción Estimada** | ~70-80% | Del código frontend |

---

## 1. ARCHIVOS ESENCIALES (MANTENER)

### 1.1 Páginas Principales
```
app/page.tsx              ← Landing Page
app/login/page.tsx        ← Login Page
app/dashboard/page.tsx    ← Dashboard Principal
app/layout.tsx            ← Layout principal
```

### 1.2 Contextos Requeridos
```
contexts/LanguageContext.tsx   ← Usado por Landing (traducciones)
contexts/ModelContext.tsx      ← Usado por Dashboard (selector de modelos)
```

### 1.3 Componentes Landing Page
```
components/landing/Navbar.tsx
components/landing/Hero.tsx
components/landing/Metrics.tsx
components/landing/Features.tsx
components/landing/HowItWorks.tsx
components/landing/Pricing.tsx
components/landing/FAQ.tsx
components/landing/Footer.tsx
```

### 1.4 Componentes Dashboard
```
components/ui/card.tsx           ← UI base
components/ui/badge.tsx          ← UI base
components/ui/button.tsx         ← UI base
components/ui/table.tsx          ← Para TradesTable
components/ui/skeleton.tsx       ← Para loading states
components/trading/TradingSummaryCard.tsx
components/trading/TradesTable.tsx
components/charts/TradingChartWithSignals.tsx  ← Lazy loaded
```

### 1.5 Hooks Esenciales
```
hooks/useModelMetrics.ts      ← Usado por TradingSummaryCard
hooks/useTradesHistory.ts     ← Usado por TradesTable
hooks/useIntegratedChart.ts   ← Usado por TradingChartWithSignals
```

### 1.6 Utilidades y Configuración
```
lib/utils.ts                        ← Función cn() para classnames
lib/motion.ts                       ← Animaciones
lib/translations.ts                 ← Traducciones es/en
lib/config/models.config.ts         ← Configuración de modelos
lib/config/training-periods.ts      ← Períodos de entrenamiento
lib/utils/logger.ts                 ← Logger para hooks
types/trading.ts                    ← Tipos TypeScript
```

---

## 2. PÁGINAS A ELIMINAR

### 2.1 app/trading/page.tsx
**Dependencias únicas:**
- `components/charts/RealDataTradingChart.tsx`
- `components/realtime/RealTimePriceDisplay.tsx`
- `components/trading/RiskStatusCard.tsx`
- `hooks/useRealTimePrice.ts`
- `hooks/useDbStats.ts`

### 2.2 app/ml-analytics/page.tsx
**Dependencias únicas:**
- `components/ml-analytics/ModelPerformanceDashboard.tsx`
- Todo el directorio `components/ml-analytics/`

### 2.3 app/agent-trading/page.tsx
**Dependencias únicas:**
- `components/charts/ChartWithPositions.tsx`
- `components/trading/AgentActionsTable.tsx`

### 2.4 app/risk/page.tsx
**Dependencias únicas:**
- `hooks/useRiskStatus.ts`
- `hooks/useLiveState.ts`
- `components/ui/progress.tsx`

### 2.5 app/trade-history/page.tsx
**Dependencias:** Usa TradesTable (compartido)

### 2.6 app/test/signal-overlay/page.tsx
**Dependencias únicas:**
- Test components específicos

---

## 3. COMPONENTES A ELIMINAR

### 3.1 Directorios Completos (ELIMINAR TODO)
```
components/charts/chart-engine/      ← No usado
components/common/                   ← No usado
components/compliance/               ← No usado
components/layout/                   ← No usado
components/metrics/                  ← No usado
components/ml-analytics/             ← Solo para ml-analytics page
components/monitoring/               ← No usado
components/navigation/               ← No usado
components/realtime/                 ← Solo para trading page
components/views/                    ← No usado
```

### 3.2 Componentes Charts (ELIMINAR)
```
components/charts/AdvancedHistoricalChart.tsx
components/charts/ChartWithPositions.tsx
components/charts/EquityCurveChart.tsx
components/charts/InteractiveTradingChart.tsx
components/charts/RealDataTradingChart.tsx
components/charts/SignalOverlay.tsx
components/charts/SingleModelEquityCurve.tsx
components/charts/real-data-chart/       ← Directorio completo
components/charts/signal-overlay/        ← Directorio completo
components/charts/index.ts
```

### 3.3 Componentes Trading (ELIMINAR - NO usados por dashboard)
```
components/trading/AgentActionsTable.tsx
components/trading/ChartToolbar.tsx
components/trading/index.ts
components/trading/KPICards.tsx
components/trading/MetricCard.tsx
components/trading/ModelMetricsPanel.tsx
components/trading/ModelSelector.tsx
components/trading/ModelTradesTable.tsx
components/trading/PaperTradingBadge.tsx
components/trading/PositionCard.tsx
components/trading/RiskStatusCard.tsx
components/trading/SignalAlerts.tsx
components/trading/TradingSignals.tsx
```

### 3.4 Componentes UI (ELIMINAR - NO usados)
```
components/ui/alert.tsx
components/ui/AlertBanner.tsx
components/ui/alert-notification.tsx
components/ui/AnimatedList.tsx
components/ui/AnimatedSidebar.tsx
components/ui/command-palette.tsx
components/ui/CustomCursor.tsx
components/ui/DataTable.tsx
components/ui/EnhancedNavigationSidebar.tsx
components/ui/export-toolbar.tsx
components/ui/help-system.tsx
components/ui/info-tooltip.tsx
components/ui/metric-card.tsx
components/ui/MetricGrid.tsx
components/ui/MobileControlsBar.tsx
components/ui/notification-manager.tsx
components/ui/OrderbookVisual.tsx
components/ui/PositionCard.tsx
components/ui/PriceTickerPro.tsx
components/ui/professional-trading-interface.tsx
components/ui/progress.tsx
components/ui/select.tsx
components/ui/SidebarToggleButtons.tsx
components/ui/slider.tsx
components/ui/smart-crosshair.tsx
components/ui/SplashScreen.tsx
components/ui/status-badge.tsx
components/ui/StatusIndicator.tsx
components/ui/tabs.tsx
components/ui/TimeRangeSelector.tsx
components/ui/tooltip.tsx
components/ui/trading-context-menu.tsx
```

### 3.5 Otros Componentes (ELIMINAR)
```
components/ViewRenderer.tsx
```

---

## 4. HOOKS A ELIMINAR

```
hooks/compliance/useAuditCompliance.ts
hooks/trading/useExecutionMetrics.ts
hooks/trading/useRealTimeMarketData.ts
hooks/trading/useTradingSession.ts
hooks/useAccessibility.tsx
hooks/useAnalytics.ts
hooks/useDbStats.ts
hooks/useEquityCurveStream.ts
hooks/useExport.ts
hooks/useFinancialMetrics.ts
hooks/useKeyboardShortcuts.ts
hooks/useLiveState.ts
hooks/useMarketStats.ts
hooks/useModelPositions.ts
hooks/useModelSignals.ts
hooks/useModelTrades.ts
hooks/usePaperTradingMetrics.ts
hooks/usePerformanceSummary.ts
hooks/useRealtimeData.ts
hooks/useRealTimePrice.ts
hooks/useResponsiveLayout.ts
hooks/useRiskStatus.ts
hooks/useSidebarState.ts
hooks/useSignalOverlay.ts
hooks/useTouchGestures.ts
hooks/useWorkspaceManager.ts
```

---

## 5. LIBRERÍAS Y SERVICIOS A ELIMINAR

### 5.1 Directorios Completos
```
lib/adapters/                  ← No usado
lib/auth/                      ← No usado (login es local)
lib/core/                      ← No usado
lib/db/                        ← Solo backend API
lib/factories/                 ← No usado
lib/services/                  ← Mayoría no usada
lib/store/                     ← No usado
lib/types/                     ← Parcial (mantener trading.ts)
```

### 5.2 Archivos Específicos
```
lib/config/api.config.ts
lib/config/market.config.ts
lib/config/realtime.config.ts
lib/config/risk.config.ts
lib/config/ui.config.ts
lib/technical-indicators.ts
lib/services/backtest-client.ts
lib/services/data-visualization-optimizer.ts
lib/services/enhanced-data-service.ts
lib/services/export-manager.ts
lib/services/hedge-fund-metrics.ts
lib/services/historical-data-manager.ts
lib/services/market-data-service.ts
lib/services/minio-client.ts
lib/services/mlmodel.ts
lib/services/pipeline.ts
lib/services/pipeline-api-client.ts
lib/services/pipeline-data-client.ts
lib/services/real-market-metrics.ts
lib/services/real-time-risk-engine.ts
lib/services/realtime-websocket-manager.ts
lib/services/twelvedata.ts
lib/services/unified-websocket-manager.ts
lib/services/websocket-manager.ts
lib/services/demo-data-generator.ts
```

---

## 6. ARCHIVOS DE CONFIGURACIÓN (REVISAR)

### 6.1 Mantener
```
config/views.config.ts     ← Puede necesitar simplificación
middleware.ts              ← Revisar si es necesario
```

### 6.2 Eliminar o Simplificar
```
archivo/                   ← Todo el directorio archive/
debug/                     ← Todo el directorio debug/
database/                  ← Todo el directorio database/
grafana-dashboards/        ← YA eliminado según git status
libs/                      ← YA eliminado según git status
```

---

## 7. ARCHIVOS QUE YA ESTÁN MARCADOS COMO ELIMINADOS (GIT STATUS)

Según el git status, estos archivos ya están programados para eliminación:
- `components/charts/` - Varios archivos obsoletos
- `components/views/` - Varios archivos obsoletos
- `libs/` - Directorio completo
- `grafana-dashboards/` - Directorio completo

---

## 8. RECOMENDACIONES DE IMPLEMENTACIÓN

### Paso 1: Backup
```bash
# Crear backup antes de eliminar
cd usdcop-trading-dashboard
git stash
# O crear branch de backup
git checkout -b backup-before-cleanup
git checkout main
```

### Paso 2: Eliminar Páginas
```bash
rm -rf app/trading
rm -rf app/ml-analytics
rm -rf app/agent-trading
rm -rf app/risk
rm -rf app/trade-history
rm -rf app/test
```

### Paso 3: Eliminar Componentes
```bash
rm -rf components/charts/chart-engine
rm -rf components/common
rm -rf components/compliance
rm -rf components/layout
rm -rf components/metrics
rm -rf components/ml-analytics
rm -rf components/monitoring
rm -rf components/navigation
rm -rf components/realtime
rm -rf components/views
# Eliminar archivos individuales según lista arriba
```

### Paso 4: Eliminar Hooks No Usados
```bash
rm -rf hooks/compliance
rm -rf hooks/trading
# Eliminar archivos individuales según lista arriba
```

### Paso 5: Eliminar Libs No Usadas
```bash
rm -rf lib/adapters
rm -rf lib/auth
rm -rf lib/core
rm -rf lib/db
rm -rf lib/factories
rm -rf lib/services  # CUIDADO: Revisar primero
rm -rf lib/store
```

### Paso 6: Verificar Build
```bash
npm run build
# Corregir cualquier import roto
```

---

## 9. RIESGO Y CONSIDERACIONES

### Bajo Riesgo
- Eliminar páginas adicionales (trading, ml-analytics, etc.)
- Eliminar componentes en directorios dedicados (ml-analytics, monitoring, etc.)

### Medio Riesgo
- Eliminar hooks (verificar que no hay imports indirectos)
- Eliminar componentes UI (algunos pueden ser usados indirectamente)

### Alto Riesgo
- Eliminar servicios en lib/services/ (algunos pueden ser usados por API routes)
- Eliminar tipos en lib/types/ (pueden ser usados por múltiples archivos)

---

## 10. CHECKLIST FINAL

Antes de eliminar, verificar:

- [ ] ¿El dashboard carga correctamente?
- [ ] ¿La landing page muestra todos sus componentes?
- [ ] ¿El login funciona?
- [ ] ¿El gráfico de velas se renderiza?
- [ ] ¿La tabla de trades muestra datos?
- [ ] ¿Las métricas KPI muestran valores?
- [ ] ¿El selector de modelos funciona?

---

**Generado automáticamente por análisis de dependencias Claude Code**
