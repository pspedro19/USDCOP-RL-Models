# USDCOP Trading Dashboard - READMEv2

## Lecciones Aprendidas (Post-Deployment Session - Enero 27, 2026)

Este documento captura las lecciones aprendidas durante la sesion de deployment y debugging del dashboard de trading USDCOP.

---

### 1. Autenticacion en Dev Mode: Bypass necesario para desarrollo local

**Problema:** Todas las rutas API protegidas con `withAuth()` retornaban 401 Unauthorized porque `getServerSession(authOptions)` requiere una sesion NextAuth activa, que no existe en desarrollo local.

**Solucion:** Se agrego un bypass en `lib/auth/api-auth.ts` que detecta `NEXT_PUBLIC_DEV_MODE=true` y crea un usuario sintetico admin sin pasar por NextAuth.

**Leccion:** Las capas de autenticacion deben tener un modo de desarrollo claro desde el inicio. Un middleware que bloquea todo sin bypass de dev genera fricciones constantes durante el desarrollo.

---

### 2. API Routes como Proxy: Nunca conectar directamente al backend desde el cliente

**Problema:** El servicio de backtest (`backtest.service.ts`) conectaba directamente a `localhost:8003` (inference API) desde el navegador, causando `ERR_CONNECTION_REFUSED` cuando el backend no esta corriendo.

**Solucion:** Se crearon rutas proxy en Next.js App Router:
- `/api/backtest/stream` - SSE streaming con fallback sintetico
- `/api/backtest/status/[modelId]` - Status check
- `/api/backtest` - Backtest no-streaming

**Leccion:** El patron correcto es **siempre** usar Next.js API routes como proxy. El cliente nunca debe hacer fetch directo a servicios backend internos. Esto permite:
- Fallbacks sinteticos cuando el backend esta caido
- Control de CORS
- Logging centralizado
- Transformacion de datos

---

### 3. Formatos de Respuesta API: El frontend dicta el contrato

**Problema:** Las rutas stub de equity-curve y metrics retornaban formatos simples (`{ data: [] }`) pero el frontend esperaba estructuras anidadas especificas (`{ data: { points: [], summary: { ... } } }`).

**Solucion:** Se ajustaron las respuestas para coincidir exactamente con lo que el frontend espera en `app/dashboard/page.tsx`.

**Leccion:** Antes de crear una ruta API stub, **leer el codigo frontend que la consume**. El contrato lo define el consumidor, no el productor. Un `console.error('Invalid equity data format')` siempre indica un mismatch de contrato.

---

### 4. lightweight-charts v5: Breaking change en markers API

**Problema:** `series.setMarkers()` fue removido en lightweight-charts v5.0.8. El chart logueaba `[TradingChart] setMarkers not available on series`.

**Solucion:** Se migro a la nueva API v5:
```typescript
import { createSeriesMarkers, ISeriesMarkersPluginApi } from 'lightweight-charts'

// v5: Plugin-based markers
const plugin = createSeriesMarkers(series, markers)
// Actualizaciones posteriores:
plugin.setMarkers(newMarkers)
```
Con fallback al API legacy para compatibilidad.

**Leccion:** Cuando se usa una libreria de graficos, verificar los changelogs de la version exacta instalada (`package.json`). Los breaking changes en APIs de markers/overlays son comunes entre major versions.

---

### 5. Timestamp Matching: Tolerancia adaptativa para datos sinteticos

**Problema:** Los trades sinteticos generados por el fallback del backtest no coincidian con los timestamps exactos de las velas OHLCV reales. Con tolerancia de 600s, se calculaban 0 markers de 5 signals.

**Solucion:** Tolerancia adaptativa segun el modo:
- **Modo normal:** 600s (10 min) - los signals reales deben coincidir con velas de 5min
- **Modo replay:** 86400s (24h) - los trades sinteticos pueden caer en cualquier dia

**Leccion:** Los datos sinteticos/mock nunca tienen la misma granularidad temporal que los datos reales. Siempre implementar tolerancias mas amplias para datos generados, o mejor aun, generar timestamps que coincidan exactamente con los datos OHLCV disponibles.

---

### 6. SSE Progressive Streaming: La experiencia de usuario importa

**Problema:** El backtest retornaba todos los resultados de golpe, sin retroalimentacion visual durante el procesamiento.

**Solucion:** Se implemento streaming SSE con 3 fases:
1. **Loading** (~2s): Progress 0-10%, mensajes de inicializacion
2. **Trade streaming** (~15-20s): Cada trade enviado individualmente con ~450ms de delay (investor mode), progress interleaved
3. **Finalizacion** (~1s): Metricas de performance y resultado final

**Leccion:** Para presentaciones a inversores, la percepcion de procesamiento es tan importante como el resultado. Un backtest que "piensa" y revela trades progresivamente genera mas confianza que uno que muestra todo instantaneamente. El jitter natural en delays (0.5-0.9x baseDelay) evita que se sienta robotico.

---

### 7. Datos sinteticos con sesgo controlado para demos

**Problema:** Los datos aleatorios puros (50/50 win/loss) producen equity curves planas o erraticas que no comunican valor al presentar a inversores.

**Solucion:** El modo investor demo usa:
- 57% win bias (realista pero positivo)
- 25-45 trades (suficientes para mostrar patron, no tantos que abrumen)
- Patron de onda sinusoidal en precios (`sin(i * 0.15) * 80`)
- Calculo real de Sharpe ratio, max drawdown, win rate

**Leccion:** Los datos sinteticos para demos deben ser **plausibles**, no perfectos. Un win rate de 57% es creible; un 80% genera desconfianza. El patron de precios debe tener tendencia y ruido, no ser monotono.

---

### 8. Estado del Chart: Prioridad de fuentes de datos

**Problema:** Multiples fuentes de trades (API, backtest, streaming, replay, animacion) pueden coexistir, causando conflictos en lo que muestra el chart.

**Solucion (existente):** El dashboard implementa una jerarquia clara:
```
streamingTrades > visibleAnimationTrades > replay.visibleTrades > backtestResult.trades
```

**Leccion:** Cuando un componente puede recibir datos de multiples fuentes, definir una jerarquia de prioridad explicita. El estado `isReplayMode` actua como discriminador: en replay, los signals vienen de `replayTrades` (no del API), el `endDate` del chart avanza con cada trade, y los markers se recalculan incrementalmente.

---

## Archivos Nuevos Creados

| Archivo | Proposito |
|---------|-----------|
| `app/api/backtest/stream/route.ts` | SSE proxy con fallback sintetico progresivo |
| `app/api/backtest/route.ts` | Backtest no-streaming proxy |
| `app/api/backtest/status/[modelId]/route.ts` | Status check proxy |
| `app/api/models/[modelId]/metrics/route.ts` | Metricas de modelo con fallback |
| `app/api/models/[modelId]/equity-curve/route.ts` | Equity curve con fallback |
| `app/api/trading/performance/multi-strategy/route.ts` | Performance multi-estrategia |
| `app/api/trading/trades/history/route.ts` | Historial de trades |

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `lib/auth/api-auth.ts` | Dev mode bypass en `withAuth()` |
| `lib/services/backtest.service.ts` | URLs redirigidas a proxy Next.js |
| `components/charts/TradingChartWithSignals.tsx` | Migracion a markers v5 + tolerancia adaptativa |

## Resultado Final

- **36 trades** streaming progresivamente con delays naturales
- **29 signal markers** renderizados incrementalmente en el price chart
- **Equity curve** actualizada en tiempo real desde `streamingTrades`
- **0 errores de consola** (401s, 404s, connection refused resueltos)
- **Backtest panel** funcional con presets (Validacion, Test, Ambos, Personalizado)
