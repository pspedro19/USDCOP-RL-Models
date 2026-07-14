# UX/IA — Navegación Definitiva, Landing y Contenido por Rol

> Complementa `rbac-monetization.md` (CTR-RBAC-001): aquel define *quién puede ver qué*;
> este define *qué se ve y cómo*. Estándar: producto cuant institucional (Darwinex/Numerai/
> TradingView). Diferenciador: **rigor verificable**, no promesas de retorno.
> Cualquier página nueva entra PRIMERO a la matriz RBAC y después se diseña aquí.

## 0. Tres principios
- **P1 Un solo número:** toda métrica viene del bundle publicado (nunca recomputada);
  tooltip de procedencia `fuente: estrategia · versión · bundle · fecha`.
- **P2 Badges LIVE/BACKTEST/PAPER siempre:** `● LIVE` verde (forward) · `◆ BACKTEST` azul
  (OOS) · `○ PAPER` ámbar. El marketing SIEMPRE lidera con LIVE. Sin badge no se publica.
  **AS-BUILT: `components/ui/MetricBadge.tsx` + tokens en `lib/contracts/ui.contract.ts`.**
- **P3 Divulgación progresiva por rol:** cliente jamás ve jerga interna (L4/votos/gates);
  free ve el producto real retrasado + candados con contenido detrás (blur), no vacío.

## 1. Mapa del sitio
PÚBLICO: `/` landing · `/metodologia` (**AS-BUILT** — arma de ventas) · `/pricing` ·
`/legal/*` (pendiente) · `/login`. APP (deny-by-default): `/hub` por rol · Señales
(ex-Producción, subscriber) · `/forecasting` + `/analysis` (frescura por plan) ·
SignalBridge (auto propio / admin global) · `/dashboard`+experimentos (dev/admin) ·
`/admin` (pendiente) · `/account`.

## 2. Landing (S1-S10)
Navbar sticky 1 CTA · Hero honesto ("Señales cuantitativas USD/COP verificadas en
producción... publicamos resultados ganen o pierdan") + widget vivo · Barra de confianza
SOLO ● LIVE · Cómo funciona (modelos→gates→señal; "la mayoría de semanas NO se opera") ·
Track record tabs LIVE/BACKTEST · Producto (capturas blur) · Metodología (4 mini-cards:
pre-registro/validación/paper→prod/protocolo de retiro) · Pricing · FAQ (incómodas
primero) · Footer legal. La landing existente (`components/landing/*`) se adapta sección
por sección a esto (pendiente).

## 3. Contenido por rol (resumen; matriz completa = rbac.contract.ts)
- **free:** Forecasting T−1 (joya de conversión: "Forecast vs Realidad"), Análisis resumen
  T+7 SIN escenarios de entrada, cards 🔒 con blur del contenido real de la semana.
- **signals:** franja de estado de la semana ("Señal activa…" / "Sin operación — gate de
  régimen"), página Señales (sin badges de aprobación ni jerga), forecasting/análisis al día.
- **auto:** + SignalBridge propio: wizard (verify anti-withdraw → límites solo-hacia-abajo →
  risk disclosure → paper obligatorio con progreso "semana 2 de 4"), P&L propio con badges
  paper/live separados, SU kill switch grande y rojo.
- **developer:** laboratorio completo sin "Promover" (no se renderiza) ni Vote 2 operable.
- **admin:** + Vote 2, Promover (modal con diff + audit), SignalBridge global, `/admin`
  (usuarios/entitlements/auditoría/MRR/semáforo freshness).

## 4. Escenarios transversales
Mercado cerrado = franja neutra + countdown (nunca parecer roto) · Semana sin operación =
decisión del sistema, no ausencia · Plan vencido = degradar a free, nunca borrar histórico ·
Kill switch activo = banner rojo persistente · Datos stale = "señales pausadas por
seguridad" (honestidad como feature) · Carga = skeletons con forma · Error = causa + acción.

## 5. Pricing
Free "Conoce el sistema" · Signals [Más popular] "Opera con el sistema" · Auto "El sistema
opera por ti". Add-ons por activo = fila aparte (mapean 1:1 al registro). Conversión:
free→signals = Forecast-vs-Realidad + candados blur; signals→auto = "esta señal se habría
ejecutado sola" con P&L real.

## 6. Gobernanza visual de métricas (CONTRACTUAL)
1. Badge LIVE/BACKTEST/PAPER en toda cifra de rendimiento (MetricBadge). 2. Tooltip de
procedencia. 3. Frontend NUNCA recalcula desde trades (regla ya enforced — Vote-2-del-
bundle). 4. N<20 trades ⇒ solo conteo y PnL. 5. Copy sin mecanismos internos (RL/PPO/L4)
— hub ya corregido.

## 7. Ejecución
Landing+/metodologia primero (estáticas, venden) · vistas por rol dependen de RBAC F1-4
(hechas) · SignalBridge cliente depende de R5 · ES primario, i18n preparado (next-intl,
no traducir aún) · Mobile: bottom-tabs, kill switch ≤2 taps · LCP<2.5s landing · AA
contraste sobre tema oscuro (grises del hub actuales no pasan — corregir con tokens).

## AS-BUILT ledger

> **2026-07-10 — GlobalMarkets Terminal (CTR-GM-UI-001):** toda la app migró al chrome GM
> (sidebar + topbar `TerminalShell`, tokens Var B, estados de datos envelope-driven).
> Este ledger describe las pasadas previas; el estado actual por vista está en
> `gm-terminal-migration.md`. Los principios P1/P2/P3 de este spec siguen vigentes y
> los cumple el DS GM (badges ●◆○ reutilizados, cifras del bundle, divulgación por rol).
| Pieza | Estado |
|---|---|
| `lib/contracts/ui.contract.ts` (tokens + procedencia) | ✅ 2026-07-06 |
| `components/ui/MetricBadge.tsx` (P1+P2) | ✅ 2026-07-06 |
| `/metodologia` pública | ✅ 2026-07-06 |
| **S1** Navbar (Metodología·Resultados·Precios) | ✅ pasada 2 |
| **S2** Hero honesto + CTAs /login+/metodologia | ✅ desplegado |
| **S3** TrustBar solo-●LIVE (`/api/public/live-stats` — agregados del bundle, sin señales) | ✅ pasada 2 |
| **S5+S7** TrackRecord tabs LIVE/BACKTEST + mini-cards metodología (`TrackRecord.tsx`) | ✅ pasada 2 |
| **S9** FAQ incómodas (¿puedo perder? / retiro / custodia / asesoría / cancelar) EN+ES | ✅ pasada 2 |
| **S10** footer tagline honesto · **/legal/{terminos,riesgo,privacidad}** (ruta dinámica DRY) | ✅ pasada 2 |
| **§3.5 /admin** (usuarios+plan mix+auditoría append-only, skeletons, `/api/admin/overview`) | ✅ pasada 2 (v1 read-only; editor de entitlements = incremento futuro) |
| **§6.1-6.2** MetricBadge cableado: ● LIVE en /production · ◆ BACKTEST en /dashboard | ✅ pasada 2 |
| **§3.1** cards 🔒 con blur real + microcopy de conversión (hub, roles cliente) | ✅ pasada 2 |
| S4 Cómo-funciona reescrito · S6 capturas blur · franja de estado semana · wizard auto · bottom-tabs · i18n · Forecast-vs-Realidad | ⏳ pasada 3 |
