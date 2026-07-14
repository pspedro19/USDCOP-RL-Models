# GlobalMarkets Terminal — migración de UI + contrato FE↔BE

> **Estado: FASES 1–3 IMPLEMENTADAS Y VERIFICADAS (2026-07-11)** — E2E 20/20 y 0 errores
> de consola tras cada fase; smoke funcional de catálogo/watchlist con persistencia en DB;
> **Fase 3 = fidelidad pixel al prototipo por vista × rol** (re-skin profundo Backtest/Análisis/
> Execution, admin de 9 pestañas, "Ver como", invitado demo, i18n ES/EN vivo, checklist
> front-end profesional); `npx tsc` 0 errores de app, `next build` exit 0, `rbac:check` OK
> (83 API / 30 páginas), cero hex fuera de `gm-tokens.ts` en superficies GM activas.
> capturas en `shots/gm-terminal/` y `shots/gm-final/`. Contract: CTR-GM-UI-001 · Basado en el handoff
> `usdcop-trading-dashboard/mejora-frontend-dashboard-trading/` (prototipo
> `GlobalMarkets Terminal - Var B.dc.html`, hi-fi) y en **CTR-FE-BE-001**
> (`frontend-backend-contract.md`, ya en `.claude/specs/platform/`).
>
> Estrategia: **strangler** — la nueva UI toma las rutas reales página a página; cada
> versión vieja queda archivada en `/legacy/<ruta>` (admin-only, matriz RBAC prefijo
> `/legacy`). El chrome nuevo (`TerminalShell`) reemplaza a `GlobalNavbar` en toda la app.

## Fundación (as-built)

| Pieza | Archivo | Qué es |
|---|---|---|
| Contrato FE↔BE | `.claude/specs/platform/frontend-backend-contract.md` + `docs/api/openapi.yaml` + `lib/api/_reference/bff-handler-reference.ts` | CTR-FE-BE-001 (Camino A del handoff aplicado) |
| Envelope | `lib/api/envelope.ts` | `ok()/fail()` + `UpstreamError` + códigos `UPSTREAM_*` (server) |
| Relay BFF | `lib/api/relay.ts` | `requirePermission()` (matriz CTR-RBAC-001 vía headers anti-spoof, NO role-rank) + `relayUpstream()` (timeout, bearer, traceparent, Idempotency-Key) |
| Cliente | `lib/api/gm-client.ts` | `apiFetch()` envelope-aware con compat legacy (strangler) + `goToLogin()` (401→`/login?next=`). El viejo `lib/api/client.ts` (zod) queda para vistas legacy |
| Tokens | `lib/ui/gm-tokens.ts` | GM/GMT: paleta Var B (#0A0D15, paneles rgba(14,21,38,.72), acento #22D3EE→#3B82F6, pos #34D399/neg #FB7185), tipos, tonos. Cero hex fuera de aquí en código GM |
| Estados de datos | `components/gm/AsyncBoundary.tsx` | Skeleton→Empty→ErrorRetry (código `UPSTREAM_*` + HTTP + Reintentar), prototipo líneas 130–159 |
| Data hook | `components/gm/useGmQuery.ts` | apiFetch + abort-on-unmount + polling + stale-while-error |
| Chrome | `components/gm/TerminalShell.tsx` | Sidebar colapsable (nav por matriz RBAC, relabel Señales, plan-gate SignalBridge) + topbar (ticker registry, Planes) + JetBrains Mono |
| Primitivas | `components/gm/primitives.tsx` | GmPanel/GmKpi/GmBadge/GmPageHeader/GmDelta |

## Estado por vista

| Vista | Ruta | Estado | Nota |
|---|---|---|---|
| Hub | `/hub` | ✅ GM (rebuild) | cards por permiso, KPIs reales, activos del registry |
| Producción/Señales | `/production` | ✅ GM (rebuild) | KPIs del bundle (P1), canShowRatios N<20, EXIT_REASON_COLORS |
| Forecasting | `/forecasting` | ✅ GM (rebuild) | contrato §5 intacto (delay por plan, layout PNG/JSON) |
| Landing | `/` | ✅ GM (rebuild) | header público + hero + demo card |
| Pricing | `/pricing` | ✅ GM (rebuild) | 3 planes |
| Backtest | `/dashboard` | ✅ GM re-skin (Fase 3) | panels GM, 0 clases slate; Vote-2 intacto (números del bundle); bloque aprobación solo admin |
| Análisis | `/analysis` | ✅ GM re-skin (Fase 3) | 23 componentes migrados a gm-tokens + AsyncBoundary; sparklines; 0 hex propio (paleta de charts = `gm-analysis.tsx`) |
| Admin | `/admin` | ✅ GM re-skin (Fase 3) | **9 pestañas** (Resumen·Ingresos·Registros·Usuarios·Modelos·Riesgo·Catálogo·Sistema·Auditoría) + "Ver como" read-only auditado; `lib/ui/tokens.ts` = adapter deprecado que compone gm-tokens |
| SignalBridge | `/execution/*` | ✅ chrome GM (Fase 2) | layout del grupo en TerminalShell + tab-strip GM; lógica WS/kill-switch intacta; `/execution/login` fuera del shell |
| Login/Register/Reset | `/login`, `/register`, `/reset-password` | ✅ GM (Fase 2) | re-skin con lógica verbatim; testids del E2E verificados contra todos los harnesses; `?next=` ahora respetado |
| Catálogo/Watchlist/Carrito | `/catalog` + `/api/{catalog,watchlist,cart}` | ✅ full-stack (Fase 2) | migración 057, BFF **enveloped** (`ok/fail` + `requireSession`), contrato `catalog.contract.ts`, CartDrawer en el shell, checkout → `lib/billing` (DIP); cart NO se limpia hasta webhook (regla 7) |
| Sistema (observabilidad) | `/admin` (Sistema) | ✅ (Fase 2) | semáforo + servicios + **deep-links** Grafana/Prometheus/AlertManager/Jaeger/Airflow/MLflow (`NEXT_PUBLIC_*_URL`) |

## Fase 2 (2026-07-10, mismo día) — as-built

- **Catálogo/Watchlist/Carrito (§4.3)**: migración `057_catalog_watchlist_cart.sql`
  (aplicada), contrato `lib/contracts/catalog.contract.ts`, servicio SSOT
  `lib/services/catalog-registry.ts` (deriva del registry; coming-soon estáticos
  marcados), BFF **enveloped** `/api/{catalog,watchlist,cart,cart/checkout}` con
  `requireSession()` (nuevo en `lib/api/relay.ts`), página `/catalog`
  (`CatalogView`) + `CartDrawer` montado en el shell (badge + `CART_CHANGED_EVENT`).
  Checkout llama `getBillingProvider().createCheckout()` directo (DIP, sin self-HTTP);
  el carrito NO se limpia hasta el webhook (regla 7 CTR-RBAC-001). Smoke verificado:
  watch-toggle persiste en `user_watchlist`.
- **Auth GM**: login/register/reset re-skineados con lógica verbatim (captcha, lockout,
  temp-password, minteo NextAuth); selectores E2E verificados contra todos los harnesses;
  `?next=` de `gm-client.goToLogin()` ahora respetado.
- **SignalBridge**: `app/execution/layout.tsx` monta TerminalShell (tab-strip GM para
  sub-secciones; `/execution/login` fuera del shell); kill-switch/WS/forms intactos.
- **Sistema**: deep-links Grafana/Prometheus/AlertManager/Jaeger/Airflow/MLflow en la
  consola admin (`NEXT_PUBLIC_*_URL` con defaults localhost).
- **Limpieza anti-redundancia**: eliminado `lib/api/client.ts` (cliente zod pre-contrato,
  **cero consumidores**) y su barrel reescrito a la superficie CTR-FE-BE-001; borrados
  duplicados del handoff (`* (2).*`) y `.tmp_*.mjs`; `GlobalNavbar` ya solo lo referencian
  las copias `/legacy`.

## Fase 3 (2026-07-11) — fidelidad por vista × rol + checklist front-end (as-built)

Objetivo: cerrar la brecha entre el chrome-swap de Fase 1–2 y el prototipo hi-fi, en
especial las **variantes por rol** (admin/subscriber/free/invitado) que difieren por página.
Ejecutada en subfases F0–F7 con archivos disjuntos.

- **F0 harness**: `docs/rbac/VISUAL-SPEC-CHECKLIST.md` (checklist por vista × rol + ítems 🔴
  del checklist profesional: contraste ≥4.5, teclado/focus, targets ≥44px, estados de UI,
  reduced-motion, zoom 200%, tabular-nums, sin alturas fijas). Baseline `qa:visual`.
- **F1 base**: `gm-tokens.ts` v2 (tipografía rem+`clamp()`, motion tokens 150/250/400ms,
  z-index nombrado, `textMuted` #7C8AA3 AA-compliant); `globals.css` primitivos CSS-vars
  `--gm-*` + `prefers-reduced-motion` global + `.gm-contain`/`.gm-prose`. **Único** archivo
  con hex propio = `gm-tokens.ts` (paletas de charts JS son carve-out: `gm-analysis.tsx`,
  `components/charts/*`, `viewport.ts`, payload `api/models`).
- **F2 shell + i18n**: `Spark.tsx` (sparkline determinista), ticker para roles cliente vía
  `/api/catalog` (arregla ticker vacío), nav con **Activos** para todos, **i18n ES/EN vivo**
  (`lib/i18n/gm-core.ts` + diccionarios `gm.ts`; toggle con persistencia; Execution con
  `EXEC_DICT`/`t()` — cero copy hardcodeado).
- **F3 variantes por rol + invitado**: HubView (card Activos todos, Backtest solo admin,
  lockMsg exactos), ProductionView (signal-card subscriber), ForecastingView (bloqueo rico
  free), **invitado demo** (bootstrap_guest en SignalBridge → rol free `is_test=TRUE`,
  `/api/auth/guest`, botones Landing/Login con testids nuevos).
- **F4 re-skin profundo**: Backtest, Análisis (23 comp.), Execution (+español) — slate→GM
  mecánico + AsyncBoundary; 0 clases legacy en las tres superficies.
- **F5 admin fiel**: 9 pestañas en orden; endpoints enveloped nuevos `/api/admin/{models,
  risk,revenue,catalog}` (+`impersonate`) con `requirePermission('admin:all')`; secciones
  Modelos (deep-link a `/dashboard`, **sin** aprobar aquí — regla una-sola-superficie),
  Riesgo (kill global confirm "STOP" + audit_log), Ingresos ("— · Fase 6", cero mocks),
  Catálogo (solo lectura); **"Ver como"** cookie firmada `gm-view-as` (HMAC) httpOnly +
  espejo legible `gm-view-as-role`, banner persistente "solo lectura", nav por rol simulado
  **sin** conceder permisos server-side; `lib/ui/tokens.ts` → adapter deprecado que compone
  gm-tokens (un solo design system).
- **F6 carrito**: `useCart()` compartido, `app/cart/page.tsx`, add-ons en pricing.
- **F7 verificación**: `tsc` 0 (app), `next build` exit 0, `rbac:check` OK (83/30), hex=0
  en superficies GM activas, Docker dashboard rebuild + stack healthy.

**"Ver como" — seguridad (importante):** la simulación es **solo de navegación/lectura**.
La cookie firmada NO toca `x-user-role`; toda mutación sigue exigiendo el rol REAL del admin
(el servidor no cambia). Activar deja fila `view_as_start` en `audit_log`; salir → `view_as_end`.

## Fase 3.1 (2026-07-11, tarde) — responsive real + "Ver como" en el shell + guardarraíles

- **Centrado en pantallas anchas (bug reportado)**: el contenido salía pegado al sidebar
  (~40% de un 2560px en negro). Causa: `w-full` + `mx-auto` (width:100% resuelve los
  auto-margins a 0 antes del `max-width`) — en `<main>` de `TerminalShell`, `LandingView`
  y `PricingView` (las públicas tienen su propio `PublicChrome`, por eso el fix del shell
  no las cubrió). Fix universal: `<main className="flex justify-center">` + hijo con
  `max-w` (nunca `w-full mx-auto`). Verificado a 2560/1920/1440 (gaps L=R).
- **Ancho por tipo de página**: prop `width` en `TerminalShell` (`default` 1600 · `wide`
  1860 · `full`). Análisis y Backtest usan `wide` (densos en gráficos/tablas).
- **"Ver como" en el menú lateral (fiel al prototipo)**: antes solo estaba enterrado en la
  pestaña Usuarios; ahora hay un `<select data-testid="view-as-switch">` en el footer del
  sidebar, **visible solo para el admin REAL** (`sessionRole === 'admin'` vía `useSession`,
  no la cookie → siempre puede volver). Cambiar de opción hace POST/DELETE a
  `/api/admin/impersonate` y recarga; el banner read-only sigue arriba.
- **aria-live en KPIs**: `GmKpi` envuelve el valor con `aria-live="polite"` (anuncia solo
  al cambiar; silencioso para KPIs estáticos — checklist 14.5).
- **Container queries (fundación)**: utilidades `.gm-cq` / `.gm-auto-grid` en `globals.css`
  (checklist 3.2/3.5) — disponibles; no se forzó sobre grids que ya funcionan.
- **Limpieza**: borrado `components/execution/_pages_reference/` + `ConnectExchangeForm`
  (huérfanos reales). `components/forecasting/{ForecastingDashboard,WeeklyInferenceView}` y
  `components/landing/*` **se conservan** — los usan las rutas `/legacy/*` (admin-only).
- **Guardarraíles nuevos**: `visual-qa.mjs --zoom` (viewport 720px = reflow 200%, falla si
  hay overflow horizontal — WCAG 1.4.4); `.stylelintrc.json` (`color-no-hex` warning,
  checklist 18.3) + `npm run lint:css`; `.github/workflows/a11y.yml` (axe-core sobre páginas
  públicas en cada PR, checklist 18.1). `qa:visual` ahora corre `--mobile --wide --zoom`.

## Gaps restantes (candidatos a próximos PR)

1. **Envelope en el BFF legacy**: `/api/public/live-stats`, `/api/registry`, admin, etc.
   devuelven JSON crudo — migrar handler a handler a `ok()/fail()` (el cliente ya tolera ambos;
   los endpoints de §4.3 ya nacen enveloped).
2. **Precio/variación por activo**: falta endpoint público de última cotización por asset
   (alimentaría ticker, catálogo y hub) — hoy el catálogo muestra "—" honesto.
3. **Precios de planes/add-ons**: placeholders (`DEFAULT_PRICES_COP_CENTS`) — decisión de
   negocio; Wompi hoy cobra solo el plan (add-ons sin monto en `amountInCents`).
4. **Limpieza del carrito post-pago**: añadir al handler del webhook de billing (donde se
   otorgan entitlements).
5. **Idempotency-Key con dedup real** en checkout/promote (hoy idempotencia efectiva).
6. ~~i18n chip "ES" decorativo~~ **RESUELTO (Fase 3)**: toggle ES/EN vivo con persistencia.
   **liveStats multi-asset**: hub agrega solo COP activa (sigue pendiente).
7. **Retiro de `/legacy`**: borrar copias + `components/navigation/GlobalNavbar.tsx` cuando
   la GM cumpla un ciclo sin regresión.
8. **Componentes forecasting huérfanos**: `components/forecasting/{ForecastingDashboard,
   WeeklyInferenceView}.tsx` ya NO se renderizan (`/forecasting` monta `ForecastingView`; el
   barrel solo exporta `/types` en uso) — conservan hex/inglés legacy. Borrar o re-skinear si
   se re-cablean.
9. **Ingresos/Modelos-DA reales**: `/api/admin/revenue` = "— · Fase 6" hasta billing Wompi;
   `da_pct` null hasta que el registry lo publique. Layouts fieles y listos.

## Lección de build (2026-07-10, no repetir)

**Tailwind v4 auto-detección NO emitió algunas clases arbitrarias** de `TerminalShell`
(`w-[248px]`, `lg:pl-[248px]`) aunque otras del mismo archivo sí compilaron — el
contenido quedó debajo del sidebar y rompió el E2E del admin. Fix doble (belt & braces):
1. `@source "../components"; @source "../lib"; @source "../app";` explícitos en
   `app/globals.css`.
2. El layout load-bearing del shell vive en **CSS explícito** (`.gm-sidebar`,
   `.gm-content` + data-attrs), nunca en utilidades escaneadas.
Regla: cualquier clase cuyo fallo rompa navegación/layout estructural va a CSS explícito.

## Lección de centrado (2026-07-11, no repetir)

**`w-full` + `mx-auto` NO centra** (contenido pegado a la izquierda, ~40% de un
monitor 2560px en negro). `width:100%` resuelve los `margin-inline:auto` a **0**
*antes* de que `max-width` recorte la caja → queda anclada a la izquierda (pegada al
sidebar fijo de 248px). Un `mx-auto` de **flex-item** en un contenedor
`flex-direction:column` (`.gm-content`) también resuelve a 0. Fix aplicado en
`TerminalShell.tsx` `<main>`: envolver los children en un bloque
`max-w-[1320px]` centrado con **`justify-content:center` en el `<main>` flex**
(no depende de auto-margins). Verificado con getBoundingClientRect a 2560/1920/1440
(gaps L=R). Regla: para centrar una columna con tope de ancho, `flex + justify-center`
en el padre, **nunca** `w-full mx-auto` en el hijo. Tope de contenido subido
**1320→1600px** (mejor uso de monitores anchos sin estirar texto; sigue centrado).
Guardrail: `scripts/visual-qa.mjs --wide` (ya en `npm run qa:visual`) captura a 2560px
y **falla si el gap izq/der difiere >8px** — el harness a 1440 fijo no detectaba el bug
(a 1440 la columna llena el área y parece bien).

## Reglas de la migración

1. Toda vista GM: `TerminalShell` + `useGmQuery` + `AsyncBoundary` por bloque de datos;
   colores solo de `gm-tokens.ts`; números con `GMT.mono` (JetBrains Mono tabular).
2. Contratos que NO se rompen (CTR-FE-BE-001 §5): delay forecasting por plan, registry
   SSOT, análisis por activo, tablas `forecast_h5_*`, mismo builder Backtest/Producción.
3. `data-testid` existentes se preservan (E2E `registration-qa.mjs` y QA visual).
4. Página archivada = copia en `components/legacy/<Nombre>Legacy.tsx` + ruta
   `/legacy/<ruta>` (admin-only). Se borra cuando la vista GM cumpla un ciclo sin regresión.
5. El BFF migra al envelope endpoint a endpoint; `apiFetch` tolera raw JSON legacy
   mientras tanto (strangler en ambas capas).

---

## Fidelity pass — as-built (2026-07-11, plan `valiant-purring-crane`)

The GM migration reached pixel-fidelity to prototipo **Var B**. Harness + gates:
`docs/rbac/VISUAL-SPEC-CHECKLIST.md` (vista × rol), consumido por `npm run qa:visual`.

**Fundación (F1) — `lib/ui/gm-tokens.ts` v2 + `app/globals.css`:**
- Tipografía `GMT` en **rem + `clamp()`** (WCAG 1.4.4 bajo zoom 200%); primitivos CSS
  vars `--gm-*` en `:root`; **motion tokens** (`--gm-dur-*`/`--gm-ease-*` → `MOTION`) y
  **z-index nombrado** (`--z-sticky..toast` → `Z`); `prefers-reduced-motion` global.
- **Gate "cero hex fuera de `gm-tokens.ts`"**: cumplido en TODA superficie viva
  (`components/gm`, `admin`, `analysis`, `production`, `charts` = 0 hex). Los valores
  crudos que Recharts/lightweight-charts exigen viven en **`GM_HEX`/`GM_CHART`/`GM_TV`**
  (único hogar de hex). Solo dirs pre-GM archivados (`legacy`, `landing`/`navigation`/
  `mlops`/`monitoring`/`forecasting` viejos, sin importadores vivos) conservan hex.

**Shell + i18n (F2):** ticker por rol (research→`/api/registry`, cliente→`/api/catalog`),
`Activos`/catalog en nav para todos; i18n ES/EN vivo (`lib/i18n/gm.ts` + `gm-core.ts`,
`useGmT/useGmLang`). Sparkline fiel = `components/gm/Spark.tsx` (56×22 / 84×34, modo
serie-real que nunca inventa datos), **re-exportado como `Sparkline.tsx`** (nombre del
checklist).

**Variantes por rol (F3):** Hub (cards + lockMsgs free), Producción ("Señales" vs
"Monitor"), Forecasting (bloqueo rico upsell free). **Invitado**: `POST /api/auth/guest`
(server-side, sin captcha — no reenvía credenciales) → SignalBridge `bootstrap_guest()`
(rol free, approved, `is_test=true`, hash bcrypt correcto en backend; **no** hay migración
SQL dashboard-side por diseño — evita duplicar hashing). Botones en Landing/Login.

**Re-skin (F4) + Admin (F5) + Cart (F6):** Backtest/Análisis/Execution a 0 clases `slate-`
(GM tokens + i18n ES); Admin 9 pestañas orden exacto sobre gm-tokens (`lib/ui/tokens.ts` =
adapter `@deprecated` que compone de gm-tokens, cero hex propio); "Ver como" admin-only
read-only auditado (`/api/admin/impersonate`); `/cart` (plan radio + add-ons + total
sticky), `useCart()` compartido, add-ons por activo en Pricing.

**Gates F7 verdes:** tsc app-scope 0 · `rbac:check` 91 rutas/30 páginas · hex-gate live 0 ·
`role-matrix-qa` 61/0 · `registration-qa` 20/20.
