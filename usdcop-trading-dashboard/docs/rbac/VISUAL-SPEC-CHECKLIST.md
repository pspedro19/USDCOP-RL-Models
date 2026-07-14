# VISUAL-SPEC-CHECKLIST — GlobalMarkets Terminal (vista × rol)

> Fuente de verdad: prototipo `mejora-frontend-dashboard-trading/project/GlobalMarkets
> Terminal - Var B.dc.html` (view-model líneas 1674–3259) + Checklist Front-End
> Profesional (gates 🔴). Lo consume la revisión de `npm run qa:visual` (capturas por
> vista × rol: anon / admin / developer / subscriber / free).
>
> Regla: si un valor no sale de un token, o un estado (hover/loading/empty/error) no
> está diseñado, el componente no está terminado.

## Gates 🔴 transversales (toda vista, todo rol)

- [ ] Contraste ≥ 4.5:1 en texto body/small (muted mínimo #94A3B8 sobre #0A0D15)
- [ ] Navegable 100% teclado; `focus-visible` anillo accent en TODO interactivo
- [ ] Touch targets ≥ 44px (topbar, botones de fila, toggles)
- [ ] Loading = skeleton con forma · Empty = causa + acción · Error = mensaje + código
      `UPSTREAM_* · HTTP` + Reintentar (AsyncBoundary)
- [ ] `prefers-reduced-motion` respetado (media query global)
- [ ] Cifras/KPIs/tablas con `tabular-nums` (JetBrains Mono)
- [ ] Sin alturas fijas en cards con texto; zoom 200% sin overflow ni pérdida
- [ ] Cero hex fuera de `lib/ui/gm-tokens.ts`; z-index solo de la escala nombrada
- [ ] Español por defecto; toggle ES/EN vivo (sin strings hardcodeados fuera del dic)

## Chrome (TerminalShell) por rol

| Elemento | admin | subscriber | free |
|---|---|---|---|
| Nav items (orden) | Inicio · Activos · Backtest · **Producción** · Forecasting · Análisis · SignalBridge · Admin | Inicio · Activos · **Señales** · Forecasting · Análisis · SignalBridge* | Inicio · Activos · Forecasting · Análisis |
| Ticker "Mis activos" | símbolos + sparkline | símbolos (vía /api/catalog) | símbolos (vía /api/catalog) |
| Carrito (badge) | ✓ | ✓ | ✓ |
| Identidad | nombre real / iniciales | nombre real | "Invitado / IN" si guest |

*SignalBridge visible para subscriber solo con plan `auto` (execution.enabled).

## Hub por rol

- [ ] Eyebrow "Bienvenido · {rol}"; título "Tu terminal"; liveStats (4 chips del bundle,
      iguales para todos, caption "cifras del bundle publicado")
- [ ] admin: cards catalog+backtest+Producción+forecasting+analysis+execution+admin, ninguna bloqueada
- [ ] subscriber: catalog+Señales+forecasting+analysis+execution; sin backtest/admin
- [ ] free: catalog+forecasting+analysis desbloqueadas; **signals y execution BLOQUEADAS**
      (atenuada + candado + lockMsg exacto: "Activa señales en vivo con un plan." /
      "Ejecución automática — plan Auto." + "Ver planes" → /pricing)
- [ ] Franja "Mis activos" (contador, cards con sparkline si hay datos) + "Gestionar" → /catalog

## Vistas con rama por rol

- [ ] **Backtest** (solo admin/dev por RBAC): bloque aprobación Voto 2/2 SOLO admin;
      gates Voto 1 visibles; KPIs del bundle; replay + rango Desde/Hasta; razones de
      salida con barras; tabla de trades GM
- [ ] **Producción**: título "Monitor de producción" (admin/dev) vs **"Señales"**
      (subscriber); signal-card semanal (dirección/confianza/entrada/TP/SL); posición
      actual + P&L; guardrails; equity con eje Y; N<20 ⇒ solo conteo+PnL
- [ ] **SignalBridge**: subtítulo "OMS global · fan-out a suscriptores" (admin) vs
      "Tu OMS · tus llaves · tu kill switch" (subscriber); form exchange con nota
      AES-256; permisos con "Retiros" bloqueado; kill switch; progreso paper X/4
- [ ] **Forecasting**: free = badge "T-1 semana · plan free" + activos no incluidos con
      panel bloqueado rico (upsell → /pricing); 4 filtros; PNGs con placeholder onError
- [ ] **Catálogo/Cart/Análisis**: idénticas por rol (server aplica delays free)

## Admin (solo admin) — 9 pestañas en orden exacto

- [ ] Resumen · **Ingresos** · Registros · Usuarios · **Modelos** · **Riesgo y bloqueos**
      · **Catálogo** · Sistema · Auditoría (badges numéricos reales en Registros /
      Modelos / Riesgo)
- [ ] Resumen: 4 KPIs negocio + "Pendientes de acción" (chips que navegan a su tab) + alertas
- [ ] Ingresos: layout fiel (KPIs MRR/ARR/ARPU/LTV, ingresos por plan, movimiento MRR,
      cobros, dunning) con valores **"— · Fase 6"** (jamás mocks)
- [ ] Usuarios: tabla con rol/plan/estado/último acceso + acciones **Ver como** (read-only,
      banner + audit) / Editar plan (fase 6) / Suspender
- [ ] Modelos: tabla real (registry+bundles: Sharpe/DA/gates/estado) + candidato con gates
      y deep-link a /dashboard (nunca segundo botón de aprobar)
- [ ] Riesgo: kill global (confirmación tipada) · límites · conexiones API por aprobar ·
      guardrails · bloqueados — datos SignalBridge reales
- [ ] Catálogo: activos del registry con toggle (deshabilitado + nota hasta registry mutable)
- [ ] Sistema: stack con deep-links · SLOs p50/p95/p99 · targets Prometheus (UP/DOWN) ·
      alertas AlertManager · pipeline Airflow L0–L6 · recursos · logs → Grafana Explore
- [ ] Auditoría: filtros + email resuelto + export CSV (ya as-built)

## Invitado (guest)

- [ ] Sin sesión: solo landing/login/register/pricing; ruta protegida → /login con
      `?next=` y nota "Inicia sesión para continuar"
- [ ] "Explorar como invitado" en Landing, Login y Register-done → sesión demo rol free
      (guest@demo.local, is_test) → hub con nav de 4 y cards bloqueadas
- [ ] Guest NUNCA ve Backtest/Admin ni en nav ni en hub

## Públicas

- [ ] Landing: header público (logo glifo degradado · Planes · idioma · Iniciar sesión ·
      Crear cuenta degradado) · hero · terminal demo marcada "Vista demo" · features ·
      cómo funciona · métricas LIVE del bundle · CTA · footer disclaimer
- [ ] Pricing: 3 tiers (badge "MÁS POPULAR" en Señales Pro) + add-ons por activo · plan
      actual resaltado si hay sesión
- [ ] Login/Register/Reset: card GM 420px, captcha, testids E2E intactos

## NO replicar (herramientas demo del prototipo)

- Tabs de data-state del sidebar (loading/empty/error demo)
- Quick-logins de rol en login (solo el botón invitado es real)
- Números mock de negocio ($49,6M MRR, 1.284 usuarios…)
