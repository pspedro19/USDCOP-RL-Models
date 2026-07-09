# PLAN FINAL — QA Funcional+Visual+Suscripciones hasta 100/100

> Consolidación 2026-07-06. Método: iterar (capturar fullPage + console logs + docker logs +
> probes funcionales) → hallazgos → batch fix → build → re-iterar. Parada = 100/100.
> Harness: `scripts/visual-qa.mjs` (v2 fullPage+console) + `scripts/functional-qa.mjs`.
> Checklist BDD: `docs/rbac/VISUAL-SPEC-CHECKLIST.md`. Usuarios: admin/Admin2026!,
> {dev,pro,free}@test.com / Test2026!.

## A — Visual (iter-1 COMPLETADA: 11 hallazgos, batch aplicado)
✅ login .local→.com · badges ●LIVE production · terminal col width · billing copy ·
TrackRecord centrado · #6 hub plan-lock SignalBridge · #9 vista "Señales" subscriber
(h1, sin Pendiente/PENDING_APPROVAL) · #10 ratios N<20 cliente · #11 Promover solo-admin ·
harness fullPage (scroll down/up) + console capture. → build iter-2 y re-verificar.

## B — Funcional página-por-página (`functional-qa.mjs`, con logs)
| F | Flujo | Método |
|---|---|---|
| F1 | Registro nuevo usuario → aprobación admin → temp-pw → reset (flujo Phase 8) | API SignalBridge + docker logs |
| F2 | Login/logout 4 roles + landing correcto | browser |
| F3 | Replay backtest: play, versión, curva avanza | browser click + screenshot mid-replay |
| F4 | Vote 2 Aprobar → approval_state → deploy | ⚠️ probar spawn python en imagen standalone (hallazgo probable: no hay python en el contenedor node) |
| F5 | Promover versión y RESTAURAR (ciclo completo) | API admin + registry check |
| F6 | SignalBridge+MEXC: tenant/me/keys (anti-withdraw), limits (techos), kill on/off, fan_out→user_executions, ejecución PAPER simulada (PreTradeGate) | API con JWT + BD + docker logs signalbridge |
| F7 | Delays por plan servidos (free: forecast T−1, análisis T+7) | API con sesión free |

## C — Suscripciones end-to-end (por tipo de plan)
| C | Entregable | Estado base |
|---|---|---|
| C1 | **Registro self-serve tier free** (signup → rol free auto-aprobado, sin admin) — el registro actual exige aprobación admin (pensado para operadores); clientes free entran solos | construir (flag en register: `self_serve=free`) |
| C2 | **Compra por plan**: /pricing → checkout (plan signals/auto + add-ons Oro/BTC) → webhook firma-verificada → entitlements | ✅ construido; probar con **webhook simulado** (sin claves Wompi reales) |
| C3 | **Vistas/funcionalidades POR PLAN activadas** (no solo por rol): free=retrasado+locks · signals=señales al día+notifs · auto=+SignalBridge propio (wizard paper-first) · add-ons=assets[] gatean análisis/forecasting por activo | núcleo ✅ (delays, plan-lock hub, asset-gate /api/data); falta: gate por-asset en forecasting, notifs, wizard auto |
| C4 | **Upgrade/downgrade en vivo**: pro@test → webhook simulado plan auto → hub desbloquea SignalBridge sin redeploy; vencimiento → degrada a free y las vistas lo reflejan | probar (lazy degradation ya ✅) |
| C5 | **/admin editor de entitlements** (cambiar plan/vencimiento manual + audit) | construir (v1 read-only ✅) |

## D — Logs integrales por iteración
Console web (report.json) + `docker logs usdcop-dashboard --since` + `docker logs usdcop-signalbridge --since` + salida de probes → TODOS los hallazgos con severidad → batch → build → repetir.

## Orden de ejecución
1. Build iter-2 (batch A) → visual re-verify (fullPage, 4 roles)
2. functional-qa.mjs F1-F7 + C2/C4 (webhook simulado) → hallazgos → batch → build
3. C1 (self-serve free) + C3 restantes (asset-gate forecasting, wizard) + C5 (editor admin)
4. Iterar hasta: visual convergido + F1-F7 verdes + C1-C5 verdes = **100/100**

## ITER-2 RESULTADOS (2026-07-06) — 8/14 PASS
✅ F1 registro(202) · F2 logins · F3 replay click+badge · F5 promote+restore(200/200)
❌ **BLOQUEANTE SEC**: POST /api/tenant/system/kill devolvió 200 A UN SUBSCRIBER y activó
   global_kill (revertido por SQL). Causa a confirmar: role-check en tenant.py no dispara
   (¿user.role enum vs 'admin' string? ¿SB conectado a OTRA database sin las tablas 055 →
   comportamiento indefinido?). Diagnóstico parcial: GET me/limits = 500 con
   `invalid UUID '1'` en logs SB → las rutas tenant fallan contra la DB que SB usa.
   **SIGUIENTE PASO OBLIGADO**: verificar SB DATABASE_URL (¿db propia signalbridge vs
   usdcop_trading?); si es otra DB → aplicar migración 055 allí O apuntar las queries tenant
   a la DB correcta; corregir role-check (comparar str(user.role).lower().endswith('admin'));
   test unitario de deny; re-correr functional.
❌ F7 free análisis actual = 500 (no 403): revisar handler /api/data (getEntitlements/
   isFresherThanAllowed crash — leer docker logs dashboard con -A10 en el stack).
❌ hallazgo #12 logs: column role en users legacy — FIX APLICADO (compat cols + mig 055 §5).
Pendiente iter-3: fixes F6(sec)+F7 → redeploy SB → functional re-run → visual iter-2 review
(capturas en scratchpad/iter2) → C1-C5 suscripciones.

### HIPÓTESIS ROOT-CAUSE F6 (confirmar en iter-3)
`user_risk_limits_v2` = 0 filas (kill global sin daño). SB DATABASE_URL = usdcop_trading
(misma DB, tablas existen). El error `invalid UUID '1'` ⇒ **user.id = 1 (INT)** en las rutas
tenant ⇒ `get_current_active_user` (app/middleware/auth.py) resolvió al ADMIN de la tabla
LEGACY `users` (id=1, is_admin) en vez del sb_user del token — por eso system/kill dio 200
(pasó el check como "admin" legacy) y me/limits 500 (str(1) no es UUID).
**FIX iter-3**: inspeccionar app/middleware/auth.py — hay doble resolución de usuario
(compat dashboard); las rutas tenant deben usar la dependencia que resuelve sb_users por el
`sub` UUID del token SB (la misma que usan las rutas exchanges que sí funcionan) + endurecer
el role-check (`str(user.role).lower()`) + test de deny. Después: docker cp tenant.py+fix →
restart SB → re-run functional-qa → esperar 14/14.

## E — Auditoría E2E de PIPELINES DE DATOS (por estrategia, con logs antes/durante/después)
| E | Flujo | Verificación |
|---|---|---|
| E1 | Trigger DAGs: `asset_xauusd_pipeline_weekly`, `asset_btcusdt_pipeline_weekly` (l0_ingest→l0c_derivatives(BTC)→l0b_chart→l4_backtest_publish→l5_forecast→l6_verify), `core_l0_02_ohlcv_realtime`, `core_l0_04_macro_update` | `airflow dags trigger` + estado por task |
| E2 | Logs por task: ANTES (estado previo de seeds/tablas), DURANTE (`airflow tasks logs` / docker logs scheduler), DESPUÉS (exit + duración) | grep de errores/warnings por task |
| E3 | Entradas→salidas cuestionadas: seeds mtime+filas Δ, `crypto_derivatives_daily` filas nuevas, bundles nuevos en registry (versión/inmutabilidad), `weekly_inference_*.json` regenerado, `asset_daily_ohlcv` Δ | SQL + fs diff pre/post |
| E4 | El dashboard refleja los outputs: screenshots de /dashboard (bundle nuevo en selector), /forecasting (inference fresca) con console logs | visual-qa dirigido |
> Cadencia: correr E tras cada iteración funcional verde; TODO hallazgo (task fallida, output
> vacío, freshness violada, error en console web) entra al batch de la iteración.

## LEDGER DE HALLAZGOS iter-2/3/4 (2026-07-06/07) — todos con fix
| # | Sev | Hallazgo | Fix | Config/Archivo |
|---|---|---|---|---|
| 12 | MAYOR | `column "role" does not exist` en cada login (AuthService fallback vs tabla legacy `users`) | columnas compat + backfill is_admin→role | mig 055 §5 (aplicada) |
| 13 | **SEC BLOQ** | `SIGNALBRIDGE_DEV_MODE=true` suplantaba TODO token con DevUser admin id=1 → subscriber "ejecutó" kill global; queries UUID rotas | **apagado permanente** (comentado el porqué) + SB recreado + archivos re-copiados | `docker-compose.yml` L1482 |
| 14 | MAYOR OPS | DAGs `asset_*_pipeline_weekly` PAUSADOS — los ciclos semanales nunca corrieron en schedule | unpause ambos | airflow |
| 15 | MAYOR | `l4_backtest_publish` fallaba en Airflow: `services/` no montado → import DSR rompía pipelines | import fail-safe (sin DSR ⇒ verdict cap REVIEW, constitucional) + **mount `./services:/opt/airflow/services`** (aplica al próximo recreate) | `run_gold/btc_pipeline.py` + compose (scheduler+webserver) |
| 16 | **BLOQ** | subscriber `/production` = pantalla blanca ("Something went wrong") — `trades.length` con trades undefined en el path cliente | `trades?.length ?? 0` (null-safe) | `app/production/page.tsx` (build iter-4) |
| 17 | MENOR | ErrorBoundary con texto blanco sobre fondo claro (ilegible) | pendiente (theme del boundary) | — |
| — | ✅ | F7 500: entitlements parciales `{"plan":"free"}` sin campos → `.assets.includes` crash | `effectiveEntitlements` MERGE sobre PLAN_DEFAULTS (fix en SSOT + test contrato) | `rbac.contract.ts` |

## HITO: FUNCTIONAL 14/14 PASS (func3, imagen iter-3)
Registro 202 · logins 4 roles · replay click+badge · promote+restore 200/200 ·
SignalBridge techos CLAMPED (99999→5000) · kill propio on/off · **system/kill 403 real** ·
free bloqueado de análisis actual (404). Console: solo los 8 proxies graceful conocidos.

## Configuraciones cambiadas (operativas)
- `SIGNALBRIDGE_DEV_MODE=false` (SEGURIDAD — nunca reactivar en prod)
- Mount `./services:/opt/airflow/services` en scheduler+webserver (DSR real en DAGs)
- DAGs asset_* despausados (ciclo semanal dominical activo)
- Usuarios QA: {dev,pro,free}@test.com / Test2026! (roles developer/subscriber/free)
- Constraint sb_users_role_chk ampliada a 5 roles (mig 055 §4)

## F1+ — Pruebas de REGISTRO end-to-end (visual + logs) [añadido 2026-07-07]
| Paso | Verificación visual (screenshot) | Verificación logs |
|---|---|---|
| R-a | Página/form de registro: campos, validaciones en pantalla (email inválido, pw débil), estados de error CON contraste legible | console web (Playwright) sin errores |
| R-b | Submit registro → mensaje "pendiente de aprobación" visible | `docker logs usdcop-signalbridge` — evento register + fila `sb_users` status=pending |
| R-c | **Admin aprueba** (UI si existe / API admin) → temp password emitida | logs SB del approve + **MailHog** (:8025) captura el correo con temp-pw (screenshot del correo) |
| R-d | Login con temp-pw → forzado `must_reset_password` → pantalla de reset visible | logs del flujo reset |
| R-e | Reset → login normal → hub del rol asignado (screenshot) | audit_log registra el ciclo |
| R-f | Negativos: email duplicado, pw corta, login antes de aprobar (mensaje claro, no 500) | logs sin stacks no manejados |
> Cadencia: correr F1+ en cada iteración funcional; TODO fallo visual (mensaje ilegible,
> estado mudo) o de logs (stack no manejado) entra al batch. MailHog ya desplegado (overlay).

## ✅ FASE E — PIPELINES: SUCCESS (2026-07-07)
Ambos DAGs corrieron E2E tras fix #15: `asset_xauusd_pipeline_weekly` SUCCESS,
`asset_btcusdt_pipeline_weekly` SUCCESS. E3 entradas→salidas verificadas vs snapshot ANTES:
- `crypto_derivatives_daily` 2492→**2493** (max_date → 2026-07-07) ✓ ingesta funcionó
- `asset_daily_ohlcv` 9304→**10335** (+1031, max → 2026-07-07) ✓ Gold+BTC daily refrescado
- seeds `xauusd/btcusdt_*` mtime → 20:01 ✓ reescritos
- `public/forecasting/xauusd/weekly_inference_*` regenerado ✓ (l5)
Conclusión: el ciclo DS por estrategia (ingest→derivados→chart→backtest→forecast→verify)
es funcional end-to-end. Verdict cap REVIEW en runtime Airflow (sin services/ hasta recreate) = correcto.

## F1++ — REGISTRO + APROBACIÓN ADMIN doble-sesión (best-practice producto) [2026-07-07]
> Dos contextos de navegador aislados en paralelo: sesión A (anónimo→registra) y sesión B
> (admin→aprueba). Cubre incluso el caso "no existe admin configurado".
| Paso | Sesión | Verificación visual + logs |
|---|---|---|
| P0 | — | **Bootstrap admin si no existe**: garantizar `admin@` role=admin approved (idempotente) — best practice: un sistema nuevo SIEMPRE tiene un admin semilla |
| P1 | A (anon) | `/registro` o `/login` signup → form validado (email/pw) → submit → "pendiente de aprobación" | SB register + `sb_users` status=pending |
| P2 | B (admin) | Panel admin: usuario nuevo aparece PENDING; botón Aprobar visible SOLO a admin | GET admin/overview o users |
| P3 | B | Click Aprobar → estado APPROVED + temp-pw | audit_log approve + MailHog correo |
| P4 | A | Login con temp-pw → forzar reset → set nueva pw → hub del rol | must_reset_password flow |
| P5 | — | Negativos: aprobar sin ser admin (403), doble aprobación idempotente, login pre-aprobación (mensaje claro) | logs sin stacks |
> Implementación: `scripts/registration-qa.mjs` (2 browser contexts). MailHog :8025.

## ✅ F1++ — REGISTRO + APROBACIÓN ADMIN doble-sesión: PASS 20/20 (2026-07-06)
Harness: `usdcop-trading-dashboard/scripts/registration-qa.mjs` (2 browser contexts aislados).
Topología de correo: SB recreado con `docker-compose.mailhog.yml` (SMTP→mailhog:1025, `SIGNALBRIDGE_DEV_MODE=false`,
bootstrap admin idempotente). Resultado (`scratchpad/reg1/registration-report.json` + screenshots A1..C4):

| Grupo | Prueba | Resultado |
|---|---|---|
| P0 | bootstrap admin autentica (aprobador existe) | PASS |
| A (anon) | /login→"Solicitar acceso"→/register, form validado, submit→panel "pendiente" | PASS A1-A4 |
| Neg | registro duplicado→409 · login pre-aprobación→403 | PASS N1-N2 |
| B (admin) | login→/admin cola de aprobación, fila PENDING del nuevo email visible | PASS B1-B3 |
| Neg | aprobar como no-admin (free)→403 · doble-aprobación→409 (sin 500) | PASS N3-N4 |
| B (admin) | botón Aprobar→flash "Aprobado · correo enviado", fila desaparece | PASS B4-B6 |
| MailHog | contraseña temporal capturada del correo de aprobación (len=16) | PASS M1 |
| A (cont.) | login temp-pw→**forzado a /reset-password**→reset→/login?reset=1→login limpio→hub del rol | PASS C1-C4 |

**Nuevas vistas/rutas creadas (journey lógico completo):**
- `app/register/page.tsx` — signup público (validación email/pw en vivo, reglas espejo de `RegisterRequest`), 202→panel "pendiente de aprobación". Link desde `/login` ("Solicitar acceso").
- `app/reset-password/page.tsx` — consumo forzado de la contraseña temporal (nunca entra a la app sin resetear).
- `app/admin/page.tsx` — **Cola de aprobación** con Aprobar/Rechazar (antes read-only).
- Proxies BFF: `app/api/admin/users/route.ts` (+`[id]/approve`,`[id]/reject`) → relay del **token admin propio** a SB `require_admin` (sin secreto de servicio; `audit_log` con el admin_id real). `app/api/execution/auth/reset-password/route.ts`.
- `lib/signalbridge/admin-proxy.ts` — helper server-only (guard `x-user-role==admin` + relay bearer).
- Login proxy: **no** acuña cookie de sesión si `must_reset_password` (fuerza el reset primero).
- `middleware.ts` + `rbac.contract.ts`: `/register`, `/reset-password` públicos (PUBLIC_PREFIXES + PAGE_ROUTES).

## Findings ledger #18–#20 (console-log QA, 2026-07-06)
- **#18 `/register` 404 (reportado por el usuario)** → no existía la vista; la journey no era lógica. **FIX**: creadas `/register` + `/reset-password` + link desde `/login`; `must_reset` gate. Verificado 20/20.
- **#19 Ruido 401 en páginas públicas** (`/api/models`, `/api/proxy/trading/stats`, `/api/market/realtime` desde `/login`,`/register`,`/reset-password`). **FIX**: (a) `ModelContext` sólo hace fetch de `/api/models` con sesión que tenga `research:read` (público y subscriber/free ya no llaman); (b) ticker de `/login` repunta a nuevo endpoint **público** `/api/public/market-price` (precio spot USD/COP desde DB, dato de mercado público — no IP monetizada).
- **#20 `/api/models` 404 en páginas autenticadas** → no existía `app/api/models/route.ts` (sólo `[modelId]/*`); la app siempre cayó a defaults RL/PPO **stale**. **FIX**: creado `/api/models/route.ts` (200) que sirve la registry o la estrategia de producción canónica **Smart Simple v2.0** (alinea plan R4: quitar copy "modelo RL"/"PPO"). Gate `research:read` evita 403 a subscriber/free.

## ✅ Regression after console-noise fixes (2026-07-06)
- `functional-qa.mjs`: **14/14 PASS, 0 console errors** (era 14/14 con 8 errores) — sin regresión en
  login por rol, replay, promote/restore, SB tenant (limits/kill/system-kill 403).
- `registration-qa.mjs`: **20/20 PASS, 0 console errors** (era 24) — journey completo limpio.
- `/api/public/market-price?symbol=USD/COP` sirve dato real (p.ej. 3376.30, −1.37%) al ticker anónimo de `/login`.

## Findings ledger #21–#25 (visual TDD role sweep, 2026-07-06, vqa-iter5)
- **#21 CRASH: subscriber /production white-screens** → `ApprovalStatusCard approval={isClientView ? null : approval}` passed null into a non-null component (`approval.strategy_name`). **FIX**: hide the entire "Estado y Gates" section for client view (`{approval && !isClientView && …}`) + `ApprovalStatusCard` returns null on null approval. Verified: subscriber /production 0 errors, renders "Señales", no gates. Aligns RBAC §8 (clients see outputs, not internals).
- **#22 RBAC: developer sees "Aprobar y Promover"/"Rechazar"** on Backtest (Vote 2 is admin-only; API already 403s but the button must not render). **FIX**: `ApprovalPanel`+`DeployPanel` gated behind `canPromote` (admin only) in `ForecastingBacktestSection`. Verified: developer /dashboard hasPromover=false, keeps Backtest/replay.
- **#23 ErrorBoundary near-invisible** (`terminal-*` theme vars → light-on-light). **FIX**: explicit high-contrast self-contained card (slate-950 bg, white text, cyan CTA), Spanish copy. Legible on any theme.
- **#24 (noted, R3) free /forecasting shows current week (2026-W27) + all 3 assets + all 9 models** — should be delay-gated (T-1) + 1 asset per entitlement. Monetization gating gap; server-side delay not yet applied to static forecasting artifacts. Backlog R3.
- **#25 (env) news ingestion slow** — GDELT rate-limits (15s waits/retries), NewsAPI disabled (no key), 5/6 sources enabled. Pipeline works; external throttle. Portafolio captured; GDELT bottleneck.

Verified working with REAL data: admin /production (2026 YTD +1.8%, Sharpe 3.55, PF 3.14, DA 62.5%, 8 trades, active weekly LONG signal from 06-22, gates 5/5) and subscriber "Señales" (same data, internals hidden, honest N<20 ratio suppression).

## Findings ledger #26–#27 + R3 forecasting gate (2026-07-06)
- **#26 (R3) forecasting NOT plan-gated** — any session saw current week + all 3 assets. **FIX**: new gated route `app/api/forecasting/[...path]/route.ts` (mirror of `/api/data`): asset scope (free/base-subscriber ⇒ `['usdcop']` ⇒ Gold/BTC **403 upgrade**) + forecast-delay filter (strips current-week entries from `weekly_inference_*.json` when `forecast_delay_hours>0`). `WeeklyInferenceView` fetches through it; 403 ⇒ "requiere plan superior" + **Ver planes** CTA. Verified: FREE gold=403/btc=403, base-SUBSCRIBER(signals) gold=403 (add-on required), ADMIN(auto+assets) gold=200/btc=200. Gold/BTC are **paid add-ons** per `PLAN_DEFAULTS` (all plans default `['usdcop']`).
- **#27 INTERNALS LEAK on /forecasting** — the no-data fallback showed `python -m scripts.pipeline.generate_asset_weekly_forecast --asset xauusd` to clients (RBAC §8 violation). **FIX**: script hint gated to admin/developer only; locked case shows the upgrade CTA instead. Verified visually (free-gold-locked.png).

Remaining R3 gap (noted): the USD/COP forecast (served as the static `bi_dashboard_unified.csv` via `ForecastingDashboard`) is free's entitled asset but its **T-1 delay is not yet applied** (CSV date-row filtering) — asset boundary enforced, same-asset delay is the follow-up.

## Findings ledger #28–#30 (consolidated visual sweep + copy/compliance pass, 2026-07-07, iter-final)
- **#28 Consolidated role sweep (33 captures, 0 fails, 0 page errors, 0 console errors)** — `visual-qa.mjs shots/iter-final --mobile`. Nav-by-role matrix now clean and matches RBAC §B.2:
  - anon: `/dashboard`→307, `/admin`→307, `/data/registry.json`→401 (deny-by-default).
  - admin: Inicio·Backtest·Produccion·Forecasting·Analisis·SignalBridge·admin.
  - developer: Backtest·Produccion·Forecasting·Analisis (NO SignalBridge, NO admin).
  - subscriber: **Señales**·Forecasting·Analisis·SignalBridge (NO Backtest).
  - free: Forecasting·Analisis only.
  Mobile (390px): landing/pricing/hub responsive single-column, all modules render.
  *(Noted, backlog)*: subscriber on the `signals` plan still shows SignalBridge in nav — per §B.5 own-account SignalBridge is `auto`-only; nav should hide it for `signals`. Low-risk (server gates `execution:self`); tracked for R5.
- **#29 STALE VERSION: strategy shown as "Smart Simple v1.1.0" everywhere** but the config already carries ALL v2.0 features (regime_gate, dynamic_leverage, effective_portfolio_cap, XGBoost) and the registry canonical name is v2.0.0 — the `version` field in `config/execution/smart_simple_v1.yaml` was never bumped. **FIX**: config `version: "1.1.0"→"2.0.0"`; export script hardcodes derive from `cfg['version']` (`generate_pngs(..., version=)`); hardcoded `STRATEGY_NAME` in `app/api/production/live/route.ts` + `lib/services/forecast-backtest.service.ts` → "Smart Simple v2.0"; 9 published JSONs patched (`Smart Simple v1.1.0`→`v2.0.0`, `smart_simple_aggr` v1.0.0 left untouched). Verified live: subscriber /production renders "Smart Simple v2.0" ×4, "v1.1" ×0.
- **#30 FALSE COMPLIANCE CLAIMS on public /login (§B.7 violation)** — "Regulado por SFC Colombia", "Compliance SOC 2 Type II", "ISO 27001" (×2), "Actividad auditada ISO 27001", plus stale "Reinforcement Learning v2.4 / RL v2.4 / © 2024". The system is NOT SFC-regulated (legal gate pending) and holds none of those certs. **FIX**: replaced with truthful statements grounded in the actual architecture — "Cifrado AES-256" (Vault AES-256-GCM is real), "Llaves en Vault", "Actividad registrada en audit log" (append-only audit_log exists), "Acceso exclusivo para cuentas aprobadas" (admin-approval flow exists), honest risk disclaimer ("Contenido informativo y educativo. No es asesoría financiera. Rendimientos pasados no garantizan resultados. Riesgo de pérdida total."), "Trading Cuantitativo • Smart Simple v2.0". Full-app grep confirms 0 remaining false regulatory/cert claims (ISO/SOC/SFC/FINRA/PCI/regulated).

## Findings ledger #31 (model-promotion visual E2E, 2026-07-07, iter-promotion) — new harness `scripts/promotion-e2e.mjs`
- **#31 Vote-2 promotion flow verified end-to-end (10/10 PASS)**. Admin UI: `/dashboard` → ApprovalPanel (5/5 gates from the **published bundle**, L4 PROMOTE 100%) → "Aprobar y Promover" → "Confirmar" → `POST /api/production/approve`. Verified:
  - pre-state `PENDING_APPROVAL`; gates 5/5 read from bundle (not recomputed — I-4/§7 discipline holds); recommendation PROMOTE.
  - post-state `APPROVED`; **`approved_by` = authenticated principal `admin@trading.usdcop.com`** — the route records `auth.user.email`, NEVER the client-supplied `reviewer` string ("dashboard_user"), per audit A4-13. `approved_at` stamped.
  - `/production` flips to "Aprobado" badge + selector "Smart Simple v2.0 APPROVED", KPIs render real 2026 data (+1.8%, Sharpe 3.55, PF 3.14, WR 63%, MaxDD -0.8%, active LONG signal 06-22). Strategy name shows **v2.0** everywhere (copy fix #29 holds).
  - Restored `PENDING_APPROVAL` from backup after the test (no unexpected state left behind).
- **Known limitation (noted, not blocking):** the approve route's fire-and-forget deploy `spawn('python3', …)` runs **inside the node/alpine dashboard container, which has no python3** → the auto-retrain no-ops silently (non-blocking by design). Real production deploy is the **manual/host path** documented in `approval-gates.md` (`python scripts/pipeline/train_and_export_smart_simple.py --phase production --seed-db`) or the Airflow H5 chain. The UI "auto-deploy on approve" is therefore a no-op in this containerized topology — the Vote-2 approval itself (the human gate) works correctly; only the downstream retrain trigger needs a container-appropriate mechanism (Airflow DAG trigger) if in-UI auto-deploy is desired. Backlog.

## Findings ledger #32 (H5 COP weekly chain — full run on fresh data, 2026-07-07)
- **#32 H5 chain L3→L5→L5vt→L7 verified end-to-end on FRESH data (OHLCV through 2026-07-06).**
  - **Pre-check**: OHLCV stale at 2026-07-02 (>3d threshold ⇒ would block L3 freshness gate). Ran `core_l0_01_ohlcv_backfill` (success ~75s) → OHLCV advanced to 2026-07-06 12:55 (age 15h). Macro 2026-07-03 (<7d, OK). *Data-cycling verified inputs→outputs.*
  - **L3 training** (`forecast_h5_l3_weekly_training`): all 6 tasks success — `validate_data_freshness` PASSED (fresh OHLCV), features built, models trained+validated+persisted. Fresh `.pkl` at `outputs/forecasting/h5_weekly_models/latest/` (ridge_h5, bayesian_ridge_h5, scaler_h5, feature_cols_h5) ts 2026-07-07 04:36.
  - **L5 signal** (`forecast_h5_l5_weekly_signal`): the ExternalTaskSensor `wait_for_h5_l3_training` mismatches on manual-trigger execution_dates (up_for_reschedule) — expected manual-trigger friction, not a bug. Bypassed by running `generate_signal`+`persist_signal` with `--ignore-all-dependencies`. Produced **week-28 LONG signal, ensemble_return +1.61%**.
  - **L5 vol-targeting** (`forecast_h5_l5_vol_targeting`, no sensor): success ~15s. Enriched the signal → **leverage 0.50, regime=trending (Hurst 0.735), regime_leverage_scaler=1.0, skip_trade=FALSE (gate OPEN), effective_HS 3.0%, TP 1.5%**. Confirms v2.0 regime-gate behavior: it OPENS in a trending regime (contrast: it blocked 11/12 mean-reverting weeks in Q1).
  - **L7 executor** (`forecast_h5_l7_multiday_executor`): success; `EXECUTION_MODE=paper` (fail-safe — non-paper entry DISABLED at line 337). Route → **skip** ("No execution and not Monday [entry window]") — correct: the manual run (Mon 23:52 COT) is outside the Mon 08–12 COT entry window and no position is open; entry records at the next window. `forecast_h5_paper_trading` cum_pnl 1.768% / 8 weeks (matches dashboard +1.8% YTD).
  - **State restored**: all H5 chain DAGs + OHLCV backfill re-paused to their found state (paused-by-design). Generated fresh models + week-28 signal remain as artifacts. Going live weekly = unpause these 5 DAGs.

## Findings ledger #33–#34 (weekly inference + analysis L8, 2026-07-07)
- **#33 Weekly inference + analysis L8 freshness — VERIFIED current.** Forecasting: Gold/BTC `weekly_inference_2026.json` current through **week 28 (2026-07-06)**; USD/COP `bi_dashboard_unified.csv` dated 07-06. Analysis (`/analysis`, file-driven from `public/data/analysis/`): `analysis_index.json` has 27 weeks, newest **W27 (2026-06-29→07-03) with 5 daily entries** — i.e. fresh through the last COMPLETED week (W28 just started). Admin `/analysis` renders full LLM content: weekly diagnosis, técnico (Bajista, RSI 25.7), trading scenarios (SHORT/LONG w/ stops+targets), macro regime (Risk-On), H5/H1 signals, current macro values (DXY 97.75, VIX 23.38, WTI 96.16, EMBI 259, Oro 4827, Brent 64.33), daily timeline Mon→Fri.
- **#34 FIX: `/analysis` macro history mini-charts showed "Sin datos" for every variable.** Root cause: the weekly-analysis exporter sources chart history from `data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet`, which is **stale (maxes 2026-03-17)** — so `macro_charts[*].data` shipped **empty (0 points × 8 vars × 27 weeks)** while the current-value snapshots (from the fresh DB) rendered fine. **Fix**: new `scripts/ops/patch_analysis_macro_charts.py` backfills every `weekly_2026_W*.json`'s 8 chart arrays from `macro_indicators_daily` (fresh to 07-03), matching the frontend `MacroChartPoint[]` shape + `get_chart_data` math (SMA20, BB(20,2), RSI(14)); idempotent, JSON-safe. Ran → **27/27 weeks × 8/8 charts filled**. Verified: admin `/analysis` "Sin datos" count **8→0**; DXY/VIX/WTI/EMBI/UST10Y/IBR/Oro/Brent now plot value+SMA20 through 07-03. **Upstream durable fix (backlog)**: refresh `MACRO_DAILY_CLEAN.parquet` via `core_l0_03_macro_backfill` (regenerates it from the DB) before each analysis regen, or point `weekly_generator` chart-history at the DB; until then re-run the patch after regen.

## Findings ledger #35 (SignalBridge multi-tenant R5 — S4 fan-out wired + verified, 2026-07-07) — new harness `scripts/validation/signalbridge_fanout_e2e.sh`
- **#35 S4 signal fan-out was DEAD CODE — now wired + verified (11/11 PASS).** The multi-tenant module `services/signalbridge_api/app/api/routes/tenant.py` already had `/tenant/me/{keys,limits,kill}` + `/tenant/system/kill` (admin) + a complete `fan_out_signal()` — but **`fan_out_signal` had zero callers** (grep-confirmed), so a published signal never reached any user. **Fix**: added `POST /tenant/system/fan-out` (admin-only, `getattr(user,'role','')!='admin'` → 403 + `fanout_denied` audit; success → `fanout` audit) invoking the existing `fan_out_signal`. Chosen as an explicit operator/system publish action (NOT auto-fanning every user-created signal, which would leak test signals to everyone). SB source is bind-mounted (`./services/signalbridge_api:/app`) → picked up on container restart.
  - **E2E** (2 paper tenants seeded with verified keys + different caps): admin fan-out (notional 5000 @ px 100) → **fanned_out_to=2**; **pro qty 50** (5000/100, within 5000 cap), **free qty 10** (1000/100, **capped at free's 1000 limit**); both rows **PENDING** (never sent — PreTradeGate paper-first). **Cascade kill**: free flips own kill → re-fan → **fanned_out_to=1**, free gets 0 rows. **Gating**: subscriber JWT → `/system/fan-out` → **403** + `fanout_denied` audit; `fanout` audit rows present. Self-cleaning.
  - **Eligibility** (in `fan_out_signal`): `user_risk_limits_v2.kill_switch=FALSE AND max_notional_usd>0` JOIN `user_exchange_keys.status='verified'` — so no key / killed / zero-cap users are silently excluded. Per-user notional cap = `min(signal_notional, user.max_notional_usd)`.
  - **Next integration (backlog)**: connect the H5 weekly published signal → this endpoint so the operator publishes once and it fans out; real-key registration (`/me/keys`) rejects withdraw-permission keys and marks unverifiable keys `pending` (fail-closed) — the E2E seeds `verified` directly to exercise fan-out logic without a live exchange.

## Findings ledger #36–#40 (backlog wave, 2026-07-07)
- **#36 (R3 complete) USD/COP CSV now plan-delayed.** `/api/forecasting/[...path]` extended to serve `.csv` with `applyCsvDelay` (strips rows whose `inference_date` > now − `forecast_delay_hours`; naive comma split is safe — CSV verified 0 quotes/embedded commas). Both consumers (`ForecastingDashboard.tsx`, `forecasting.service.ts`) repointed from the static path to the gated route. **Verified live: FREE max inference_date = 2026-06-26 (1827 rows, T-1) vs ADMIN = 2026-07-02 (1954 rows, live).**
- **#37 (#28 closed) SignalBridge nav is now plan-gated.** `GlobalNavbar` mirrors the hub's pattern: subscriber role fetches `/api/billing/me` and the SignalBridge entry renders only when `execution.enabled` (true only for `auto` plan). **Verified: pro@test.com (signals plan) nav = Inicio·Señales·Forecasting·Analisis (no SignalBridge); admin keeps it.** Hub already showed the locked upsell teaser.
- **#38 (R5 integration) H5 signal → tenant fan-out wired.** New `fan_out_signal_to_tenants()` in `airflow/dags/utils/signalbridge_client.py` (best-effort by contract: any failure logs + returns fanned_out_to:0, NEVER blocks the DAG; auth via `SIGNALBRIDGE_SERVICE_TOKEN` or `SIGNALBRIDGE_ADMIN_EMAIL/PASSWORD` env, else skipped). Called from `forecast_h5_l5_vol_targeting.py::persist_leverage` AFTER the signal is final (leverage+regime+stops) and only when `skip_trade=false`; notional = `FANOUT_BASE_NOTIONAL_USD` (default 10000) × adjusted_leverage; signal_id `h5_<date>`. **Verified from inside the scheduler: login→fan-out→1 tenant→`user_executions` row pending/paper; no-creds path → skipped gracefully.** Test rows cleaned.
- **#39 (R7 complete + BUG FIX) rate limiter was BYPASSED for `authenticated`-permission routes.** Middleware returned early for `authenticated`/null-permission routes (e.g. `/api/data/**`) BEFORE the rate-limit block — a 130-request burst got 130× served, 0× 429. **Fix: moved the R7 block before that early return. Verified: burst → exactly 120 served + 10× 429.** Security headers verified live (X-Frame-Options DENY, nosniff, Referrer-Policy, Permissions-Policy); CORS closed (no Access-Control-Allow-Origin anywhere); added npm scripts `qa:functional|visual|registration|promotion` + composite **`qa:gate` = rbac:check + rbac:test + qa:functional** (all green).
- **#40 (deploy honesty confirmed — no fix needed).** The in-UI auto-deploy after Vote-2 DOES surface its failure: `deploy_status.json` from the promotion E2E shows `status=failed, error="Deploy timed out after 10 minutes"` (python3 absent in the node container; watchdog caught it). Not a silent no-op. Real in-container auto-deploy would need Airflow REST basic-auth (backend is session-only today) + a `--phase production` DAG (L4 DAG covers backtest only) — an infra feature for the operator to green-light, not a QA gap.

## Findings ledger #41 (MACRO_DAILY_CLEAN upstream fix — 3 real bugs found + fixed, 2026-07-07)
- **#41 `core_l0_03_macro_backfill` could never regenerate MACRO_DAILY_CLEAN in the container — 3 stacked bugs, all fixed and verified:**
  1. **Missing Airflow connection**: `health_check` requires `timescale_conn`, which was never defined → the whole DAG short-circuited (run state "success" was misleading — only `send_report` ran via trigger-rule). Fixed: `airflow connections add timescale_conn` (postgres @ usdcop-postgres-timescale). NOTE for cold boot: this connection lives in the Airflow metadata DB — re-create it after a full reset (or add to bootstrap).
  2. **Container-blind PROJECT_ROOT + sys.path shadowing**: `PROJECT_ROOT = DAGS_DIR.parent.parent` resolves to `/opt` in the container (dags mounted at /opt/airflow/dags) → `export_consolidated_files` wrote to nonexistent `/opt/data`. Fixed layout-aware (pick the parent that holds `data/`). The first correction attempt exposed a second trap: front-inserting SRC_PATH put `src/services`+`src/utils` BEFORE the dags packages → ModuleNotFoundError storm. Final fix: **append** (never insert-front) — Airflow already prioritizes the dags dir.
  3. **Regen task SQL/contract drift**: `regenerate_macro_clean_parquet` selected short column names (`dxy, vix, …`) that don't exist (DB uses lowercase SSOT long names) AND wrote a wrong shape (fecha as column, short names). Rewritten to `SELECT *` minus metadata, UPPERCASE SSOT names, DatetimeIndex on fecha, TPM legacy D→M rename — byte-compatible with the historical CLEAN contract consumed by `weekly_generator`/`_find_column`.
  - **Result**: extract(7 sources)→upsert→export MASTERs→regen all green; `MACRO_DAILY_CLEAN.parquet` now **fresh to 2026-07-07** (was 2026-03-17), 10,882 rows × 18 cols, atomic write, host-visible. The `/analysis` chart-data source is fixed at origin (the earlier JSON patch #34 remains valid for already-published weeks). DAG re-paused (found state).

## Findings ledger #42–#44 (backlog wave 2 — in-UI deploy real + audit deferred, 2026-07-07)
- **#42 (closes #40) In-UI auto-deploy is now REAL and container-native — full E2E verified.** New **H5-L4b DAG** `forecast_h5_l4b_production_deploy` (registered in `dag_registry`, schedule=None event-driven, unpaused-safe): `guard_approved` (HARD gate: approval_state must be APPROVED — Vote-2 integrity re-checked server-side) → `run_production` (manifest-driven `--phase production --no-png --seed-db`, 25-min timeout) → `validate_output` (summary.json fresh) → `register_bundle` (exit gate). Progress is mirrored into `deploy_status.json` (`runner: airflow`, phases) so the panel UX is identical. Dashboard `deploy/route.ts` now PREFERS the Airflow REST path (`AIRFLOW_API_*` env in compose; basic_auth backend enabled `AIRFLOW__API__AUTH_BACKENDS=basic_auth,session`) and falls back to local spawn for host dev; airflow-run deploys 409 correctly on double-click. **E2E: admin POST deploy → "delegated to Airflow" → DAG 4/4 SUCCESS (retrain 9.5 min) → deploy_status completed/done/airflow → summary.json fresh (v2.0.0, 2026 +1.77% refreshed with data through 07-06) → registry refreshed.** Approval restored to found PENDING; refreshed production artifacts kept (legitimate weekly-ops output of the frozen config). Also durable: `AIRFLOW_CONN_TIMESCALE_CONN` env added to compose (survives metadata-DB resets — the missing-connection class of #41 can't recur on cold boot).
- **#43 (audit A7-01 + A4-02 closed).** A7-01: `HoldSignalCheck` now short-circuits **CLOSE/FLAT/EXIT** too — a position-closing signal can never be blocked by entry-oriented risk checks (verified inline: HOLD/CLOSE/FLAT/EXIT pass-through, BUY continues chain). A4-02: `register_strategy_bundle()` wired as the **DAG EXIT gate** of `forecast_h5_l4_backtest_promotion` (validate→backtest→validate_output→**register_bundle**→report→notify) and of the new L4b deploy DAG — the CTR-STRAT-REGISTRY-001 §6.2 exit rule now actually runs in pipelines (smoke: publishes 10 strategies from the container).
- **#44 (audit A4-01 closed).** `BundlePublisher.publish()` gained `phase="production"`: the LIVE year now writes **MUTABLE** `strategies/<sid>/production/{summary,trades}.json` and sets the **`manifest.production` pointer** (model_version/year/paths/updated_at) instead of freezing the first weekly write forever in immutable `backtests/`. `train_and_export_smart_simple.py` production call passes `phase="production"`. Sandbox-verified: live year absent from backtests[], pointer set, weekly overwrite works, backtest phase untouched.

Final regression: **functional 14/14 PASS, 0 console errors** after dashboard rebuild + Airflow recreate + both new/modified DAGs.

## Findings ledger #45 (audit A4-03 closed — typed manifest contract, 2026-07-07)
- **#45 New `lib/contracts/strategy-manifest.contract.ts`** — field-for-field TS mirror of the Python SSOT `src/contracts/strategy_manifest.py` (`StrategyBundleManifest`, `BacktestEntry`, `ModelVersionEntry`, `ProductionPointer` (A4-01 shape), `RegistryIndex`, `RegistryStrategyEntry`, + `activeVersion()/activeBacktest()` read-helpers mirroring the Python @properties). The three consuming routes (`/api/registry`, `/api/registry/promote`, `/api/strategies/[id]/manifest`) are now fully typed — zero `any` on the manifest path. tsc clean on all touched files (the only repo TS errors are the pre-existing `types/generated/schemas.ts`, untouched since 2026-01-14). Note: after the real L4b deploy, the registry's active version is legitimately **2.0.0** (config bump #29) — F5 promote cycle re-verified against it (14/14, active=2.0.0 ⇄ 1.0.0 cycle OK).

## Findings ledger #46–#49 (loop wave 3 — specs reconciled + decisions executed + audit WIP batch, 2026-07-07)
- **#46 Specs reconciled to as-built**: `approval-gates.md` (Airflow L4b deploy path), `elite-operations.md` (event-driven DAGs + Airflow bootstrap note), `registry-lifecycle.md` (production pointer A4-01, exit gate wired A4-02, TS contract A4-03), `rbac-monetization.md` (R3 CSV ✅, R4 nav plan-gate ✅, R5 núcleo ✅, R7 ✅ with statuses), `data-freshness.md` (macro recovery 3-bug fix note), `risk-management.md` (subsystem landscape: mlops RiskManager documented as 4th subsystem A7-04, Command Pattern DEPRECATED A7-02, kill-switch audit = audit_log A7-03, exit passthrough A7-01), `CLAUDE.md` (H5 = 7 DAGs incl. L4b).
- **#47 Wire-or-delete decisions executed**: **A2-02** was already resolved (NRT services deleted + regression guard 8/8). **A7-02** → DEPRECATED (deprecation docstring on `src/risk/commands.py`; dormant library, no consumers without ADR). **A7-03** → SUPERSEDED (live audit surface = append-only `audit_log`; mig-014 table kept unused). **A7-04** → documented.
- **#48 Audit WIP batch (13 findings resolved/hardened)**: A8-06 **refresh one-time-use** (rotation blacklists presented jti + /refresh rejects blacklisted — E2E 200→replay 401); A8-08 broken `/execution/login` → redirect shim; A8-09 `password123`/demo creds stripped (NODE_ENV-guarded mock → prod DCE); A8-12 fake build hash/`Environment: PROD`/admin placeholders removed; A8-05 secret guard extended to staging + loud dev warning (live vault key STILL placeholder — rotate with the exposed exchange keys, checklist §D); A8-03 mitigated by admin-approval flow; A8-07 documented fail-open decision; A8-10 already done (role column+claim); **A5-07 load-race guard** (loadSeqRef token in both loaders); A5-02 already resolved (bundle decision numbers); **A10-01/02/03 Gold regime honesty** (classify_regime parametrized + profile-driven, yaml nulls documented as explicit 0.5 pivot, D1 test added — 7/7). Regression: **functional 14/14, 0 console errors** after SB restart + dashboard rebuild.
- **#49 OLA 4 status (constitution-compliant)**: COP-NULL suite was already RUN + registered 2026-07-06 (`cop_null_suite.py`; H-COP-V11-01: ΔCalmar IC95 includes 0 → cannot claim v11 beats always-short; NULL-B Calmar 7.67 > v11 4.19; DECOMP: TP/HS mechanics SUBTRACT from always-short spot). Remaining OLA 4-7 rows are PRE-REGISTERED in `assets/usdcop/HYPOTHESIS-REGISTRY.md` awaiting the operator: **H-COP-CARRY-00 (broker swap measurement, 0 compute) gates the whole carry thesis**; COP-CORE sensitivities pre-registered (+6 trials); LATAM TSMOM already refuted (H-LATAM-02). No autonomous trials were run (cada trial = presupuesto DSR + Vote humano). Legal gate: `docs/legal/SFC-GATE-CHECKLIST.md` created; paper-only enforcement verified live (trading_mode=PAPER, all user modes paper).

## Findings ledger #50–#52 (engineering backlog wave — 8/8 items, 2026-07-07)
- **#50 paper→live enforcement (rbac.md §6)**: new `POST /tenant/me/accept-risk` + `POST /tenant/me/go-live` — go-live HARD-GATED on (1) risk disclosure accepted, (2) ≥4 weeks of REAL paper history (oldest `user_executions` row), (3) kill off; 403 lists concrete missing reasons; every outcome audited. **E2E: denied with reasons → accept-risk → still denied (paper 0/4) → seeded 5-week history → `mode:live` @5.0 weeks; audit rows `risk_accepted`/`go_live_denied`/`go_live`; state restored.** Global SFC gate unaffected (trading_mode=PAPER still simulates).
- **#51 R3 100% closed — PNG per-file delay**: `/api/forecasting/[...path]` now serves `.png` (binary) applying `forecast_delay_hours` by the `_YYYY_WNN` in the filename (403 upgrade when too new — checked BEFORE existence, no probing oracle); middleware REWRITES static `/forecasting/*.png` → the gated route **stamping x-user-id/role on the rewritten request** (first attempt failed closed to free for everyone — caught by role probe). **Verified: free W28=403/W27=200/W10=200; admin W28=404 (no delay)/rest 200.**
- **#52 Remaining hardening batch**: `rbac-gate.yml` CI workflow (coverage + contract on PRs touching app/contracts/middleware — static, no live stack); **register throttle** per-IP reusing Redis lockout (verified: 5×202 → 429, trusted-proxy XFF only per A8-13 — internet peers can't spoof their bucket); **A8-04 mitigation**: httpOnly `sb-token` cookie set at login, `bearerFrom()` falls back to it (additive — localStorage path still works, future removal enabled); **anti-withdraw verifier ready-to-run** `scripts/validation/verify_key_permissions.py` (SAFE/DANGEROUS/UNKNOWN fail-closed; env-only creds; blocked on user's key rotation); **R6 daily degradation DAG** `rbac_entitlements_daily` (00:00 COT; E2E: expired `signals` → `free` + `entitlement_degraded` audit row); **A8-11** DevUser id → fixed UUID. Final regression: **functional 14/14, 0 console errors**.

## Findings ledger #53 (cold-boot destroy/rebuild cycles, 2026-07-07) — new harness `scripts/validation/coldboot_verify.sh`
- **#53 Full `docker compose down -v` → `up -d --build` reconstruction VERIFIED (cycle 1: 31/31 + functional 14/14).**
  - **Chain audited first**: init-scripts apply schema + migrations glob `04[3-9]/05[0-9]` (043–055 incl. RBAC ✔); `data-seeder` container seeds OHLCV/macro from `data/backups/seeds/` (priority: daily backup → git seed → MinIO) then runs `feature_data_backup --mode restore` (empty-table-only); SB startup bootstraps admin.
  - **Gap #1 fixed pre-cycle**: QA role users (dev/pro/free@test.com) were manual DB rows → wiped by `-v`. Added idempotent non-production bootstrap in SB startup (role+entitlements+approved). Verified on virgin DB: 4 users correct.
  - **Fresh backup taken first**: `core_l0_05_seed_backup` manual run → seeds manifest + 13-table feature backup, minutes old.
  - **Cycle 1 result**: BEFORE vs AFTER **byte-exact** on all data tables (OHLCV 130,850 · macro 10,882 · news 28 · H5 signals 10 / executions 8 · asset_daily 10,335 · crypto_derivatives 2,493). sb_users 26→4 = by design (22 were throwaway registration-test accounts; essentials bootstrap). All 6 core services healthy; dashboard/SB logins live; migrations constraints present (uq_h5_subtrade, entitlements).
  - **Gap #2 found by the cycle**: `audit_log` 65→0 — the append-only compliance trail was NOT in the feature backup (an audit log that vanishes on rebuild defeats its purpose). **Fixed**: added `("audit_log","created_at")` to `feature_data_backup.TABLES` (now 14) + fresh dump taken. Pre-wipe history unrecoverable (accepted); mechanism durable going forward.
  - Airflow: metadata DB rebuilt by airflow-init (admin user + pools recreated); `timescale_conn` survives via `AIRFLOW_CONN_TIMESCALE_CONN` env (ledger #42) — the class of "connection lost on reset" is closed.
- **Cycle 2 (repeatability)**: `down -v --remove-orphans` (32 resources) → `up -d` (cached images) → seeder Exited(0) → **coldboot_verify 31/31 again** + **functional 14/14, 0 console errors**. Note: a cold boot re-pauses ALL DAGs (fresh Airflow metadata) — correct for the paused-by-design trading DAGs; `rbac_entitlements_daily` now sets `is_paused_upon_creation=False` (safe maintenance) so R6 survives rebuilds. MailHog was an overlay orphan — removed on down; SB falls back to ConsoleEmailSender (re-add `docker-compose.mailhog.yml` overlay when testing email flows).

## Findings ledger #54 (internal-role entitlements — found by user post-coldboot, 2026-07-07)
- **#54 Admin/developer lost Gold+BTC forecasting after cold boot.** Migration 055 re-seeds admin with `plan=auto` but `assets=['usdcop']`; the pre-wipe manual asset grant was DB state → wiped by `down -v`. The old special-case in `lib/auth/entitlements.ts` lifted delays but NOT asset scope. **Fix (role-based, survives any DB reset)**: `getEntitlements` now grants INTERNAL roles (admin **and** developer, per §B.2 — internals see all data) `assets=[usdcop,xauusd,btcusdt]` + zero delays + signals_realtime; execution remains plan/permission-based (developer still has no execution:self). **Verified live matrix**: admin gold/btc=200 (UI renders Gold weekly inference fully), developer=200, subscriber(signals)=403 upsell, free=403 — paid add-on boundary intact for clients.

## Findings ledger #55–#57 (user-reported wave: admin menu/captcha/rebrand + promotion + Calmar, 2026-07-07)
- **#55 Admin approval queue now REACHABLE + captcha + rebrand.** (a) GlobalNavbar lacked the `/admin` entry (contract had it; page existed but was URL-only) → added `Admin` nav item (`admin:all`). Verified: queue renders 3 pending with Aprobar/Rechazar + audit trail. (b) **Self-hosted CAPTCHA** (`lib/auth/captcha.ts`: HMAC-signed, expiring, one-time arithmetic challenge; answer never leaves the server) enforced in the register + login proxies (missing/wrong → 400; fields stripped from forward); UI on both forms with refresh button; `/api/captcha` public in the RBAC matrix; NextAuth `/api/auth/callback` (captcha-free secondary) added to the rate-limited prefixes. QA harnesses solve the on-page challenge; registration-qa also self-clears its register-throttle bucket (the A8-03 throttle locked repeated runs — working as designed). **registration 20/20 · functional 14/14** (MailHog overlay restored for email capture). (c) **Rebrand**: platform title `USD/COP` → **GlobalMarkets** (navbar logo, login titles/footer, HTML metadata) — data symbols untouched.
- **#56 REGRESSION (mine, same-day): admin could not promote.** The A5-07 load-race guard used ONE shared sequence — the on-mount `loadVersionData` (manifest default) invalidated `loadStrategyData`'s approval fetch/write → `approval=null` → ApprovalPanel never rendered (promotion-e2e dropped to 4/9). **Fix**: separate strategy-level sequence for `approval` (only loadStrategyData writes it) + guard WRITES only, never early-return before the approval fetch. **promotion-e2e back to 10/10.**
- **#57 Calmar ratio added as first-class KPI** (constitution §2: Calmar is the PRIMARY graduation metric — it was displayed NOWHERE). New `ui.contract.calmarRatio()` (annualized via trading_days, null-safe, never Infinity) + KPI cards on `/dashboard` (Backtest) and `/production` next to Profit Factor, client-gated by the same N<20 rule. Verified: admin sees Calmar+PF on both; PF was already visible to admin (the N<20 suppression only applies to client views — by design).

## Findings ledger #58 (operator's live Vote-2 deploy stuck + Promover button removal, 2026-07-07)
- **#58a OPERATOR'S REAL DEPLOY STUCK IN 'queued' — root-caused live in minutes.** The user cast a real Vote-2 (Aprobar) → deploy API delegated to Airflow → panel froze at "desplegando/exportando". Diagnosis chain: `deploy_status.json` (running/retraining, run_id set) → `airflow dags list-runs` → run **queued** → the L4b DAG was **paused** (cold-boot side effect: fresh Airflow metadata re-pauses every DAG; only the R6 job had been hardened). **Fix**: unpause (the queued run started immediately and completed **4/4 SUCCESS** — guard validated the real vote by admin@trading.usdcop.com, retrain 9.2 min, validate, register_bundle; deploy_status → completed/done; summary.json regenerated with data through 07-07, 2026 YTD honestly recomputed; DB reseeded) + `is_paused_upon_creation=False` on the L4b DAG so a rebuild can never strand a Vote-2 deploy again.
- **#58b "Promover a activa" (blue, version selector) REMOVED** per operator request — it also resolves the audit A4-04 surface (TS-side promote racing the Python registry rebuild). The version dropdown stays for review/replay; the ONLY promotion path is now the two-vote flow (Aprobar y Promover → L4b). `/api/registry/promote` kept for tooling (F5 still exercises it). Verified: button absent for admin, dropdown + deploy panel intact, **functional 14/14, 0 console errors**.

## Findings ledger #59 ("Signal source unavailable" on /production + selector question, 2026-07-07)
- **#59 /production signals were sourced from LEGACY RL tables — cold boot exposed it.** Chain: chart → `useIntegratedChart` → `/api/trading/signals` → primary `public.trades_history` (RL paper-trading; **regenerable, deliberately not in backups → 0 rows post-rebuild**) → fallback `dw.fact_rl_inference` whose query referenced pre-rebuild columns (`inference_id/timestamp_utc/...` vs the init-script truth `id/timestamp/action/model_version`) → **SQL error → "Signal source unavailable"** on the page of the strategy that IS live. **Fix**: (1) new FIRST source = `forecast_h5_executions` (entry/exit → BUY/SELL markers with confidence tier + exit reason — the production strategy's real signals, restored by the feature backup); (2) RL fallback rewritten to the actual schema and wrapped — no data/schema drift now degrades to a clean empty success (never fabricates, never scares the UI; A5-01 discipline kept). **Verified: API `{ok:true, n:16, src:forecast_h5_executions}`, banner gone, APPROVED badge renders; functional 14/14.**
- **Selector note (not a bug)**: `/production` lists only strategies with PRODUCTION exports (Vote-2 approved + deployed) — today that is `smart_simple_v11` alone. Gold/BTC are **experimental registry strategies** (backtest+web, honest gate pending: must beat B1+B1′ on Sharpe AND Calmar OOS before any production promotion) — they appear in `/dashboard` (Backtest selector) and `/forecasting`, by design.

## Findings ledger #60 (BTC production-paper + strategy pruning + hypothesis verdicts, 2026-07-07)
- **#60a BTC paper-production WIRED E2E**: `WITHDRAWAL-PROTOCOL-BTC.md` templated for signature (16/26-week window, Calmar≥0.75, DD caps, retiro inmediato rules); `run_btc_pipeline --phase production` exports per-sid production files (2026 slice via oos-slicer: −2.62%, 3 trades — honest paper start); new `GET /api/production/strategies` (scans singleton + per-sid summaries); `/production` multi-strategy selector (dropdown lists COP APPROVED/live + BTC PENDING/paper; per-sid files via /api/data; COP singleton untouched; live endpoint COP-only); Vote-2 per-strategy (approve/deploy routes accept `strategy_id`; L4b guard reads `dag_run.conf.strategy_id` → `approval_state_<sid>.json`; validate checks `summary_<sid>.json`); intraday COP chart gated off for non-default strategies (daily-bar note instead). Verified: selector shows both, BTC KPIs render (incl. Calmar −1.00), functional 14/14 (1 warmup-transient console reset).
- **#60b Hypothesis verdicts (both pre-registered strongest were ALREADY RUN + published)**: **H-POS-01 BTC funding-gate REFUTADA** (S4 Calmar 1.825/Sharpe 1.33 vs B2 1.833/1.40; OOS-2025 −1.42% igual de plano → no añade). **H-XAU-TREND-01 parcial**: ENS beats B2 everywhere (Calmar 0.171 vs 0.128, DSR 0.989>0.95 — Gold's first DSR pass) **pero falla B1′** (0.171 < 0.223) → constitución §3.3: en Oro el baseline ES la estrategia; `gold_long_only_b1` (exposición constante vol-targeted) es la ganadora honesta.
- **#60c Pruning (operator directive)**: ARCHIVED `gold_regime_gated_v1`, `btc_exposure_s3`, `btc_trend_funding_s4` (manifest.status + archived_reason; registry rebuilt) — hidden from all pages per StrategyStatus contract. Kept: winners + baselines (B1s, B2s, ENS experimental). Full history + lineage + schedules: **new SSOT `.claude/specs/assets/_strategy-history.md`**.

## Findings ledger #61 (hypothesis round on verified lineage, 2026-07-07)
- **#61** Lineage audit: BTC OI/long-short forward-only (31d — NOT backtestable; only funding 2019→, already refuted); DXY/VIX daily 2020→, IBR 2008→ ⇒ Gold-DXY tilt + COP-CORE feasible (design 2020-24, OOS full-2025). **H-BTC-VOLBRK-01 RUN + NOT ADOPTED** (prior z=2.0 declared ex-ante; S5 ≡ B2 — Calmar 1.832=1.832, breaker redundant with ADX/SMA flatness; N_trials→5, B2 DSR holds 0.998). H-XAU-DXY-01 + H-COP-CORE-01 pre-registered with ex-ante priors in `_strategy-history.md §5` awaiting run (next session or operator go). S5 kept in code as documented refuted variant (not published to registry).

## Findings ledger #62 (hypothesis round RUN — 3 honest non-adoptions, 2026-07-07)
- **#62** All three pre-registered hypotheses run with frozen priors, full-2025 evaluation, every cell reported (no picking): **H-BTC-VOLBRK-01** no añade (S5≡B2); **H-XAU-DXY-01** consistent sign (+0.04 Sharpe all 3 cells) but Calmar 0.113 < bar 0.223 → not adopted; **H-COP-CORE-01** carry gate DEAD in 2025 regime (0 active weeks at prior 2pp; loosest cell Calmar 0.32 vs NULL-A 1.52; prime-vs-FFR proxy caveat declared + FEDFUNDS refinement registered). +7 trials to the DSR budget. New runner `scripts/analysis/cop_core_hypothesis.py`. Champions unchanged. Full tables in `_strategy-history.md §5`.

## Findings ledger #63 (operator pruning: champions-only, 2026-07-07)
- **#63** Backtest selector = ONLY the 3 champions (`smart_simple_v11`, `btc_trend_b2`, `gold_long_only_b1`) — 6 more manifests archived (aggr, vt_trailing/rl legacy were never registry entries; hodl_b1/gold_b2/ens archived) + `/api/registry` now filters `archived` (StrategyStatus contract enforced at the API). `/api/production/strategies` lists non-default entries ONLY when APPROVED/LIVE — BTC's PENDING paper export stays OUT of /production until the operator's manual Vote-2 from /dashboard (its approval file remains so that flow works). Verified: backtest=[btc_trend_b2, gold_long_only_b1, smart_simple_v11], production=[smart_simple_v11:APPROVED]. Functional 14/14.

## Findings ledger #64 (FINAL role×surface truth-table, 2026-07-07) — new harness `scripts/role-matrix-qa.mjs` (npm `qa:matrix`, now part of `qa:gate`)
- **#64** The RBAC contract IS a matrix — the final validation asserts it cell-by-cell with REAL sessions: 5 principals (anon/free/subscriber/developer/admin) × 12 critical surfaces (registry, production list/live, **Vote-2 approve/deploy** — admin-only at the edge, admin probing a fake sid gets 404 never 401/403 —, admin users, per-asset forecasting xauusd/btcusdt/COP-csv, current-week PNG delay, billing/me, captcha) + page-level /admin. **61/61 PASS.** One initial "failure" was a wrong EXPECTATION, not the system: subscriber plan `signals` is al-día (delay 0) → correctly passes the PNG gate to a 404 (only free/T-1 gets 403) — the test now documents that nuance. `qa:gate` = rbac:check + rbac:test + functional (14) + matrix (61).

## Findings ledger #65 (gold replay + OOS-default metrics + asset forward views, 2026-07-07)
- **#65a Gold replay was dead**: `filteredTrades` filtered by ENTRY timestamp only — an always-in strategy (gold B1: ONE trade entered 2004-12-22) never "entered" in the 2025 window → 0 trades to animate. **Fix: span-overlap filter** (entry ≤ window-end AND exit ≥ window-start). Also **DEFAULT metrics for registry bundles now come from the `oos` (2025) slice** (KPIs + p-value), matching the operator methodology (trained ≤2024, Backtest = full 2025); full-history stays in the bundle.
- **#65b BTC 2025 anatomy confirmed from data** (the operator's read is correct): 7 trades, net −1.73%; four Feb whipsaws (regime_flip churn in range, −0.7/−1.0/−1.1/−1.7%), one real trend +7.31% (Apr27–May31) **sized at only 0.27x by vol-targeting** (high-vol breakout ⇒ small size) while the losers ran 0.64–0.75x; Oct −3.69%. Same-day rows with equal entry/exit price but nonzero PnL = synthesis artifact of turning a daily-rebalanced exposure series into discrete "trades" (fills at next open, intra-trade resizing). This is EXACTLY why btc_trend_b2 is paper-gated behind the withdrawal protocol, not real money.
- **#65c Gold/BTC forward forecast like COP (operator request)**: new `scripts/pipeline/generate_asset_forward_charts.py` → per-asset fan-chart PNG (90d history + 68/95% realized-vol bands, RW drift-0, dark theme) + `forward.json` (direction/exposure of the champion + per-horizon {1,5,10,20,30}d expected move + **measured walk-forward DA-2025 per horizon**). Rendered as a "Forward Forecast" panel atop WeeklyInferenceView (image via the gated PNG route → free tier gets the delay). Honest labeling: "NO es una predicción ML". Live numbers: Gold LONG 0.42x (DA-1d 59.8% → DA-30d 88.1%); **BTC FLAT 0.00x** (B2 out of market — consistent with the whipsaw regime), DA-1d 47.5%.

## Findings ledger #66 (gold replay window deep-fix, 2026-07-07 — operator report #2)
- **#66** Root-caused the "replay 2025 shows Dec-2025→Jul-2026" nonsense: gold B1 is ONE trade (2004-12-21 → 2026-07-03); its raw exit marker dragged the chart axis into 2026, and the table/equity/summary consumed the RAW trade (entry price $439 of 2004, +209%, 5805 days) as if it were the window's. **Fixes**: (1) span-overlap filter + (2) trades CLIPPED to the replay window for display/replay/chart (timestamps clamped; chart stays in 2025; "1 trades en rango"); (3) table now consumes the clipped window trades (was raw when not replaying); (4) clipped pass-through rows BLANK their prices/P&L and tag `posicion_continua` — full-trade numbers must not masquerade as window numbers; the 2025 truth is the bundle's OOS slice (+38%, Sharpe 3.05) already shown in the metric cards (ledger #65a). Functional 14/14.

## Findings ledger #67 (replay/metrics DEFINITIVE fix — daily-series SSOT, 2026-07-07)
- **#67** Root cause of every replay artifact (Gold 2004 prices as "2025", BTC clipped row with $0/+0.00%, metrics not reacting to date ranges): **trade records cannot represent partial windows** — the only correct source for window metrics/equity of daily strategies is the DAILY EQUITY SERIES. **Fix end-to-end**: (1) both pipelines now publish `signals_<year>.json` = daily equity series into the bundles (publisher's existing `signals=` param; file-level immutability allowed adding it to existing versions); (2) frontend fetches the series and computes **window metrics live** (return/Sharpe(ann inferred 365 vs 252)/MaxDD/Calmar/days) over exactly `[replayStart, effEnd]` — `effEnd` tracks the last visible trade during replay, so cards animate AND respond to any custom range; (3) equity curve renders the daily series (window-sliced) instead of trade steps; (4) table cells null-safe ('—' not $0/+0.00%). **Verified (Playwright + console logs + screenshots)**: Gold full-2025 = +38.0% (== OOS bundle), Gold H1-2025 = **+13.6%** (dynamic ✓), BTC 2025 = −0.8% (series window semantics incl. boundary day; bundle slice −1.37% uses year-first-row base — both honest, difference documented), no fake values, 0 blocking console errors. Refuted-variant bundles republished as side effect were re-archived (registry 3 champions only). Functional 14/14.
