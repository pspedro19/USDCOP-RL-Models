# RBAC + Monetización — AS-BUILT + roadmap (CTR-RBAC-001)

> **Estado 2026-07-06: núcleo IMPLEMENTADO** (primera pasada de OLA 8 del plan maestro,
> `../audit/PLAN-completar-sistema-2026-07.md` APÉNDICE B = diseño completo). Regla thin
> auto-cargada: `.claude/rules/rbac.md`. Complementa `authentication.md` + `approval-gates.md`.

## Principio
El **rol** dice quién eres (`admin`/`developer`/`subscriber`/`free`); el **plan** dice qué
pagaste (`sb_users.entitlements` JSONB). Ambos se validan **server-side por request**; la UI
solo refleja. Deny-by-default.

## AS-BUILT (qué existe y dónde)

| Pieza | Archivo | Estado |
|---|---|---|
| **Contrato SSOT** (roles, permisos, matrices página/API, planes, nav) | `lib/contracts/rbac.contract.ts` | ✅ |
| **Middleware deny-by-default** (contract-driven; gatea también `/data/**` + `/forecasting/**` estáticos; identidad en request headers anti-spoof) | `middleware.ts` | ✅ |
| **Test de cobertura** (ruta real sin entrada en matriz ⇒ exit 1) | `scripts/check-rbac-coverage.mjs` (`npm run rbac:check`) | ✅ (53 APIs + 14 páginas) |
| **Modelo de datos** (entitlements JSONB, `audit_log` append-only con trigger, `user_exchange_keys`, `user_risk_limits_v2`, `user_executions`) | `database/migrations/055_rbac_monetization.sql` | ✅ aplicada |
| **Entitlements server-side** (cache 60s, degradación por expiry, fail-closed a free) | `lib/auth/entitlements.ts` | ✅ |
| **Delay tiering** (analysis: asset ∈ plan + freshness gate por nombre de archivo `YYYY-Www`/`YYYY-MM-DD`) | `app/api/data/[...path]/route.ts` | ✅ |
| **Hub por rol** (cards filtradas por permiso; subscriber ve "Señales"; copy RL→Smart Simple) | `app/hub/page.tsx` | ✅ |
| **Billing DIP** (interface + factory + Wompi con checkout hosted + webhook SHA256) | `lib/billing/{provider,wompi,index}.ts` | ✅ (necesita env: `WOMPI_PUBLIC_KEY`, `WOMPI_EVENTS_SECRET`, `WOMPI_INTEGRITY_SECRET`, `BILLING_PRICES_COP`) |
| **Webhook → entitlements + audit** | `app/api/billing/webhook/route.ts` | ✅ |
| **Checkout** | `app/api/billing/checkout/route.ts` | ✅ |
| **/pricing** (3 tiers desde el SSOT, disclaimer, paper-first) | `app/pricing/page.tsx` | ✅ |
| **PreTradeGate** (SignalBridge: modo global PAPER-default simula, kill-switch usuario default-off, caps; fail-safe BLOCK; 8 tests) | `services/signalbridge_api/app/services/pretrade.py` + `tests/test_pretrade_gate.py` | ✅ desplegado |
| **Sizing correcto** (`quantize_amount` ccxt en base adapter; fix TICK_SIZE MEXC) | `app/adapters/{base,mexc}.py` | ✅ |

### Roles → permisos (resumen; SSOT en el contrato)
`admin` todo · `developer` research+lecturas (sin Vote2/promote/llaves/kill) · `subscriber`
señales+forecast+análisis+execution:self · `free` forecast/análisis **retrasados**.

### Precios
Placeholder en `BILLING_PRICES_COP` (env JSON, centavos COP) — decisión comercial pendiente.

### UX/IA (spec hermana: ux-navigation.md)
Gobernanza visual P1/P2 implementada: `ui.contract.ts` (tokens + canShowRatios N<20) +
`MetricBadge.tsx` (LIVE/BACKTEST/PAPER + procedencia). `/metodologia` publica en produccion.

## PENDIENTE (siguientes pasadas de OLA 8) — actualizado 2026-07-07
- ~~R0~~ ✅ hecho: `docs/rbac/PHASE0-surface-audit.md` (generador: `scripts/generate-surface-audit.mjs`).
- ~~R3~~ ✅ **completo 2026-07-07**: JSON delay vía `/api/data` + `/api/forecasting` (asset scope +
  `forecast_delay_hours` sobre `weekly_inference_*.json`) **+ CSV**: `bi_dashboard_unified.csv`
  se sirve por la ruta gateada con filtro por `inference_date` (free = T-1: verificado 06-26 vs
  admin 07-02). Restante menor: PNGs estáticos exigen sesión pero no delay per-file.
- **R4 casi completo**: GlobalNavbar contract-driven ✅ + **plan-gate de SignalBridge en nav** ✅
  (2026-07-07: subscriber plan `signals` no la ve; solo `execution.enabled`/plan `auto`, vía
  `/api/billing/me` — hub ya tenía el teaser); resta barrer jerga interna página-a-página (§3.2).
- **R5** ✅ **núcleo hecho 2026-07-07**: `/api/tenant/me/{keys,limits,kill}` + `/tenant/system/kill`
  + **`POST /tenant/system/fan-out`** (admin-only, audit) que invoca `fan_out_signal()` (elegible =
  llave `verified` + límites + kill off; cap por usuario; filas PENDING paper — E2E 11/11:
  `scripts/validation/signalbridge_fanout_e2e.sh`). **Integración pipeline**: el DAG H5 vol-targeting
  publica la señal final vía `fan_out_signal_to_tenants()` (best-effort, skip sin creds).
  Restante: verify anti-withdraw contra exchange real (llaves vivas) + `paper_weeks_required`
  enforcement en el paso paper→live.
- **R6**: `/account/billing` página ✅ existe; job diario de degradación (hoy: lazy por request ✅).
- **R7** ✅ **hecho 2026-07-07**: rate-limit por usuario en `/api/{market,data,analysis,forecasting}`
  (120/min; **bug corregido**: el early-return de rutas `authenticated` lo bypasseaba — verificado
  120 servidos + 429s en ráfaga), headers de seguridad en middleware, CORS cerrado (ningún ACAO),
  gates npm: `rbac:check` + `rbac:test` + **`qa:gate`** compuesto. Restante: correr `qa:gate` en CI
  como bloqueo de merge.
- **Gate legal** (SFC) antes de `auto` con dinero real de terceros — paper-only hasta entonces.
  Checklist operativo: `docs/legal/SFC-GATE-CHECKLIST.md`.

---

## AS-BUILT (2026-07-06) — Registration → Admin-Approval → Forced-Reset journey

> Self-serve signup with human admin approval, verified E2E in two concurrent browser
> contexts (`scripts/registration-qa.mjs`, 20/20). SignalBridge is the single authority for
> account state + temp-password + email; the dashboard is a BFF that relays the admin's own
> SB token — no shared service secret, and `audit_log` records the real admin_id.

### Views (public unless noted)
| Route | Purpose | Gate |
|---|---|---|
| `/register` | Signup form (email/name/pw, live validation mirroring `RegisterRequest`) → 202 PENDING → "pendiente de aprobación" panel. No tokens issued. | public |
| `/reset-password` | Forced consumption of the admin-issued temp password (temp bearer from the temp-pw login). Wipes temp creds → `/login?reset=1`. | public |
| `/admin` → **Cola de aprobación** | Pending queue with Aprobar/Rechazar (was read-only). | `admin:all` |
| `/login` | Adds "Solicitar acceso"→`/register`; on `must_reset_password` routes to `/reset-password` (NO session cookie minted yet). | public |

### Endpoints
| Route | Auth | Notes |
|---|---|---|
| `POST /api/execution/auth/register` | public | proxy → SB `/api/auth/register` (202) |
| `POST /api/execution/auth/reset-password` | public + bearer | proxy → SB `/api/auth/reset-password` (consumes temp pw) |
| `GET /api/admin/users?status=` · `POST /api/admin/users/:id/{approve,reject}` | `admin:all` | `lib/signalbridge/admin-proxy.ts` relays the caller's SB admin bearer → SB `require_admin` |
| `GET /api/public/market-price?symbol=` | public | spot FX quote from DB (public market data) — feeds the anon `/login` ticker without 401s |

### Flow (state machine)
```
/register → SB register → sb_users.status=PENDING (email: "registration received")
   → /admin Aprobar → SB approve → status=APPROVED, must_reset_password=TRUE,
       temp pw generated + emailed (MailHog in test), audit_log(approve, admin_id)
   → login(temp pw): must_reset=TRUE ⇒ NO session cookie ⇒ /reset-password
   → reset-password(temp, new): consumes temp, must_reset=FALSE
   → login(new pw) ⇒ session cookie minted ⇒ role hub
```
Negatives locked by the suite: duplicate register→409, pre-approval login→403, approve-as-non-admin→403,
double-approve→409 (no 500).

### Email topology (test)
`docker-compose.mailhog.yml` overlay points SB SMTP→`mailhog:1025`, `SIGNALBRIDGE_DEV_MODE=false`,
idempotent bootstrap admin. Temp pw is captured from the approval email via the MailHog JSON API
(`/api/v2/messages`). In prod, set `SMTP_HOST`/`SMTP_PORT` to the real relay.

---

## As-built increment (2026-07-11) — prices surfaced + chat quota

- **Public prices**: `GET /api/billing/prices` (RBAC `public`) serves plan+add-on COP from the `lib/billing/prices.ts` SSOT (signals 99k, auto 299k, add-ons 39k/mo). Pricing/Cart/Catalog render real amounts (Intl es-CO); Pricing's feature matrix is **derived from `PLAN_DEFAULTS`** (can't drift). Catalog joins live `/api/public/market-price` for FX. Follow-up: cart displays plan+add-ons while Wompi charges plan only (add-on amount not yet in the charge).
- **Chat as a paid lever (CTR-CHAT-001)**: the analysis chatbot (`analysis:read`) is quota-gated per plan server-side (`lib/chat/quota.ts`: free 15/day · signals 100 · auto 250). Provider abstraction `lib/chat/*` (Azure primary → Anthropic fallback); free gets delayed content per entitlements, paid gets higher limits.

---

## Dynamic RBAC — as-built (2026-07-12, migración 056, CTR-RBAC-001)

El mapeo **rol→permiso** y los **overrides por usuario** son ahora DATOS editables desde
la consola admin ("Roles y vistas"), sin tocar el piso estático:

- **Estático (inmutable, invariante `rbac:check`)**: `PAGE_ROUTES`/`API_ROUTES` + enums
  `PERMISSIONS`/`ROLES` en `rbac.contract.ts`. Semilla y *fallback* de la matriz.
- **Dinámico (DB, migración 056)**: `rbac_role_permissions` (matriz permiso×rol) +
  `rbac_user_overrides` (`grant`/`deny` por usuario). Resolver `lib/auth/rbac-resolver.ts`
  (`effectivePermissions = rolePerms ∪ grants − denies`, cache 60s, fallback a la matriz
  estática si la tabla está vacía o la DB cae — **fail-to-code, nunca OPEN**).
- **Aplicación (edge-safe)**: el set efectivo se **hornea en el JWT al login** (los 3 mint
  paths: NextAuth callback + proxy `/api/execution/auth/login` + guest). `middleware.ts` lo
  lee (fallback estático) y lo estampa en `x-user-perms`; `relay.ts::requirePermission`
  chequea ese header (fallback `roleHasPermission`). Cambios aplican en el **próximo login**
  del usuario (JWT re-mint) / dentro de 60s para lecturas server-side.
- **Preview real "Ver como" (downgrade-only)**: para admin real + petición GET + cookie
  `gm-view-as-role` válida, el middleware autoriza LECTURAS contra `realPerms ∩ rol
  previsualizado` — solo restringe, nunca escala (una cookie forjada no da privilegios). Las
  MUTACIONES siguen exigiendo el rol admin real. Nav/hub del cliente usan el set horneado
  (`session.user.permissions`) para reflejar ediciones; en preview usan el rol simulado.
- **Guardrails**: no se puede quitar `admin:all` del rol `admin` (servidor 403); no se puede
  auto-denegar `admin:all` (self-lockout). Todo edit → `audit_log` (`role_perm_change` /
  `user_override_change`).
- **Endpoints** (`admin:all`, auditados): `GET|PATCH /api/admin/roles`,
  `GET|PATCH /api/admin/users/[id]/overrides`. Sección `RolesSection.tsx` (tab "Roles y
  vistas") + editor de overrides en el drawer de `UsersSection`.
