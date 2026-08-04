---
kind: as-built
status: IMPLEMENTED
contract: CTR-AUTH-001
version: 1.1.0
last_verified: 2026-08-04
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/lib/auth/api-auth.ts
  - usdcop-trading-dashboard/lib/services/execution/auth.service.ts
  - services/signalbridge_api/app/services/user.py
  - database/migrations/053_sb_user_approval.sql
  - database/migrations/055_rbac_monetization.sql
---
# SDD Spec: Authentication & User Management

> **Responsibility**: Authoritative source for how users authenticate and are created across the
> system — the SignalBridge API (JWT auth, registration, lockout, token revocation) and the Next.js
> dashboard (login pages, middleware, session). Documents the AS-BUILT flow and the security backlog.
>
> Contract: CTR-AUTH-001
> Version: 1.0.0 (as-built documented from audit agent A8, 2026-07)
> Status: IMPLEMENTED (as-built current; remediation tasks in `../audit/AUDIT-2026-07-remediation.md` §A8)
> Cross-refs: `execution-bridge.md` (SignalBridge service), `dashboard-integration.md`,
> `../audit/AUDIT-2026-07-remediation.md` (§A8 findings)
> Scope note: some auth code is active WIP; this spec describes what is on disk today.

---

## 1. As-Built Authentication (SignalBridge API — the auth SSOT)

- **Registration** — `POST /api/auth/register` (`auth.py:42`) → `UserService.create` (`user.py:43`) inserts into **`sb_users`** (UUID PK, `21-signalbridge-users-schema.sql:22`) + a default `sb_trading_configs` row; returns access+refresh tokens immediately. **Open self-registration** — no invite, no email verification (`is_verified` never enforced), no lockout/rate-limit on this route.
- **Password hashing** — **bcrypt** via passlib `CryptContext(schemes=["bcrypt"])` (`security.py:15`). No argon2, no explicit rounds, no server-side complexity policy.
- **Login** — `POST /api/auth/login` (`auth.py:71`): Redis lockout check → `UserService.authenticate` (bcrypt verify) → failure `LoginThrottle.record_failure` / success `clear` + issue tokens + update `last_login`.
- **JWT** — **HS256** (`config.py:52`) signed with `settings.jwt_secret_key`. Access TTL **30 min**, refresh **7 days**. Claims: `sub, email, **role**, exp, iat, type, jti` — el `role` **sí** viaja en el token desde `auth.py:151` (verificado 2026-08-04); la afirmación previa de que no existía quedó stale.
- **Refresh** — `POST /api/auth/refresh` re-verifies + re-issues both tokens (full rotation, but the old refresh token is **not** blacklisted → replay window, A8-06).
- **Logout / revocation** — `POST /api/auth/logout` (`auth.py:181`) blacklists the access `jti` in Redis (TTL=remaining life); `get_current_user` rejects blacklisted jti. Refresh tokens are **not** revoked on logout.
- **Account lockout** — `LoginThrottle` (`login_security.py`): per-email AND per-IP counters, **5 failures / 15-min window → 15-min lockout**, HTTP 429 + `Retry-After`. Redis-backed and **fails OPEN if Redis is down** (A8-07).
- **DEV bypass** — `SIGNALBRIDGE_DEV_MODE=true` → `get_current_user` returns a dummy `DevUser` (id=`1` int, `admin@trading.usdcop.com`), no token check (`middleware/auth.py:29`). Hard-guarded off when `app_env==production`. Set `true` in `docker-compose.compact.yml:529`; the testauth override flips it `false`.
- **Global rate limiting** — `RateLimitMiddleware` added only when NOT development (`main.py:150`); compact runs `APP_ENV=development` → global IP rate-limit OFF (only login lockout active).
- **Roles / RBAC** — **A8-10 corregido** (verificado 2026-08-04): `sb_users` tiene `role` (modelo en `app/models.py:38`), el JWT lo transporta (`auth.py:151`) y **hay enforcement real** — `require_admin` (`app/api/routes/admin.py:30-36`) devuelve **403** a todo principal no-admin. La API ya **no** es single-tier y sí puede honrar el rol del dashboard.

## 2. As-Built Authentication (Dashboard, Next.js)

- Primary `/login` (`app/login/page.tsx`) POSTs to `/api/execution/auth/login` (proxy → SB `/auth/login`), stores `access_token`/`refresh_token` in **localStorage** + `isAuthenticated` flags, falls back to NextAuth `signIn('credentials')`. Password-strength meter/trading-ID validators are **cosmetic** (submit only needs non-empty + `minLength=8`).
- `middleware.ts` protects pages/APIs via NextAuth JWT (`getToken`), adds security headers, gates `ADMIN_ROUTES` on `token.role==='admin'`. **`/execution` and `/api/auth` are PUBLIC** — the execution module self-authenticates via localStorage.
- `AUTH_BYPASS_ENABLED=true` skips middleware auth, guarded by `NODE_ENV!=='production'`. **A8-02 corregido** (verificado 2026-08-04): el helper de API `protectApiRoute` (`lib/auth/api-auth.ts:98-103`) lleva **la misma guarda** y su comentario cita el hallazgo. En el contenedor actual conviven `AUTH_BYPASS_ENABLED=true` y `NODE_ENV=production`, así que el bypass está **inerte** — la combinación alarma al leerla, pero no abre nada.
- `authService` (`lib/services/execution/auth.service.ts:68-71`) tiene `MOCK_MODE`, pero **A8-09 está corregido** (verificado 2026-08-04): ya **no** hay credencial hardcodeada en el bundle; el login mock es dev-only (`NODE_ENV!=='production'`) y **no acepta contraseña** — sólo devuelve la sesión mock para trabajo local de UI. `MOCK_MODE` sale de `NEXT_PUBLIC_MOCK_MODE`.

## 3. Data Model

- **`sb_users`** (`init-scripts/21-signalbridge-users-schema.sql:22`) — UUID PK, email (unique), bcrypt password hash, `is_verified`, `last_login`, timestamps. **Ya NO carece de rol** (verificado contra la DB viva, 2026-08-04): tras aplicar `platform-bootstrap-v1` la tabla suma `role` (default `'user'`), `status` (default `'pending'`), `must_reset_password`, `approved_by`/`approved_at`, `rejected_at`/`rejection_reason` (migración `053`) y `entitlements` JSONB (default `{"plan":"free"}`, migración `055`).
  > **Deadlock de arranque conocido** (2026-08-04): el esquema soporta admin, pero no existe CLI de
  > creación y el registro deja al usuario en `status='pending'`; aprobar exige un `admin_id` que
  > sea fila real de `sb_users` (`services/signalbridge_api/app/services/user.py:210-222`). Con la
  > tabla vacía **el primer admin no puede crearse por la vía normal**, y `/dashboard` responde
  > `307 → /login`. Decisión pendiente del operador; no se resuelve acuñando credenciales desde un
  > agente.
- **`sb_trading_configs`** — per-user default trading config created at registration.
- Token state is Redis-only (blacklist by `jti`, lockout counters) — no DB sessions table. Managed by Alembic + the init-script schema.

## 4. Security Posture & Backlog

**Hardened (verified OK)**: bcrypt hashing; per-email+IP login lockout; access-token blacklist on logout; `SIGNALBRIDGE_DEV_MODE` hard-guarded off in production; Vault AES-256-GCM (+PBKDF2) for exchange API keys; prod-secret validator rejects placeholder secrets when `app_env==production`.

**Open gaps** — tracked as tasks in `../audit/AUDIT-2026-07-remediation.md` §A8:
- **CRITICAL** A8-01 — JWT secret env-name mismatch (`JWT_SECRET` vs `JWT_SECRET_KEY`) → API signs with a public default → all tokens forgeable.
- **HIGH** A8-03 (open registration + public `/api/auth`, no throttle), A8-04 (JWTs in localStorage).
- **MEDIUM** A8-05 (Vault key default in dev), A8-06 (refresh not blacklisted on rotation), A8-07 (revocation fails OPEN on Redis down), A8-08 (broken execution login).

**Cerrados desde la auditoría** (verificados contra el código el 2026-08-04, no declarados):
- **A8-02** — `protectApiRoute` lleva la guarda `NODE_ENV!=='production'` (`lib/auth/api-auth.ts:98-103`).
- **A8-09** — sin credencial hardcodeada; el login mock es dev-only y no acepta contraseña (`auth.service.ts:68-71`).
- **A8-10** — rol en modelo + JWT + `require_admin` con 403 (`admin.py:30-36`).

> Los demás siguen abiertos: **no** se han verificado en esta pasada y su ausencia de esta lista
> significaría lo contrario de lo que este documento pretende.
- **LOW** A8-11 (DevUser int id vs UUID), A8-12 (fake compliance/build UI claims), A8-13 (spoofable X-Forwarded-For lockout key).

## DO NOT
- Do NOT commit `.env` / `secrets/*` — they are gitignored; auth secrets stay out of the repo.
- Do NOT ship `AUTH_BYPASS_ENABLED=true` or `SIGNALBRIDGE_DEV_MODE=true` to any non-dev environment.
- Do NOT store credentials or JWT/Vault secrets with weak/default values — the prod validator must reject placeholders in every non-test env, not only `production`.
- Do NOT store JWTs in localStorage for a trading surface — use httpOnly Secure SameSite cookies.
- Do NOT expose `/auth/register` without an invite/allowlist + throttle.
