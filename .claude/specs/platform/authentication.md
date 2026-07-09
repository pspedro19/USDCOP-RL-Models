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
- **JWT** — **HS256** (`config.py:52`) signed with `settings.jwt_secret_key`. Access TTL **30 min**, refresh **7 days**. Claims: `sub, email, exp, iat, type, jti`. **No `role`/scope claim.**
- **Refresh** — `POST /api/auth/refresh` re-verifies + re-issues both tokens (full rotation, but the old refresh token is **not** blacklisted → replay window, A8-06).
- **Logout / revocation** — `POST /api/auth/logout` (`auth.py:181`) blacklists the access `jti` in Redis (TTL=remaining life); `get_current_user` rejects blacklisted jti. Refresh tokens are **not** revoked on logout.
- **Account lockout** — `LoginThrottle` (`login_security.py`): per-email AND per-IP counters, **5 failures / 15-min window → 15-min lockout**, HTTP 429 + `Retry-After`. Redis-backed and **fails OPEN if Redis is down** (A8-07).
- **DEV bypass** — `SIGNALBRIDGE_DEV_MODE=true` → `get_current_user` returns a dummy `DevUser` (id=`1` int, `admin@trading.usdcop.com`), no token check (`middleware/auth.py:29`). Hard-guarded off when `app_env==production`. Set `true` in `docker-compose.compact.yml:529`; the testauth override flips it `false`.
- **Global rate limiting** — `RateLimitMiddleware` added only when NOT development (`main.py:150`); compact runs `APP_ENV=development` → global IP rate-limit OFF (only login lockout active).
- **Roles / RBAC** — `sb_users` has **no role column** and SB JWTs carry no role → SB API is single-tier. RBAC exists only on the dashboard (NextAuth `token.role`), which the SB API cannot honor (A8-10).

## 2. As-Built Authentication (Dashboard, Next.js)

- Primary `/login` (`app/login/page.tsx`) POSTs to `/api/execution/auth/login` (proxy → SB `/auth/login`), stores `access_token`/`refresh_token` in **localStorage** + `isAuthenticated` flags, falls back to NextAuth `signIn('credentials')`. Password-strength meter/trading-ID validators are **cosmetic** (submit only needs non-empty + `minLength=8`).
- `middleware.ts` protects pages/APIs via NextAuth JWT (`getToken`), adds security headers, gates `ADMIN_ROUTES` on `token.role==='admin'`. **`/execution` and `/api/auth` are PUBLIC** — the execution module self-authenticates via localStorage.
- `AUTH_BYPASS_ENABLED=true` skips middleware auth, guarded by `NODE_ENV!=='production'` — **but** the API-route helper `protectApiRoute` (`lib/auth/api-auth.ts:98`) bypass has **no production guard** and returns a synthetic `role:'admin'` user (A8-02).
- `authService` (`lib/services/execution/auth.service.ts`) has a `MOCK_MODE` accepting `password123`/`demo@signalbridge.com` and minting a fake JWT (A8-09).

## 3. Data Model

- **`sb_users`** (`init-scripts/21-signalbridge-users-schema.sql:22`) — UUID PK, email (unique), bcrypt password hash, `is_verified`, `last_login`, timestamps. **No role/is_admin column.**
- **`sb_trading_configs`** — per-user default trading config created at registration.
- Token state is Redis-only (blacklist by `jti`, lockout counters) — no DB sessions table. Managed by Alembic + the init-script schema.

## 4. Security Posture & Backlog

**Hardened (verified OK)**: bcrypt hashing; per-email+IP login lockout; access-token blacklist on logout; `SIGNALBRIDGE_DEV_MODE` hard-guarded off in production; Vault AES-256-GCM (+PBKDF2) for exchange API keys; prod-secret validator rejects placeholder secrets when `app_env==production`.

**Open gaps** — tracked as tasks in `../audit/AUDIT-2026-07-remediation.md` §A8:
- **CRITICAL** A8-01 — JWT secret env-name mismatch (`JWT_SECRET` vs `JWT_SECRET_KEY`) → API signs with a public default → all tokens forgeable.
- **HIGH** A8-02 (API auth-bypass with no prod guard), A8-03 (open registration + public `/api/auth`, no throttle), A8-04 (JWTs in localStorage).
- **MEDIUM** A8-05 (Vault key default in dev), A8-06 (refresh not blacklisted on rotation), A8-07 (revocation fails OPEN on Redis down), A8-08 (broken execution login), A8-09 (mock-auth creds in bundle), A8-10 (no SB-side RBAC).
- **LOW** A8-11 (DevUser int id vs UUID), A8-12 (fake compliance/build UI claims), A8-13 (spoofable X-Forwarded-For lockout key).

## DO NOT
- Do NOT commit `.env` / `secrets/*` — they are gitignored; auth secrets stay out of the repo.
- Do NOT ship `AUTH_BYPASS_ENABLED=true` or `SIGNALBRIDGE_DEV_MODE=true` to any non-dev environment.
- Do NOT store credentials or JWT/Vault secrets with weak/default values — the prod validator must reject placeholders in every non-test env, not only `production`.
- Do NOT store JWTs in localStorage for a trading surface — use httpOnly Secure SameSite cookies.
- Do NOT expose `/auth/register` without an invite/allowlist + throttle.
