# Auth Load, Stress & Integration Test Harness

Tools to load/stress test and validate the **registration + login** flow after
consolidating auth on **SignalBridge (SSOT)**.

- Backend under test: `services/signalbridge_api` (FastAPI, default `http://localhost:8085`)
- Store of record: Postgres table `sb_users` + Redis (lockout + token blacklist)

---

## 0. Prerequisites (do this once)

### 0.1 Create the `sb_users` tables
The ORM uses `sb_*` tables that **were never created** by any init-script. Apply
the new idempotent script:

```bash
# Fresh DB: init-scripts/21-signalbridge-users-schema.sql runs automatically.
# Existing DB: apply manually.
docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  < init-scripts/21-signalbridge-users-schema.sql

# verify
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT count(*) FROM sb_users;"
```

### 0.2 Run the API with real auth (not dev bypass)
For **security/revocation tests to be meaningful**, the API must NOT bypass auth:

```bash
# In the signalbridge container / process environment:
SIGNALBRIDGE_DEV_MODE=false     # else get_current_user returns a fake DevUser
APP_ENV=staging                 # keeps rate-limit middleware active; not 'development'
```

Point `DATABASE_URL` at the DB that has `sb_users`, and `REDIS_URL` at a live
Redis (lockout + blacklist live there).

### 0.3 Install k6
`k6` is a single binary — https://k6.io/docs/get-started/installation/
(`winget install k6` / `choco install k6` / `brew install k6`).

---

## 1. Load / stress tests (k6)

All scripts take `BASE_URL` (default `http://localhost:8085`). To go through the
Next.js proxy instead, use `BASE_URL=http://localhost:5000 AUTH_BASE=/api/execution/auth`.

| Script | What it measures | Command |
|--------|------------------|---------|
| `register_storm.js` | Signup ceiling. **bcrypt cost 12 is CPU-bound by design** — this finds the real per-core throughput. | `k6 run tests/load/auth/register_storm.js` |
| `login_ramp.js` | Sustained login throughput + p95/p99. Seeds a user pool in `setup()`. | `PEAK_VUS=100 k6 run tests/load/auth/login_ramp.js` |
| `token_lifecycle.js` | Full flow register→login→refresh→protected→logout→**revoked**. Validates the blacklist under load. | `k6 run tests/load/auth/token_lifecycle.js` |
| `login_bruteforce.js` | Security: lockout must return **429** after repeated failures. | `k6 run tests/load/auth/login_bruteforce.js` |

Useful env knobs: `VUS`, `PEAK_VUS`, `DURATION`, `POOL`, `RAMP`, `ATTEMPTS`,
`RUN_ID` (set a fixed one to reuse seeded users across runs), `PROTECTED_PATH`.

### Reading the results
- `register_storm` p95 will be **high** (hundreds of ms → seconds under load).
  That is bcrypt, not a bug. If you need higher signup throughput, scale API
  replicas / CPU — do NOT lower the bcrypt cost.
- `login_ramp` thresholds: `p95<800ms`, `p99<1500ms`, `<1%` failures.
- `token_lifecycle` threshold `auth_revocation_enforced > 0.99` proves logout
  actually revokes tokens (needs Redis + `SIGNALBRIDGE_DEV_MODE=false`).
- `login_bruteforce` threshold `auth_lockouts_seen > 0` proves brute-force
  protection fires (needs Redis).

> ⚠️ The lockout also has a **per-IP** counter. Running `login_bruteforce` or a
> failing login test from one host will lock that IP for ~15 min — expected.
> `login_ramp` uses correct credentials so it does not trip the lock.

---

## 2. Integration tests (pytest)

Hit a running API; skipped automatically if unreachable.

```bash
# stack up first, then:
pytest services/signalbridge_api/tests/test_auth_flow.py -v
AUTH_BASE_URL=http://localhost:8085 pytest services/signalbridge_api/tests/test_auth_flow.py -v
```

Covers: register (+tokens), duplicate email → 409, weak passwords → 422,
login ok / wrong-pw 401 / unknown-user 401, brute-force → 429, refresh,
and logout revoking the access token.

---

## 3. What changed to make this real (summary)

| Area | Before | After |
|------|--------|-------|
| `sb_users` tables | never created → register/login 500 on real DB | `init-scripts/21-signalbridge-users-schema.sql` |
| Login brute-force | none | Redis lockout (5 fails / 15 min → 429), `app/core/login_security.py` |
| Logout | no-op | Redis token blacklist (jti), enforced in `get_current_user` |
| Prod secrets | `change-me` defaults silently accepted | startup fails in production (`config.py`) |
| Dev bypass | `SIGNALBRIDGE_DEV_MODE` could disable auth anywhere | ignored in production |
| Dashboard `/login` | hardcoded `admin/admin123`, password logged, hint leaked | real SignalBridge login, no hardcoded creds |
| `middleware.ts` | auth skipped whenever `NODE_ENV==='development'` | explicit `AUTH_BYPASS_ENABLED`, never in prod |
| `execution/login` | "Demo Account" fake-token bypass | removed |
| `auth.service.ts` | POSTed to non-existent `/auth/signup` | fixed to `/auth/register` |

---

## 4. Still open (recommended next, not blocking tests)

- Move browser tokens from `localStorage` to `httpOnly` cookies (XSS hardening).
- Set real `NEXTAUTH_SECRET`, `jwt_secret_key`, `vault_encryption_key` in prod env.
- Set `useSecureCookies: true` in NextAuth behind TLS.
- Decide whether `/execution` should be gated by middleware once the whole app
  shares SignalBridge sessions (today it uses its own client guard).
