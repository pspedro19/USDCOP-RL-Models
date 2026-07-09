/*
 * k6 load test — Full token lifecycle
 * ===================================
 * Exercises the whole flow per iteration:
 *   register -> login -> refresh -> call protected endpoint -> logout ->
 *   confirm the revoked access token is now rejected (401).
 * This validates the Redis-backed blacklist under concurrency, not just latency.
 *
 * Run:
 *   k6 run tests/load/auth/token_lifecycle.js
 *   BASE_URL=http://localhost:8085 PROTECTED_PATH=/api/users/me k6 run tests/load/auth/token_lifecycle.js
 */
import http from 'k6/http';
import { check, group } from 'k6';
import { Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8085';
const AUTH_BASE = __ENV.AUTH_BASE || '/api/auth';
const PROTECTED_PATH = __ENV.PROTECTED_PATH || '/api/users/me';
const RUN_ID = __ENV.RUN_ID || `${Date.now()}`;
const PASSWORD = 'LoadTest123';

const revoked = new Rate('auth_revocation_enforced');

export const options = {
  scenarios: {
    lifecycle: {
      executor: 'constant-vus',
      vus: Number(__ENV.VUS || 10),
      duration: __ENV.DURATION || '45s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],
    // The blacklist must actually revoke tokens under load.
    auth_revocation_enforced: ['rate>0.99'],
    'http_req_duration{endpoint:login}': ['p(95)<800'],
  },
};

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export default function () {
  const email = `life_${RUN_ID}_${__VU}_${__ITER}@trading.usdcop.com`;
  let accessToken = null;
  let refreshToken = null;

  group('register', () => {
    const res = http.post(
      `${BASE_URL}${AUTH_BASE}/register`,
      JSON.stringify({ email, password: PASSWORD, name: `Life ${__VU}` }),
      { headers: JSON_HEADERS, tags: { endpoint: 'register' } }
    );
    check(res, { 'register ok': (r) => r.status === 201 || r.status === 409 });
  });

  group('login', () => {
    const res = http.post(
      `${BASE_URL}${AUTH_BASE}/login`,
      JSON.stringify({ email, password: PASSWORD }),
      { headers: JSON_HEADERS, tags: { endpoint: 'login' } }
    );
    check(res, { 'login 200': (r) => r.status === 200 });
    try {
      accessToken = res.json('access_token');
      refreshToken = res.json('refresh_token');
    } catch (_e) { /* leave null */ }
  });

  if (!accessToken) return;

  group('refresh', () => {
    const res = http.post(
      `${BASE_URL}${AUTH_BASE}/refresh`,
      JSON.stringify({ refresh_token: refreshToken }),
      { headers: JSON_HEADERS, tags: { endpoint: 'refresh' } }
    );
    check(res, { 'refresh 200': (r) => r.status === 200 });
  });

  const authHeaders = { headers: { Authorization: `Bearer ${accessToken}` } };

  group('protected-before-logout', () => {
    const res = http.get(`${BASE_URL}${PROTECTED_PATH}`, {
      ...authHeaders,
      tags: { endpoint: 'protected' },
    });
    // Accept 200; 404 means PROTECTED_PATH is wrong for this build (adjust env).
    check(res, { 'protected reachable (200)': (r) => r.status === 200 });
  });

  group('logout', () => {
    const res = http.post(`${BASE_URL}${AUTH_BASE}/logout`, null, {
      ...authHeaders,
      tags: { endpoint: 'logout' },
    });
    check(res, { 'logout 200': (r) => r.status === 200 });
  });

  group('protected-after-logout', () => {
    const res = http.get(`${BASE_URL}${PROTECTED_PATH}`, {
      ...authHeaders,
      tags: { endpoint: 'protected_revoked' },
    });
    // The blacklisted token must now be rejected.
    const enforced = res.status === 401;
    check(res, { 'revoked token -> 401': () => enforced });
    revoked.add(enforced);
  });
}
