/*
 * k6 security test — Brute-force lockout
 * ======================================
 * Fires repeated WRONG-password logins for a single account and asserts the
 * server starts returning HTTP 429 (locked out) rather than an unbounded stream
 * of 401s. Requires Redis to be up (that is where the lockout counters live).
 *
 * Run:
 *   k6 run tests/load/auth/login_bruteforce.js
 *   BASE_URL=http://localhost:8085 ATTEMPTS=15 k6 run tests/load/auth/login_bruteforce.js
 *
 * Expectation: after MAX_LOGIN_ATTEMPTS (default 5) failures the identity/IP is
 * locked, so at least one 429 must appear within ATTEMPTS tries.
 */
import http from 'k6/http';
import { check } from 'k6';
import { Counter } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8085';
const AUTH_BASE = __ENV.AUTH_BASE || '/api/auth';
const RUN_ID = __ENV.RUN_ID || `${Date.now()}`;
const ATTEMPTS = Number(__ENV.ATTEMPTS || 15);

const lockouts = new Counter('auth_lockouts_seen');
const JSON_HEADERS = { 'Content-Type': 'application/json' };

export const options = {
  scenarios: {
    bruteforce: {
      executor: 'per-vu-iterations',
      vus: 1,
      iterations: 1,
      maxDuration: '2m',
    },
  },
  thresholds: {
    // The whole point: the account must get locked.
    auth_lockouts_seen: ['count>0'],
  },
};

export default function () {
  const email = `victim_${RUN_ID}@trading.usdcop.com`;

  // Seed the account so failures are "wrong password", not "unknown user"
  // (both count toward lockout, but this makes the test intent explicit).
  http.post(
    `${BASE_URL}${AUTH_BASE}/register`,
    JSON.stringify({ email, password: 'CorrectHorse1', name: 'Victim' }),
    { headers: JSON_HEADERS }
  );

  let saw429 = false;
  let saw401 = false;
  for (let i = 0; i < ATTEMPTS; i++) {
    const res = http.post(
      `${BASE_URL}${AUTH_BASE}/login`,
      // Must be >= 8 chars or the request is rejected with 422 (validation)
      // before authenticate() runs, so no failure would ever be counted.
      JSON.stringify({ email, password: `wrongpass-${i}` }),
      { headers: JSON_HEADERS, tags: { endpoint: 'login_bad' } }
    );
    if (res.status === 429) {
      saw429 = true;
      lockouts.add(1);
      check(res, { 'has Retry-After': (r) => !!r.headers['Retry-After'] });
      break;
    }
    if (res.status === 401) saw401 = true;
  }

  check(null, {
    'got 401s before lockout': () => saw401,
    'lockout (429) triggered': () => saw429,
  });
}
