/*
 * k6 load test — Login ramp
 * =========================
 * Pre-seeds a pool of real users in setup(), then ramps concurrent logins and
 * measures throughput + p95/p99 latency. Uses correct credentials so the
 * brute-force lockout is NOT tripped (see login_bruteforce.js for that path).
 *
 * Run:
 *   k6 run tests/load/auth/login_ramp.js
 *   BASE_URL=http://localhost:8085 POOL=50 PEAK_VUS=100 k6 run tests/load/auth/login_ramp.js
 */
import http from 'k6/http';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8085';
const AUTH_BASE = __ENV.AUTH_BASE || '/api/auth';
const RUN_ID = __ENV.RUN_ID || `${Date.now()}`;
const POOL = Number(__ENV.POOL || 50);
const PASSWORD = 'LoadTest123';

const loginFail = new Rate('auth_login_failed');
const loginDur = new Trend('auth_login_duration', true);

export const options = {
  scenarios: {
    login: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '20s', target: Number(__ENV.PEAK_VUS || 100) },
        { duration: __ENV.DURATION || '1m', target: Number(__ENV.PEAK_VUS || 100) },
        { duration: '15s', target: 0 },
      ],
    },
  },
  thresholds: {
    auth_login_duration: ['p(95)<800', 'p(99)<1500'],
    auth_login_failed: ['rate<0.01'],
    http_req_failed: ['rate<0.02'],
  },
};

// Register POOL users once; hand their emails to the VUs.
export function setup() {
  const emails = [];
  for (let i = 0; i < POOL; i++) {
    const email = `ramp_${RUN_ID}_${i}@trading.usdcop.com`;
    const res = http.post(
      `${BASE_URL}${AUTH_BASE}/register`,
      JSON.stringify({ email, password: PASSWORD, name: `Ramp ${i}` }),
      { headers: { 'Content-Type': 'application/json' } }
    );
    // 201 = created, 409 = already exists from a previous run — both are usable.
    if (res.status === 201 || res.status === 409) {
      emails.push(email);
    }
  }
  if (emails.length === 0) {
    throw new Error('setup failed: could not seed any users — is the API up and sb_users created?');
  }
  return { emails };
}

export default function (data) {
  const email = data.emails[(__VU + __ITER) % data.emails.length];
  const res = http.post(
    `${BASE_URL}${AUTH_BASE}/login`,
    JSON.stringify({ email, password: PASSWORD }),
    { headers: { 'Content-Type': 'application/json' }, tags: { endpoint: 'login' } }
  );

  loginDur.add(res.timings.duration);
  const ok = check(res, {
    'login 200': (r) => r.status === 200,
    'has access_token': (r) => {
      try {
        return !!r.json('access_token');
      } catch (_e) {
        return false;
      }
    },
  });
  loginFail.add(!ok);
}
