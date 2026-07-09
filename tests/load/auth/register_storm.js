/*
 * k6 load test — Registration storm
 * =================================
 * Hammers POST /api/auth/register with unique users to find the real signup
 * ceiling. NOTE: registration hashes the password with bcrypt (cost 12), which
 * is CPU-bound BY DESIGN — expect low per-core throughput and rising latency
 * under concurrency. That is the number you want to size capacity against.
 *
 * Run:
 *   k6 run tests/load/auth/register_storm.js
 *   BASE_URL=http://localhost:8085 VUS=20 DURATION=1m k6 run tests/load/auth/register_storm.js
 *
 * Targets the SignalBridge backend directly (default :8085). Point BASE_URL at
 * the dashboard proxy (http://localhost:5000, path /api/execution/auth) to
 * include Next.js proxy overhead.
 */
import http from 'k6/http';
import { check } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8085';
const AUTH_BASE = __ENV.AUTH_BASE || '/api/auth';
// Unique per-run prefix so re-runs don't collide on the unique email constraint.
const RUN_ID = __ENV.RUN_ID || `${Date.now()}`;

const registered = new Counter('auth_registered');
const regFail = new Rate('auth_register_failed');
const regDur = new Trend('auth_register_duration', true);

export const options = {
  scenarios: {
    register: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: __ENV.RAMP || '20s', target: Number(__ENV.VUS || 20) },
        { duration: __ENV.DURATION || '40s', target: Number(__ENV.VUS || 20) },
        { duration: '10s', target: 0 },
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    // Registration is intentionally expensive; keep a generous but real bound.
    auth_register_duration: ['p(95)<2000'],
    auth_register_failed: ['rate<0.05'],
    http_req_failed: ['rate<0.05'],
  },
};

export default function () {
  // Globally-unique identity: run id + VU + iteration.
  const email = `load_${RUN_ID}_${__VU}_${__ITER}@trading.usdcop.com`;
  const payload = JSON.stringify({
    email,
    password: 'LoadTest123',
    name: `Load User ${__VU}-${__ITER}`,
  });

  const res = http.post(`${BASE_URL}${AUTH_BASE}/register`, payload, {
    headers: { 'Content-Type': 'application/json' },
    tags: { endpoint: 'register' },
  });

  regDur.add(res.timings.duration);
  const ok = check(res, {
    'register 201': (r) => r.status === 201,
    'has access_token': (r) => {
      try {
        return !!r.json('access_token');
      } catch (_e) {
        return false;
      }
    },
  });
  registered.add(ok ? 1 : 0);
  regFail.add(!ok);
}
