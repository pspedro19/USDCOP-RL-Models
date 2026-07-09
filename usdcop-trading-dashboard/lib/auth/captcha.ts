/**
 * Self-hosted CAPTCHA (no external service — CSP-safe, no API keys).
 *
 * v1: server-issued arithmetic challenge with an HMAC-signed, expiring, one-time token.
 * The answer never travels to the client; the token carries HMAC(answer|nonce|exp).
 * Deters naive credential-stuffing/spam bots and composes with the per-IP rate limits
 * (middleware + SB register throttle). Upgrade path: swap the generator for a
 * proof-of-work (ALTCHA-style) challenge without touching the verify contract.
 */
import { createHmac, randomBytes } from 'crypto';

const SECRET = process.env.NEXTAUTH_SECRET || process.env.CAPTCHA_SECRET || 'captcha-dev-secret';
const TTL_MS = 5 * 60_000;

// One-time-use guard (per instance): nonces seen within TTL cannot be replayed.
const used = new Map<string, number>();
function gcUsed() {
  const now = Date.now();
  for (const [n, exp] of used) if (exp < now) used.delete(n);
  if (used.size > 10_000) used.clear(); // memory guard
}

function sign(answer: string, nonce: string, exp: number): string {
  return createHmac('sha256', SECRET).update(`${answer}|${nonce}|${exp}`).digest('hex');
}

export interface CaptchaChallenge {
  question: string;
  token: string; // base64(json{n,e,h})
}

export function issueCaptcha(): CaptchaChallenge {
  const a = 2 + Math.floor(Math.random() * 8); // 2..9
  const b = 2 + Math.floor(Math.random() * 8);
  const op = Math.random() < 0.5 ? '+' : '×';
  const answer = String(op === '+' ? a + b : a * b);
  const nonce = randomBytes(12).toString('hex');
  const exp = Date.now() + TTL_MS;
  const token = Buffer.from(JSON.stringify({ n: nonce, e: exp, h: sign(answer, nonce, exp) }))
    .toString('base64url');
  return { question: `¿Cuánto es ${a} ${op} ${b}?`, token };
}

export function verifyCaptcha(token: string | undefined, answer: string | undefined): boolean {
  if (!token || answer == null || answer === '') return false;
  try {
    const { n, e, h } = JSON.parse(Buffer.from(token, 'base64url').toString('utf-8')) as {
      n: string; e: number; h: string;
    };
    if (!n || !e || !h || Date.now() > e) return false;
    gcUsed();
    if (used.has(n)) return false; // one-time use
    const ok = sign(String(answer).trim(), n, e) === h;
    if (ok) used.set(n, e);
    return ok;
  } catch {
    return false;
  }
}
