/**
 * POST /api/auth/guest — "Explorar como invitado" (CTR-RBAC-001).
 *
 * Public route (matrix: '/api/auth' → public). Logs the shared demo account
 * (GUEST_BOOTSTRAP_EMAIL / GUEST_BOOTSTRAP_PASSWORD — the same env pair that
 * SignalBridge's bootstrap_guest() re-ensures on startup: rol 'free', approved,
 * is_test=true) into SignalBridge SERVER-SIDE and mints the exact same cookies
 * the login proxy mints (next-auth.session-token + sb-token), so the middleware
 * and every downstream route see a normal 'free' session.
 *
 * CAPTCHA: intentionally NOT required here. The captcha on the login proxy
 * protects USER-SUPPLIED credentials from bots/stuffing; this route forwards
 * no client credentials at all (the guest password lives only in server env),
 * so there is nothing to brute-force through it. It shares the global per-IP
 * rate limit like any API route.
 */

import { NextResponse } from 'next/server';
import { encode } from 'next-auth/jwt';
import { effectivePermissions } from '@/lib/auth/rbac-resolver';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://localhost:8085';
const SESSION_MAX_AGE = 24 * 60 * 60; // 24h — matches next-auth-options session.maxAge
const SESSION_COOKIE = 'next-auth.session-token';

// Defaults mirror the docker-compose GUEST_BOOTSTRAP_* fallbacks so host-dev
// (npm run dev against the compose SignalBridge) works without a .env edit.
const GUEST_EMAIL = process.env.GUEST_BOOTSTRAP_EMAIL || 'guest@demo.local';
const GUEST_PASSWORD = process.env.GUEST_BOOTSTRAP_PASSWORD || 'Guest2026!';

/** Decode a JWT payload without verifying (SignalBridge just issued it — trusted). */
function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const part = token.split('.')[1];
    if (!part) return null;
    const b64 = part.replace(/-/g, '+').replace(/_/g, '/');
    return JSON.parse(Buffer.from(b64, 'base64').toString('utf8'));
  } catch {
    return null;
  }
}

/** Best-effort role lookup from SignalBridge /api/users/me (guest row is 'free'). */
async function fetchRole(accessToken: string): Promise<string> {
  try {
    const res = await fetch(`${BACKEND_URL}/api/users/me`, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (res.ok) {
      const me = await res.json();
      if (me?.role) return String(me.role);
    }
  } catch {
    /* ignore — fall through to default */
  }
  return 'free';
}

export async function POST() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: GUEST_EMAIL, password: GUEST_PASSWORD }),
    });

    const data = await response.json().catch(() => null);

    if (!response.ok || !data?.access_token) {
      // Guest account missing/disabled (e.g. GUEST_BOOTSTRAP_PASSWORD unset on
      // SignalBridge) — surface a clean, non-enumerable failure.
      return NextResponse.json(
        { error: 'guest access unavailable' },
        { status: 503 },
      );
    }

    const accessToken: string = data.access_token;
    const secret = process.env.NEXTAUTH_SECRET;
    if (!secret) {
      return NextResponse.json({ error: 'auth not configured' }, { status: 500 });
    }

    const claims = decodeJwtPayload(accessToken) || {};
    const sub = (claims.sub as string) || '';
    const email = (claims.email as string) || GUEST_EMAIL;
    const role = (claims.role as string) || (await fetchRole(accessToken));

    // Shape mirrors the login proxy / next-auth-options jwt() callback.
    let permissions: string[] | undefined;
    try { permissions = await effectivePermissions(sub, role); } catch { /* static fallback */ }
    const sessionToken = await encode({
      secret,
      maxAge: SESSION_MAX_AGE,
      token: { id: sub, sub, email, username: email, name: 'Invitado', role, permissions },
    });

    const res = NextResponse.json({ ok: true, email, role });
    res.cookies.set({
      name: SESSION_COOKIE,
      value: sessionToken,
      httpOnly: true,
      sameSite: 'lax',
      path: '/',
      secure: false, // HTTP dev/local — matches middleware cookieName (non-__Secure)
      maxAge: SESSION_MAX_AGE,
    });
    res.cookies.set({
      name: 'sb-token',
      value: accessToken,
      httpOnly: true,
      sameSite: 'lax',
      path: '/',
      secure: false,
      maxAge: SESSION_MAX_AGE,
    });
    return res;
  } catch (error) {
    console.error('[API] guest login error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
