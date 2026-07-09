/**
 * SignalBridge Auth Login API Route
 * ==================================
 *
 * Proxy route to SignalBridge backend /auth/login.
 *
 * In addition to proxying, on a successful login this route mints a NextAuth
 * session JWT and sets it as the `next-auth.session-token` cookie. This is what
 * unblocks the browser: `middleware.ts` gates every protected page on
 * `getToken({ cookieName: 'next-auth.session-token' })`, but the SignalBridge
 * login only returns a bearer token (stored client-side in localStorage). Without
 * this cookie the middleware bounces the user back to /login in a loop even
 * though the credentials are valid. Minting the cookie here keeps the whole fix
 * inside this one (non-WIP) route — the login page and middleware are untouched.
 */

import { NextRequest, NextResponse } from 'next/server';
import { encode } from 'next-auth/jwt';

import { verifyCaptcha } from '@/lib/auth/captcha';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://localhost:8085';
const SESSION_MAX_AGE = 24 * 60 * 60; // 24h — matches next-auth-options session.maxAge

// NextAuth v4 (HTTP) session cookie name. Must match middleware's `cookieName`.
const SESSION_COOKIE = 'next-auth.session-token';

/** Decode a JWT payload without verifying (SignalBridge just issued it — already trusted). */
function decodeJwtPayload(token: string): Record<string, unknown> | null {
  try {
    const part = token.split('.')[1];
    if (!part) return null;
    const b64 = part.replace(/-/g, '+').replace(/_/g, '/');
    const json = Buffer.from(b64, 'base64').toString('utf8');
    return JSON.parse(json);
  } catch {
    return null;
  }
}

/** Best-effort role lookup from SignalBridge /api/users/me (defaults to 'user'). */
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
  return 'user';
}

export async function POST(request: NextRequest) {
  try {
    const raw = await request.json();
    // CAPTCHA gate (public endpoint): verify the signed one-time challenge before any
    // credential reaches SignalBridge; captcha fields are stripped from the forward.
    // Composes with the SB per-identity lockout (5 fails/15 min) — captcha stops bots,
    // lockout stops targeted brute force.
    const { captcha_token, captcha_answer, ...body } = raw ?? {};
    if (!verifyCaptcha(captcha_token, captcha_answer)) {
      return NextResponse.json(
        { error: 'captcha inválido o vencido', captcha: true },
        { status: 400 },
      );
    }

    const response = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    // Temp-password login: DO NOT mint a full session cookie. The account must first
    // consume its admin-issued password on /reset-password. The bearer is still returned
    // so the reset page can authenticate the reset call; the client routes there.
    if (data?.must_reset_password) {
      return NextResponse.json(data);
    }

    // Successful SignalBridge login — mint a NextAuth session cookie so the
    // middleware recognises the browser session and stops the /login redirect loop.
    const accessToken: string | undefined = data?.access_token;
    const secret = process.env.NEXTAUTH_SECRET;

    if (accessToken && secret) {
      const claims = decodeJwtPayload(accessToken) || {};
      const sub = (claims.sub as string) || '';
      const email = (claims.email as string) || body?.email || '';
      // Prefer the role embedded in the SignalBridge token; fall back to /me.
      const role = (claims.role as string) || (await fetchRole(accessToken));

      // Shape mirrors next-auth-options jwt() callback (id/role are read by middleware).
      const sessionToken = await encode({
        secret,
        maxAge: SESSION_MAX_AGE,
        token: {
          id: sub,
          sub,
          email,
          username: email,
          name: email,
          role,
        },
      });

      const res = NextResponse.json(data);
      res.cookies.set({
        name: SESSION_COOKIE,
        value: sessionToken,
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        secure: false, // HTTP dev/local — matches middleware cookieName (non-__Secure)
        maxAge: SESSION_MAX_AGE,
      });
      // A8-04 mitigation: also park the SB access token in an httpOnly cookie so
      // server routes (admin relay, tenant proxies) can auth WITHOUT the client
      // reading it from localStorage. localStorage copy remains for now (legacy
      // consumers); server paths prefer the cookie → future removal is additive.
      if (data?.access_token) {
        res.cookies.set({
          name: 'sb-token',
          value: data.access_token,
          httpOnly: true,
          sameSite: 'lax',
          path: '/',
          secure: false,
          maxAge: SESSION_MAX_AGE,
        });
      }
      return res;
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] SignalBridge auth login error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
