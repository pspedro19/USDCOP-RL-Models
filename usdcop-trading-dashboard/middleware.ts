/**
 * Next.js Middleware — RBAC edge gate (CTR-RBAC-001).
 * ====================================================
 *
 * DENY-BY-DEFAULT, driven entirely by `lib/contracts/rbac.contract.ts` (the SSOT):
 *   - PAGE_ROUTES / API_ROUTES map every route to a required permission.
 *   - A route with no matrix entry ⇒ authenticated minimum (pages) / 401 (APIs).
 *   - Role→permission resolution is `roleHasPermission()`; unknown/legacy roles get NO
 *     permissions (they can still see public pages).
 *
 * Monetization edge-gate (audit finding R3): `/data/**` and `/forecasting/**` static
 * artifacts (bundles, weekly inference, analysis charts) were served ANONYMOUSLY —
 * 716+ monetizable files. They now require a session at the edge. Entitlement DELAY
 * tiering (free ⇒ T−1/T+7) is applied by the `/api/data` handler, not here.
 *
 * The UI hides what a role can't use; THIS file is the actual defense.
 */

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';

import {
  API_ROUTES,
  PAGE_ROUTES,
  ROLE_PERMISSIONS,
  intersectPerms,
  isRole,
  permsHave,
  requiredPermissionFor,
  type Permission,
} from '@/lib/contracts/rbac.contract';

// Truly public assets (branding for landing/login) — everything else is matrix-gated.
const PUBLIC_PREFIXES = [
  '/login', '/register', '/reset-password', '/pricing', '/metodologia', '/legal', '/api/public', '/api/auth', '/api/health',
  '/api/proxy/trading/health', '/_next', '/favicon.ico', '/images', '/fonts',
];

// Monetized static artifact roots: session required at the edge (R3 edge-gate).
const GATED_STATIC_PREFIXES = ['/data/', '/forecasting/'];

// ── R7: per-user rate limit (fixed window, in-memory per edge instance — adequate
// single-node; move to Redis when horizontally scaled). Applies to data-heavy APIs.
const RATE_LIMITED_PREFIXES = ['/api/market', '/api/data', '/api/analysis', '/api/forecasting',
  // NextAuth credentials callback: captcha-free secondary login path — throttle it so it
  // cannot be used to sidestep the captcha-gated primary proxy (SB lockout still applies).
  '/api/auth/callback'];
const RATE_LIMIT_MAX = 120;          // requests per window per user
const RATE_WINDOW_MS = 60_000;
const rateBuckets = new Map<string, { count: number; resetAt: number }>();

function rateLimited(userKey: string): boolean {
  const now = Date.now();
  const b = rateBuckets.get(userKey);
  if (!b || now > b.resetAt) {
    rateBuckets.set(userKey, { count: 1, resetAt: now + RATE_WINDOW_MS });
    if (rateBuckets.size > 5000) rateBuckets.clear(); // memory guard
    return false;
  }
  b.count += 1;
  return b.count > RATE_LIMIT_MAX;
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (pathname === '/' || PUBLIC_PREFIXES.some((p) => pathname.startsWith(p))) {
    return withSecurityHeaders(NextResponse.next());
  }

  // Explicit opt-in bypass, never in production (kept from previous middleware).
  if (process.env.AUTH_BYPASS_ENABLED === 'true' && process.env.NODE_ENV !== 'production') {
    return withSecurityHeaders(NextResponse.next());
  }

  const token = await getToken({
    req: request,
    secret: process.env.NEXTAUTH_SECRET,
    cookieName: 'next-auth.session-token',
  });
  const role = (token?.role as string | undefined) ?? null;

  // ---- monetized static artifacts: any session required (delay tiering lives in /api/data)
  if (GATED_STATIC_PREFIXES.some((p) => pathname.startsWith(p))) {
    if (!token) return deny(request, pathname);
    // Forecasting PNGs get per-file week-delay enforcement (R3): rewrite the static
    // path to the fs-backed gated route, which applies the plan's forecast_delay_hours
    // by the `_YYYY_WNN` in the filename. Other static artifacts stay session-only.
    if (pathname.startsWith('/forecasting/') && pathname.endsWith('.png')) {
      const url = request.nextUrl.clone();
      url.pathname = `/api${pathname}`;
      // Stamp identity on the rewritten REQUEST — the gated route resolves the plan
      // from x-user-id; without it getEntitlements fails closed to free for everyone.
      const headers = new Headers(request.headers);
      headers.delete('x-user-id');
      headers.delete('x-user-role');
      headers.delete('x-user-perms'); // anti-spoof: only middleware asserts identity/perms
      if (token.id) headers.set('x-user-id', String(token.id));
      if (role) headers.set('x-user-role', role);
      return withSecurityHeaders(NextResponse.rewrite(url, { request: { headers } }));
    }
    return withSecurityHeaders(NextResponse.next());
  }

  // ---- matrix resolution (deny-by-default)
  const isApi = pathname.startsWith('/api');
  const required = requiredPermissionFor(pathname, isApi ? API_ROUTES : PAGE_ROUTES);

  if (required === 'public') return withSecurityHeaders(NextResponse.next());

  if (!token) return deny(request, pathname);

  // Effective permission set (dynamic RBAC, migration 056): the JWT-baked `permissions`
  // claim, else the static role matrix (legacy tokens / bake failure ⇒ never OPEN).
  const realPerms: string[] = Array.isArray((token as { permissions?: unknown }).permissions)
    ? ((token as { permissions?: string[] }).permissions as string[])
    : (isRole(role) ? [...ROLE_PERMISSIONS[role]] : []);

  // Role PREVIEW ("Ver como", admin-only): for READ requests, authorize against
  // real ∩ preview-role (downgrade-only — a forged cookie can only restrict, never
  // escalate; the admin is already a superset). Mutations always keep the real role.
  const viewAs = request.cookies.get('gm-view-as-role')?.value;
  const isRead = request.method === 'GET' || request.method === 'HEAD';
  const effectivePerms: string[] =
    role === 'admin' && isRead && isRole(viewAs) && viewAs !== 'admin'
      ? intersectPerms(realPerms, ROLE_PERMISSIONS[viewAs])
      : realPerms;

  // R7 rate limit on data-heavy APIs (per user id, fallback IP). MUST run before the
  // `authenticated`-permission early return below — /api/data & friends resolve to
  // 'authenticated', which used to bypass the limiter entirely (found by burst smoke test).
  if (RATE_LIMITED_PREFIXES.some((p) => pathname.startsWith(p))) {
    const key = String(token.id ?? request.headers.get('x-forwarded-for') ?? 'anon');
    if (rateLimited(key)) {
      return NextResponse.json({ error: 'rate limit exceeded', retry_in_s: 60 }, { status: 429 });
    }
  }

  if (required === 'authenticated' || required === null) {
    // Unmatched routes still require a session (deny-by-default floor). Unmatched APIs
    // are additionally flagged by the CI coverage test so the matrix stays exhaustive.
    return withUserHeaders(request, token, role, effectivePerms);
  }

  if (!permsHave(effectivePerms, required as Permission)) {
    if (isApi) {
      return NextResponse.json(
        { error: 'Forbidden', required, timestamp: new Date().toISOString() },
        { status: 403 },
      );
    }
    return NextResponse.redirect(new URL('/hub', request.url)); // page: bounce to hub
  }

  return withUserHeaders(request, token, role, effectivePerms);
}

// ─────────────────────────────────────────────────────────────── helpers

function deny(request: NextRequest, pathname: string) {
  if (pathname.startsWith('/api') || GATED_STATIC_PREFIXES.some((p) => pathname.startsWith(p))) {
    return NextResponse.json(
      { error: 'Authentication required', timestamp: new Date().toISOString() },
      { status: 401 },
    );
  }
  const loginUrl = new URL('/login', request.url);
  loginUrl.searchParams.set('callbackUrl', pathname);
  return NextResponse.redirect(loginUrl);
}

function withUserHeaders(
  request: NextRequest,
  token: Record<string, unknown>,
  role: string | null,
  effectivePerms: string[] = [],
) {
  // Identity must travel on the REQUEST headers so downstream route handlers can read it
  // (the legacy middleware set these on the response — the client saw them, handlers never
  // did). Strip any spoofed inbound values first: only the middleware may assert identity.
  const headers = new Headers(request.headers);
  headers.delete('x-user-id');
  headers.delete('x-user-role');
  headers.delete('x-user-perms');
  if (request.nextUrl.pathname.startsWith('/api')) {
    if (token.id) headers.set('x-user-id', String(token.id));
    if (role) headers.set('x-user-role', role);
    // Effective (dynamic + preview-downgraded) permission set for in-handler re-checks.
    headers.set('x-user-perms', effectivePerms.join(','));
  }
  const response = NextResponse.next({ request: { headers } });
  addSecurityHeaders(response);
  return response;
}

function withSecurityHeaders(response: NextResponse) {
  addSecurityHeaders(response);
  return response;
}

function addSecurityHeaders(response: NextResponse): void {
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=(), interest-cohort=()',
  );
}

// Run on everything except Next internals + public branding assets. NOTE: unlike the
// previous matcher, .json/.csv/.png are NOT excluded — that exclusion is what left the
// monetized /data + /forecasting artifacts publicly readable (audit R3).
export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|images|fonts).*)'],
};
