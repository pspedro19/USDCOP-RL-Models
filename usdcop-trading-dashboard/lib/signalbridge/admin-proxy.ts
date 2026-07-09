/**
 * Server-only helper for the /admin approval console.
 *
 * Design: the dashboard is a BFF, NOT a second identity provider. The admin who is
 * clicking Approve already holds a SignalBridge admin JWT (minted at login, kept in
 * the browser's `auth-token`). The /admin page forwards it as `Authorization: Bearer`;
 * these handlers relay it verbatim to SignalBridge, whose `require_admin` is the single
 * authority. No shared service secret, and the audit log records the REAL admin_id.
 *
 * Defense in depth: middleware already gates /api/admin/** behind `admin:all` (the
 * NextAuth session role); the relayed bearer is validated again by SignalBridge.
 */
import { NextResponse } from 'next/server';

const SB = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';

export function bearerFrom(req: Request): string | null {
  const h = req.headers.get('authorization') || req.headers.get('Authorization');
  if (h && /^Bearer\s+.+/i.test(h)) return h;
  // A8-04: prefer the httpOnly `sb-token` cookie (set by the login proxy) when the
  // client sent no Authorization header — lets callers stop touching localStorage.
  const cookie = req.headers.get('cookie') || '';
  const m = cookie.match(/(?:^|;\s*)sb-token=([^;]+)/);
  return m ? `Bearer ${decodeURIComponent(m[1])}` : null;
}

/** Relay a request to SignalBridge's admin API with the caller's bearer token. */
export async function relayAdmin(
  req: Request,
  path: string,
  init: { method?: string; body?: unknown } = {},
): Promise<NextResponse> {
  // Session-level guard (middleware stamps the NextAuth role on the request headers).
  if (req.headers.get('x-user-role') !== 'admin') {
    return NextResponse.json({ error: 'admin only' }, { status: 403 });
  }
  const bearer = bearerFrom(req);
  if (!bearer) {
    return NextResponse.json(
      { error: 'missing SignalBridge token', hint: 'log in again to refresh your session' },
      { status: 401 },
    );
  }
  try {
    const res = await fetch(`${SB}${path}`, {
      method: init.method ?? 'GET',
      headers: { 'Content-Type': 'application/json', Authorization: bearer },
      body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
      cache: 'no-store',
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: 'signalbridge unreachable', detail: String(e) }, { status: 502 });
  }
}
