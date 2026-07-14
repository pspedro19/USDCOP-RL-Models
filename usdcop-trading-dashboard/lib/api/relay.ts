/**
 * BFF relay + in-handler RBAC (CTR-FE-BE-001 §1–§3) — adapted to this repo's AS-BUILT:
 *
 * - Identity arrives on anti-spoof headers stamped by `middleware.ts` (`x-user-id`,
 *   `x-user-role`) — NOT re-resolved per handler.
 * - Permission gates use the CTR-RBAC-001 matrix (`roleHasPermission`), never role ranks.
 * - The SignalBridge bearer travels as `Authorization` header or the httpOnly `sb-token`
 *   cookie (same convention as `lib/signalbridge/admin-proxy.ts`).
 *
 * Every upstream call is bounded (timeout + client-abort fan-in), propagates W3C
 * `traceparent`, forwards `Idempotency-Key` on mutations, and maps errors to the
 * stable envelope codes (`UPSTREAM_*`).
 */
import { permsHave, roleHasPermission, type Permission, type Role } from '@/lib/contracts/rbac.contract';

import { fail, UpstreamError, upstreamCode } from './envelope';

const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS ?? 10_000);

export interface HandlerIdentity {
  userId: string | null;
  role: Role;
}

/**
 * In-handler permission re-check (defense in depth — middleware is not the only gate).
 * Usage:
 *   const gate = requirePermission(req, 'admin:all');
 *   if (gate instanceof Response) return gate;
 *   const who = gate;
 */
export function requirePermission(req: Request, perm: Permission): HandlerIdentity | Response {
  const role = req.headers.get('x-user-role');
  if (!role) return fail('UNAUTHENTICATED', 'Inicia sesión para continuar.', 401);
  // Prefer the effective set stamped by middleware (dynamic RBAC + preview downgrade,
  // migration 056); fall back to the static role matrix for legacy/unstamped requests.
  const stamped = req.headers.get('x-user-perms');
  const allowed = stamped !== null
    ? permsHave(stamped.split(',').filter(Boolean), perm)
    : roleHasPermission(role, perm);
  if (!allowed) {
    return fail('FORBIDDEN', 'No tienes permiso para esta acción.', 403, { required: perm, actual: role });
  }
  return { userId: req.headers.get('x-user-id'), role: role as Role };
}

/**
 * Session-only gate for 'authenticated' matrix routes (catalog/watchlist/cart):
 * any role is fine, but a concrete user identity is required (per-user DB rows).
 * Same anti-spoof source as requirePermission — middleware-stamped headers.
 */
export function requireSession(req: Request): { userId: string; role: Role } | Response {
  const role = req.headers.get('x-user-role');
  const userId = req.headers.get('x-user-id');
  if (!role || !userId) return fail('UNAUTHENTICATED', 'Inicia sesión para continuar.', 401);
  return { userId, role: role as Role };
}

/** SignalBridge bearer: `Authorization` header, else the httpOnly `sb-token` cookie. */
export function bearerFrom(req: Request): string | null {
  const h = req.headers.get('authorization') || req.headers.get('Authorization');
  if (h && /^Bearer\s+.+/i.test(h)) return h.replace(/^Bearer\s+/i, '');
  const cookie = req.headers.get('cookie') || '';
  const m = cookie.match(/(?:^|;\s*)sb-token=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

export interface RelayOpts {
  method?: string;
  bearer?: string | null;
  body?: unknown;
  clientSignal?: AbortSignal;
  incomingHeaders?: Headers;
  idempotencyKey?: string;
}

/** Fetch an upstream JSON endpoint; returns parsed data or throws UpstreamError. */
export async function relayUpstream<T>(url: string, opts: RelayOpts = {}): Promise<{ data: T; traceId?: string }> {
  const ac = new AbortController();
  const timer = setTimeout(
    () => ac.abort(new DOMException('upstream timeout', 'TimeoutError')),
    UPSTREAM_TIMEOUT_MS,
  );
  opts.clientSignal?.addEventListener('abort', () => ac.abort(), { once: true });

  const headers: Record<string, string> = { accept: 'application/json' };
  if (opts.body !== undefined) headers['content-type'] = 'application/json';
  if (opts.bearer) headers['authorization'] = `Bearer ${opts.bearer}`;
  const tp = opts.incomingHeaders?.get('traceparent');
  if (tp) headers['traceparent'] = tp;
  if (opts.idempotencyKey) headers['idempotency-key'] = opts.idempotencyKey;

  try {
    const res = await fetch(url, {
      method: opts.method ?? 'GET',
      headers,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
      signal: ac.signal,
      cache: 'no-store',
    });
    const traceId = res.headers.get('traceparent') ?? tp ?? undefined;
    const text = await res.text();
    const parsed = text ? safeJson(text) : undefined;

    if (!res.ok) {
      const p = parsed as { error?: { code?: string; message?: string }; message?: string } | undefined;
      throw new UpstreamError(
        p?.error?.code ?? upstreamCode(res.status),
        p?.error?.message ?? p?.message ?? res.statusText,
        res.status,
        traceId,
      );
    }
    const data = (parsed && typeof parsed === 'object' && 'data' in (parsed as object)
      ? (parsed as { data: T }).data
      : parsed) as T;
    return { data, traceId };
  } catch (e) {
    if (e instanceof UpstreamError) throw e;
    if ((e as Error)?.name === 'TimeoutError') throw new UpstreamError('UPSTREAM_TIMEOUT', 'El servicio tardó demasiado.', 504);
    if ((e as Error)?.name === 'AbortError') throw new UpstreamError('CLIENT_ABORTED', 'Solicitud cancelada.', 499);
    throw new UpstreamError('UPSTREAM_UNAVAILABLE', 'Servicio no disponible.', 502);
  } finally {
    clearTimeout(timer);
  }
}

function safeJson(s: string): unknown {
  try { return JSON.parse(s); } catch { return undefined; }
}
