/**
 * =============================================================================
 * BFF handler + client — reference implementation
 * Contract: CTR-FE-BE-001  ·  companion to openapi.yaml + frontend-backend-contract.md
 * =============================================================================
 *
 * This is the CANONICAL pattern every `app/api/**` route and every browser
 * data-fetch should follow. It is intentionally framework-light (works with
 * Next.js App Router route handlers) and dependency-free so the backend and
 * frontend teams share one mental model.
 *
 * The ten practices it enforces, in one place:
 *   1. Uniform envelope        — ok(data, meta) / fail(code, msg): never 200-with-error.
 *   2. Auth boundary           — session on browser→BFF; Bearer relayed BFF→upstream.
 *   3. RBAC re-check in-handler — middleware is defense-in-depth, NOT the only gate.
 *   4. Timeout + abort         — every upstream call is bounded; client aborts propagate.
 *   5. Error mapping           — upstream status → stable machine code + safe message.
 *   6. Trace propagation       — W3C `traceparent` in and out → Jaeger correlation.
 *   7. Cache discipline        — authorized/live reads are `no-store`; static is cacheable.
 *   8. Idempotency             — mutating POSTs forward an Idempotency-Key.
 *   9. Rate-limit + captcha     — public auth routes gate before touching upstream.
 *  10. Secret hygiene          — API keys/secrets live server-side; never in a GET response.
 *
 * NOTE: types are illustrative (`unknown`/generics). Wire them to your generated
 * openapi types (`orval`/`openapi-typescript`) so the contract is compiler-checked.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

// ---------------------------------------------------------------------------
// 0. Config — read once, fail fast if a required var is missing.
// ---------------------------------------------------------------------------
const ENV = {
  SIGNALBRIDGE_URL: reqEnv('BACKEND_URL'),          // signalbridge-api (auth, users, exchanges)
  BACKTEST_API_URL: reqEnv('BACKTEST_API_URL'),     // backtest-api
  UPSTREAM_TIMEOUT_MS: Number(process.env.UPSTREAM_TIMEOUT_MS ?? 10_000),
} as const;

function reqEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`[config] missing required env ${name}`);
  return v;
}

// ---------------------------------------------------------------------------
// 1. Envelope — the ONLY two shapes the browser ever parses.
// ---------------------------------------------------------------------------
export interface Meta { nextCursor?: string | null; asOf?: string }
export interface ApiError {
  code: string;          // stable machine code, SCREAMING_SNAKE (switch on this, not on message)
  message: string;       // human-readable, safe to surface
  status?: number;
  details?: Record<string, unknown>;
  traceId?: string;
  path?: string;
  timestamp?: string;
}
export type Envelope<T> = { ok: true; data: T; meta?: Meta } | { ok: false; error: ApiError };

/** Success envelope. */
export function ok<T>(data: T, init?: { meta?: Meta; status?: number; headers?: HeadersInit }) {
  return json({ ok: true, data, meta: init?.meta } satisfies Envelope<T>, init?.status ?? 200, init?.headers);
}

/** Error envelope. `status` drives the HTTP code; `code` is the stable client contract. */
export function fail(code: string, message: string, status = 400, details?: Record<string, unknown>) {
  const error: ApiError = { code, message, status, details, timestamp: new Date().toISOString() };
  return json({ ok: false, error } satisfies Envelope<never>, status);
}

function json(body: unknown, status: number, headers?: HeadersInit) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers },
  });
}

// ---------------------------------------------------------------------------
// 2–3. Auth boundary + RBAC re-check.
//    getSession() reads the NextAuth session. requireRole() is called at the TOP
//    of every protected handler — middleware gating is not sufficient on its own
//    (a route can be reached by internal redirects/rewrites that skip middleware).
// ---------------------------------------------------------------------------
export type Role = 'admin' | 'developer' | 'subscriber' | 'free';
export interface Session {
  userId: string; name: string; email: string;
  role: Role; plan: 'free' | 'signals' | 'auto'; status: 'active' | 'pending' | 'suspended';
  bearer: string;                 // SignalBridge token — relayed upstream, NEVER sent to the browser
}

/** Replace body with your real NextAuth accessor (`getServerSession(authOptions)`). */
export async function getSession(_req: Request): Promise<Session | null> {
  // return await resolveNextAuthSession(_req);
  return null; // placeholder
}

const ROLE_RANK: Record<Role, number> = { free: 0, subscriber: 1, developer: 2, admin: 3 };

/** Returns the session or a ready-to-return 401/403 Response. Usage:
 *    const gate = await requireRole(req, 'admin'); if (gate instanceof Response) return gate;
 *    const session = gate; // typed Session
 */
export async function requireRole(req: Request, min: Role): Promise<Session | Response> {
  const session = await getSession(req);
  if (!session) return fail('UNAUTHENTICATED', 'Inicia sesión para continuar.', 401);
  if (session.status !== 'active') return fail('ACCOUNT_INACTIVE', 'Tu cuenta no está activa.', 403);
  if (ROLE_RANK[session.role] < ROLE_RANK[min]) {
    return fail('FORBIDDEN', 'No tienes permiso para esta acción.', 403, { required: min, actual: session.role });
  }
  return session;
}

// ---------------------------------------------------------------------------
// 4–6. relayUpstream — the single choke-point for BFF→microservice calls.
//    Bounded (timeout+abort), relays bearer + trace, maps errors to the envelope.
// ---------------------------------------------------------------------------
interface RelayOpts {
  method?: string;
  bearer?: string;              // session.bearer — added as Authorization
  body?: unknown;
  clientSignal?: AbortSignal;   // propagate the browser's abort so we don't orphan upstream work
  incomingHeaders?: Headers;    // to forward traceparent
  idempotencyKey?: string;      // for mutating requests
}

/** Fetch an upstream JSON endpoint and return parsed data, or throw UpstreamError. */
export async function relayUpstream<T>(url: string, opts: RelayOpts = {}): Promise<{ data: T; traceId?: string }> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(new DOMException('upstream timeout', 'TimeoutError')), ENV.UPSTREAM_TIMEOUT_MS);
  // Fan the client's abort into ours so a closed tab cancels upstream work.
  opts.clientSignal?.addEventListener('abort', () => ac.abort(), { once: true });

  const headers: Record<string, string> = { accept: 'application/json' };
  if (opts.body !== undefined) headers['content-type'] = 'application/json';
  if (opts.bearer) headers['authorization'] = `Bearer ${opts.bearer}`;
  // Trace propagation: forward inbound W3C traceparent so upstream spans join our trace.
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
      // Map upstream shape → our stable code. Prefer an upstream-provided code.
      throw new UpstreamError(
        (parsed as any)?.error?.code ?? upstreamCode(res.status),
        (parsed as any)?.error?.message ?? (parsed as any)?.message ?? res.statusText,
        res.status,
        traceId,
      );
    }
    // Upstream may already be enveloped ({ok,data}) or raw — normalize to data.
    const data = (parsed && typeof parsed === 'object' && 'data' in (parsed as any) ? (parsed as any).data : parsed) as T;
    return { data, traceId };
  } catch (e) {
    if (e instanceof UpstreamError) throw e;
    if ((e as any)?.name === 'TimeoutError') throw new UpstreamError('UPSTREAM_TIMEOUT', 'El servicio tardó demasiado.', 504);
    if ((e as any)?.name === 'AbortError') throw new UpstreamError('CLIENT_ABORTED', 'Solicitud cancelada.', 499);
    throw new UpstreamError('UPSTREAM_UNAVAILABLE', 'Servicio no disponible.', 502);
  } finally {
    clearTimeout(timer);
  }
}

export class UpstreamError extends Error {
  constructor(public code: string, message: string, public status: number, public traceId?: string) { super(message); }
  toResponse() { return fail(this.code, this.message, this.status, this.traceId ? { traceId: this.traceId } : undefined); }
}

function upstreamCode(status: number): string {
  if (status === 401) return 'UPSTREAM_UNAUTHENTICATED';
  if (status === 403) return 'FORBIDDEN';
  if (status === 404) return 'NOT_FOUND';
  if (status === 409) return 'CONFLICT';
  if (status === 429) return 'RATE_LIMITED';
  if (status >= 500) return 'UPSTREAM_ERROR';
  return 'UPSTREAM_BAD_REQUEST';
}
function safeJson(s: string): unknown { try { return JSON.parse(s); } catch { return undefined; } }

// ---------------------------------------------------------------------------
// EXAMPLE ROUTE — POST /api/admin/users/[id]/approve
//    Shows the full pattern end-to-end: RBAC gate → relay → envelope → error map.
// ---------------------------------------------------------------------------
export async function POST_approveUser(req: Request, ctx: { params: { id: string } }) {
  const gate = await requireRole(req, 'admin');       // 3. re-check in-handler
  if (gate instanceof Response) return gate;
  const session = gate;

  try {
    const { data } = await relayUpstream(
      `${ENV.SIGNALBRIDGE_URL}/api/admin/users/${encodeURIComponent(ctx.params.id)}/approve`,
      {
        method: 'POST',
        bearer: session.bearer,                       // 2. relay bearer (never exposed to client)
        incomingHeaders: req.headers,                 // 6. propagate traceparent
        idempotencyKey: req.headers.get('idempotency-key') ?? undefined, // 8. idempotent mutation
      },
    );
    return ok(data);                                  // 1. uniform envelope
  } catch (e) {
    if (e instanceof UpstreamError) return e.toResponse(); // 5. mapped error
    return fail('INTERNAL', 'Error inesperado.', 500);
  }
}

// ---------------------------------------------------------------------------
// BROWSER CLIENT — the only fetch wrapper the UI should use.
//    Unwraps the envelope, throws a typed ApiError on !ok, handles 401 centrally,
//    and supports AbortSignal so components can cancel on unmount.
// ---------------------------------------------------------------------------
export class ClientApiError extends Error {
  constructor(public code: string, message: string, public status?: number, public traceId?: string) { super(message); }
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit & { onUnauthenticated?: () => void } = {},
): Promise<{ data: T; meta?: Meta }> {
  const res = await fetch(path, {
    ...init,
    headers: { accept: 'application/json', ...(init.body ? { 'content-type': 'application/json' } : {}), ...init.headers },
    credentials: 'same-origin',   // send the session cookie; the client never holds a bearer
  });

  if (res.status === 401) { init.onUnauthenticated?.(); /* e.g. redirect to /login?next=… */ }

  const body = (await res.json().catch(() => null)) as Envelope<T> | null;
  if (!body) throw new ClientApiError('PARSE_ERROR', 'Respuesta no válida del servidor.', res.status);
  if (!body.ok) throw new ClientApiError(body.error.code, body.error.message, body.error.status ?? res.status, body.error.traceId);
  return { data: body.data, meta: body.meta };
}

/* Usage in a component/hook (with cancellation):
 *   const ac = new AbortController();
 *   apiFetch<Entitlements>('/api/billing/me', { signal: ac.signal, onUnauthenticated: goToLogin })
 *     .then(({ data }) => setEntitlements(data))
 *     .catch((e: ClientApiError) => setError(e.code === 'FORBIDDEN' ? 'upgrade' : 'retry'));
 *   return () => ac.abort();
 *
 * Live streams (backtest replay, assistant): use EventSource against the SSE route
 * (GET /api/backtest/real/stream) — do NOT poll. Persist the last position in
 * localStorage and resume from it on reload.
 *
 * Live prices / signals ticker: one shared WebSocket (see useNRTWebSocket), not
 * one socket per widget; fan out via a store.
 */
