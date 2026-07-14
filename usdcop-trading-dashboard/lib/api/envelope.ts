/**
 * Response envelope (CTR-FE-BE-001 §2.2) — the ONLY two shapes the browser parses.
 * Server-side helpers for `app/api/**` route handlers. Reference implementation:
 * `lib/api/_reference/bff-handler-reference.ts`; spec:
 * `.claude/specs/platform/frontend-backend-contract.md`.
 */

export interface Meta {
  nextCursor?: string | null;
  asOf?: string;
  requestId?: string;
}

export interface ApiErrorShape {
  code: string;          // stable machine code, SCREAMING_SNAKE — clients switch on this
  message: string;       // human-readable, safe to surface
  status?: number;
  details?: Record<string, unknown>;
  traceId?: string;
  path?: string;
  timestamp?: string;
}

export type Envelope<T> = { ok: true; data: T; meta?: Meta } | { ok: false; error: ApiErrorShape };

function json(body: unknown, status: number, headers?: HeadersInit): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers },
  });
}

/** Success envelope. */
export function ok<T>(data: T, init?: { meta?: Meta; status?: number; headers?: HeadersInit }): Response {
  return json({ ok: true, data, meta: init?.meta } satisfies Envelope<T>, init?.status ?? 200, init?.headers);
}

/** Error envelope. `status` drives HTTP; `code` is the stable client contract. */
export function fail(code: string, message: string, status = 400, details?: Record<string, unknown>): Response {
  const error: ApiErrorShape = { code, message, status, details, timestamp: new Date().toISOString() };
  return json({ ok: false, error } satisfies Envelope<never>, status);
}

/** Upstream failure carrying the stable code — throw inside handlers, `.toResponse()` at the edge. */
export class UpstreamError extends Error {
  constructor(public code: string, message: string, public status: number, public traceId?: string) {
    super(message);
  }
  toResponse(): Response {
    return fail(this.code, this.message, this.status, this.traceId ? { traceId: this.traceId } : undefined);
  }
}

export function upstreamCode(status: number): string {
  if (status === 401) return 'UPSTREAM_UNAUTHENTICATED';
  if (status === 403) return 'FORBIDDEN';
  if (status === 404) return 'NOT_FOUND';
  if (status === 409) return 'CONFLICT';
  if (status === 429) return 'RATE_LIMITED';
  if (status >= 500) return 'UPSTREAM_ERROR';
  return 'UPSTREAM_BAD_REQUEST';
}
