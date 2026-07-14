/**
 * Browser API client for the GlobalMarkets terminal (CTR-FE-BE-001) — the ONLY
 * fetch wrapper the GM views use. Unwraps the `ok/fail` envelope, throws a typed
 * ClientApiError on !ok, hooks 401 centrally (redirect to /login?next=…), and
 * supports AbortSignal for unmount cancellation.
 *
 * Backward-compat: legacy endpoints that predate the envelope return raw JSON —
 * `apiFetch` detects the absence of an `ok` discriminator and passes the body
 * through as `data`, so views migrate before the BFF does (strangler pattern).
 */
import type { Envelope, Meta } from './envelope';

export class ClientApiError extends Error {
  constructor(public code: string, message: string, public status?: number, public traceId?: string) {
    super(message);
  }
}

export interface ApiResult<T> {
  data: T;
  meta?: Meta;
}

export async function apiFetch<T>(
  path: string,
  init: RequestInit & { onUnauthenticated?: () => void } = {},
): Promise<ApiResult<T>> {
  const res = await fetch(path, {
    ...init,
    headers: {
      accept: 'application/json',
      ...(init.body ? { 'content-type': 'application/json' } : {}),
      ...init.headers,
    },
    credentials: 'same-origin',
    cache: 'no-store',
  });

  if (res.status === 401) init.onUnauthenticated?.();

  const body = (await res.json().catch(() => null)) as Envelope<T> | T | null;
  if (body === null) {
    throw new ClientApiError('PARSE_ERROR', 'Respuesta no válida del servidor.', res.status);
  }

  // Enveloped response (CTR-FE-BE-001)
  if (typeof body === 'object' && body !== null && 'ok' in body) {
    const env = body as Envelope<T>;
    if (!env.ok) {
      throw new ClientApiError(env.error.code, env.error.message, env.error.status ?? res.status, env.error.traceId);
    }
    return { data: env.data, meta: env.meta };
  }

  // Legacy raw JSON (pre-envelope endpoints) — non-2xx still becomes a typed error.
  if (!res.ok) {
    const legacy = body as { error?: string; message?: string };
    throw new ClientApiError(
      res.status === 401 ? 'UNAUTHENTICATED'
        : res.status === 403 ? 'FORBIDDEN'
        : res.status >= 500 ? 'UPSTREAM_ERROR'
        : 'UPSTREAM_BAD_REQUEST',
      legacy?.error ?? legacy?.message ?? `HTTP ${res.status}`,
      res.status,
    );
  }
  return { data: body as T };
}

/** Default 401 handler: redirect to login preserving the destination (§3.1). */
export function goToLogin(): void {
  if (typeof window !== 'undefined') {
    const next = encodeURIComponent(window.location.pathname + window.location.search);
    window.location.href = `/login?next=${next}`;
  }
}
