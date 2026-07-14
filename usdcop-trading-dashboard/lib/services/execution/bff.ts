/**
 * Execution BFF Helpers (SERVER-ONLY)
 * ===================================
 *
 * Shared server-side utilities for the `/api/execution/**` relay routes that
 * proxy to the SignalBridge FastAPI backend.
 *
 * Responsibilities (single source of truth):
 *  - `SIGNALBRIDGE_BACKEND_URL` — the ONE backend base URL default (in-container
 *    `:8000`). Previously each route hard-coded its own default (a mix of
 *    `:8085`, `:8000` and `localhost`), which silently pointed some routes at a
 *    dead host. Every owned route now imports this constant.
 *  - `resolveExecutionIdentity()` — resolves the caller's user id + SignalBridge
 *    bearer SERVER-SIDE from the NextAuth session (and, as a fallback, the signed
 *    SB token). The client-supplied `[userId]` path segment is NEVER trusted.
 *
 * This module imports `next-auth` server APIs and must NOT be re-exported from
 * `index.ts` or imported by any client component.
 */

import type { NextRequest } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth/next-auth-options';

/** Single default for the SignalBridge backend (in-container port 8000). */
export const SIGNALBRIDGE_BACKEND_URL =
  process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isUuid(value: string | null | undefined): value is string {
  return !!value && UUID_RE.test(value);
}

/** Extract the SignalBridge bearer: the incoming Authorization header, else the
 *  httpOnly `sb-token` cookie set at login. Returns a full `Bearer <jwt>` string. */
export function getSbAuthHeader(request: NextRequest): string | null {
  const header = request.headers.get('authorization');
  if (header) return header;
  const cookieToken = request.cookies.get('sb-token')?.value;
  return cookieToken ? `Bearer ${cookieToken}` : null;
}

/** Decode (without verifying — SignalBridge is the real gate) the `sub`/`id`
 *  claim from a `Bearer <jwt>` string. Used only as a fallback identity source. */
function decodeSubject(authHeader: string | null): string | null {
  if (!authHeader) return null;
  const token = authHeader.replace(/^Bearer\s+/i, '');
  const part = token.split('.')[1];
  if (!part) return null;
  try {
    const json = Buffer.from(
      part.replace(/-/g, '+').replace(/_/g, '/'),
      'base64',
    ).toString('utf8');
    const claims = JSON.parse(json) as Record<string, unknown>;
    const sub = (claims.sub as string) || (claims.id as string) || null;
    return isUuid(sub) ? sub : null;
  } catch {
    return null;
  }
}

export interface ExecutionIdentity {
  /** The resolved SignalBridge user UUID, or null if the caller is unauthenticated. */
  userId: string | null;
  /** The `Bearer <jwt>` to forward to SignalBridge, or null if none was found. */
  authHeader: string | null;
}

/**
 * Resolve the caller's identity SERVER-SIDE.
 *
 * Priority: NextAuth session (httpOnly cookie, authoritative) → `sub` claim of
 * the SignalBridge bearer. The `[userId]` route param is intentionally ignored
 * so a client can never act on behalf of another user.
 */
export async function resolveExecutionIdentity(
  request: NextRequest,
): Promise<ExecutionIdentity> {
  const authHeader = getSbAuthHeader(request);

  let userId: string | null = null;
  try {
    const session = await getServerSession(authOptions);
    if (isUuid(session?.user?.id)) {
      userId = session!.user.id;
    }
  } catch {
    /* fall through to token-derived identity */
  }

  if (!userId) {
    userId = decodeSubject(authHeader);
  }

  return { userId, authHeader };
}

/** Standard SignalBridge fetch: forwards JSON + the resolved bearer. */
export function sbFetch(
  path: string,
  init: { method?: string; authHeader?: string | null; body?: unknown } = {},
): Promise<Response> {
  const { method = 'GET', authHeader, body } = init;
  return fetch(`${SIGNALBRIDGE_BACKEND_URL}${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(authHeader ? { Authorization: authHeader } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
}
