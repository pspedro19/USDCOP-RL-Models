/**
 * User Risk Limits API Route (per-user, self-identifying)
 * =======================================================
 *
 * Relays to SignalBridge `GET/PUT /api/signal-bridge/user/{id}/limits`
 * (table `user_risk_limits_v2`).
 *
 * The `[userId]` path segment is IGNORED — the real user id is resolved
 * server-side from the NextAuth session (see `resolveExecutionIdentity`). This
 * removes the previous `'current'` / hard-coded default-UUID / DEFAULT_LIMITS
 * fallbacks that made every user share one risk profile.
 */

import { NextRequest, NextResponse } from 'next/server';

import { resolveExecutionIdentity, sbFetch } from '@/lib/services/execution/bff';

async function relayError(response: Response, fallback: string) {
  const error = await response.json().catch(() => ({}));
  return NextResponse.json(
    { error: error.detail || fallback },
    { status: response.status },
  );
}

export async function GET(request: NextRequest) {
  try {
    const { userId, authHeader } = await resolveExecutionIdentity(request);
    if (!userId) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const response = await sbFetch(`/api/signal-bridge/user/${userId}/limits`, {
      authHeader,
    });

    if (!response.ok) {
      return relayError(response, 'Failed to get user limits');
    }

    // SignalBridge already returns sane defaults for a user with no row yet.
    return NextResponse.json(await response.json());
  } catch (error) {
    console.error('[API] User limits GET error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const { userId, authHeader } = await resolveExecutionIdentity(request);
    if (!userId) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const body = await request.json();

    const response = await sbFetch(`/api/signal-bridge/user/${userId}/limits`, {
      method: 'PUT',
      authHeader,
      body,
    });

    if (!response.ok) {
      return relayError(response, 'Failed to update user limits');
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    console.error('[API] User limits PUT error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
