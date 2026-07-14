/**
 * Current User Profile API Route
 * ==============================
 *
 * GET/PATCH `/api/execution/users/me` → SignalBridge `GET/PATCH /api/users/me`.
 * SignalBridge derives the user from the bearer, so no id is passed in the path.
 * The profile the backend supports is `{ name, email }` (UserProfileUpdate).
 */

import { NextRequest, NextResponse } from 'next/server';

import { getSbAuthHeader, sbFetch } from '@/lib/services/execution/bff';

export async function GET(request: NextRequest) {
  try {
    const authHeader = getSbAuthHeader(request);
    if (!authHeader) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const response = await sbFetch('/api/users/me', { authHeader });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get profile' },
        { status: response.status },
      );
    }

    return NextResponse.json({ data: await response.json() });
  } catch (error) {
    console.error('[API] Profile GET error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const authHeader = getSbAuthHeader(request);
    if (!authHeader) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    const body = await request.json();
    // Only forward the fields SignalBridge accepts (name, email).
    const patch: Record<string, unknown> = {};
    if (typeof body?.name === 'string') patch.name = body.name;
    if (typeof body?.email === 'string') patch.email = body.email;

    const response = await sbFetch('/api/users/me', {
      method: 'PATCH',
      authHeader,
      body: patch,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to update profile' },
        { status: response.status },
      );
    }

    return NextResponse.json({ data: await response.json() });
  } catch (error) {
    console.error('[API] Profile PATCH error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
