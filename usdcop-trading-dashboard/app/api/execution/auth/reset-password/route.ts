/**
 * POST /api/execution/auth/reset-password — proxy to SignalBridge /api/auth/reset-password.
 * Consumes the temporary (admin-issued) password. Requires the bearer token minted at the
 * temp-password login (forwarded as Authorization). Public prefix, but useless without a
 * valid temp-session bearer — SignalBridge is the authority.
 */
import { NextRequest, NextResponse } from 'next/server';

import { SIGNALBRIDGE_BACKEND_URL as BACKEND_URL } from '@/lib/services/execution/bff';

export async function POST(request: NextRequest) {
  try {
    const auth = request.headers.get('authorization') || request.headers.get('Authorization');
    if (!auth) {
      return NextResponse.json({ error: 'missing bearer token' }, { status: 401 });
    }
    const body = await request.json();
    const res = await fetch(`${BACKEND_URL}/api/auth/reset-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: auth },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    console.error('[API] SignalBridge reset-password error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
