/**
 * Exchange Disconnect API Route
 * =============================
 *
 * DELETE `/api/execution/exchanges/{exchange}/disconnect`
 *
 * Resolves the caller server-side, finds THAT user's credential(s) for the
 * given exchange via SignalBridge `GET /api/exchanges/credentials?exchange=…`,
 * and deletes each via `DELETE /api/exchanges/credentials/{id}`.
 *
 * Previously this route did not exist, so the UI's disconnect button 404'd.
 */

import { NextRequest, NextResponse } from 'next/server';

import { resolveExecutionIdentity, sbFetch } from '@/lib/services/execution/bff';

interface Params {
  params: Promise<{ exchange: string }>;
}

export async function DELETE(request: NextRequest, { params }: Params) {
  try {
    const { exchange } = await params;
    const { userId, authHeader } = await resolveExecutionIdentity(request);
    if (!userId) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 });
    }

    // Find the caller's credentials for this exchange (scoped to the user by
    // SignalBridge via the bearer — never by a client-supplied id).
    const listResponse = await sbFetch(
      `/api/exchanges/credentials?exchange=${encodeURIComponent(exchange)}`,
      { authHeader },
    );

    if (!listResponse.ok) {
      const error = await listResponse.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to list credentials' },
        { status: listResponse.status },
      );
    }

    const credentials = (await listResponse.json()) as Array<{ id: string; exchange: string }>;
    const targets = (credentials || []).filter((c) => c.exchange === exchange);

    if (targets.length === 0) {
      return NextResponse.json(
        { error: `No connected ${exchange} credential found` },
        { status: 404 },
      );
    }

    // Delete each matching credential. Surface the first failure honestly.
    for (const cred of targets) {
      const del = await sbFetch(`/api/exchanges/credentials/${cred.id}`, {
        method: 'DELETE',
        authHeader,
      });
      if (!del.ok) {
        const error = await del.json().catch(() => ({}));
        return NextResponse.json(
          { error: error.detail || 'Failed to delete credential' },
          { status: del.status },
        );
      }
    }

    return NextResponse.json({ data: { exchange, deleted: targets.length } });
  } catch (error) {
    console.error('[API] Exchange disconnect error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
