/**
 * Signal Bridge Status API Route
 * ===============================
 *
 * Proxy route to SignalBridge backend /signal-bridge/status
 */

import { NextRequest, NextResponse } from 'next/server';

import { SIGNALBRIDGE_BACKEND_URL as BACKEND_URL } from '@/lib/services/execution/bff';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');

    const response = await fetch(`${BACKEND_URL}/api/signal-bridge/status`, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get bridge status' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] Signal Bridge status error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
