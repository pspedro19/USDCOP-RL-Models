/**
 * Executions API Route
 * ====================
 *
 * Proxy route to SignalBridge backend /executions
 */

import { NextRequest, NextResponse } from 'next/server';

import { SIGNALBRIDGE_BACKEND_URL as BACKEND_URL } from '@/lib/services/execution/bff';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    const { searchParams } = new URL(request.url);

    // Forward query params
    const queryString = searchParams.toString();
    const url = queryString
      ? `${BACKEND_URL}/api/executions?${queryString}`
      : `${BACKEND_URL}/api/executions`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get executions' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] Executions error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
