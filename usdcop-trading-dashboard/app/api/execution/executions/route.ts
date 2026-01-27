/**
 * Executions API Route
 * ====================
 *
 * Proxy route to SignalBridge backend /executions
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8085';

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
