/**
 * Signal Bridge History API Route
 * ================================
 *
 * Proxy route to SignalBridge backend /signal-bridge/history
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';

// Default empty response for when backend fails
const DEFAULT_HISTORY = {
  items: [],
  total: 0,
  page: 1,
  limit: 20,
  has_more: false,
};

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    const searchParams = request.nextUrl.searchParams;

    // Forward all query params
    const queryString = searchParams.toString();
    const url = queryString
      ? `${BACKEND_URL}/api/signal-bridge/history?${queryString}`
      : `${BACKEND_URL}/api/signal-bridge/history`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!response.ok) {
      // Return empty history on error to keep UI functional
      if (response.status === 500) {
        console.warn('[API] Signal Bridge history backend error, returning empty');
        return NextResponse.json(DEFAULT_HISTORY);
      }
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get execution history' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] Signal Bridge history error:', error);
    // Return empty history on error to keep UI functional
    return NextResponse.json(DEFAULT_HISTORY);
  }
}
