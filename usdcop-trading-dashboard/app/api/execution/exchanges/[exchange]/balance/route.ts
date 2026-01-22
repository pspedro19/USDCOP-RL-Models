/**
 * Exchange Balance API Route
 * ==========================
 *
 * Proxy route to SignalBridge backend /exchanges/{exchange}/balance
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://localhost:8080';

interface Params {
  params: Promise<{ exchange: string }>;
}

export async function GET(request: NextRequest, { params }: Params) {
  try {
    const { exchange } = await params;
    const authHeader = request.headers.get('authorization');

    const response = await fetch(`${BACKEND_URL}/exchanges/${exchange}/balance`, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get balance' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json({ data });
  } catch (error) {
    console.error('[API] Exchange balance error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
