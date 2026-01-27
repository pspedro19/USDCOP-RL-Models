/**
 * SignalBridge Auth Login API Route
 * ==================================
 *
 * Proxy route to SignalBridge backend /auth/login
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://localhost:8085';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] SignalBridge auth login error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
