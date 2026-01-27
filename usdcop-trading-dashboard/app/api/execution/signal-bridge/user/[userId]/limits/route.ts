/**
 * User Risk Limits API Route
 * ==========================
 *
 * Proxy route to SignalBridge backend /signal-bridge/user/{userId}/limits
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';

// Default risk limits for when userId is "current" or invalid
// Using a valid RFC 4122 v4 UUID format (version 4 at pos 13, variant 8/9/a/b at pos 17)
const DEFAULT_LIMITS = {
  user_id: '00000001-0000-4000-8000-000000000001',
  max_daily_loss_pct: 2.0,
  max_trades_per_day: 10,
  max_position_size_usd: 1000,
  cooldown_minutes: 15,
  enable_short: false,
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const { userId } = await params;
    const authHeader = request.headers.get('authorization');

    // If userId is "current" or not a valid UUID, return defaults
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    if (userId === 'current' || !uuidRegex.test(userId)) {
      console.log('[API] Returning default limits for non-UUID userId:', userId);
      return NextResponse.json(DEFAULT_LIMITS);
    }

    const response = await fetch(`${BACKEND_URL}/api/signal-bridge/user/${userId}/limits`, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!response.ok) {
      // Return defaults on error to keep UI functional
      if (response.status === 404 || response.status === 422) {
        return NextResponse.json(DEFAULT_LIMITS);
      }
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get user limits' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] User limits GET error:', error);
    // Return defaults on error to keep UI functional
    return NextResponse.json(DEFAULT_LIMITS);
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const { userId } = await params;
    const authHeader = request.headers.get('authorization');
    const body = await request.json();

    // Use default UUID if userId is "current" or invalid
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const actualUserId = (userId === 'current' || !uuidRegex.test(userId))
      ? '00000001-0000-4000-8000-000000000001'
      : userId;

    const response = await fetch(`${BACKEND_URL}/api/signal-bridge/user/${actualUserId}/limits`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to update user limits' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] User limits PUT error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
