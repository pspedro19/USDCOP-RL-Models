/**
 * Signal Bridge Statistics API Route
 * ===================================
 *
 * Proxy route to SignalBridge backend /signal-bridge/statistics
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8085';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    const { searchParams } = new URL(request.url);
    const days = searchParams.get('days') || '7';

    const response = await fetch(`${BACKEND_URL}/api/signal-bridge/statistics?days=${days}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader && { Authorization: authHeader }),
      },
    });

    if (!response.ok) {
      // If backend returns 500, return mock statistics to keep UI functional
      if (response.status === 500) {
        console.warn('[API] Signal Bridge statistics backend error, returning defaults');
        return NextResponse.json({
          total_signals_received: 0,
          total_executions: 0,
          successful_executions: 0,
          failed_executions: 0,
          blocked_by_risk: 0,
          total_volume_usd: 0,
          total_pnl_usd: 0,
          avg_execution_time_ms: 0,
          period_start: new Date(Date.now() - parseInt(days) * 24 * 60 * 60 * 1000).toISOString(),
          period_end: new Date().toISOString(),
        });
      }

      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: error.detail || 'Failed to get statistics' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('[API] Signal Bridge statistics error:', error);
    // Return default statistics on error to keep UI functional
    return NextResponse.json({
      total_signals_received: 0,
      total_executions: 0,
      successful_executions: 0,
      failed_executions: 0,
      blocked_by_risk: 0,
      total_volume_usd: 0,
      total_pnl_usd: 0,
      avg_execution_time_ms: 0,
      period_start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
      period_end: new Date().toISOString(),
    });
  }
}
