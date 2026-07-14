/**
 * Signal Bridge Statistics API Route
 * ===================================
 *
 * Relays to SignalBridge `GET /api/signal-bridge/statistics`.
 *
 * On a backend failure this route STILL returns HTTP 200 (so the dashboard does
 * not blow up), but it marks the payload `degraded: true` and zero-fills the
 * metrics. The dashboard reads `degraded` and shows a "data unavailable"
 * indicator instead of presenting the zeros as real $0/0 figures.
 */

import { NextRequest, NextResponse } from 'next/server';

import { getSbAuthHeader, sbFetch } from '@/lib/services/execution/bff';

function degradedPayload(days: number) {
  return {
    total_signals_received: 0,
    total_executions: 0,
    successful_executions: 0,
    failed_executions: 0,
    blocked_by_risk: 0,
    total_volume_usd: 0,
    total_pnl_usd: 0,
    avg_execution_time_ms: 0,
    period_start: new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString(),
    period_end: new Date().toISOString(),
    degraded: true,
  };
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const daysParam = searchParams.get('days') || '7';
  const days = Number.parseInt(daysParam, 10) || 7;

  try {
    const authHeader = getSbAuthHeader(request);
    const response = await sbFetch(`/api/signal-bridge/statistics?days=${days}`, {
      authHeader,
    });

    if (!response.ok) {
      // Backend down/erroring — degrade honestly instead of faking zeros.
      console.warn('[API] Signal Bridge statistics backend error:', response.status);
      return NextResponse.json(degradedPayload(days));
    }

    const data = await response.json();
    return NextResponse.json({ ...data, degraded: false });
  } catch (error) {
    console.error('[API] Signal Bridge statistics error:', error);
    return NextResponse.json(degradedPayload(days));
  }
}
