/**
 * Real Backtest API - Proxy to Python BacktestEngine
 * ===================================================
 *
 * This endpoint proxies to the backtest-api Python service which runs
 * the SAME BacktestEngine as L4 validation.
 *
 * Contract: CTR-BACKTEST-REAL-001
 */

import { NextRequest, NextResponse } from 'next/server';

const BACKTEST_API_URL = process.env.INFERENCE_API_URL || 'http://backtest-api:8000';

interface BacktestRequest {
  proposal_id: string;
  start_date: string;
  end_date: string;
}

/**
 * POST /api/backtest/real
 *
 * Proxy to Python backtest-api service for REAL backtest using L4 BacktestEngine
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as BacktestRequest;
    const { proposal_id, start_date, end_date } = body;

    if (!proposal_id || !start_date || !end_date) {
      return NextResponse.json(
        { error: 'Missing required fields: proposal_id, start_date, end_date' },
        { status: 400 }
      );
    }

    console.log(`[RealBacktest] Calling Python service for ${proposal_id}: ${start_date} to ${end_date}`);

    // Call the Python backtest-api service
    const url = `${BACKTEST_API_URL}/api/v1/backtest/real?proposal_id=${encodeURIComponent(proposal_id)}&start_date=${start_date}&end_date=${end_date}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // Timeout after 3 minutes
      signal: AbortSignal.timeout(180000),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[RealBacktest] Python service error: ${response.status} - ${errorText}`);
      return NextResponse.json(
        { error: `Backtest service error: ${errorText}`, success: false },
        { status: response.status }
      );
    }

    const result = await response.json();

    console.log(`[RealBacktest] Complete: ${result.trade_count} trades, ${result.summary?.win_rate}% WR`);

    return NextResponse.json(result);

  } catch (error) {
    console.error('[RealBacktest] Error:', error);

    // Check for timeout
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json(
        { error: 'Backtest timed out after 3 minutes', success: false },
        { status: 504 }
      );
    }

    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : 'Backtest failed',
        success: false
      },
      { status: 500 }
    );
  }
}

/**
 * GET for health check
 */
export async function GET() {
  // Check if Python service is available
  try {
    const healthUrl = `${BACKTEST_API_URL}/api/v1/health`;
    const response = await fetch(healthUrl, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    const pythonServiceStatus = response.ok ? 'connected' : 'unavailable';

    return NextResponse.json({
      service: 'Real Backtest API Proxy',
      status: 'ready',
      pythonService: pythonServiceStatus,
      description: 'Proxies to Python backtest-api which uses REAL BacktestEngine with model from proposal lineage',
      config: {
        stop_loss: '-2.5%',
        take_profit: '+3%',
        trailing_stop: true,
        thresholds: '±0.50',
        transaction_cost: '2.5 bps'
      }
    });
  } catch {
    return NextResponse.json({
      service: 'Real Backtest API Proxy',
      status: 'degraded',
      pythonService: 'unavailable',
      error: 'Cannot reach Python backtest service'
    });
  }
}
