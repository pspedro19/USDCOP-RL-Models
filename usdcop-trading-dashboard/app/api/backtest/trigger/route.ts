/**
 * Backtest Trigger API Endpoint
 *
 * Triggers new backtest execution via Airflow DAG
 */

import { NextRequest, NextResponse } from 'next/server';

/**
 * POST /api/backtest/trigger
 *
 * Trigger a new backtest run via Airflow DAG
 */
export async function POST(request: NextRequest) {
  console.log('[BacktestTrigger] POST /api/backtest/trigger');

  try {
    const body = await request.json();
    const { forceRebuild = false } = body;

    // In production, this would trigger the Airflow DAG
    // Example: curl -X POST "http://airflow:8080/api/v1/dags/usdcop_m5__07_l6_backtest_referencia/dagRuns"

    const mockRunId = `L6_${new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14)}_${Math.random().toString(36).substr(2, 6)}`;

    console.log(`[BacktestTrigger] Triggered backtest run: ${mockRunId}, forceRebuild: ${forceRebuild}`);

    // Mock successful response
    return NextResponse.json({
      success: true,
      message: 'Backtest triggered successfully',
      runId: mockRunId,
      estimatedDuration: '5-10 minutes',
      dagId: 'usdcop_m5__07_l6_backtest_referencia',
      forceRebuild
    });

  } catch (error) {
    console.error('[BacktestTrigger] Error triggering backtest:', error);

    return NextResponse.json({
      success: false,
      error: 'Failed to trigger backtest',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

/**
 * GET /api/backtest/trigger
 *
 * Get trigger status and available parameters
 */
export async function GET(request: NextRequest) {
  return NextResponse.json({
    success: true,
    message: 'Backtest trigger endpoint available',
    methods: ['POST'],
    parameters: {
      forceRebuild: 'boolean - Force rebuild of all data layers (default: false)'
    },
    example: {
      method: 'POST',
      body: {
        forceRebuild: false
      }
    }
  });
}