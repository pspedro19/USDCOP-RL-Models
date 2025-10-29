import { NextRequest, NextResponse } from 'next/server';

/**
 * L6 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L6 layer status (Backtest Results)
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

export async function GET(request: NextRequest) {
  try {
    // Call backend API for L6 backtest results
    const response = await fetch(`${PIPELINE_API}/api/pipeline/l6/backtest-results?split=test`, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      return NextResponse.json({
        layer: 'L6',
        name: 'Backtest',
        status: 'unknown',
        pass: false,
        quality_metrics: {
          total_trades: null,
          win_rate_pct: null,
          sharpe_ratio: null,
          sortino_ratio: null,
          max_drawdown_pct: null,
          cagr_pct: null
        },
        last_update: new Date().toISOString(),
        error: 'L6 pipeline not executed yet or backend unavailable',
        message: 'Run DAG: usdcop_m5__07_l6_backtest_referencia in Airflow'
      }, {
        status: 503,
        headers: { 'Cache-Control': 'no-store, max-age=0' }
      });
    }

    const data = await response.json();

    // Check if backtest has valid results
    const hasResults = data.kpis && Object.keys(data.kpis).length > 0;
    const passesThreshold = hasResults && data.kpis.sortino_ratio >= 1.0;

    return NextResponse.json({
      layer: 'L6',
      name: 'Backtest',
      status: passesThreshold ? 'pass' : (hasResults ? 'warning' : 'unknown'),
      pass: passesThreshold,
      quality_metrics: {
        total_trades: data.kpis?.total_trades || null,
        win_rate_pct: data.kpis?.win_rate_pct || null,
        sharpe_ratio: data.kpis?.sharpe_ratio || null,
        sortino_ratio: data.kpis?.sortino_ratio || null,
        max_drawdown_pct: data.kpis?.max_drawdown_pct || null,
        cagr_pct: data.kpis?.cagr_pct || null,
        calmar_ratio: data.kpis?.calmar_ratio || null
      },
      last_update: data.timestamp || new Date().toISOString(),
      run_id: data.run_id
    }, {
      headers: {
        'Cache-Control': 'no-store, max-age=0',
      },
    });

  } catch (error: unknown) {
    console.error('[L6 Status API] Error:', error);

    return NextResponse.json({
      layer: 'L6',
      name: 'Backtest',
      status: 'error',
      pass: false,
      quality_metrics: {},
      last_update: new Date().toISOString(),
      error: error instanceof Error ? error.message : 'Unknown error',
      message: 'Backend API unavailable'
    }, {
      status: 503,
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });
  }
}
