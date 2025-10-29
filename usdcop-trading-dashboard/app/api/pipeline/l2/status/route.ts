import { NextRequest, NextResponse } from 'next/server';

/**
 * L2 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L2 layer status
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

export async function GET(request: NextRequest) {
  try {
    // Call backend API for L2 status
    const response = await fetch(`${PIPELINE_API}/api/pipeline/l2/status`, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      // Backend not ready or L2 not executed
      return NextResponse.json({
        layer: 'L2',
        name: 'Prepared',
        status: 'unknown',
        pass: false,
        quality_metrics: {
          indicators_count: null,
          winsorization_pct: null,
          missing_values_pct: null,
          rows: null,
          columns: null
        },
        last_update: new Date().toISOString(),
        error: 'L2 pipeline not executed yet or backend unavailable',
        message: 'Run DAG: usdcop_m5__03_l2_prepare in Airflow'
      }, {
        status: 503,
        headers: { 'Cache-Control': 'no-store, max-age=0' }
      });
    }

    const data = await response.json();

    return NextResponse.json(data, {
      headers: {
        'Cache-Control': 'no-store, max-age=0',
      },
    });

  } catch (error: unknown) {
    console.error('[L2 Status API] Error:', error);

    return NextResponse.json({
      layer: 'L2',
      name: 'Prepared',
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
