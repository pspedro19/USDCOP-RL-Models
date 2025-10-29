import { NextRequest, NextResponse } from 'next/server';

/**
 * L3 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L3 layer status (Feature Engineering)
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

export async function GET(request: NextRequest) {
  try {
    // Call backend API for L3 status
    const response = await fetch(`${PIPELINE_API}/api/pipeline/l3/status`, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      return NextResponse.json({
        layer: 'L3',
        name: 'Features',
        status: 'unknown',
        pass: false,
        quality_metrics: {
          features_count: null,
          correlations_computed: null,
          forward_ic_passed: null,
          max_ic: null,
          leakage_tests_passed: null
        },
        last_update: new Date().toISOString(),
        error: 'L3 pipeline not executed yet or backend unavailable',
        message: 'Run DAG: usdcop_m5__04_l3_feature in Airflow'
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
    console.error('[L3 Status API] Error:', error);

    return NextResponse.json({
      layer: 'L3',
      name: 'Features',
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
