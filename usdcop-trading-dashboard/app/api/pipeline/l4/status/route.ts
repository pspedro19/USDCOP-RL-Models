import { NextRequest, NextResponse } from 'next/server';

/**
 * L4 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L4 layer status (RL-Ready Dataset)
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

export async function GET(request: NextRequest) {
  try {
    // Call backend API for L4 status
    const response = await fetch(`${PIPELINE_API}/api/pipeline/l4/status`, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      return NextResponse.json({
        layer: 'L4',
        name: 'RL-Ready',
        status: 'unknown',
        pass: false,
        quality_metrics: {
          episodes: null,
          train_episodes: null,
          val_episodes: null,
          test_episodes: null,
          observation_features: null,
          max_clip_rate_pct: null,
          reward_rmse: null
        },
        last_update: new Date().toISOString(),
        error: 'L4 pipeline not executed yet or backend unavailable',
        message: 'Run DAG: usdcop_m5__05_l4_rlready in Airflow'
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
    console.error('[L4 Status API] Error:', error);

    return NextResponse.json({
      layer: 'L4',
      name: 'RL-Ready',
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
