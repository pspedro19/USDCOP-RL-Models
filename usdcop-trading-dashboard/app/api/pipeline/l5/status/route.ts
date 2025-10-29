import { NextRequest, NextResponse } from 'next/server';

/**
 * L5 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L5 layer status (Model Serving)
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

export async function GET(request: NextRequest) {
  try {
    // Call backend API for L5 status
    const response = await fetch(`${PIPELINE_API}/api/pipeline/l5/models`, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      return NextResponse.json({
        layer: 'L5',
        name: 'Serving',
        status: 'unknown',
        pass: false,
        quality_metrics: {
          active_models: 0,
          model_ready: false,
          inference_latency_ms: null,
          last_training: null
        },
        last_update: new Date().toISOString(),
        error: 'L5 pipeline not executed yet or backend unavailable',
        message: 'Run DAG: usdcop_m5__06_l5_serving in Airflow'
      }, {
        status: 503,
        headers: { 'Cache-Control': 'no-store, max-age=0' }
      });
    }

    const data = await response.json();

    // Transform L5 models response to status format
    const hasModels = data.count > 0;

    return NextResponse.json({
      layer: 'L5',
      name: 'Serving',
      status: hasModels ? 'pass' : 'unknown',
      pass: hasModels,
      quality_metrics: {
        active_models: data.count || 0,
        model_ready: hasModels,
        inference_latency_ms: data.models?.[0]?.inference_latency_ms || null,
        last_training: data.models?.[0]?.training_date || null,
        model_path: data.models?.[0]?.path || null
      },
      last_update: data.timestamp || new Date().toISOString()
    }, {
      headers: {
        'Cache-Control': 'no-store, max-age=0',
      },
    });

  } catch (error: unknown) {
    console.error('[L5 Status API] Error:', error);

    return NextResponse.json({
      layer: 'L5',
      name: 'Serving',
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
