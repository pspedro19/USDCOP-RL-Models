import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * L5 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L5 layer status (Model Serving)
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

const circuitBreaker = getCircuitBreaker('l5-status-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  try {
    const data = await circuitBreaker.execute(async () => {
      const response = await fetch(`${PIPELINE_API}/api/pipeline/l5/models`, {
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error('L5 pipeline not executed yet or backend unavailable');
      }

      return response.json();
    });

    // Transform L5 models response to status format
    const hasModels = data.count > 0;

    const statusData = {
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
    };

    return NextResponse.json(
      createApiResponse(true, 'live', { data: statusData }),
      { headers: { 'Cache-Control': 'no-store, max-age=0' } }
    );

  } catch (error: unknown) {
    console.error('[L5 Status API] Error:', error);

    const isCircuitOpen = error instanceof CircuitOpenError;

    return NextResponse.json(
      createApiResponse(false, 'none', {
        data: {
          layer: 'L5',
          name: 'Serving',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
        },
        error: isCircuitOpen
          ? 'Pipeline API circuit breaker is open'
          : (error instanceof Error ? error.message : 'Unknown error'),
        message: 'Run DAG: usdcop_m5__06_l5_serving in Airflow'
      }),
      { status: 503, headers: { 'Cache-Control': 'no-store, max-age=0' } }
    );
  }
});
