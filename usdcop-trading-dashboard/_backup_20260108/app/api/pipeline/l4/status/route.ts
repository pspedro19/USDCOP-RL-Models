import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * L4 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L4 layer status (RL-Ready Dataset)
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

const circuitBreaker = getCircuitBreaker('l4-status-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  try {
    const data = await circuitBreaker.execute(async () => {
      const response = await fetch(`${PIPELINE_API}/api/pipeline/l4/status`, {
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error('L4 pipeline not executed yet or backend unavailable');
      }

      return response.json();
    });

    return NextResponse.json(
      createApiResponse(true, 'live', { data }),
      { headers: { 'Cache-Control': 'no-store, max-age=0' } }
    );

  } catch (error: unknown) {
    console.error('[L4 Status API] Error:', error);

    const isCircuitOpen = error instanceof CircuitOpenError;

    return NextResponse.json(
      createApiResponse(false, 'none', {
        data: {
          layer: 'L4',
          name: 'RL-Ready',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
        },
        error: isCircuitOpen
          ? 'Pipeline API circuit breaker is open'
          : (error instanceof Error ? error.message : 'Unknown error'),
        message: 'Run DAG: usdcop_m5__05_l4_rlready in Airflow'
      }),
      { status: 503, headers: { 'Cache-Control': 'no-store, max-age=0' } }
    );
  }
});
