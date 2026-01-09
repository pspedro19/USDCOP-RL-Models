import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * L2 Pipeline Status - Dynamic endpoint
 * Calls backend pipeline API to get L2 layer status
 */

const PIPELINE_API = process.env.PIPELINE_API_URL || 'http://usdcop-pipeline-api:8002';

const circuitBreaker = getCircuitBreaker('l2-status-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  try {
    const data = await circuitBreaker.execute(async () => {
      const response = await fetch(`${PIPELINE_API}/api/pipeline/l2/status`, {
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error('L2 pipeline not executed yet or backend unavailable');
      }

      return response.json();
    });

    return NextResponse.json(
      createApiResponse(true, 'live', { data }),
      { headers: { 'Cache-Control': 'no-store, max-age=0' } }
    );

  } catch (error: unknown) {
    console.error('[L2 Status API] Error:', error);

    const isCircuitOpen = error instanceof CircuitOpenError;

    return NextResponse.json(
      createApiResponse(false, 'none', {
        data: {
          layer: 'L2',
          name: 'Prepared',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
        },
        error: isCircuitOpen
          ? 'Pipeline API circuit breaker is open'
          : (error instanceof Error ? error.message : 'Unknown error'),
        message: 'Run DAG: usdcop_m5__03_l2_prepare in Airflow'
      }),
      { status: 503, headers: { 'Cache-Control': 'no-store, max-age=0' } }
    );
  }
});
