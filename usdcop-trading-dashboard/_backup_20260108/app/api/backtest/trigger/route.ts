// CRITICAL: NO MOCK DATA - Real Airflow DAG triggers only
/**
 * Backtest Trigger API
 *
 * CRITICAL: NO MOCK DATA
 * This endpoint triggers real Airflow DAG runs.
 * If backend unavailable, return 503 error.
 */

import { NextRequest, NextResponse } from 'next/server';
import { withAuth } from '@/lib/auth/api-auth';
import { createApiResponse } from '@/lib/types/api';

/**
 * POST /api/backtest/trigger
 *
 * Trigger a new backtest run via Airflow DAG
 */
export const POST = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { force = false } = body;

    const backendUrl = process.env.BACKTEST_API_URL;

    if (!backendUrl) {
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Backtest API not configured',
          message: 'BACKTEST_API_URL environment variable is required'
        }),
        { status: 503 }
      );
    }

    const response = await fetch(`${backendUrl}/api/backtest/trigger`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force }),
      signal: AbortSignal.timeout(10000)
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    const successResponse = createApiResponse(true, 'live', { data });
    successResponse.metadata.latency = Date.now() - startTime;
    return NextResponse.json(successResponse);

  } catch (error) {
    // NO MOCK FALLBACK - Return clear error
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Backtest service unavailable',
        message: 'Cannot trigger backtest. Ensure Backtest API is running.'
      }),
      { status: 503 }
    );
  }
});

/**
 * GET /api/backtest/trigger
 *
 * Get trigger status and available parameters
 */
export const GET = withAuth(async (request, { user }) => {
  return NextResponse.json(
    createApiResponse(true, 'live', {
      data: {
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
      }
    })
  );
});