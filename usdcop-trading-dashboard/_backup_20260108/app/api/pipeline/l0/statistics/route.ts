/**
 * L0 Statistics Endpoint
 * GET /api/pipeline/l0/statistics
 *
 * Proxies to Pipeline Data API backend for L0 statistics
 */

import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

const circuitBreaker = getCircuitBreaker('l0-statistics-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

export const GET = withAuth(async (request, { user }) => {
  try {
    const backendUrl = process.env.PIPELINE_DATA_API_URL || 'http://pipeline-data-api:8002';

    const backendData = await circuitBreaker.execute(async () => {
      const response = await fetch(`${backendUrl}/api/pipeline/l0/statistics`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(10000)
      });

      if (!response.ok) {
        throw new Error(`Backend API returned ${response.status}`);
      }

      return response.json();
    });

    // Transform backend response to match frontend expectations
    const statistics = {
      overview: {
        totalRecords: backendData.total_records,
        dateRange: {
          earliest: backendData.date_range.earliest,
          latest: backendData.date_range.latest,
          tradingDays: backendData.date_range.days,
        },
        priceMetrics: {
          min: backendData.price_stats.min,
          max: backendData.price_stats.max,
          avg: backendData.price_stats.avg,
        },
        volume: {
          total: backendData.avg_volume || 0,
        },
      },
      sources: [{
        source: 'twelvedata',
        count: backendData.total_records,
        percentage: '100%',
        firstRecord: backendData.date_range.earliest,
        lastRecord: backendData.date_range.latest,
      }],
      dataQuality: {
        completeness: [],
        avgRecordsPerDay: Math.round(backendData.total_records / backendData.date_range.days)
      },
      hourlyDistribution: []
    };

    return NextResponse.json(
      createApiResponse(true, 'live', { data: { statistics } })
    );

  } catch (error) {
    console.error('[L0 Statistics API] Error:', error);

    const isCircuitOpen = error instanceof CircuitOpenError;

    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: isCircuitOpen
          ? 'Pipeline API circuit breaker is open'
          : 'Failed to retrieve L0 statistics',
        message: error instanceof Error ? error.message : 'Unknown error',
      }),
      { status: 500 }
    );
  }
});
