/**
 * L0 Statistics Endpoint
 * GET /api/pipeline/l0/statistics
 *
 * Proxies to Pipeline Data API backend for L0 statistics
 */

import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    // Call backend Pipeline Data API
    const backendUrl = process.env.PIPELINE_DATA_API_URL || 'http://pipeline-data-api:8002';

    const response = await fetch(`${backendUrl}/api/pipeline/l0/statistics`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(10000)
    });

    if (!response.ok) {
      throw new Error(`Backend API returned ${response.status}: ${await response.text()}`);
    }

    const backendData = await response.json();

    // Transform backend response to match frontend expectations
    return NextResponse.json({
      success: true,
      time: new Date().toISOString(),
      statistics: {
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
      },
    });

  } catch (error) {
    console.error('[L0 Statistics API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L0 statistics',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
