import { NextRequest, NextResponse } from 'next/server';

/**
 * Multi-Strategy Performance Endpoint
 * Returns comparative performance metrics for all strategies
 */

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const periodDays = searchParams.get('period_days') || '30';

  try {
    const response = await fetch(
      `${MULTI_MODEL_API}/api/models/performance/comparison?period_days=${periodDays}`,
      {
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store'
      }
    );

    if (!response.ok) {
      return NextResponse.json({
        success: false,
        error: 'Multi-model API unavailable',
        strategies: [],
        portfolio_total: {},
        timestamp: new Date().toISOString()
      }, {
        status: 503,
        headers: { 'Cache-Control': 'no-store' }
      });
    }

    const data = await response.json();

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });

  } catch (error: any) {
    console.error('[Multi-Strategy Performance API] Error:', error);

    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to fetch performance metrics',
      strategies: [],
      portfolio_total: {},
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
