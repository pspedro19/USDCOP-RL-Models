import { NextRequest, NextResponse } from 'next/server';

/**
 * Multi-Strategy Equity Curves Endpoint
 * Returns time-series equity data for all strategies
 */

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const hours = searchParams.get('hours') || '24';
  const resolution = searchParams.get('resolution') || '5m';

  try {
    const response = await fetch(
      `${MULTI_MODEL_API}/api/models/equity-curves?hours=${hours}&resolution=${resolution}`,
      {
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store'
      }
    );

    if (!response.ok) {
      return NextResponse.json({
        success: false,
        error: 'Multi-model API unavailable',
        equity_curves: [],
        metadata: {},
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
    console.error('[Multi-Strategy Equity Curves API] Error:', error);

    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to fetch equity curves',
      equity_curves: [],
      metadata: {},
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
