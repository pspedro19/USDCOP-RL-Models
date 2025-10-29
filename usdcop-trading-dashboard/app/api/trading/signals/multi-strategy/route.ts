import { NextRequest, NextResponse } from 'next/server';

/**
 * Multi-Strategy Signals Endpoint
 * Proxies to multi-model trading API (port 8006)
 */

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://usdcop-multi-model-api:8006';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);

  try {
    const response = await fetch(
      `${MULTI_MODEL_API}/api/models/signals/latest?${searchParams}`,
      {
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store'
      }
    );

    if (!response.ok) {
      return NextResponse.json({
        success: false,
        error: 'Multi-model API unavailable',
        signals: [],
        market_price: 0,
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
    console.error('[Multi-Strategy Signals API] Error:', error);

    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to fetch signals',
      signals: [],
      market_price: 0,
      timestamp: new Date().toISOString()
    }, {
      status: 500,
      headers: { 'Cache-Control': 'no-store' }
    });
  }
}
