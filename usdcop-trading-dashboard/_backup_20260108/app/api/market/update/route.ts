/**
 * Market Data Update API Endpoint
 * Handles real-time data updates from WebSocket to database
 * Ensures seamless integration between live data and historical records
 */

import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse, measureLatency } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

interface MarketUpdateData {
  timestamp: string;
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  source: string;
}

/**
 * POST /api/market/update
 * Update market data in real-time
 */
export const POST = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const data: MarketUpdateData = await request.json();

    // Validate required fields
    if (!data.timestamp || !data.symbol || !data.price || !data.source) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Missing required fields: timestamp, symbol, price, source',
          latency
        }),
        { status: 400 }
      );
    }

    // Validate data types and ranges
    if (typeof data.price !== 'number' || data.price <= 0) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Invalid price value',
          latency
        }),
        { status: 400 }
      );
    }

    if (data.bid && (typeof data.bid !== 'number' || data.bid <= 0)) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Invalid bid value',
          latency
        }),
        { status: 400 }
      );
    }

    if (data.ask && (typeof data.ask !== 'number' || data.ask <= 0)) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Invalid ask value',
          latency
        }),
        { status: 400 }
      );
    }

    // Validate timestamp
    const timestamp = new Date(data.timestamp);
    if (isNaN(timestamp.getTime())) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Invalid timestamp format',
          latency
        }),
        { status: 400 }
      );
    }

    // Check if timestamp is too old or too far in the future
    const now = new Date();
    const timeDiff = Math.abs(now.getTime() - timestamp.getTime());
    const maxTimeDiff = 24 * 60 * 60 * 1000; // 24 hours

    if (timeDiff > maxTimeDiff) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Timestamp is too old or too far in the future',
          latency
        }),
        { status: 400 }
      );
    }

    // Forward to backend API for database update
    const backendUrl = 'http://localhost:8000';
    const backendResponse = await fetch(`${backendUrl}/api/market/update`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'USDCOP-Dashboard/1.0',
        'X-Forwarded-For': request.ip || 'unknown',
        'X-Real-IP': request.ip || 'unknown'
      },
      body: JSON.stringify({
        timestamp: data.timestamp,
        symbol: data.symbol.toUpperCase(),
        price: Number(data.price.toFixed(4)),
        bid: data.bid ? Number(data.bid.toFixed(4)) : null,
        ask: data.ask ? Number(data.ask.toFixed(4)) : null,
        volume: data.volume ? Math.max(0, Math.floor(data.volume)) : null,
        source: data.source,
        created_at: new Date().toISOString()
      }),
      // Add timeout to prevent hanging
      signal: AbortSignal.timeout(5000)
    });

    if (!backendResponse.ok) {
      console.error('Backend API error:', backendResponse.statusText);
      const latency = Date.now() - startTime;

      // Don't return success when backend is unavailable - return 503 with dataSource: 'none'
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Backend API error',
          message: 'Data received, but backend update failed',
          latency,
          backendUrl
        }),
        { status: 503 }
      );
    }

    const backendResult = await backendResponse.json();
    const latency = Date.now() - startTime;

    return NextResponse.json(
      createApiResponse(true, 'live', {
        data: {
          message: 'Market data updated successfully',
          symbol: data.symbol,
          price: data.price,
          timestamp: data.timestamp,
          source: data.source,
          backend_response: backendResult
        },
        latency,
        backendUrl
      })
    );

  } catch (error: any) {
    console.error('Error updating market data:', error);
    const latency = Date.now() - startTime;

    // Distinguish between different types of errors
    if (error.name === 'AbortError') {
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Backend timeout',
          message: 'Database update timed out, but data was received',
          latency
        }),
        { status: 408 }
      );
    }

    if (error.name === 'SyntaxError') {
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Invalid JSON format in request body',
          latency
        }),
        { status: 400 }
      );
    }

    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Internal server error',
        message: 'Failed to update market data',
        latency
      }),
      { status: 500 }
    );
  }
});

/**
 * GET /api/market/update
 * Get current update status and statistics
 */
export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const backendUrl = 'http://localhost:8000';

    // Get backend health status
    const healthResponse = await fetch(`${backendUrl}/api/health`, {
      method: 'GET',
      headers: {
        'User-Agent': 'USDCOP-Dashboard/1.0'
      },
      signal: AbortSignal.timeout(3000)
    });

    let backendHealth = null;
    if (healthResponse.ok) {
      backendHealth = await healthResponse.json();
    }

    // Get latest market data
    const latestResponse = await fetch(`${backendUrl}/api/latest/USDCOP`, {
      method: 'GET',
      headers: {
        'User-Agent': 'USDCOP-Dashboard/1.0'
      },
      signal: AbortSignal.timeout(3000)
    });

    let latestData = null;
    if (latestResponse.ok) {
      latestData = await latestResponse.json();
    }

    const latency = Date.now() - startTime;
    const dataSource = (backendHealth && latestData) ? 'live' : 'none';

    return NextResponse.json(
      createApiResponse(true, dataSource, {
        data: {
          update_endpoint_status: 'operational',
          backend_health: backendHealth,
          latest_data: latestData,
          market_status: {
            is_open: isMarketOpen(),
            trading_hours: '08:00 - 12:55 COT',
            timezone: 'America/Bogota'
          },
          update_statistics: {
            last_update: latestData?.timestamp || null,
            total_records: backendHealth?.total_records || 0,
            data_source: latestData?.source || 'unknown'
          }
        },
        latency,
        backendUrl
      })
    );

  } catch (error: any) {
    console.error('Error getting update status:', error);
    const latency = Date.now() - startTime;

    return NextResponse.json(
      createApiResponse(false, 'none', {
        data: {
          update_endpoint_status: 'degraded',
          backend_health: null,
          latest_data: null,
          market_status: {
            is_open: isMarketOpen(),
            trading_hours: '08:00 - 12:55 COT',
            timezone: 'America/Bogota'
          }
        },
        error: error.message,
        latency
      })
    );
  }
});

/**
 * Check if market is currently open
 */
function isMarketOpen(): boolean {
  try {
    const now = new Date();
    const bogotaTime = new Date(now.toLocaleString("en-US", {timeZone: "America/Bogota"}));
    const currentHour = bogotaTime.getHours() + (bogotaTime.getMinutes() / 60);
    const dayOfWeek = bogotaTime.getDay();

    // Market is open Monday to Friday, 8:00 AM to 12:55 PM COT
    return dayOfWeek >= 1 && dayOfWeek <= 5 &&
           currentHour >= 8 &&
           currentHour <= 12.92; // 12:55 PM
  } catch (error) {
    console.error('Error checking market status:', error);
    return false;
  }
}

/**
 * OPTIONS /api/market/update
 * Handle CORS preflight requests
 */
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
}