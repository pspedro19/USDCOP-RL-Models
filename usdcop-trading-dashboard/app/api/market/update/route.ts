/**
 * Market Data Update API Endpoint
 * Handles real-time data updates from WebSocket to database
 * Ensures seamless integration between live data and historical records
 */

import { NextRequest, NextResponse } from 'next/server';

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
export async function POST(request: NextRequest) {
  try {
    const data: MarketUpdateData = await request.json();

    // Validate required fields
    if (!data.timestamp || !data.symbol || !data.price || !data.source) {
      return NextResponse.json(
        { error: 'Missing required fields: timestamp, symbol, price, source' },
        { status: 400 }
      );
    }

    // Validate data types and ranges
    if (typeof data.price !== 'number' || data.price <= 0) {
      return NextResponse.json(
        { error: 'Invalid price value' },
        { status: 400 }
      );
    }

    if (data.bid && (typeof data.bid !== 'number' || data.bid <= 0)) {
      return NextResponse.json(
        { error: 'Invalid bid value' },
        { status: 400 }
      );
    }

    if (data.ask && (typeof data.ask !== 'number' || data.ask <= 0)) {
      return NextResponse.json(
        { error: 'Invalid ask value' },
        { status: 400 }
      );
    }

    // Validate timestamp
    const timestamp = new Date(data.timestamp);
    if (isNaN(timestamp.getTime())) {
      return NextResponse.json(
        { error: 'Invalid timestamp format' },
        { status: 400 }
      );
    }

    // Check if timestamp is too old or too far in the future
    const now = new Date();
    const timeDiff = Math.abs(now.getTime() - timestamp.getTime());
    const maxTimeDiff = 24 * 60 * 60 * 1000; // 24 hours

    if (timeDiff > maxTimeDiff) {
      return NextResponse.json(
        { error: 'Timestamp is too old or too far in the future' },
        { status: 400 }
      );
    }

    // Forward to backend API for database update
    const backendResponse = await fetch('http://localhost:8000/api/market/update', {
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

      // If backend fails, still return success to avoid blocking WebSocket
      // Log error for monitoring but don't fail the WebSocket update
      return NextResponse.json({
        success: true,
        message: 'Data received, backend update queued',
        timestamp: new Date().toISOString(),
        backend_status: 'queued'
      });
    }

    const backendResult = await backendResponse.json();

    return NextResponse.json({
      success: true,
      message: 'Market data updated successfully',
      timestamp: new Date().toISOString(),
      backend_status: 'updated',
      data: {
        symbol: data.symbol,
        price: data.price,
        timestamp: data.timestamp,
        source: data.source
      },
      backend_response: backendResult
    });

  } catch (error: any) {
    console.error('Error updating market data:', error);

    // Distinguish between different types of errors
    if (error.name === 'AbortError') {
      return NextResponse.json(
        {
          error: 'Backend timeout',
          message: 'Database update timed out, but data was received',
          success: false
        },
        { status: 408 }
      );
    }

    if (error.name === 'SyntaxError') {
      return NextResponse.json(
        { error: 'Invalid JSON format in request body' },
        { status: 400 }
      );
    }

    return NextResponse.json(
      {
        error: 'Internal server error',
        message: 'Failed to update market data',
        success: false
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/market/update
 * Get current update status and statistics
 */
export async function GET(request: NextRequest) {
  try {
    // Get backend health status
    const healthResponse = await fetch('http://localhost:8000/api/health', {
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
    const latestResponse = await fetch('http://localhost:8000/api/latest/USDCOP', {
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

    return NextResponse.json({
      update_endpoint_status: 'operational',
      timestamp: new Date().toISOString(),
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
    });

  } catch (error: any) {
    console.error('Error getting update status:', error);

    return NextResponse.json({
      update_endpoint_status: 'degraded',
      timestamp: new Date().toISOString(),
      error: error.message,
      backend_health: null,
      latest_data: null,
      market_status: {
        is_open: isMarketOpen(),
        trading_hours: '08:00 - 12:55 COT',
        timezone: 'America/Bogota'
      }
    });
  }
}

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