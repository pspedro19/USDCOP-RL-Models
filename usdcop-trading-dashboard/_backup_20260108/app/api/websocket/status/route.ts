import { NextRequest, NextResponse } from 'next/server';
import { apiConfig } from '@/lib/config/api.config';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';

const websocketCircuitBreaker = getCircuitBreaker('websocket-status-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

/**
 * WebSocket Status API
 * ====================
 *
 * Connects to REAL WebSocket server to get actual connection status
 * NO MOCK DATA - All metrics from real WebSocket monitoring
 */

const TRADING_BASE = apiConfig.trading.baseUrl;

interface WebSocketStatusResponse {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  connection: {
    active: boolean;
    clientsConnected: number;
    uptime: string;
    lastRestart: string;
    reconnectAttempts: number;
  };
  readySignal: {
    active: boolean;
    waiting: boolean;
    handoverComplete: boolean;
    lastHandover: string;
    pendingRecords: number;
    handoverLatency: number;
  };
  dataFlow: {
    recordsPerMinute: number;
    totalRecords: number;
    lastUpdate: string;
    bufferSize: number;
    maxBufferSize: number;
  };
  marketHours: {
    isOpen: boolean;
    nextOpen: string;
    nextClose: string;
    timezone: string;
  };
  performance: {
    avgLatency: number;
    maxLatency: number;
    errorRate: number;
    throughput: number;
  };
}

// Calculate actual market hours for USD/COP
function getMarketHoursStatus(): { isOpen: boolean; nextOpen: string; nextClose: string } {
  const now = new Date();
  const bogotaTime = new Date(now.toLocaleString("en-US", { timeZone: "America/Bogota" }));
  const hours = bogotaTime.getHours();
  const minutes = bogotaTime.getMinutes();
  const dayOfWeek = bogotaTime.getDay();
  const currentTime = hours + minutes / 60;

  // Market hours: Mon-Fri 8:00 AM - 12:55 PM COT
  const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
  const isOpen = isWeekday && currentTime >= 8 && currentTime <= 12.917;

  // Calculate next open/close
  let nextOpen = new Date(bogotaTime);
  let nextClose = new Date(bogotaTime);

  if (isOpen) {
    // Market is open, next close is today at 12:55
    nextClose.setHours(12, 55, 0, 0);
    // Next open is tomorrow at 8:00
    nextOpen.setDate(nextOpen.getDate() + 1);
    nextOpen.setHours(8, 0, 0, 0);
    if (nextOpen.getDay() === 0) nextOpen.setDate(nextOpen.getDate() + 1); // Skip Sunday
    if (nextOpen.getDay() === 6) nextOpen.setDate(nextOpen.getDate() + 2); // Skip Saturday
  } else {
    // Market is closed
    if (currentTime < 8 && isWeekday) {
      // Before market opens today
      nextOpen.setHours(8, 0, 0, 0);
    } else {
      // After market closed or weekend
      nextOpen.setDate(nextOpen.getDate() + 1);
      while (nextOpen.getDay() === 0 || nextOpen.getDay() === 6) {
        nextOpen.setDate(nextOpen.getDate() + 1);
      }
      nextOpen.setHours(8, 0, 0, 0);
    }
    nextClose = new Date(nextOpen);
    nextClose.setHours(12, 55, 0, 0);
  }

  return {
    isOpen,
    nextOpen: nextOpen.toISOString(),
    nextClose: nextClose.toISOString()
  };
}

export const GET = withAuth(async (request, { user }) => {
  try {
    // Try to get status from real WebSocket server
    const wsStatusUrl = `${TRADING_BASE}/api/websocket/status`;

    try {
      const data = await websocketCircuitBreaker.execute(async () => {
        const response = await fetch(wsStatusUrl, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(5000),
          cache: 'no-store'
        });

        if (response.ok) {
          return response.json();
        }
        throw new Error('Backend not available');
      });

      return NextResponse.json({
        ...data,
        source: 'websocket_server',
        timestamp: new Date().toISOString()
      }, {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      });
    } catch (backendError) {
      const isCircuitOpen = backendError instanceof CircuitOpenError;
      console.warn('[WebSocket Status API] Backend unavailable:', isCircuitOpen ? 'Circuit open' : backendError);
    }

    // Return actual market hours status even if WebSocket server is unavailable
    const marketHours = getMarketHoursStatus();

    return NextResponse.json({
      success: false,
      error: 'WebSocket server unavailable',
      message: 'Cannot connect to WebSocket server for status. The server may not be running.',
      marketHours: {
        ...marketHours,
        timezone: 'America/Bogota'
      },
      troubleshooting: {
        backend_url: TRADING_BASE,
        expected_endpoint: '/api/websocket/status',
        how_to_start: 'npm run ws (starts server/websocket-server.js)'
      },
      timestamp: new Date().toISOString()
    }, {
      status: 503,
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate'
      }
    });

  } catch (error) {
    console.error('[WebSocket Status] Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'WebSocket status check failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
});

export const POST = withAuth(async (request, { user }) => {
  try {
    const body = await request.json();

    // Forward to real WebSocket server
    const wsControlUrl = `${TRADING_BASE}/api/websocket/control`;

    try {
      const response = await fetch(wsControlUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000)
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json({
          ...data,
          source: 'websocket_server'
        });
      }
    } catch (backendError) {
      console.warn('[WebSocket Status POST] Backend unavailable:', backendError);
    }

    // Return error for operations that require the server (NO MOCK RESPONSES)
    return NextResponse.json({
      success: false,
      error: 'WebSocket server unavailable',
      message: 'Cannot perform WebSocket operations without server connection.',
      action_attempted: body.action,
      timestamp: new Date().toISOString()
    }, { status: 503 });

  } catch (error) {
    console.error('[WebSocket Status POST] Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process WebSocket request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
});
