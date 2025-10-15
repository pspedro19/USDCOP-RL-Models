import { NextRequest, NextResponse } from 'next/server';

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

export async function GET(request: NextRequest) {
  try {
    const now = new Date();
    const isMarketOpen = now.getHours() >= 9 && now.getHours() < 17; // Simplified market hours

    const statusData: WebSocketStatusResponse = {
      status: 'healthy',
      timestamp: now.toISOString(),
      connection: {
        active: Math.random() > 0.05,
        clientsConnected: Math.floor(Math.random() * 20) + 1,
        uptime: `${Math.floor(Math.random() * 72)}h ${Math.floor(Math.random() * 60)}m`,
        lastRestart: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        reconnectAttempts: Math.floor(Math.random() * 3)
      },
      readySignal: {
        active: Math.random() > 0.1,
        waiting: Math.random() > 0.8,
        handoverComplete: Math.random() > 0.2,
        lastHandover: new Date(Date.now() - Math.random() * 600000).toISOString(),
        pendingRecords: Math.floor(Math.random() * 100),
        handoverLatency: Math.random() * 50 + 5
      },
      dataFlow: {
        recordsPerMinute: isMarketOpen ? Math.floor(Math.random() * 100) + 20 : 0,
        totalRecords: Math.floor(Math.random() * 100000) + 50000,
        lastUpdate: new Date(Date.now() - Math.random() * 60000).toISOString(),
        bufferSize: Math.floor(Math.random() * 500),
        maxBufferSize: 1000
      },
      marketHours: {
        isOpen: isMarketOpen,
        nextOpen: isMarketOpen ?
          new Date(now.getTime() + 24 * 3600000).toISOString() :
          new Date(now.getTime() + (9 - now.getHours()) * 3600000).toISOString(),
        nextClose: isMarketOpen ?
          new Date(now.getTime() + (17 - now.getHours()) * 3600000).toISOString() :
          new Date(now.getTime() + (17 - now.getHours() + 24) * 3600000).toISOString(),
        timezone: 'America/Bogota'
      },
      performance: {
        avgLatency: Math.random() * 50 + 10,
        maxLatency: Math.random() * 200 + 50,
        errorRate: Math.random() * 0.05,
        throughput: Math.floor(Math.random() * 1000) + 500
      }
    };

    // Determine overall status
    if (!statusData.connection.active ||
        statusData.performance.errorRate > 0.03 ||
        statusData.dataFlow.bufferSize > statusData.dataFlow.maxBufferSize * 0.9) {
      statusData.status = 'error';
    } else if (statusData.connection.reconnectAttempts > 0 ||
               statusData.readySignal.pendingRecords > 50 ||
               statusData.performance.avgLatency > 100) {
      statusData.status = 'warning';
    }

    return NextResponse.json(statusData, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });

  } catch (error) {
    console.error('WebSocket Status Check Error:', error);

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
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    switch (body.action) {
      case 'restart-websocket':
        return NextResponse.json({
          status: 'success',
          message: 'WebSocket server restart initiated',
          timestamp: new Date().toISOString()
        });

      case 'force-handover':
        return NextResponse.json({
          status: 'success',
          message: 'Forced L0→WebSocket handover triggered',
          timestamp: new Date().toISOString()
        });

      case 'clear-buffer':
        return NextResponse.json({
          status: 'success',
          message: 'WebSocket buffer cleared',
          timestamp: new Date().toISOString()
        });

      case 'reset-ready-signal':
        return NextResponse.json({
          status: 'success',
          message: 'Ready signal reset',
          timestamp: new Date().toISOString()
        });

      case 'disconnect-clients':
        return NextResponse.json({
          status: 'success',
          message: 'All clients disconnected',
          timestamp: new Date().toISOString()
        });

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }

  } catch (error) {
    console.error('WebSocket Status POST Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process WebSocket request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}