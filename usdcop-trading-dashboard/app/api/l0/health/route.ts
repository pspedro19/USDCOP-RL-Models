import { NextRequest, NextResponse } from 'next/server';

interface HealthCheckResponse {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  pipeline: {
    status: 'running' | 'completed' | 'failed' | 'idle';
    recordsProcessed: number;
    errors: number;
    warnings: number;
    lastRun: string;
    nextRun?: string;
  };
  backup: {
    exists: boolean;
    lastUpdated: string;
    recordCount: number;
    gapsDetected: number;
    size: string;
    integrity: 'good' | 'warning' | 'error';
  };
  readySignal: {
    active: boolean;
    waiting: boolean;
    handoverComplete: boolean;
    websocketReady: boolean;
    lastHandover: string;
    pendingRecords: number;
  };
  dataQuality: {
    completeness: number;
    latency: number;
    gapsCount: number;
    lastTimestamp: string;
    duplicates: number;
    outliers: number;
  };
  apiUsage: {
    callsUsed: number;
    rateLimit: number;
    remainingCalls: number;
    resetTime: string;
    keyRotationDue: boolean;
    keyAge: number;
  };
}

export async function GET(request: NextRequest) {
  try {
    // In a real implementation, these would come from actual monitoring systems
    // This is mock data for demonstration purposes

    const healthData: HealthCheckResponse = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      pipeline: {
        status: Math.random() > 0.1 ? 'running' : 'failed',
        recordsProcessed: Math.floor(Math.random() * 10000) + 5000,
        errors: Math.floor(Math.random() * 3),
        warnings: Math.floor(Math.random() * 5),
        lastRun: new Date(Date.now() - 3600000).toISOString(),
        nextRun: new Date(Date.now() + 1800000).toISOString()
      },
      backup: {
        exists: true,
        lastUpdated: new Date(Date.now() - 300000).toISOString(),
        recordCount: Math.floor(Math.random() * 1000) + 500,
        gapsDetected: Math.floor(Math.random() * 3),
        size: `${(Math.random() * 5 + 1).toFixed(1)} GB`,
        integrity: Math.random() > 0.2 ? 'good' : 'warning'
      },
      readySignal: {
        active: Math.random() > 0.1,
        waiting: Math.random() > 0.8,
        handoverComplete: Math.random() > 0.2,
        websocketReady: Math.random() > 0.1,
        lastHandover: new Date(Date.now() - 120000).toISOString(),
        pendingRecords: Math.floor(Math.random() * 100)
      },
      dataQuality: {
        completeness: 95 + Math.random() * 5,
        latency: Math.random() * 50 + 10,
        gapsCount: Math.floor(Math.random() * 5),
        lastTimestamp: new Date().toISOString(),
        duplicates: Math.floor(Math.random() * 10),
        outliers: Math.floor(Math.random() * 20)
      },
      apiUsage: {
        callsUsed: Math.floor(Math.random() * 500) + 1200,
        rateLimit: 2000,
        remainingCalls: 800 - Math.floor(Math.random() * 200),
        resetTime: new Date(Date.now() + 3600000).toISOString(),
        keyRotationDue: Math.random() > 0.8,
        keyAge: Math.floor(Math.random() * 30) + 1
      }
    };

    // Determine overall status based on component health
    if (healthData.pipeline.status === 'failed' ||
        healthData.backup.integrity === 'error' ||
        !healthData.readySignal.websocketReady ||
        healthData.dataQuality.completeness < 90) {
      healthData.status = 'error';
    } else if (healthData.pipeline.warnings > 0 ||
               healthData.backup.gapsDetected > 0 ||
               healthData.apiUsage.keyRotationDue ||
               healthData.dataQuality.gapsCount > 2) {
      healthData.status = 'warning';
    }

    return NextResponse.json(healthData, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });

  } catch (error) {
    console.error('L0 Health Check Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'Health check failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Handle health check triggers or configuration updates
    if (body.action === 'refresh') {
      // Trigger a manual health check refresh
      return NextResponse.json({
        status: 'success',
        message: 'Health check refresh triggered',
        timestamp: new Date().toISOString()
      });
    }

    if (body.action === 'reset-alerts') {
      // Reset alert counters
      return NextResponse.json({
        status: 'success',
        message: 'Alerts reset successfully',
        timestamp: new Date().toISOString()
      });
    }

    return NextResponse.json(
      { error: 'Invalid action' },
      { status: 400 }
    );

  } catch (error) {
    console.error('L0 Health Check POST Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process health check request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}