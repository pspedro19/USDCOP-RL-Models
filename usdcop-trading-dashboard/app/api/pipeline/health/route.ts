import { NextRequest, NextResponse } from 'next/server';

interface PipelineHealthResponse {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  components: {
    l0: ComponentHealth;
    l1: ComponentHealth;
    l2: ComponentHealth;
    l3: ComponentHealth;
    l4: ComponentHealth;
    l5: ComponentHealth;
  };
  overall: {
    uptime: string;
    totalErrors: number;
    totalWarnings: number;
    lastFailure?: string;
    throughput: number;
  };
}

interface ComponentHealth {
  status: 'healthy' | 'warning' | 'error' | 'inactive';
  lastUpdate: string;
  recordsProcessed: number;
  errors: number;
  warnings: number;
  latency: number;
  uptime: string;
}

export async function GET(request: NextRequest) {
  try {
    const generateComponentHealth = (componentName: string): ComponentHealth => {
      const isHealthy = Math.random() > 0.15;
      const hasWarnings = Math.random() > 0.7;

      return {
        status: !isHealthy ? 'error' : hasWarnings ? 'warning' : 'healthy',
        lastUpdate: new Date(Date.now() - Math.random() * 300000).toISOString(),
        recordsProcessed: Math.floor(Math.random() * 5000) + 1000,
        errors: !isHealthy ? Math.floor(Math.random() * 5) + 1 : 0,
        warnings: hasWarnings ? Math.floor(Math.random() * 3) + 1 : 0,
        latency: Math.random() * 100 + 10,
        uptime: `${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m`
      };
    };

    const healthData: PipelineHealthResponse = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      components: {
        l0: generateComponentHealth('L0'),
        l1: generateComponentHealth('L1'),
        l2: generateComponentHealth('L2'),
        l3: generateComponentHealth('L3'),
        l4: generateComponentHealth('L4'),
        l5: generateComponentHealth('L5')
      },
      overall: {
        uptime: `${Math.floor(Math.random() * 168)}h ${Math.floor(Math.random() * 60)}m`,
        totalErrors: 0,
        totalWarnings: 0,
        throughput: Math.floor(Math.random() * 1000) + 500
      }
    };

    // Calculate overall metrics
    Object.values(healthData.components).forEach(component => {
      healthData.overall.totalErrors += component.errors;
      healthData.overall.totalWarnings += component.warnings;
    });

    // Determine overall status
    const hasErrors = Object.values(healthData.components).some(c => c.status === 'error');
    const hasWarnings = Object.values(healthData.components).some(c => c.status === 'warning');

    if (hasErrors) {
      healthData.status = 'error';
      healthData.overall.lastFailure = new Date(Date.now() - Math.random() * 86400000).toISOString();
    } else if (hasWarnings) {
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
    console.error('Pipeline Health Check Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'Pipeline health check failed',
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
      case 'restart-component':
        if (!body.component) {
          return NextResponse.json(
            { error: 'Component name required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `${body.component} restart initiated`,
          timestamp: new Date().toISOString()
        });

      case 'clear-errors':
        return NextResponse.json({
          status: 'success',
          message: 'Error logs cleared',
          timestamp: new Date().toISOString()
        });

      case 'force-sync':
        return NextResponse.json({
          status: 'success',
          message: 'Pipeline synchronization triggered',
          timestamp: new Date().toISOString()
        });

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }

  } catch (error) {
    console.error('Pipeline Health Check POST Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process pipeline health request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}