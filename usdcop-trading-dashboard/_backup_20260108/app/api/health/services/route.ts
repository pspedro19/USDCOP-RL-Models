// Health Check API - All Services Status

import { NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getHealthChecker, getServiceRegistry } from '@/lib/services/health';

// Initialize services on first import
const healthChecker = getHealthChecker();
const registry = getServiceRegistry();

// Register default services
if (registry.getServiceCount().total === 0) {
  // PostgreSQL Database
  registry.register({
    name: 'postgres',
    url: process.env.POSTGRES_URL || 'http://localhost:5432',
    checkInterval: 30000,
    timeout: 5000,
  });

  // Trading API
  registry.register({
    name: 'trading-api',
    url: process.env.TRADING_API_URL || 'http://localhost:8001/health',
    checkInterval: 30000,
    timeout: 5000,
  });

  // ML Analytics API
  registry.register({
    name: 'ml-analytics',
    url: process.env.ML_ANALYTICS_URL || 'http://localhost:8002/health',
    checkInterval: 30000,
    timeout: 5000,
  });

  // Pipeline API
  registry.register({
    name: 'pipeline-api',
    url: process.env.PIPELINE_API_URL || 'http://localhost:8000/health',
    checkInterval: 30000,
    timeout: 5000,
  });

  // WebSocket Server
  registry.register({
    name: 'websocket',
    url: process.env.WS_URL || 'http://localhost:8080',
    checkInterval: 30000,
    timeout: 5000,
  });
}

export async function GET() {
  const startTime = Date.now();

  try {
    const systemHealth = await healthChecker.checkAllServices();
    const latency = Date.now() - startTime;
    const isHealthy = systemHealth.overall_status === 'healthy';

    return NextResponse.json(
      createApiResponse(isHealthy, 'live', {
        data: systemHealth,
        latency
      }),
      {
        status: isHealthy ? 200 : 503,
      }
    );
  } catch (error) {
    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to check service health',
        message: (error as Error).message,
        latency
      }),
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { service } = body;

    if (!service) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Service name is required',
          latency
        }),
        { status: 400 }
      );
    }

    const serviceHealth = await healthChecker.checkService(service);
    const latency = Date.now() - startTime;
    const isHealthy = serviceHealth.status === 'healthy';

    return NextResponse.json(
      createApiResponse(isHealthy, 'live', {
        data: serviceHealth,
        latency
      }),
      {
        status: isHealthy ? 200 : 503,
      }
    );
  } catch (error) {
    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to check service health',
        message: (error as Error).message,
        latency
      }),
      { status: 500 }
    );
  }
}
