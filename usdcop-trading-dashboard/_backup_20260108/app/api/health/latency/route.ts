// Latency Metrics API

import { NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getLatencyMonitor } from '@/lib/services/health';

const latencyMonitor = getLatencyMonitor();

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const service = searchParams.get('service');
    const operation = searchParams.get('operation');
    const minutes = parseInt(searchParams.get('minutes') || '5');
    const type = searchParams.get('type') || 'stats'; // stats, trend, recent, end-to-end

    if (type === 'end-to-end') {
      const endToEndLatency = latencyMonitor.getEndToEndLatency(minutes);
      return NextResponse.json(createApiResponse({
        type: 'end-to-end',
        minutes,
        latency: endToEndLatency,
      }, 'live'));
    }

    if (type === 'recent') {
      const limit = parseInt(searchParams.get('limit') || '100');
      const measurements = latencyMonitor.getRecentMeasurements(limit);

      return NextResponse.json(createApiResponse({
        type: 'recent',
        count: measurements.length,
        measurements,
      }, 'live'));
    }

    if (type === 'trend') {
      if (!service || !operation) {
        return NextResponse.json(
          createApiResponse(null, 'none', 'Service and operation are required for trend data'),
          { status: 400 }
        );
      }

      const trend = latencyMonitor.getLatencyTrend(service, operation, minutes);

      return NextResponse.json(createApiResponse({
        type: 'trend',
        service,
        operation,
        interval_minutes: minutes,
        data: trend,
      }, 'live'));
    }

    if (type === 'high-latency') {
      const limit = parseInt(searchParams.get('limit') || '5');
      const highLatency = latencyMonitor.getHighLatencyServices(limit);

      return NextResponse.json(createApiResponse({
        type: 'high-latency',
        limit,
        services: highLatency,
      }, 'live'));
    }

    // Default: return stats
    if (!service) {
      return NextResponse.json(
        createApiResponse(null, 'none', 'Service name is required for stats'),
        { status: 400 }
      );
    }

    const stats = latencyMonitor.getLatencyStats(service, operation || undefined);

    return NextResponse.json(createApiResponse({
      type: 'stats',
      service,
      operation: operation || 'all',
      stats,
    }, 'live'));
  } catch (error) {
    return NextResponse.json(
      createApiResponse(null, 'none', `Failed to get latency metrics: ${(error as Error).message}`),
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { service, operation, latency_ms, success, metadata } = body;

    if (!service || !operation || latency_ms === undefined) {
      return NextResponse.json(
        createApiResponse(null, 'none', 'Service, operation, and latency_ms are required'),
        { status: 400 }
      );
    }

    latencyMonitor.recordLatency(
      service,
      operation,
      latency_ms,
      success ?? true,
      metadata
    );

    return NextResponse.json(createApiResponse({
      message: 'Latency recorded successfully',
      service,
      operation,
      latency_ms,
    }, 'live'));
  } catch (error) {
    return NextResponse.json(
      createApiResponse(null, 'none', `Failed to record latency: ${(error as Error).message}`),
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const service = searchParams.get('service');

    latencyMonitor.clearMeasurements(service || undefined);

    return NextResponse.json(createApiResponse({
      message: service
        ? `Latency measurements cleared for service: ${service}`
        : 'All latency measurements cleared',
    }, 'live'));
  } catch (error) {
    return NextResponse.json(
      createApiResponse(null, 'none', `Failed to clear latency measurements: ${(error as Error).message}`),
      { status: 500 }
    );
  }
}
