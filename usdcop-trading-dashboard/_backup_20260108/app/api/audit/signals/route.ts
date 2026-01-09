// Audit Log API for Signals

import { NextRequest, NextResponse } from 'next/server';
import { getAuditLogger } from '@/lib/services/logging';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

const auditLogger = getAuditLogger();

export const GET = withAuth(async (request, { user }) => {
  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type') || 'trail'; // trail, stats, export

    if (type === 'stats') {
      const stats = await auditLogger.getAuditStats();

      return NextResponse.json(
        createApiResponse({ type: 'stats', stats }, 'postgres')
      );
    }

    if (type === 'export') {
      const format = (searchParams.get('format') || 'json') as 'json' | 'csv';
      const data = await auditLogger.exportAuditTrail(format);

      const headers: Record<string, string> = {
        'Content-Disposition': `attachment; filename=audit_trail_${Date.now()}.${format}`,
      };

      if (format === 'csv') {
        headers['Content-Type'] = 'text/csv';
      } else {
        headers['Content-Type'] = 'application/json';
      }

      return new Response(data, { headers });
    }

    // Default: return audit trail
    const eventType = searchParams.get('eventType') as any;
    const symbol = searchParams.get('symbol');
    const userId = searchParams.get('userId');
    const limit = parseInt(searchParams.get('limit') || '100');
    const startTime = searchParams.get('startTime');
    const endTime = searchParams.get('endTime');

    const filters: any = {};

    if (eventType) filters.eventType = eventType;
    if (symbol) filters.symbol = symbol;
    if (userId) filters.userId = userId;
    if (limit) filters.limit = limit;
    if (startTime) filters.startTime = new Date(startTime);
    if (endTime) filters.endTime = new Date(endTime);

    const auditTrail = await auditLogger.getAuditTrail(filters);

    return NextResponse.json(
      createApiResponse(
        { type: 'trail', count: auditTrail.length, filters, entries: auditTrail },
        'postgres'
      )
    );
  } catch (error) {
    return NextResponse.json(
      createApiResponse(null, 'none', 'Failed to get audit log', {
        message: (error as Error).message,
      }),
      { status: 500 }
    );
  }
});

export const POST = withAuth(async (request, { user }) => {
  try {
    const body = await request.json();
    const { event_type, data, metadata } = body;

    if (!event_type || !data) {
      return NextResponse.json(
        createApiResponse(null, 'none', 'Event type and data are required'),
        { status: 400 }
      );
    }

    switch (event_type) {
      case 'SIGNAL_GENERATED':
        await auditLogger.logSignalGenerated(data, metadata);
        break;

      case 'SIGNAL_EXECUTED':
        await auditLogger.logSignalExecuted(data.signal, data.execution, metadata);
        break;

      case 'POSITION_OPENED':
        await auditLogger.logPositionOpened(data, metadata);
        break;

      case 'POSITION_CLOSED':
        await auditLogger.logPositionClosed(data.position, data.pnl, metadata);
        break;

      case 'RISK_ALERT':
        await auditLogger.logRiskAlert(data, metadata);
        break;

      case 'SYSTEM_EVENT':
        await auditLogger.logSystemEvent(data.event, metadata);
        break;

      default:
        return NextResponse.json(
          createApiResponse(null, 'none', 'Invalid event type'),
          { status: 400 }
        );
    }

    return NextResponse.json(
      createApiResponse(
        { event_type },
        'postgres',
        undefined,
        { message: 'Audit log entry created successfully' }
      )
    );
  } catch (error) {
    return NextResponse.json(
      createApiResponse(null, 'none', 'Failed to create audit log entry', {
        message: (error as Error).message,
      }),
      { status: 500 }
    );
  }
});

export const DELETE = withAuth(async (request, { user }) => {
  try {
    await auditLogger.clearAuditTrail();

    return NextResponse.json(
      createApiResponse({}, 'postgres', undefined, {
        message: 'Audit trail cleared successfully',
      })
    );
  } catch (error) {
    return NextResponse.json(
      createApiResponse(null, 'none', 'Failed to clear audit trail', {
        message: (error as Error).message,
      }),
      { status: 500 }
    );
  }
});
