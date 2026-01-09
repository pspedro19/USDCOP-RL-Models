import { NextRequest, NextResponse } from 'next/server';
import { apiConfig } from '@/lib/config/api.config';
import { pgQuery } from '@/lib/db/postgres-client';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';

const alertsCircuitBreaker = getCircuitBreaker('alerts-system-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

/**
 * Alert System API
 * ================
 *
 * Connects to REAL alert monitoring systems
 * NO MOCK DATA - All alerts from real monitoring
 */

const PIPELINE_BASE = apiConfig.pipeline.baseUrl;

interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  component: string;
  message: string;
  timestamp: string;
  status: 'active' | 'acknowledged' | 'resolved';
  acknowledgedBy?: string;
  resolvedAt?: string;
  details: Record<string, unknown>;
}

interface AlertRule {
  id: string;
  name: string;
  component: string;
  condition: string;
  threshold: number;
  severity: 'critical' | 'warning' | 'info';
  enabled: boolean;
  cooldown: number;
  lastTriggered?: string;
}

// Query real alerts from database if available
async function getRealAlertsFromDB(): Promise<Alert[]> {
  try {
    // Try to query alert_log table if it exists
    const result = await pgQuery(`
      SELECT
        id,
        severity,
        component,
        message,
        created_at as timestamp,
        status,
        acknowledged_by,
        resolved_at,
        details
      FROM alert_log
      WHERE created_at > NOW() - INTERVAL '24 hours'
      ORDER BY created_at DESC
      LIMIT 100
    `);

    if (result.rows.length > 0) {
      return result.rows.map(row => ({
        id: row.id,
        severity: row.severity || 'info',
        component: row.component || 'unknown',
        message: row.message || '',
        timestamp: row.timestamp?.toISOString() || new Date().toISOString(),
        status: row.status || 'active',
        acknowledgedBy: row.acknowledged_by,
        resolvedAt: row.resolved_at?.toISOString(),
        details: row.details || {}
      }));
    }
    return [];
  } catch (error) {
    // Table might not exist - this is expected in some setups
    console.warn('[Alert System] Database query failed (table may not exist):', error);
    return [];
  }
}

// Check pipeline health and generate real alerts
async function checkPipelineHealth(): Promise<Alert[]> {
  const alerts: Alert[] = [];

  try {
    // Check L0 data freshness
    const l0Result = await pgQuery(`
      SELECT MAX(timestamp) as last_update
      FROM usdcop_m5_ohlcv
    `);

    if (l0Result.rows.length > 0 && l0Result.rows[0].last_update) {
      const lastUpdate = new Date(l0Result.rows[0].last_update);
      const ageMinutes = (Date.now() - lastUpdate.getTime()) / 60000;

      if (ageMinutes > 30) {
        alerts.push({
          id: `l0-stale-${Date.now()}`,
          severity: 'critical',
          component: 'L0-Pipeline',
          message: `L0 data is ${Math.round(ageMinutes)} minutes stale`,
          timestamp: new Date().toISOString(),
          status: 'active',
          details: {
            lastUpdate: lastUpdate.toISOString(),
            ageMinutes: Math.round(ageMinutes),
            threshold: 30
          }
        });
      } else if (ageMinutes > 10) {
        alerts.push({
          id: `l0-warning-${Date.now()}`,
          severity: 'warning',
          component: 'L0-Pipeline',
          message: `L0 data update delayed by ${Math.round(ageMinutes)} minutes`,
          timestamp: new Date().toISOString(),
          status: 'active',
          details: {
            lastUpdate: lastUpdate.toISOString(),
            ageMinutes: Math.round(ageMinutes),
            threshold: 10
          }
        });
      }
    }
  } catch (error) {
    console.warn('[Alert System] Pipeline health check failed:', error);
  }

  return alerts;
}

// Static alert rules (these define what we monitor)
const alertRules: AlertRule[] = [
  {
    id: 'pipeline-gap',
    name: 'Pipeline Data Gap Detection',
    component: 'L0-Pipeline',
    condition: 'data_age_minutes > threshold',
    threshold: 30,
    severity: 'critical',
    enabled: true,
    cooldown: 300
  },
  {
    id: 'pipeline-warning',
    name: 'Pipeline Data Delay Warning',
    component: 'L0-Pipeline',
    condition: 'data_age_minutes > threshold',
    threshold: 10,
    severity: 'warning',
    enabled: true,
    cooldown: 180
  },
  {
    id: 'api-rate-limit',
    name: 'API Rate Limit Warning',
    component: 'API-Usage',
    condition: 'usage_percentage > threshold',
    threshold: 80,
    severity: 'warning',
    enabled: true,
    cooldown: 900
  },
  {
    id: 'websocket-latency',
    name: 'WebSocket High Latency',
    component: 'WebSocket',
    condition: 'avg_latency_ms > threshold',
    threshold: 100,
    severity: 'warning',
    enabled: true,
    cooldown: 300
  }
];

export const GET = withAuth(async (request, { user }) => {
  try {
    // Try to get alerts from real backend
    const backendUrl = `${PIPELINE_BASE}/api/alerts`;

    let backendAlerts: Alert[] = [];

    try {
      const data = await alertsCircuitBreaker.execute(async () => {
        const response = await fetch(backendUrl, {
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

      if (data.alerts) {
        backendAlerts = data.alerts;
      }
    } catch (backendError) {
      const isCircuitOpen = backendError instanceof CircuitOpenError;
      console.warn('[Alert System] Backend unavailable:', isCircuitOpen ? 'Circuit open' : backendError);
    }

    // Get alerts from database
    const dbAlerts = await getRealAlertsFromDB();

    // Check pipeline health for real-time alerts
    const pipelineAlerts = await checkPipelineHealth();

    // Combine all alerts
    const allAlerts = [...backendAlerts, ...dbAlerts, ...pipelineAlerts];

    // Calculate metrics
    const criticalAlerts = allAlerts.filter(a => a.severity === 'critical' && a.status === 'active').length;
    const warningAlerts = allAlerts.filter(a => a.severity === 'warning' && a.status === 'active').length;
    const resolvedAlerts = allAlerts.filter(a => a.status === 'resolved').length;

    // Determine overall status
    let status: 'active' | 'warning' | 'error' = 'active';
    if (criticalAlerts > 0) {
      status = 'error';
    } else if (warningAlerts > 0) {
      status = 'warning';
    }

    return NextResponse.json({
      status,
      timestamp: new Date().toISOString(),
      source: backendAlerts.length > 0 ? 'backend' : (dbAlerts.length > 0 ? 'database' : 'pipeline_check'),
      activeAlerts: allAlerts,
      alertRules,
      metrics: {
        totalAlerts24h: allAlerts.length,
        criticalAlerts,
        warningAlerts,
        resolvedAlerts,
        avgResponseTime: 0 // Not available without real data
      }
    }, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });

  } catch (error) {
    console.error('[Alert System] Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'Alert system check failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
});

export const POST = withAuth(async (request, { user }) => {
  try {
    const body = await request.json();

    // Forward to real backend
    const backendUrl = `${PIPELINE_BASE}/api/alerts`;

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000)
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json({
          ...data,
          source: 'backend'
        });
      }
    } catch (backendError) {
      console.warn('[Alert System POST] Backend unavailable:', backendError);
    }

    // Return error for critical operations (NO MOCK RESPONSES)
    return NextResponse.json({
      success: false,
      error: 'Alert backend unavailable',
      message: 'Cannot perform alert operations without backend connection.',
      action_attempted: body.action,
      timestamp: new Date().toISOString()
    }, { status: 503 });

  } catch (error) {
    console.error('[Alert System POST] Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process alert request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
});
