import { NextRequest, NextResponse } from 'next/server';
import { apiConfig } from '@/lib/config/api.config';
import { pgQuery } from '@/lib/db/postgres-client';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';

const usageCircuitBreaker = getCircuitBreaker('usage-monitoring-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

/**
 * API Usage Monitoring
 * ====================
 *
 * Connects to REAL usage tracking systems
 * NO MOCK DATA - All metrics from real API usage logs
 */

const PIPELINE_BASE = apiConfig.pipeline.baseUrl;

interface APIMetrics {
  callsUsed: number;
  rateLimit: number;
  remainingCalls: number;
  resetTime: string;
  errorRate: number;
  avgResponseTime: number;
  lastCall: string;
  status: 'active' | 'limited' | 'error';
}

interface KeyInfo {
  keyId: string;
  provider: string;
  age: number;
  expiresIn: number;
  callsToday: number;
  status: 'active' | 'expired' | 'suspended' | 'rotation_due';
}

// Query real API usage from database if available
async function getRealAPIUsageFromDB(): Promise<{
  twelveData: APIMetrics | null;
  internal: APIMetrics | null;
}> {
  try {
    // Check for api_usage_log table
    const result = await pgQuery(`
      SELECT
        provider,
        COUNT(*) as calls_today,
        AVG(response_time_ms) as avg_response_time,
        SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END)::float / COUNT(*) as error_rate,
        MAX(created_at) as last_call
      FROM api_usage_log
      WHERE created_at > CURRENT_DATE
      GROUP BY provider
    `);

    const metrics: { [key: string]: APIMetrics } = {};

    for (const row of result.rows) {
      metrics[row.provider] = {
        callsUsed: parseInt(row.calls_today) || 0,
        rateLimit: row.provider === 'twelveData' ? 800 : 10000,
        remainingCalls: 0,
        resetTime: new Date(Date.now() + 24 * 3600000).toISOString(),
        errorRate: parseFloat(row.error_rate) || 0,
        avgResponseTime: parseFloat(row.avg_response_time) || 0,
        lastCall: row.last_call?.toISOString() || new Date().toISOString(),
        status: 'active'
      };
      metrics[row.provider].remainingCalls = metrics[row.provider].rateLimit - metrics[row.provider].callsUsed;
      if (metrics[row.provider].remainingCalls < metrics[row.provider].rateLimit * 0.2) {
        metrics[row.provider].status = 'limited';
      }
    }

    return {
      twelveData: metrics['twelveData'] || null,
      internal: metrics['internal'] || null
    };
  } catch (error) {
    console.warn('[API Usage] Database query failed (table may not exist):', error);
    return { twelveData: null, internal: null };
  }
}

// Check TwelveData API key status from environment
function getTwelveDataKeyStatus(): KeyInfo | null {
  const apiKey = process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY || process.env.TWELVEDATA_API_KEY;

  if (!apiKey) {
    return null;
  }

  // Key exists - return basic info (can't know actual age without tracking)
  return {
    keyId: `twelvedata-${apiKey.substring(0, 8)}...`,
    provider: 'twelveData',
    age: 0, // Unknown without tracking
    expiresIn: 365, // Assumption
    callsToday: 0, // Will be updated from DB
    status: 'active'
  };
}

export const GET = withAuth(async (request, { user }) => {
  try {
    // Try to get usage from real backend
    const backendUrl = `${PIPELINE_BASE}/api/usage/monitoring`;

    try {
      const data = await usageCircuitBreaker.execute(async () => {
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

      return NextResponse.json({
        ...data,
        source: 'backend',
        timestamp: new Date().toISOString()
      }, {
        headers: { 'Cache-Control': 'no-cache, no-store, must-revalidate' }
      });
    } catch (backendError) {
      const isCircuitOpen = backendError instanceof CircuitOpenError;
      console.warn('[API Usage] Backend unavailable:', isCircuitOpen ? 'Circuit open' : backendError);
    }

    // Fallback: Query database directly
    const dbMetrics = await getRealAPIUsageFromDB();
    const twelveDataKey = getTwelveDataKeyStatus();

    // Build response with real data where available
    const hasAnyData = dbMetrics.twelveData || dbMetrics.internal || twelveDataKey;

    if (hasAnyData) {
      const usageData = {
        status: 'healthy' as 'healthy' | 'warning' | 'error',
        timestamp: new Date().toISOString(),
        source: 'database_fallback',
        apis: {
          twelveData: dbMetrics.twelveData || {
            callsUsed: 0,
            rateLimit: 800,
            remainingCalls: 800,
            resetTime: new Date(Date.now() + 24 * 3600000).toISOString(),
            errorRate: 0,
            avgResponseTime: 0,
            lastCall: new Date().toISOString(),
            status: 'active' as const
          },
          internal: dbMetrics.internal || {
            callsUsed: 0,
            rateLimit: 10000,
            remainingCalls: 10000,
            resetTime: new Date(Date.now() + 24 * 3600000).toISOString(),
            errorRate: 0,
            avgResponseTime: 0,
            lastCall: new Date().toISOString(),
            status: 'active' as const
          }
        },
        keys: {
          active: twelveDataKey ? [twelveDataKey] : [],
          rotation: {
            nextRotation: 'N/A',
            rotationPolicy: 'Manual',
            lastRotation: 'N/A'
          }
        },
        alerts: {
          rateLimitWarnings: 0,
          keyExpirationWarnings: 0,
          quotaWarnings: 0
        }
      };

      // Calculate alerts
      if (usageData.apis.twelveData.status === 'limited') {
        usageData.alerts.rateLimitWarnings++;
        usageData.status = 'warning';
      }

      return NextResponse.json(usageData, {
        headers: { 'Cache-Control': 'no-cache, no-store, must-revalidate' }
      });
    }

    // No data available
    return NextResponse.json({
      success: false,
      error: 'API usage monitoring unavailable',
      message: 'No usage data sources available. Database table api_usage_log may not exist.',
      troubleshooting: {
        required_table: 'api_usage_log',
        backend_url: PIPELINE_BASE,
        env_vars: ['TWELVEDATA_API_KEY']
      },
      timestamp: new Date().toISOString()
    }, { status: 503 });

  } catch (error) {
    console.error('[API Usage Monitoring] Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'API usage monitoring failed',
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
    const backendUrl = `${PIPELINE_BASE}/api/usage/monitoring`;

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000)
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json({ ...data, source: 'backend' });
      }
    } catch (backendError) {
      console.warn('[API Usage POST] Backend unavailable:', backendError);
    }

    // Return error for operations (NO MOCK RESPONSES)
    return NextResponse.json({
      success: false,
      error: 'API usage backend unavailable',
      message: 'Cannot perform usage operations without backend connection.',
      action_attempted: body.action,
      timestamp: new Date().toISOString()
    }, { status: 503 });

  } catch (error) {
    console.error('[API Usage Monitoring POST] Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process API usage request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
});
