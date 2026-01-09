import { NextRequest, NextResponse } from 'next/server';
import { apiConfig } from '@/lib/config/api.config';
import { pgQuery } from '@/lib/db/postgres-client';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { createApiResponse } from '@/lib/types/api';

/**
 * Pipeline Health API
 * ===================
 *
 * Connects to REAL Pipeline API backend (Port 8002) and PostgreSQL
 * NO MOCK DATA - All health metrics from real pipeline monitoring
 */

const PIPELINE_BASE = apiConfig.pipeline.baseUrl;

// Circuit breaker for Pipeline API
const pipelineCircuitBreaker = getCircuitBreaker('pipeline-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

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

// Query L0 health from PostgreSQL
async function getL0HealthFromDB(): Promise<ComponentHealth | null> {
  try {
    const result = await pgQuery(`
      SELECT
        MAX(timestamp) as last_update,
        COUNT(*) as records_processed,
        MIN(timestamp) as first_record
      FROM usdcop_m5_ohlcv
      WHERE timestamp > NOW() - INTERVAL '24 hours'
    `);

    if (result.rows.length > 0) {
      const row = result.rows[0];
      const lastUpdate = row.last_update ? new Date(row.last_update) : new Date();
      const ageMinutes = (Date.now() - lastUpdate.getTime()) / 60000;

      return {
        status: ageMinutes < 10 ? 'healthy' : ageMinutes < 30 ? 'warning' : 'error',
        lastUpdate: lastUpdate.toISOString(),
        recordsProcessed: parseInt(row.records_processed) || 0,
        errors: 0,
        warnings: ageMinutes > 10 ? 1 : 0,
        latency: ageMinutes * 60 * 1000, // Convert to ms
        uptime: '24h' // Based on query window
      };
    }
    return null;
  } catch (error) {
    console.error('[Pipeline Health] L0 DB query failed:', error);
    return null;
  }
}

export async function GET(request: NextRequest) {
  try {
    // Try to fetch from real Pipeline API backend with circuit breaker protection
    const backendUrl = `${PIPELINE_BASE}/api/pipeline/health`;

    let backendData = null;

    try {
      backendData = await pipelineCircuitBreaker.execute(async () => {
        const response = await fetch(backendUrl, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(10000),
          cache: 'no-store'
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
      });
    } catch (backendError) {
      if (backendError instanceof CircuitOpenError) {
        console.warn('[Pipeline Health API] Circuit breaker open');
      } else {
        console.warn('[Pipeline Health API] Backend unavailable:', backendError);
      }
    }

    // If backend returned data, use it
    if (backendData) {
      return NextResponse.json({
        ...backendData,
        source: 'pipeline_backend',
        timestamp: new Date().toISOString()
      }, {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      });
    }

    // Fallback: Query PostgreSQL directly for L0 health
    const l0Health = await getL0HealthFromDB();

    if (l0Health) {
      // Build partial response from what we can query
      const healthData: PipelineHealthResponse = {
        status: l0Health.status === 'healthy' ? 'healthy' : 'warning',
        timestamp: new Date().toISOString(),
        components: {
          l0: l0Health,
          l1: { status: 'inactive', lastUpdate: '', recordsProcessed: 0, errors: 0, warnings: 0, latency: 0, uptime: 'N/A' },
          l2: { status: 'inactive', lastUpdate: '', recordsProcessed: 0, errors: 0, warnings: 0, latency: 0, uptime: 'N/A' },
          l3: { status: 'inactive', lastUpdate: '', recordsProcessed: 0, errors: 0, warnings: 0, latency: 0, uptime: 'N/A' },
          l4: { status: 'inactive', lastUpdate: '', recordsProcessed: 0, errors: 0, warnings: 0, latency: 0, uptime: 'N/A' },
          l5: { status: 'inactive', lastUpdate: '', recordsProcessed: 0, errors: 0, warnings: 0, latency: 0, uptime: 'N/A' }
        },
        overall: {
          uptime: 'N/A',
          totalErrors: l0Health.errors,
          totalWarnings: l0Health.warnings,
          throughput: l0Health.recordsProcessed
        }
      };

      return NextResponse.json({
        ...healthData,
        source: 'postgresql_fallback',
        warning: 'Pipeline API unavailable - showing L0 data from PostgreSQL only'
      }, {
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate'
        }
      });
    }

    // No data available from any source
    return NextResponse.json({
      success: false,
      error: 'Pipeline health data unavailable',
      message: 'Cannot connect to Pipeline API (port 8002) or PostgreSQL database.',
      troubleshooting: {
        backend_url: PIPELINE_BASE,
        expected_endpoint: '/api/pipeline/health',
        required_services: ['pipeline_api (port 8002)', 'PostgreSQL (port 5432)'],
        database_table: 'usdcop_m5_ohlcv'
      },
      timestamp: new Date().toISOString()
    }, { status: 503 });

  } catch (error) {
    console.error('[Pipeline Health] Error:', error);

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

    // Forward to real backend with circuit breaker protection
    const backendUrl = `${PIPELINE_BASE}/api/pipeline/health`;

    try {
      const data = await pipelineCircuitBreaker.execute(async () => {
        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(10000)
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
      });

      return NextResponse.json({
        ...data,
        source: 'pipeline_backend'
      });
    } catch (backendError) {
      if (backendError instanceof CircuitOpenError) {
        console.warn('[Pipeline Health POST] Circuit breaker open');
        return NextResponse.json({
          success: false,
          error: 'Service temporarily unavailable',
          message: 'Pipeline API is experiencing issues. Please try again later.',
          action_attempted: body.action,
          timestamp: new Date().toISOString()
        }, { status: 503 });
      }
      console.warn('[Pipeline Health POST] Backend unavailable:', backendError);
    }

    // Return error for critical operations (NO MOCK RESPONSES)
    return NextResponse.json({
      success: false,
      error: 'Pipeline API unavailable',
      message: 'Cannot perform pipeline operations without backend connection.',
      action_attempted: body.action,
      timestamp: new Date().toISOString()
    }, { status: 503 });

  } catch (error) {
    console.error('[Pipeline Health POST] Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process pipeline health request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
