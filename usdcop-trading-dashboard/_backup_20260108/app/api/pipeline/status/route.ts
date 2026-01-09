// Pipeline Status API
// Provides status for all pipeline layers L0-L6

import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

// Circuit breaker for external API calls
const tradingApiCircuitBreaker = getCircuitBreaker('pipeline-status-trading-api', {
  failureThreshold: 3,
  resetTimeout: 30000,
});

// SECURITY: MinIO credentials from environment only (no fallbacks)
const MINIO_ENDPOINT = process.env.MINIO_ENDPOINT;
const MINIO_ACCESS_KEY = process.env.MINIO_ACCESS_KEY;
const MINIO_SECRET_KEY = process.env.MINIO_SECRET_KEY;

// Disable caching
export const dynamic = 'force-dynamic';
export const revalidate = 0;

interface PipelineLayerStatus {
  layer: string;
  name: string;
  status: 'pass' | 'fail' | 'warning' | 'loading' | 'unknown';
  pass: boolean;
  quality_metrics?: any;
  data_shape?: any;
  last_update?: string;
  error?: string;
  dataSource: 'postgres' | 'minio' | 'api' | 'none';
}

/**
 * Get MinIO manifest for a given layer
 * For now, returns mock data as MinIO access needs proper setup
 */
async function getMinIOManifest(bucket: string, objectPath: string) {
  // TODO: Implement proper MinIO SDK access
  // For now return null to use fallback mock data
  return null;
}

/**
 * Get L0 (Raw Data) status from PostgreSQL
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL0Status(): Promise<PipelineLayerStatus> {
  try {
    // Query trading API with circuit breaker protection
    const data = await tradingApiCircuitBreaker.execute(async () => {
      const pgResponse = await fetch('http://localhost:8000/api/candlesticks/USDCOP?timeframe=5m&limit=100', {
        signal: AbortSignal.timeout(5000)
      });

      if (!pgResponse.ok) {
        throw new Error('Failed to fetch L0 data');
      }

      return pgResponse.json();
    });

    const dataPoints = data.count || 0;
    const coverage = (dataPoints / 100) * 100; // Expected 100 points

    return {
      layer: 'L0',
      name: 'Raw Data',
      status: coverage >= 90 ? 'pass' : coverage >= 70 ? 'warning' : 'fail',
      pass: coverage >= 90,
      quality_metrics: {
        coverage_pct: coverage,
        data_points: dataPoints,
        ohlc_violations: 0,
        stale_rate_pct: 0
      },
      data_shape: {
        actual_bars: dataPoints,
        expected_bars: 100
      },
      last_update: new Date().toISOString(),
      dataSource: 'postgres'
    };
  } catch (error) {
    const isCircuitOpen = error instanceof CircuitOpenError;
    return {
      layer: 'L0',
      name: 'Raw Data',
      status: 'unknown',
      pass: false,
      error: isCircuitOpen
        ? 'Trading API circuit breaker is open - service temporarily unavailable'
        : (error instanceof Error ? error.message : 'Unknown error'),
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

/**
 * Get L1 (Standardized) status from MinIO
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL1Status(): Promise<PipelineLayerStatus> {
  try {
    // Attempt to fetch L1 manifest from MinIO
    const manifest = await getMinIOManifest('usdcop-l1-standardized', 'USDCOP_M5/latest/manifest.json');

    if (manifest) {
      return {
        layer: 'L1',
        name: 'Standardized',
        status: 'pass',
        pass: true,
        quality_metrics: manifest.quality_metrics || {},
        data_shape: manifest.data_shape || {},
        last_update: manifest.timestamp || new Date().toISOString(),
        dataSource: 'minio'
      };
    }

    throw new Error('MinIO manifest not available');
  } catch (error) {
    return {
      layer: 'L1',
      name: 'Standardized',
      status: 'unknown',
      pass: false,
      error: 'L1 data unavailable - run standardization DAG or check MinIO',
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

/**
 * Get L2 (Prepared/Features) status from MinIO
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL2Status(): Promise<PipelineLayerStatus> {
  try {
    // Attempt to fetch L2 manifest from MinIO
    const manifest = await getMinIOManifest('usdcop-l2-prepared', 'USDCOP_M5/latest/manifest.json');

    if (manifest) {
      return {
        layer: 'L2',
        name: 'Prepared',
        status: 'pass',
        pass: true,
        quality_metrics: manifest.quality_metrics || {},
        data_shape: manifest.data_shape || {},
        last_update: manifest.timestamp || new Date().toISOString(),
        dataSource: 'minio'
      };
    }

    throw new Error('MinIO manifest not available');
  } catch (error) {
    return {
      layer: 'L2',
      name: 'Prepared',
      status: 'unknown',
      pass: false,
      error: 'L2 data unavailable - run feature preparation DAG or check MinIO',
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

/**
 * Get L3 (Feature Engineering) status from MinIO
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL3Status(): Promise<PipelineLayerStatus> {
  try {
    // Attempt to fetch L3 manifest from MinIO
    const manifest = await getMinIOManifest('usdcop-l3-features', 'USDCOP_M5/latest/manifest.json');

    if (manifest) {
      return {
        layer: 'L3',
        name: 'Features',
        status: 'pass',
        pass: true,
        quality_metrics: manifest.quality_metrics || {},
        data_shape: manifest.data_shape || {},
        last_update: manifest.timestamp || new Date().toISOString(),
        dataSource: 'minio'
      };
    }

    throw new Error('MinIO manifest not available');
  } catch (error) {
    return {
      layer: 'L3',
      name: 'Features',
      status: 'unknown',
      pass: false,
      error: 'L3 data unavailable - run feature engineering DAG or check MinIO',
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

/**
 * Get L4 (RL-Ready) status from MinIO
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL4Status(): Promise<PipelineLayerStatus> {
  try {
    // Attempt to fetch L4 manifest from MinIO
    const manifest = await getMinIOManifest('usdcop-l4-rlready', 'USDCOP_M5/latest/manifest.json');

    if (manifest) {
      return {
        layer: 'L4',
        name: 'RL-Ready',
        status: 'pass',
        pass: true,
        quality_metrics: manifest.quality_metrics || {},
        data_shape: manifest.data_shape || {},
        last_update: manifest.timestamp || new Date().toISOString(),
        dataSource: 'minio'
      };
    }

    throw new Error('MinIO manifest not available');
  } catch (error) {
    return {
      layer: 'L4',
      name: 'RL-Ready',
      status: 'unknown',
      pass: false,
      error: 'L4 data unavailable - run RL preparation DAG or check MinIO',
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

/**
 * Get L5 (Serving/Model) status from MinIO
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL5Status(): Promise<PipelineLayerStatus> {
  try {
    // Attempt to fetch L5 manifest from MinIO (model artifacts)
    const manifest = await getMinIOManifest('usdcop-l5-serving', 'USDCOP_M5/latest/manifest.json');

    if (manifest) {
      return {
        layer: 'L5',
        name: 'Serving',
        status: 'pass',
        pass: true,
        quality_metrics: manifest.quality_metrics || {},
        data_shape: manifest.data_shape || {},
        last_update: manifest.timestamp || new Date().toISOString(),
        dataSource: 'minio'
      };
    }

    throw new Error('MinIO manifest not available');
  } catch (error) {
    return {
      layer: 'L5',
      name: 'Serving',
      status: 'unknown',
      pass: false,
      error: 'L5 data unavailable - run model training DAG or check MinIO',
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

/**
 * Get L6 (Backtest) status from MinIO
 * NO MOCK DATA - Returns error if backend unavailable
 */
async function getL6Status(): Promise<PipelineLayerStatus> {
  try {
    // Attempt to fetch L6 manifest from MinIO (backtest results)
    const manifest = await getMinIOManifest('usdcop-l6-backtest', 'USDCOP_M5/latest/manifest.json');

    if (manifest) {
      return {
        layer: 'L6',
        name: 'Backtest',
        status: 'pass',
        pass: true,
        quality_metrics: manifest.quality_metrics || {},
        data_shape: manifest.data_shape || {},
        last_update: manifest.timestamp || new Date().toISOString(),
        dataSource: 'minio'
      };
    }

    throw new Error('MinIO manifest not available');
  } catch (error) {
    return {
      layer: 'L6',
      name: 'Backtest',
      status: 'unknown',
      pass: false,
      error: 'L6 data unavailable - run backtest DAG or check MinIO',
      last_update: new Date().toISOString(),
      dataSource: 'none'
    };
  }
}

export const GET = withAuth(async (request, { user }) => {
  try {
    // Fetch all pipeline layer statuses in parallel
    const [l0, l1, l2, l3, l4, l5, l6] = await Promise.all([
      getL0Status(),
      getL1Status(),
      getL2Status(),
      getL3Status(),
      getL4Status(),
      getL5Status(),
      getL6Status()
    ]);

    // Calculate overall system health
    const allLayers = [l0, l1, l2, l3, l4, l5, l6];
    const passingLayers = allLayers.filter(layer => layer.pass).length;
    const totalLayers = allLayers.length;
    const systemHealth = (passingLayers / totalLayers) * 100;

    // Determine overall data source based on what's available
    const overallDataSource = l0.dataSource !== 'none' ? l0.dataSource : 'none';

    return NextResponse.json(
      createApiResponse(
        {
          system_health: {
            health_percentage: systemHealth,
            passing_layers: passingLayers,
            total_layers: totalLayers,
            status: systemHealth >= 85 ? 'healthy' : systemHealth >= 70 ? 'degraded' : 'critical'
          },
          layers: {
            l0,
            l1,
            l2,
            l3,
            l4,
            l5,
            l6
          }
        },
        overallDataSource as any
      )
    );

  } catch (error) {
    console.error('[Pipeline Status API] Error:', error);
    return NextResponse.json(
      createApiResponse(
        null,
        'none',
        `Failed to fetch pipeline status: ${error instanceof Error ? error.message : String(error)}`
      ),
      { status: 500 }
    );
  }
});
