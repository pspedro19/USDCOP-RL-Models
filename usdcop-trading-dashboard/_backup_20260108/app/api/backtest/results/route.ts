/**
 * Backtest Results API Endpoint
 * 
 * Provides access to hedge fund grade backtest results from the L6 MinIO bucket.
 * This endpoint connects the frontend dashboard to the Airflow DAG outputs 
 * containing comprehensive performance metrics, trade analytics, and risk analysis.
 */

import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse, measureLatency } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

// MinIO client for accessing L6 backtest data
// SECURITY: No hardcoded credentials - all from environment variables
class MinIOBacktestClient {
  private endpoint: string;
  private accessKey: string;
  private secretKey: string;
  private bucket: string = 'usdcop-l6-backtest';

  constructor() {
    this.endpoint = process.env.MINIO_ENDPOINT || '';
    this.accessKey = process.env.MINIO_ACCESS_KEY || '';
    this.secretKey = process.env.MINIO_SECRET_KEY || '';

    if (!this.endpoint || !this.accessKey || !this.secretKey) {
      console.warn('[MinIOBacktestClient] MinIO credentials not configured');
    }
  }

  /**
   * Get the latest backtest run ID from the L6 bucket
   */
  async getLatestBacktestRunId(): Promise<string | null> {
    try {
      // In production, this would scan the L6 bucket for latest run
      // For now, we'll look for the most recent date folder
      const response = await fetch(`http://${this.endpoint}/${this.bucket}/?list-type=2&prefix=date=`);
      
      if (!response.ok) {
        console.warn('[BacktestAPI] MinIO connection failed, using fallback');
        return null;
      }

      // Parse S3 XML response to find latest date folder
      const text = await response.text();
      const dateMatches = text.match(/<Key>date=(\d{4}-\d{2}-\d{2})\/[^<]*<\/Key>/g);
      
      if (!dateMatches || dateMatches.length === 0) {
        return null;
      }

      // Extract dates and find the most recent
      const dates = dateMatches.map(match => {
        const dateMatch = match.match(/date=(\d{4}-\d{2}-\d{2})/);
        return dateMatch ? dateMatch[1] : null;
      }).filter(Boolean);

      const latestDate = dates.sort().reverse()[0];
      
      // Now find the latest run_id for this date
      const runResponse = await fetch(`http://${this.endpoint}/${this.bucket}/?list-type=2&prefix=date=${latestDate}/run_id=`);
      
      if (!runResponse.ok) return null;
      
      const runText = await runResponse.text();
      const runMatches = runText.match(/<Key>date=\d{4}-\d{2}-\d{2}\/run_id=([^\/]+)\/[^<]*<\/Key>/g);
      
      if (!runMatches || runMatches.length === 0) {
        return null;
      }

      const runIds = runMatches.map(match => {
        const runMatch = match.match(/run_id=([^\/]+)/);
        return runMatch ? runMatch[1] : null;
      }).filter(Boolean);

      return runIds.sort().reverse()[0]; // Get latest run_id
      
    } catch (error) {
      console.error('[BacktestAPI] Error getting latest run ID:', error);
      return null;
    }
  }

  /**
   * Fetch backtest results from MinIO L6 bucket
   */
  async getBacktestResults(runId?: string): Promise<any> {
    try {
      const targetRunId = runId || await this.getLatestBacktestRunId();
      
      if (!targetRunId) {
        throw new Error('No backtest runs found');
      }

      // Find the date for this run_id
      const latestDate = new Date().toISOString().split('T')[0]; // Today as fallback
      const basePath = `date=${latestDate}/run_id=${targetRunId}`;

      // Fetch key files from L6 bucket
      const results = {
        runId: targetRunId,
        timestamp: new Date().toISOString(),
        test: await this.fetchSplitData(basePath, 'test'),
        val: await this.fetchSplitData(basePath, 'val'),
        metadata: await this.fetchMetadata(basePath)
      };

      return results;
      
    } catch (error) {
      console.error('[BacktestAPI] Error fetching backtest results:', error);
      throw error;
    }
  }

  /**
   * Fetch data for a specific split (test or val)
   */
  private async fetchSplitData(basePath: string, split: string): Promise<any> {
    const splitPath = `${basePath}/split=${split}`;
    
    try {
      // Fetch KPIs
      const kpisResponse = await fetch(`http://${this.endpoint}/${this.bucket}/${splitPath}/metrics/kpis_${split}.json`);
      const kpis = kpisResponse.ok ? await kpisResponse.json() : null;

      // Fetch rolling metrics
      const rollingResponse = await fetch(`http://${this.endpoint}/${this.bucket}/${splitPath}/metrics/kpis_${split}_rolling.json`);
      const rolling = rollingResponse.ok ? await rollingResponse.json() : null;

      // Fetch trade ledger (would need to convert parquet to JSON in production)
      const tradesResponse = await fetch(`http://${this.endpoint}/${this.bucket}/${splitPath}/trades/trade_ledger.parquet`);
      const trades = tradesResponse.ok ? await this.parseParquetToJson(await tradesResponse.arrayBuffer()) : [];

      // Fetch daily returns (would need to convert parquet to JSON in production)
      const returnsResponse = await fetch(`http://${this.endpoint}/${this.bucket}/${splitPath}/returns/daily_returns.parquet`);
      const dailyReturns = returnsResponse.ok ? await this.parseParquetToJson(await returnsResponse.arrayBuffer()) : [];

      // Fetch manifest
      const manifestResponse = await fetch(`http://${this.endpoint}/${this.bucket}/${splitPath}/meta/backtest_manifest.json`);
      const manifest = manifestResponse.ok ? await manifestResponse.json() : null;

      return {
        kpis,
        rolling,
        trades,
        dailyReturns,
        manifest
      };

    } catch (error) {
      console.warn(`[BacktestAPI] Error fetching ${split} data:`, error);
      return null;
    }
  }

  /**
   * Fetch backtest metadata
   */
  private async fetchMetadata(basePath: string): Promise<any> {
    try {
      const indexResponse = await fetch(`http://${this.endpoint}/${this.bucket}/${basePath}/index.json`);
      return indexResponse.ok ? await indexResponse.json() : null;
    } catch (error) {
      console.warn('[BacktestAPI] Error fetching metadata:', error);
      return null;
    }
  }

  /**
   * Parse Parquet buffer to JSON (simplified - in production would use proper parquet library)
   */
  private async parseParquetToJson(buffer: ArrayBuffer): Promise<any[]> {
    // In production, this would use a proper parquet parser
    // For now, return empty array as placeholder
    console.warn('[BacktestAPI] Parquet parsing not implemented, returning empty array');
    return [];
  }
}

// Circuit breaker for Backtest API backend
const backtestCircuitBreaker = getCircuitBreaker('backtest-api', {
  failureThreshold: 3,
  resetTimeout: 30000, // 30 seconds
  monitorInterval: 5000,
});

/**
 * GET /api/backtest/results
 *
 * Query parameters:
 * - runId: specific backtest run ID (optional, defaults to latest)
 * - split: test or val (optional, defaults to both)
 */
export const GET = withAuth(async (request, { user }) => {
  console.log('[BacktestAPI] GET /api/backtest/results');
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const runId = searchParams.get('runId') || undefined;
    const split = searchParams.get('split') || undefined;

    // Execute through circuit breaker for resilience
    try {
      const data = await backtestCircuitBreaker.execute(async () => {
        const backendUrl = process.env.BACKTEST_API_URL || 'http://localhost:8006';
        const params = new URLSearchParams();
        if (split) params.append('split', split);

        const response = await fetch(`${backendUrl}/api/backtest/results?${params.toString()}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          signal: AbortSignal.timeout(10000)
        });

        if (!response.ok) {
          throw new Error(`Backend error: ${response.status}`);
        }

        return response.json();
      });

      const latency = Date.now() - startTime;
      const backendUrl = process.env.BACKTEST_API_URL || 'http://localhost:8006';

      return NextResponse.json(
        createApiResponse(true, 'live', {
          data,
          latency,
          backendUrl
        })
      );

    } catch (backendError) {
      if (backendError instanceof CircuitOpenError) {
        console.warn('[BacktestAPI] Circuit breaker OPEN - using MinIO fallback');
      } else {
        console.warn('[BacktestAPI] Backend service unavailable, using MinIO fallback');
      }

      // Fallback to MinIO
      const client = new MinIOBacktestClient();
      const results = await client.getBacktestResults(runId);

      // Filter by split if requested
      if (split && (split === 'test' || split === 'val')) {
        const resultData = {
          runId: results.runId,
          timestamp: results.timestamp,
          [split]: results[split],
          metadata: results.metadata
        };
        const latency = Date.now() - startTime;
        return NextResponse.json(
          createApiResponse(true, 'minio', {
            data: resultData,
            latency,
            circuitState: backendError instanceof CircuitOpenError ? 'OPEN' : undefined
          })
        );
      }

      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(true, 'minio', {
          data: results,
          latency,
          circuitState: backendError instanceof CircuitOpenError ? 'OPEN' : undefined
        })
      );
    }

  } catch (error) {
    console.error('[BacktestAPI] Error in GET /api/backtest/results:', error);
    const latency = Date.now() - startTime;

    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to fetch backtest results',
        message: error instanceof Error ? error.message : 'Unknown error',
        latency
      }),
      { status: 500 }
    );
  }
});

/**
 * POST /api/backtest/trigger
 *
 * Trigger a new backtest run via Airflow DAG
 */
export const POST = withAuth(async (request, { user }) => {
  console.log('[BacktestAPI] POST /api/backtest/trigger');
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { force = false } = body;

    // Execute through circuit breaker for resilience
    const data = await backtestCircuitBreaker.execute(async () => {
      const backendUrl = process.env.BACKTEST_API_URL || 'http://localhost:8006';

      const response = await fetch(`${backendUrl}/api/backtest/trigger`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ force }),
        signal: AbortSignal.timeout(10000)
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      return response.json();
    });

    const latency = Date.now() - startTime;
    const backendUrl = process.env.BACKTEST_API_URL || 'http://localhost:8006';

    return NextResponse.json(
      createApiResponse(true, 'live', {
        data,
        latency,
        backendUrl
      })
    );

  } catch (error) {
    const latency = Date.now() - startTime;

    if (error instanceof CircuitOpenError) {
      // Fast-fail when circuit is open
      console.error('[BacktestAPI] Circuit breaker OPEN - service unavailable');
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Backtest API temporarily unavailable',
          message: 'Circuit breaker is open. Service is experiencing issues.',
          circuitState: 'OPEN',
          latency
        }),
        { status: 503 }
      );
    }

    // Other errors
    console.error('[BacktestAPI] Error triggering backtest:', error);
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to trigger backtest',
        message: error instanceof Error ? error.message : 'Backend service not responding. Please ensure Backtest API is running on port 8006.',
        latency
      }),
      { status: 503 }
    );
  }
});