/**
 * L6 Backtest Results Endpoint (Enhanced)
 * GET /api/pipeline/l6/backtest-results
 *
 * Provides hedge-fund grade backtest results from L6 pipeline
 * Primary: Multi-Model Trading API (port 8006)
 * Fallback: MinIO bucket usdcop-l6-backtest
 *
 * Metrics include:
 * - Sharpe Ratio, Sortino Ratio, Calmar Ratio
 * - Maximum Drawdown, Win Rate
 * - Trade ledger, Daily returns
 * - Rolling performance metrics
 * - Test and Validation split results
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

const MULTI_MODEL_API = process.env.MULTI_MODEL_API_URL || 'http://localhost:8006';

export const GET = withAuth(async (request, { user }) => {
  const searchParams = request.nextUrl.searchParams;
  const runId = searchParams.get('run_id');
  const split = searchParams.get('split') || 'test'; // 'test' or 'val'
  const metric = searchParams.get('metric'); // specific metric to retrieve
  const startTime = Date.now();

  // --- PRIMARY: Try Multi-Model Trading API ---
  try {
    const backendParams = new URLSearchParams({ split });

    const response = await fetch(
      `${MULTI_MODEL_API}/api/backtest/results?${backendParams}`,
      {
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(5000),
        cache: 'no-store'
      }
    );

    if (response.ok) {
      const data = await response.json();
      const latency = Date.now() - startTime;

      // Return data in format expected by L6BacktestResults component
      return NextResponse.json(
        createApiResponse(
          {
            sharpe_ratio: data.sharpe_ratio,
            sortino_ratio: data.sortino_ratio,
            calmar_ratio: data.calmar_ratio,
            max_drawdown: data.max_drawdown,
            win_rate: data.win_rate,
            profit_factor: data.profit_factor,
            total_trades: data.total_trades,
            total_return: data.total_return,
            avg_trade_pnl: data.avg_trade_pnl,
            winning_trades: data.winning_trades,
            losing_trades: data.losing_trades,
          },
          'live'
        )
      );
    }
  } catch (backendError) {
    console.warn('[L6 Backtest API] Multi-model API unavailable, trying MinIO fallback:', backendError);
  }

  // --- FALLBACK: Try MinIO bucket ---
  try {
    const bucket = 'usdcop-l6-backtest';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      // No MinIO bucket either - return simulated fallback data
      console.warn('[L6 Backtest API] MinIO bucket not found, using simulated data');
      return NextResponse.json(
        createApiResponse(
          {
            sharpe_ratio: 1.42,
            sortino_ratio: 1.85,
            calmar_ratio: 0.92,
            max_drawdown: 0.125,
            win_rate: 0.52,
            profit_factor: 1.68,
            total_trades: 287,
            total_return: 0.0845,
            avg_trade_pnl: 0.000295,
            winning_trades: 149,
            losing_trades: 138,
          },
          'simulated'
        )
      );
    }

    // List all runs
    const allObjects = await minioClient.listObjects(bucket, '');

    // Find latest run if not specified
    let targetRunId = runId;
    if (!targetRunId) {
      // Extract unique run_ids
      const runIds = new Set<string>();
      allObjects.forEach(obj => {
        const match = obj.name.match(/run_id=([^\/]+)/);
        if (match) runIds.add(match[1]);
      });

      if (runIds.size === 0) {
        // No runs found - return simulated data
        return NextResponse.json(
          createApiResponse(
            {
              sharpe_ratio: 1.35,
              sortino_ratio: 1.72,
              calmar_ratio: 0.88,
              max_drawdown: 0.138,
              win_rate: 0.51,
              profit_factor: 1.55,
              total_trades: 245,
              total_return: 0.0723,
              avg_trade_pnl: 0.000295,
              winning_trades: 125,
              losing_trades: 120,
            },
            'simulated'
          )
        );
      }

      // Get latest run_id (sorted)
      targetRunId = Array.from(runIds).sort().reverse()[0];
    }

    // Filter objects for this run
    const runObjects = allObjects.filter(obj => obj.name.includes(`run_id=${targetRunId}`));

    if (runObjects.length === 0) {
      return NextResponse.json(
        createApiResponse(null, 'none', `Run ${targetRunId} not found`),
        { status: 404 }
      );
    }

    // Organize by split
    const splits = ['test', 'val'];
    const results: any = {
      runId: targetRunId,
      timestamp: new Date().toISOString(),
    };

    for (const splitName of splits) {
      if (!split || split === splitName) {
        const splitObjects = runObjects.filter(obj => obj.name.includes(`split=${splitName}`));

        // Get KPIs
        const kpisFile = splitObjects.find(obj => obj.name.includes(`kpis_${splitName}.json`));
        const kpis = kpisFile ? await minioClient.getObject(bucket, kpisFile.name) : null;

        // Get rolling metrics
        const rollingFile = splitObjects.find(obj => obj.name.includes(`kpis_${splitName}_rolling.json`));
        const rolling = rollingFile ? await minioClient.getObject(bucket, rollingFile.name) : null;

        // Get manifest
        const manifestFile = splitObjects.find(obj => obj.name.includes('backtest_manifest.json'));
        const manifest = manifestFile ? await minioClient.getObject(bucket, manifestFile.name) : null;

        // List trade files and return files
        const tradeFiles = splitObjects.filter(obj => obj.name.includes('/trades/'));
        const returnFiles = splitObjects.filter(obj => obj.name.includes('/returns/'));

        results[splitName] = {
          kpis,
          rolling,
          manifest,
          files: {
            trades: tradeFiles.map(f => ({
              name: f.name,
              size: f.size,
              lastModified: f.lastModified,
            })),
            returns: returnFiles.map(f => ({
              name: f.name,
              size: f.size,
              lastModified: f.lastModified,
            })),
          },
        };
      }
    }

    // Extract metrics for component format if KPIs exist
    if (results.test?.kpis || results.val?.kpis) {
      const kpis = results[split]?.kpis || results.test?.kpis;
      if (kpis) {
        return NextResponse.json(
          createApiResponse(
            {
              sharpe_ratio: kpis.sharpe_ratio || 0,
              sortino_ratio: kpis.sortino_ratio || 0,
              calmar_ratio: kpis.calmar_ratio || 0,
              max_drawdown: kpis.max_drawdown || 0,
              win_rate: kpis.win_rate || 0,
              profit_factor: kpis.profit_factor || 0,
              total_trades: kpis.total_trades || 0,
              total_return: kpis.total_return || 0,
              avg_trade_pnl: kpis.avg_trade_pnl || 0,
              winning_trades: kpis.winning_trades || 0,
              losing_trades: kpis.losing_trades || 0,
            },
            'minio'
          )
        );
      }
    }

    // If specific metric requested, filter response
    if (metric && results.test?.kpis) {
      return NextResponse.json(
        createApiResponse(
          {
            runId: targetRunId,
            metric,
            test: results.test.kpis[metric],
            val: results.val?.kpis?.[metric],
          },
          'minio'
        )
      );
    }

    return NextResponse.json(
      createApiResponse(
        {
          results,
          summary: {
            runId: targetRunId,
            splits: split ? [split] : ['test', 'val'],
            availableMetrics: results.test?.kpis ? Object.keys(results.test.kpis) : [],
          },
        },
        'minio'
      )
    );

  } catch (error) {
    console.error('[L6 Backtest Results API] Error:', error);

    // Ultimate fallback - return simulated data so dashboard works
    return NextResponse.json(
      createApiResponse(
        {
          sharpe_ratio: 1.28,
          sortino_ratio: 1.65,
          calmar_ratio: 0.82,
          max_drawdown: 0.145,
          win_rate: 0.50,
          profit_factor: 1.45,
          total_trades: 210,
          total_return: 0.0612,
          avg_trade_pnl: 0.000291,
          winning_trades: 105,
          losing_trades: 105,
        },
        'simulated'
      )
    );
  }
});
