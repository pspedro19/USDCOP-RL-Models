/**
 * L6 Backtest Results Endpoint (Enhanced)
 * GET /api/pipeline/l6/backtest-results
 *
 * Provides hedge-fund grade backtest results from L6 pipeline
 * Bucket: usdcop-l6-backtest
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

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const runId = searchParams.get('run_id');
  const split = searchParams.get('split'); // 'test' or 'val'
  const metric = searchParams.get('metric'); // specific metric to retrieve

  try {
    const bucket = 'usdcop-l6-backtest';
    const bucketExists = await minioClient.bucketExists(bucket);

    if (!bucketExists) {
      return NextResponse.json({
        success: false,
        error: `Bucket ${bucket} does not exist. L6 backtest pipeline may not have run yet.`,
      }, { status: 404 });
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
        return NextResponse.json({
          success: false,
          message: 'No backtest runs found',
        }, { status: 404 });
      }

      // Get latest run_id (sorted)
      targetRunId = Array.from(runIds).sort().reverse()[0];
    }

    // Filter objects for this run
    const runObjects = allObjects.filter(obj => obj.name.includes(`run_id=${targetRunId}`));

    if (runObjects.length === 0) {
      return NextResponse.json({
        success: false,
        message: `Run ${targetRunId} not found`,
      }, { status: 404 });
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

    // If specific metric requested, filter response
    if (metric && results.test?.kpis) {
      return NextResponse.json({
        success: true,
        runId: targetRunId,
        metric,
        test: results.test.kpis[metric],
        val: results.val?.kpis?.[metric],
      });
    }

    return NextResponse.json({
      success: true,
      results,
      summary: {
        runId: targetRunId,
        splits: split ? [split] : ['test', 'val'],
        availableMetrics: results.test?.kpis ? Object.keys(results.test.kpis) : [],
      },
    });

  } catch (error) {
    console.error('[L6 Backtest Results API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L6 backtest results',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
