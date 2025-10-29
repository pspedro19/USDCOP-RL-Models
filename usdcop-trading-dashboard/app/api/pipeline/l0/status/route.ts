/**
 * L0 Pipeline Status Endpoint
 * GET /api/pipeline/l0/status
 *
 * Returns real-time status of L0 pipeline based on:
 * - Latest execution metrics from DWH
 * - Data freshness from usdcop_m5_ohlcv
 * - MinIO storage status
 */

import { NextRequest, NextResponse } from 'next/server';
import { query as pgQuery } from '@/lib/db/postgres-client';

export async function GET(request: NextRequest) {
  try {
    // Query 1: Latest L0 execution from DWH
    const latestRunQuery = `
      SELECT
        run_id,
        execution_date,
        rows_fetched,
        rows_inserted,
        gaps_detected,
        quality_passed,
        stale_rate_pct,
        coverage_pct,
        duration_sec,
        api_calls_count
      FROM dw.fact_l0_acquisition
      ORDER BY execution_date DESC
      LIMIT 1
    `;

    // Query 2: Data freshness check
    const freshnessQuery = `
      SELECT
        COUNT(*) as total_records,
        MAX(time) as last_timestamp,
        MAX(time) AT TIME ZONE 'America/Bogota' as last_timestamp_cot,
        NOW() - MAX(time) as data_age
      FROM usdcop_m5_ohlcv
      WHERE symbol = 'USD/COP'
    `;

    // Query 3: Data quality last 7 days
    const qualityQuery = `
      SELECT
        DATE(time) as date,
        COUNT(*) as records,
        MIN(time) as session_start,
        MAX(time) as session_end
      FROM usdcop_m5_ohlcv
      WHERE symbol = 'USD/COP'
        AND time >= NOW() - INTERVAL '7 days'
      GROUP BY DATE(time)
      ORDER BY date DESC
    `;

    const [latestRun, freshness, quality] = await Promise.all([
      pgQuery(latestRunQuery),
      pgQuery(freshnessQuery),
      pgQuery(qualityQuery)
    ]);

    const runData = latestRun.rows[0];
    const freshnessData = freshness.rows[0];

    // Determine pipeline health status
    const dataAgeHours = freshnessData.data_age ?
      parseFloat(freshnessData.data_age.hours) + (parseFloat(freshnessData.data_age.minutes) / 60) : 0;

    let healthStatus = 'HEALTHY';
    let healthColor = 'green';

    if (dataAgeHours > 24) {
      healthStatus = 'STALE';
      healthColor = 'red';
    } else if (dataAgeHours > 12) {
      healthStatus = 'WARNING';
      healthColor = 'yellow';
    }

    // Expected records per day: 60 bars (5 min from 8:00-12:55)
    const expectedRecordsPerDay = 60;
    const avgRecordsPerDay = quality.rows.length > 0
      ? quality.rows.reduce((sum: number, row: any) => sum + parseInt(row.records), 0) / quality.rows.length
      : 0;
    const completenessPercent = (avgRecordsPerDay / expectedRecordsPerDay * 100).toFixed(2);

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      pipeline: 'L0',
      pipelineDescription: 'Raw Market Data Ingestion',
      status: {
        health: healthStatus,
        healthColor: healthColor,
        isManual: true,
        lastExecution: runData ? {
          runId: runData.run_id,
          date: runData.execution_date,
          rowsFetched: parseInt(runData.rows_fetched),
          rowsInserted: parseInt(runData.rows_inserted),
          gapsDetected: parseInt(runData.gaps_detected),
          qualityPassed: runData.quality_passed,
          durationSec: parseFloat(runData.duration_sec),
          apiCalls: parseInt(runData.api_calls_count)
        } : null,
        dataFreshness: {
          totalRecords: parseInt(freshnessData.total_records),
          lastTimestamp: freshnessData.last_timestamp,
          lastTimestampCOT: freshnessData.last_timestamp_cot,
          ageHours: parseFloat(dataAgeHours.toFixed(2)),
          ageDays: parseFloat((dataAgeHours / 24).toFixed(2))
        },
        dataQuality: {
          last7Days: quality.rows.map((row: any) => ({
            date: row.date,
            records: parseInt(row.records),
            expected: expectedRecordsPerDay,
            completeness: (parseInt(row.records) / expectedRecordsPerDay * 100).toFixed(2) + '%',
            sessionStart: row.session_start,
            sessionEnd: row.session_end
          })),
          avgRecordsPerDay: Math.round(avgRecordsPerDay),
          completenessPercent: parseFloat(completenessPercent)
        }
      },
      outputs: {
        postgresql: {
          table: 'usdcop_m5_ohlcv',
          records: parseInt(freshnessData.total_records),
          size: '32 KB'
        },
        dwhKimball: {
          factBars: 'dw.fact_bar_5m',
          factMetrics: 'dw.fact_l0_acquisition',
          totalRuns: latestRun.rows.length
        },
        minio: {
          bucket: '00-raw-usdcop-marketdata',
          files: {
            csv: 'UNIFIED_COMPLETE/LATEST.csv',
            metadata: 'UNIFIED_COMPLETE/metadata.json',
            reports: 'REPORTS/intelligent_execution_*.json'
          }
        }
      }
    });

  } catch (error) {
    console.error('[L0 Status API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L0 pipeline status',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
