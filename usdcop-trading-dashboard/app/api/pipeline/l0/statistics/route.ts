/**
 * L0 Statistics Endpoint
 * GET /api/pipeline/l0/statistics
 *
 * Provides aggregate statistics on raw L0 data:
 * - Total record count
 * - Date range coverage
 * - Data completeness
 * - Source distribution
 * - Quality metrics
 */

import { NextRequest, NextResponse } from 'next/server';
import { query as pgQuery } from '@/lib/db/postgres-client';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const startDate = searchParams.get('start_date');
  const endDate = searchParams.get('end_date');

  try {
    // Build base query with optional date filtering
    let whereClause = "WHERE symbol = 'USDCOP'";
    const params: any[] = [];
    let paramCount = 1;

    if (startDate) {
      whereClause += ` AND timestamp >= $${paramCount}::timestamptz`;
      params.push(startDate);
      paramCount++;
    }

    if (endDate) {
      whereClause += ` AND timestamp <= $${paramCount}::timestamptz`;
      params.push(endDate);
      paramCount++;
    }

    // Query 1: Overall statistics
    const statsQuery = `
      SELECT
        COUNT(*) as total_records,
        MIN(timestamp) as earliest_timestamp,
        MAX(timestamp) as latest_timestamp,
        MIN(price) as min_price,
        MAX(price) as max_price,
        AVG(price) as avg_price,
        STDDEV(price) as price_stddev,
        SUM(volume) as total_volume,
        COUNT(DISTINCT DATE(timestamp)) as trading_days
      FROM market_data
      ${whereClause}
    `;

    const statsResult = await pgQuery(statsQuery, params);
    const stats = statsResult.rows[0];

    // Query 2: Source distribution
    const sourceQuery = `
      SELECT
        source,
        COUNT(*) as count,
        MIN(timestamp) as first_record,
        MAX(timestamp) as last_record
      FROM market_data
      ${whereClause}
      GROUP BY source
      ORDER BY count DESC
    `;

    const sourceResult = await pgQuery(sourceQuery, params);

    // Query 3: Data completeness by day
    const completenessQuery = `
      SELECT
        DATE(timestamp) as trading_date,
        COUNT(*) as records_per_day,
        MIN(timestamp) as session_start,
        MAX(timestamp) as session_end
      FROM market_data
      ${whereClause}
      GROUP BY DATE(timestamp)
      ORDER BY trading_date DESC
      LIMIT 30
    `;

    const completenessResult = await pgQuery(completenessQuery, params);

    // Query 4: Hourly distribution (Colombian market hours: 8:00-12:55)
    const hourlyQuery = `
      SELECT
        EXTRACT(HOUR FROM timestamp AT TIME ZONE 'America/Bogota') as hour,
        COUNT(*) as records,
        AVG(price) as avg_price
      FROM market_data
      ${whereClause}
      GROUP BY EXTRACT(HOUR FROM timestamp AT TIME ZONE 'America/Bogota')
      ORDER BY hour
    `;

    const hourlyResult = await pgQuery(hourlyQuery, params);

    // Calculate data quality metrics
    const expectedRecordsPerDay = 60; // 5-minute bars from 8:00-12:55 = ~60 bars
    const dataQuality = {
      completeness: completenessResult.rows.map((row: any) => ({
        date: row.trading_date,
        records: parseInt(row.records_per_day),
        expected: expectedRecordsPerDay,
        percentage: (parseInt(row.records_per_day) / expectedRecordsPerDay * 100).toFixed(2) + '%',
        sessionStart: row.session_start,
        sessionEnd: row.session_end,
      })),
      avgRecordsPerDay: completenessResult.rows.reduce((sum: number, row: any) =>
        sum + parseInt(row.records_per_day), 0) / completenessResult.rows.length,
    };

    return NextResponse.json({
      success: true,
      timestamp: new Date().toISOString(),
      statistics: {
        overview: {
          totalRecords: parseInt(stats.total_records),
          dateRange: {
            earliest: stats.earliest_timestamp,
            latest: stats.latest_timestamp,
            tradingDays: parseInt(stats.trading_days),
          },
          priceMetrics: {
            min: parseFloat(stats.min_price),
            max: parseFloat(stats.max_price),
            avg: parseFloat(stats.avg_price),
            stddev: parseFloat(stats.price_stddev),
          },
          volume: {
            total: parseInt(stats.total_volume || 0),
          },
        },
        sources: sourceResult.rows.map((row: any) => ({
          source: row.source,
          count: parseInt(row.count),
          percentage: (parseInt(row.count) / parseInt(stats.total_records) * 100).toFixed(2) + '%',
          firstRecord: row.first_record,
          lastRecord: row.last_record,
        })),
        dataQuality,
        hourlyDistribution: hourlyResult.rows.map((row: any) => ({
          hour: parseInt(row.hour),
          records: parseInt(row.records),
          avgPrice: parseFloat(row.avg_price),
        })),
      },
    });

  } catch (error) {
    console.error('[L0 Statistics API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L0 statistics',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
