/**
 * L1 Pipeline Completeness Endpoint
 * GET /api/pipeline/l1/completeness
 *
 * Returns REAL completeness metrics based on expected trading bars
 * - Considers trading hours: Monday-Friday, 8:00 AM - 12:55 PM COT
 * - Expected: 60 bars per trading day
 * - Excludes weekends automatically
 *
 * Query params:
 *  - period: 'week' | 'month' | 'custom' (default: 'month')
 *  - start_date: ISO date string (for custom period)
 *  - end_date: ISO date string (for custom period)
 */

import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';

// PostgreSQL connection pool
const pool = new Pool({
  host: process.env.POSTGRES_HOST || 'usdcop-postgres-timescale',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
  database: process.env.POSTGRES_DB || 'usdcop_trading',
  user: process.env.POSTGRES_USER || 'admin',
  password: process.env.POSTGRES_PASSWORD || 'admin123',
  max: 10,
  idleTimeoutMillis: 30000,
});

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const period = searchParams.get('period') || 'month';
    const symbol = searchParams.get('symbol') || 'USD/COP';
    const startDate = searchParams.get('start_date');
    const endDate = searchParams.get('end_date');

    let query: string;
    let queryParams: any[] = [symbol];

    // Select appropriate query based on period
    switch (period) {
      case 'week':
        query = 'SELECT * FROM get_weekly_completeness($1)';
        break;

      case 'month':
        query = 'SELECT * FROM get_monthly_completeness($1)';
        break;

      case 'custom':
        if (!startDate || !endDate) {
          return NextResponse.json({
            success: false,
            error: 'start_date and end_date are required for custom period'
          }, { status: 400 });
        }
        query = 'SELECT * FROM calculate_completeness($1, $2::DATE, $3::DATE)';
        queryParams = [symbol, startDate, endDate];
        break;

      default:
        return NextResponse.json({
          success: false,
          error: 'Invalid period. Use: week, month, or custom'
        }, { status: 400 });
    }

    // Execute query
    const result = await pool.query(query, queryParams);

    if (result.rows.length === 0) {
      return NextResponse.json({
        success: false,
        error: 'No data found for the specified period'
      }, { status: 404 });
    }

    const data = result.rows[0];

    // Calculate additional metrics
    const avgBarsPerDay = data.trading_days > 0
      ? (data.actual_bars / data.trading_days).toFixed(2)
      : '0.00';

    const missingBars = data.expected_bars - data.actual_bars;
    const missingDaysEquivalent = (missingBars / 60).toFixed(2);

    // Response
    const response = {
      success: true,
      timestamp: new Date().toISOString(),
      period: period,
      symbol: data.symbol,
      dateRange: {
        start: period === 'week' ? data.week_start : (period === 'month' ? data.month_start : data.start_date),
        end: period === 'week' ? data.week_end : (period === 'month' ? data.month_end : data.end_date),
      },
      completeness: {
        percentage: parseFloat(data.completeness_pct),
        expectedBars: data.expected_bars,
        actualBars: parseInt(data.actual_bars),
        missingBars: missingBars,
        missingDaysEquivalent: parseFloat(missingDaysEquivalent),
      },
      tradingDays: {
        total: data.trading_days,
        barsPerDay: 60,
        avgActualBarsPerDay: parseFloat(avgBarsPerDay),
      },
      quality: {
        status: data.completeness_pct >= 95 ? 'excellent' :
                data.completeness_pct >= 90 ? 'good' :
                data.completeness_pct >= 80 ? 'acceptable' :
                data.completeness_pct >= 70 ? 'poor' : 'critical',
        threshold: 95,
        passed: data.completeness_pct >= 95
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    console.error('[L1 Completeness API] Error:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to retrieve L1 completeness metrics',
      details: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}

// Health check endpoint
export async function HEAD(request: NextRequest) {
  try {
    await pool.query('SELECT 1');
    return new NextResponse(null, { status: 200 });
  } catch (error) {
    return new NextResponse(null, { status: 503 });
  }
}
