/**
 * Filtered Candlesticks API
 * =========================
 * Returns only candles within market hours (8am-12:55pm COT, Mon-Fri).
 * COT (Colombia Time) = UTC-5
 *
 * STRICT RULE: No data outside market hours allowed.
 *
 * NOTE: Database has two timestamp formats:
 * - Old data (before Dec 17, 2025): UTC format (13:00-17:55)
 * - New data (after Dec 17, 2025): COT stored as UTC offset (08:00-12:55)
 * This API handles both formats.
 */

import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';

const pool = new Pool({
  host: process.env.POSTGRES_HOST || 'usdcop-postgres-timescale',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
  database: process.env.POSTGRES_DB || 'usdcop_trading',
  user: process.env.POSTGRES_USER || 'admin',
  password: process.env.POSTGRES_PASSWORD || 'admin123',
  max: 5,
  idleTimeoutMillis: 30000,
});

// Market hours - two formats in database
// OLD FORMAT (UTC): 13:00-17:55 UTC = 8:00-12:55 COT
// NEW FORMAT (COT as UTC): 08:00-12:55 stored with +00 offset
const MARKET_OPEN_UTC = '13:00:00';
const MARKET_CLOSE_UTC = '17:55:00';
const MARKET_OPEN_COT = '08:00:00';
const MARKET_CLOSE_COT = '12:55:00';

// Colombian holidays 2025-2026 (exclude from trading data)
const COLOMBIA_HOLIDAYS = [
  '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
  '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-20',
  '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03', '2025-11-17',
  '2025-12-08', '2025-12-25',
  '2026-01-01', '2026-01-12', '2026-03-23', '2026-04-02', '2026-04-03',
  '2026-05-01', '2026-05-18', '2026-06-08', '2026-06-15', '2026-06-29',
  '2026-07-20', '2026-08-07', '2026-08-17', '2026-10-12', '2026-11-02',
  '2026-11-16', '2026-12-08', '2026-12-25',
];

/**
 * Generate fallback candlestick data when database is unavailable
 * Returns realistic USDCOP price data for display purposes
 */
function generateFallbackCandlesticks(startDate: string, endDate: string, limit: number) {
  const candles = [];
  const start = new Date(startDate);
  const end = new Date(endDate);

  // Base price around current USDCOP levels
  let price = 4250 + Math.random() * 100;
  const volatility = 0.0008; // 0.08% per bar typical

  let currentDate = new Date(start);
  let barCount = 0;

  while (currentDate <= end && barCount < limit) {
    const dayOfWeek = currentDate.getDay();
    const dateStr = currentDate.toISOString().split('T')[0];

    // Skip weekends and holidays
    if (dayOfWeek === 0 || dayOfWeek === 6 || COLOMBIA_HOLIDAYS.includes(dateStr)) {
      currentDate.setDate(currentDate.getDate() + 1);
      currentDate.setHours(8, 0, 0, 0);
      continue;
    }

    // Generate bars for market hours (8:00 - 12:55 COT = 60 5-minute bars)
    for (let minute = 0; minute < 300 && barCount < limit; minute += 5) {
      const hour = 8 + Math.floor(minute / 60);
      const min = minute % 60;

      // Stop at 12:55
      if (hour > 12 || (hour === 12 && min > 55)) break;

      const barTime = new Date(currentDate);
      barTime.setHours(hour, min, 0, 0);

      // Random walk with slight mean reversion
      const change = (Math.random() - 0.5) * 2 * volatility * price;
      const open = price;
      price = price + change;

      // Ensure price stays in realistic range
      price = Math.max(3800, Math.min(4600, price));

      const high = Math.max(open, price) * (1 + Math.random() * 0.0003);
      const low = Math.min(open, price) * (1 - Math.random() * 0.0003);
      const close = price;
      const volume = Math.floor(1000 + Math.random() * 5000);

      candles.push({
        time: barTime.getTime(),
        open: Math.round(open * 100) / 100,
        high: Math.round(high * 100) / 100,
        low: Math.round(low * 100) / 100,
        close: Math.round(close * 100) / 100,
        volume
      });

      barCount++;
    }

    // Move to next day
    currentDate.setDate(currentDate.getDate() + 1);
    currentDate.setHours(8, 0, 0, 0);
  }

  return candles;
}

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const startDate = searchParams.get('start_date') || '2025-01-01';
    const endDate = searchParams.get('end_date') || new Date().toISOString().split('T')[0];
    const limit = Math.min(parseInt(searchParams.get('limit') || '5000'), 100000);

    const client = await pool.connect();

    try {
      // Query bars within market hours using BOTH timestamp formats
      // Exclude Colombian holidays and weekends
      const holidaysArray = COLOMBIA_HOLIDAYS.map(d => `'${d}'`).join(',');

      // NOTE: Removed filter (open != close OR high != low) because historical data
      // from TwelveData (2020-Dec 2025) has single-tick bars where open=high=low=close.
      // These bars still show price movement between bars, just not within each bar.
      const result = await client.query(
        `SELECT
          time,
          open::float,
          high::float,
          low::float,
          close::float,
          volume::int
        FROM public.usdcop_m5_ohlcv
        WHERE time >= $1
          AND time <= ($2::date + interval '1 day')
          AND EXTRACT(DOW FROM time) BETWEEN 1 AND 5
          AND DATE(time)::text NOT IN (${holidaysArray})
          AND (
            -- OLD FORMAT: UTC times (13:00-17:55)
            (time::time >= $3::time AND time::time <= $4::time)
            OR
            -- NEW FORMAT: COT times stored as UTC (08:00-12:55)
            (time::time >= $5::time AND time::time <= $6::time)
          )
        ORDER BY time ASC
        LIMIT $7`,
        [startDate, endDate, MARKET_OPEN_UTC, MARKET_CLOSE_UTC, MARKET_OPEN_COT, MARKET_CLOSE_COT, limit]
      );

      const data = result.rows.map(row => ({
        time: new Date(row.time).getTime(),
        open: row.open,
        high: row.high,
        low: row.low,
        close: row.close,
        volume: row.volume
      }));

      return NextResponse.json({
        success: true,
        symbol: 'USDCOP',
        timeframe: '5m',
        start_date: startDate,
        end_date: endDate,
        count: data.length,
        data,
        metadata: {
          source: 'database_filtered',
          latency: Date.now() - startTime,
          marketHours: '8:00 AM - 12:55 PM COT (Mon-Fri)',
          note: 'Market hours only, excluding Colombian holidays. Historical data (2020-Dec 2025) shows single-tick bars (doji) from TwelveData.'
        }
      }, {
        headers: { 'Cache-Control': 'no-store, max-age=0' }
      });

    } finally {
      client.release();
    }

  } catch (error: any) {
    console.error('[Candlesticks Filtered API] Database error, using fallback:', error.message);

    // Generate fallback data when database is unavailable
    const searchParams = request.nextUrl.searchParams;
    const startDate = searchParams.get('start_date') || '2025-01-01';
    const endDate = searchParams.get('end_date') || new Date().toISOString().split('T')[0];
    const limit = Math.min(parseInt(searchParams.get('limit') || '5000'), 50000);

    const fallbackData = generateFallbackCandlesticks(startDate, endDate, limit);

    return NextResponse.json({
      success: true,
      symbol: 'USDCOP',
      timeframe: '5m',
      start_date: startDate,
      end_date: endDate,
      count: fallbackData.length,
      data: fallbackData,
      metadata: {
        source: 'fallback_generated',
        latency: Date.now() - startTime,
        marketHours: '8:00 AM - 12:55 PM COT (Mon-Fri)',
        note: 'Demo data - database unavailable. Prices are simulated for visualization only.'
      }
    }, {
      headers: { 'Cache-Control': 'no-store, max-age=0' }
    });
  }
}
