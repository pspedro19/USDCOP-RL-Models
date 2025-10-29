import { NextRequest, NextResponse } from 'next/server';

/**
 * Volume Profile API Endpoint
 *
 * Calculates volume distribution across price levels from OHLCV data
 * This provides a better representation of OTC market liquidity than a fake order book
 *
 * @route GET /api/market/volume-profile?hours=24
 */

const POSTGRES_CONFIG = {
  host: process.env.POSTGRES_HOST || 'usdcop-postgres-timescale',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
  database: process.env.POSTGRES_DB || 'usdcop_trading',
  user: process.env.POSTGRES_USER || 'admin',
  password: process.env.POSTGRES_PASSWORD || 'admin123',
};

async function queryPostgres(query: string, params: any[] = []) {
  const { Client } = require('pg');
  const client = new Client(POSTGRES_CONFIG);

  try {
    await client.connect();
    const result = await client.query(query, params);
    return result.rows;
  } finally {
    await client.end();
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const hours = parseInt(searchParams.get('hours') || '24');

    // Validate hours parameter
    if (hours < 1 || hours > 720) { // Max 30 days
      return NextResponse.json(
        { success: false, error: 'Hours must be between 1 and 720' },
        { status: 400 }
      );
    }

    // Calculate volume profile from OHLCV data
    // Group by price levels (rounded to nearest 10 COP)
    const volumeProfileQuery = `
      WITH price_levels AS (
        SELECT
          FLOOR(close / 10) * 10 as price_level,
          SUM(volume) as total_volume,
          COUNT(*) as bar_count
        FROM usdcop_m5_ohlcv
        WHERE time >= NOW() - INTERVAL '${hours} hours'
          AND symbol = 'USD/COP'
          AND volume > 0
        GROUP BY FLOOR(close / 10) * 10
        HAVING SUM(volume) > 0
      ),
      total_volume_calc AS (
        SELECT SUM(total_volume) as total
        FROM price_levels
      )
      SELECT
        pl.price_level,
        pl.total_volume,
        pl.bar_count,
        ROUND((pl.total_volume::numeric / NULLIF(tv.total, 0)::numeric * 100), 2) as percentage
      FROM price_levels pl
      CROSS JOIN total_volume_calc tv
      ORDER BY pl.price_level ASC;
    `;

    // Get current price
    const currentPriceQuery = `
      SELECT close as current_price
      FROM usdcop_m5_ohlcv
      WHERE symbol = 'USD/COP'
      ORDER BY time DESC
      LIMIT 1;
    `;

    // Execute queries
    const [volumeLevels, currentPriceResult] = await Promise.all([
      queryPostgres(volumeProfileQuery),
      queryPostgres(currentPriceQuery)
    ]);

    // Check if we have data
    if (!volumeLevels || volumeLevels.length === 0) {
      return NextResponse.json({
        success: false,
        error: `No OHLCV data found in last ${hours} hours. Run L0 DAG to populate data.`,
        data: null
      });
    }

    const currentPrice = currentPriceResult[0]?.current_price || volumeLevels[volumeLevels.length - 1]?.price_level || 0;

    // Find POC (Point of Control) - price level with highest volume
    const pocLevel = volumeLevels.reduce((max, level) =>
      (level.total_volume > max.total_volume) ? level : max
    , volumeLevels[0]);

    // Calculate total volume
    const totalVolume = volumeLevels.reduce((sum, level) => sum + parseFloat(level.total_volume), 0);

    // Build response
    const response = {
      success: true,
      data: {
        levels: volumeLevels.map((level: any) => ({
          priceLevel: parseFloat(level.price_level),
          totalVolume: parseFloat(level.total_volume),
          barCount: parseInt(level.bar_count),
          percentage: parseFloat(level.percentage)
        })),
        currentPrice: parseFloat(currentPrice),
        pocPrice: parseFloat(pocLevel.price_level),
        totalVolume: totalVolume,
        timestamp: new Date().toISOString(),
        timeRange: `${hours}h`,
        dataPoints: volumeLevels.length
      }
    };

    return NextResponse.json(response, {
      headers: {
        'Cache-Control': 'no-store, max-age=0',
      },
    });

  } catch (error: any) {
    console.error('[Volume Profile API] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to calculate volume profile',
        message: error.message,
        details: process.env.NODE_ENV === 'development' ? error.stack : undefined
      },
      { status: 500 }
    );
  }
}
