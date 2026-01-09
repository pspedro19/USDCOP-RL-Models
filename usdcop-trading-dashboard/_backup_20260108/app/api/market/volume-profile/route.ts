import { NextRequest, NextResponse } from 'next/server';
import { query as pgQuery } from '@/lib/db/postgres-client';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * Volume Profile API Endpoint
 *
 * Calculates volume distribution across price levels from OHLCV data
 * This provides a better representation of OTC market liquidity than a fake order book
 *
 * SECURITY: Uses shared postgres-client (no hardcoded credentials)
 *
 * @route GET /api/market/volume-profile?hours=24
 */

async function queryPostgres(query: string, params: any[] = []) {
  const result = await pgQuery(query, params);
  return result.rows;
}

export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const hours = parseInt(searchParams.get('hours') || '24');

    // Validate hours parameter
    if (hours < 1 || hours > 720) { // Max 30 days
      return NextResponse.json(
        createApiResponse(null, 'none', 'Hours must be between 1 and 720'),
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
      return NextResponse.json(
        createApiResponse(
          null,
          'postgres',
          `No OHLCV data found in last ${hours} hours. Run L0 DAG to populate data.`
        )
      );
    }

    const currentPrice = currentPriceResult[0]?.current_price || volumeLevels[volumeLevels.length - 1]?.price_level || 0;

    // Find POC (Point of Control) - price level with highest volume
    const pocLevel = volumeLevels.reduce((max, level) =>
      (level.total_volume > max.total_volume) ? level : max
    , volumeLevels[0]);

    // Calculate total volume
    const totalVolume = volumeLevels.reduce((sum, level) => sum + parseFloat(level.total_volume), 0);

    // Build response data
    const responseData = {
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
    };

    // Calculate latency
    const latency = Date.now() - startTime;

    return NextResponse.json(
      createApiResponse(responseData, 'postgres', undefined, { latency }),
      {
        headers: {
          'Cache-Control': 'no-store, max-age=0',
        },
      }
    );

  } catch (error: any) {
    console.error('[Volume Profile API] Error:', error);

    const errorMessage = process.env.NODE_ENV === 'development'
      ? `Failed to calculate volume profile: ${error.message}`
      : 'Failed to calculate volume profile';

    return NextResponse.json(
      createApiResponse(null, 'postgres', errorMessage),
      { status: 500 }
    );
  }
});
