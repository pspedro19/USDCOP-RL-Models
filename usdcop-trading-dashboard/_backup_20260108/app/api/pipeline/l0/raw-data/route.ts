/**
 * L0 Raw Data Endpoint
 * GET /api/pipeline/l0/raw-data
 *
 * Provides access to raw OHLC market data from:
 * 1. PostgreSQL usdcop_m5_ohlcv table (primary source, 81K+ rows)
 * 2. MinIO bucket: 00-raw-usdcop-marketdata (fallback/archive)
 * 3. TwelveData API (real-time/live data)
 *
 * Query Parameters:
 * - start_date: ISO date string (e.g., "2024-01-01")
 * - end_date: ISO date string (e.g., "2024-12-31")
 * - limit: Max number of records (default: 1000, max: 10000)
 * - offset: Pagination offset (default: 0)
 * - source: Data source preference ('postgres' | 'minio' | 'twelvedata' | 'all')
 */

import { NextRequest, NextResponse } from 'next/server';
import { query as pgQuery } from '@/lib/db/postgres-client';
import { minioClient } from '@/lib/services/minio-client';
import { fetchTimeSeries } from '@/lib/services/twelvedata';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';

export const GET = withAuth(async (request, { user }) => {
  const searchParams = request.nextUrl.searchParams;

  // Parse query parameters
  const startDate = searchParams.get('start_date') || searchParams.get('startDate');
  const endDate = searchParams.get('end_date') || searchParams.get('endDate');
  const limit = Math.min(parseInt(searchParams.get('limit') || '1000'), 10000);
  const offset = parseInt(searchParams.get('offset') || '0');
  const source = searchParams.get('source') || 'postgres';

  try {
    // Call the Pipeline Data API backend service
    const backendUrl = process.env.PIPELINE_DATA_API_URL || 'http://pipeline-data-api:8002';
    const params = new URLSearchParams();
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    params.append('source', source);

    const response = await fetch(`${backendUrl}/api/pipeline/l0/raw-data?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(15000)
    });

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    }

    // If backend fails, use fallback logic
    console.warn('[L0 API] Backend service unavailable, using fallback');

    let data: any[] = [];
    let metadata: any = {
      source: source,
      timestamp: new Date().toISOString(),
    };

    // Source 1: PostgreSQL (Primary - TimescaleDB with 92K+ rows)
    if (source === 'postgres' || source === 'all') {
      try {
        let sqlQuery = `
          SELECT
            time as timestamp,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            source,
            created_at
          FROM usdcop_m5_ohlcv
          WHERE symbol = 'USD/COP'
        `;

        const params: any[] = [];
        let paramCount = 1;

        if (startDate) {
          sqlQuery += ` AND time >= $${paramCount}::timestamptz`;
          params.push(startDate);
          paramCount++;
        }

        if (endDate) {
          sqlQuery += ` AND time <= $${paramCount}::timestamptz`;
          params.push(endDate);
          paramCount++;
        }

        sqlQuery += ` ORDER BY time DESC LIMIT $${paramCount} OFFSET $${paramCount + 1}`;
        params.push(limit, offset);

        const result = await pgQuery(sqlQuery, params);
        data = result.rows;

        metadata.postgres = {
          count: data.length,
          hasMore: data.length === limit,
          table: 'usdcop_m5_ohlcv',
        };

        console.log(`[L0 API] Retrieved ${data.length} rows from PostgreSQL`);
      } catch (pgError) {
        console.error('[L0 API] PostgreSQL error:', pgError);
        metadata.postgres = { error: 'Failed to query PostgreSQL' };
      }
    }

    // Source 2: MinIO (Archive/Backup)
    if ((source === 'minio' || source === 'all') && data.length === 0) {
      try {
        const bucket = '00-raw-usdcop-marketdata';
        const bucketExists = await minioClient.bucketExists(bucket);

        if (bucketExists) {
          // List objects in date range
          const prefix = startDate
            ? `data/${startDate.substring(0, 10).replace(/-/g, '')}/`
            : 'data/';

          const objects = await minioClient.listObjects(bucket, prefix);

          // Load first few objects (limit to prevent memory issues)
          const objectsToLoad = objects.slice(0, Math.min(10, Math.ceil(limit / 60)));

          for (const obj of objectsToLoad) {
            try {
              const rawData = await minioClient.getObject(bucket, obj.name);

              // Convert to standard format
              const priceData = {
                timestamp: rawData.timestamp || rawData.datetime,
                symbol: 'USDCOP',
                close: rawData.mid || rawData.close,
                open: rawData.open,
                high: rawData.high,
                low: rawData.low,
                volume: rawData.volume || 0,
                source: 'minio',
              };

              data.push(priceData);
            } catch (err) {
              console.warn(`[L0 API] Failed to parse ${obj.name}:`, err);
            }
          }

          metadata.minio = {
            count: data.length,
            objectsFound: objects.length,
            objectsLoaded: objectsToLoad.length,
            bucket: bucket,
          };

          console.log(`[L0 API] Retrieved ${data.length} rows from MinIO`);
        } else {
          metadata.minio = { error: 'Bucket does not exist' };
        }
      } catch (minioError) {
        console.error('[L0 API] MinIO error:', minioError);
        metadata.minio = { error: 'Failed to query MinIO' };
      }
    }

    // Source 3: TwelveData API (Real-time/Live)
    if ((source === 'twelvedata' || source === 'all') && data.length === 0) {
      try {
        const timeSeries = await fetchTimeSeries('USD/COP', '5min', limit);

        if (timeSeries && timeSeries.length > 0) {
          // Filter by date range if provided
          let filteredData = timeSeries;

          if (startDate || endDate) {
            const start = startDate ? new Date(startDate) : new Date(0);
            const end = endDate ? new Date(endDate) : new Date();

            filteredData = timeSeries.filter((item: any) => {
              const itemDate = new Date(item.datetime);
              return itemDate >= start && itemDate <= end;
            });
          }

          data = filteredData.map((item: any) => ({
            timestamp: item.datetime,
            symbol: 'USDCOP',
            close: parseFloat(item.close),
            open: parseFloat(item.open),
            high: parseFloat(item.high),
            low: parseFloat(item.low),
            volume: parseInt(item.volume || '0'),
            source: 'twelvedata',
          }));

          metadata.twelvedata = {
            count: data.length,
            total: timeSeries.length,
            filtered: filteredData.length,
          };

          console.log(`[L0 API] Retrieved ${data.length} rows from TwelveData`);
        }
      } catch (apiError) {
        console.error('[L0 API] TwelveData API error:', apiError);
        metadata.twelvedata = { error: 'Failed to query TwelveData API' };
      }
    }

    // Return response
    if (data.length === 0) {
      return NextResponse.json(
        createApiResponse(
          null,
          'none',
          'No data found for the specified parameters',
          { message: `Tried sources: ${source}` }
        ),
        { status: 404 }
      );
    }

    // Determine the actual data source used
    const actualSource = metadata.postgres?.count ? 'postgres' :
                        metadata.minio?.count ? 'minio' :
                        metadata.twelvedata?.count ? 'live' : 'none';

    return NextResponse.json(
      createApiResponse(
        {
          count: data.length,
          data: data,
          metadata,
          pagination: {
            limit,
            offset,
            hasMore: data.length === limit,
          },
        },
        actualSource as any
      )
    );

  } catch (error) {
    console.error('[L0 API] Error:', error);
    return NextResponse.json(
      createApiResponse(
        null,
        'none',
        `Failed to retrieve L0 raw data: ${error instanceof Error ? error.message : 'Unknown error'}`
      ),
      { status: 500 }
    );
  }
});
