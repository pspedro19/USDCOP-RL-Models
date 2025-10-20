/**
 * L0 Raw Data Endpoint (Legacy)
 * GET /api/pipeline/l0
 *
 * Multi-source fallback: MinIO → TwelveData API
 *
 * NOTE: This is a legacy endpoint. New code should use:
 * - /api/pipeline/l0/raw-data (PostgreSQL primary)
 * - /api/pipeline/l0/statistics (Aggregated metrics)
 */

import { NextRequest, NextResponse } from 'next/server';
import { minioClient } from '@/lib/services/minio-client';
import { fetchTimeSeries } from '@/lib/services/twelvedata';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const startDate = searchParams.get('startDate');
    const endDate = searchParams.get('endDate');

    if (!startDate || !endDate) {
      return NextResponse.json({
        error: 'startDate and endDate are required',
        example: '/api/pipeline/l0?startDate=2024-01-01&endDate=2024-12-31'
      }, { status: 400 });
    }

    // First, try to get data from MinIO using our client wrapper
    try {
      const bucket = '00-raw-usdcop-marketdata';
      const data: any[] = [];

      console.log(`[L0 API] Attempting to fetch from MinIO bucket: ${bucket}`);

      // List objects in bucket
      const objects = await minioClient.listObjects(bucket);

      if (objects.length === 0) {
        console.log(`[L0 API] No objects found in MinIO bucket ${bucket}, falling back to TwelveData`);
        throw new Error('No data in MinIO');
      }

      // Filter objects by date range
      const start = new Date(startDate);
      const end = new Date(endDate);

      const filteredObjects = objects.filter((obj) => {
        const dateMatch = obj.name.match(/data\/(\d{8})\//);
        if (dateMatch) {
          const objDateStr = dateMatch[1];
          const objDate = new Date(
            parseInt(objDateStr.substring(0, 4)),
            parseInt(objDateStr.substring(4, 6)) - 1,
            parseInt(objDateStr.substring(6, 8))
          );
          return objDate >= start && objDate <= end;
        }
        return false;
      });

      if (filteredObjects.length === 0) {
        console.log('[L0 API] No data found in MinIO for date range, falling back to TwelveData');
        throw new Error('No data in MinIO for date range');
      }

      // Sort by name to ensure chronological order
      filteredObjects.sort((a, b) => a.name.localeCompare(b.name));

      // Load objects (limit to 100 to prevent memory issues)
      for (const obj of filteredObjects.slice(0, 100)) {
        try {
          const rawData = await minioClient.getObject(bucket, obj.name);

          // Convert to standard format
          const priceData = {
            datetime: rawData.timestamp || rawData.datetime,
            open: rawData.mid ? rawData.mid - 0.5 : rawData.open,
            high: rawData.mid ? rawData.mid + 1 : rawData.high,
            low: rawData.mid ? rawData.mid - 1 : rawData.low,
            close: rawData.mid || rawData.close,
            volume: rawData.volume || 0
          };

          // Only include data within trading hours (8am-12:55pm COT)
          const dataTime = new Date(priceData.datetime);
          const colombiaTime = new Date(dataTime.toLocaleString("en-US", { timeZone: "America/Bogota" }));
          const hours = colombiaTime.getHours();
          const minutes = colombiaTime.getMinutes();
          const day = colombiaTime.getDay();

          // Check if within trading hours (Mon-Fri, 8:00-12:55)
          if (day >= 1 && day <= 5) {
            const totalMinutes = hours * 60 + minutes;
            if (totalMinutes >= 480 && totalMinutes <= 775) { // 8:00 to 12:55
              data.push(priceData);
            }
          }
        } catch (error) {
          console.warn(`[L0 API] Failed to parse object ${obj.name}:`, error);
        }
      }

      if (data.length > 0) {
        console.log(`[L0 API] Successfully fetched ${data.length} records from MinIO`);
        return NextResponse.json({
          success: true,
          count: data.length,
          data: data,
          source: 'minio',
          message: 'Legacy endpoint. Consider using /api/pipeline/l0/raw-data for PostgreSQL primary source.'
        });
      }
    } catch (minioError) {
      console.log('[L0 API] MinIO error, falling back to TwelveData:', minioError);
    }

    // Fallback to TwelveData API (now with REAL API integration)
    try {
      console.log('[L0 API] Fetching data from TwelveData API...');
      const timeSeries = await fetchTimeSeries('USD/COP', '5min', 100);

      if (timeSeries && timeSeries.length > 0) {
        // Filter by date range
        const start = new Date(startDate);
        const end = new Date(endDate);

        const filteredData = timeSeries.filter((item: any) => {
          const itemDate = new Date(item.timestamp);
          return itemDate >= start && itemDate <= end;
        });

        console.log(`[L0 API] Successfully fetched ${filteredData.length} records from TwelveData`);
        return NextResponse.json({
          success: true,
          count: filteredData.length,
          data: filteredData,
          source: 'twelvedata',
          message: 'Data from TwelveData API. Legacy endpoint - consider using /api/pipeline/l0/raw-data.'
        });
      }

      // If no data from either source
      console.warn('[L0 API] No data available from any source');
      return NextResponse.json({
        success: true,
        count: 0,
        data: [],
        message: 'No data available for the specified date range. Try /api/pipeline/l0/raw-data for PostgreSQL data.'
      });

    } catch (apiError) {
      console.error('[L0 API] TwelveData API error:', apiError);
      return NextResponse.json({
        success: false,
        error: 'Failed to load data from both MinIO and TwelveData',
        details: apiError instanceof Error ? apiError.message : 'Unknown error',
        suggestion: 'Try /api/pipeline/l0/raw-data for PostgreSQL data (92K+ records available)'
      }, { status: 500 });
    }

  } catch (error) {
    console.error('[L0 API] Error in L0 route:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to load L0 data',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}
