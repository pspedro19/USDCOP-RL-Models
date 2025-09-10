import { NextRequest, NextResponse } from 'next/server';
import * as Minio from 'minio';
import { fetchTimeSeries } from '@/lib/services/twelvedata';

const client = new Minio.Client({
  endPoint: 'localhost',
  port: 9000,
  accessKey: 'minioadmin',
  secretKey: 'minioadmin123',
  useSSL: false
});

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const startDate = searchParams.get('startDate');
    const endDate = searchParams.get('endDate');
    
    if (!startDate || !endDate) {
      return NextResponse.json({ error: 'startDate and endDate are required' }, { status: 400 });
    }
    
    // First, try to get data from MinIO
    try {
      const bucket = '00-raw-usdcop-marketdata';
      const data: any[] = [];
      
      // Check if bucket exists
      const bucketExists = await client.bucketExists(bucket);
      if (!bucketExists) {
        console.log(`Bucket ${bucket} does not exist, falling back to TwelveData`);
        throw new Error('Bucket not found');
      }
      
      // List objects in date range
      const stream = client.listObjectsV2(bucket, '', true);
      const objects: any[] = [];
      
      for await (const obj of stream) {
        // Filter by date range based on object name pattern
        const dateMatch = obj.name.match(/data\/(\d{8})\//);
        if (dateMatch) {
          const objDateStr = dateMatch[1];
          const objDate = new Date(
            parseInt(objDateStr.substring(0, 4)),
            parseInt(objDateStr.substring(4, 6)) - 1,
            parseInt(objDateStr.substring(6, 8))
          );
          
          const start = new Date(startDate);
          const end = new Date(endDate);
          
          if (objDate >= start && objDate <= end) {
            objects.push(obj);
          }
        }
      }
      
      // If no objects found, fallback to TwelveData
      if (objects.length === 0) {
        console.log('No data found in MinIO for date range, falling back to TwelveData');
        throw new Error('No data in MinIO');
      }
      
      // Sort by name to ensure chronological order
      objects.sort((a, b) => a.name.localeCompare(b.name));
      
      // Load and parse each object (limit to prevent memory issues)
      for (const obj of objects.slice(0, 100)) {
        try {
          const dataStream = await client.getObject(bucket, obj.name);
          const chunks: Buffer[] = [];
          
          for await (const chunk of dataStream) {
            chunks.push(chunk);
          }
          
          const content = Buffer.concat(chunks).toString();
          const rawData = JSON.parse(content);
          
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
          console.warn(`Failed to parse object ${obj.name}:`, error);
        }
      }
      
      if (data.length > 0) {
        return NextResponse.json({
          success: true,
          count: data.length,
          data: data,
          source: 'minio'
        });
      }
    } catch (minioError) {
      console.log('MinIO error, falling back to TwelveData:', minioError);
    }
    
    // Fallback to TwelveData API
    try {
      console.log('Fetching data from TwelveData API...');
      const timeSeries = await fetchTimeSeries('USD/COP', '5min', 100);
      
      if (timeSeries && timeSeries.length > 0) {
        // Filter by date range
        const start = new Date(startDate);
        const end = new Date(endDate);
        
        const filteredData = timeSeries.filter((item: any) => {
          const itemDate = new Date(item.datetime);
          return itemDate >= start && itemDate <= end;
        });
        
        return NextResponse.json({
          success: true,
          count: filteredData.length,
          data: filteredData,
          source: 'twelvedata'
        });
      }
      
      // If no data from either source
      return NextResponse.json({
        success: true,
        count: 0,
        data: [],
        message: 'No data available for the specified date range'
      });
      
    } catch (apiError) {
      console.error('TwelveData API error:', apiError);
      return NextResponse.json({ 
        success: false,
        error: 'Failed to load data from both MinIO and TwelveData',
        details: apiError instanceof Error ? apiError.message : 'Unknown error'
      }, { status: 500 });
    }
    
  } catch (error) {
    console.error('Error in L0 route:', error);
    return NextResponse.json({ 
      success: false,
      error: 'Failed to load L0 data',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}