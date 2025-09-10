import { NextResponse } from 'next/server';
import { marketDataService } from '@/lib/services/market-data-service';

// Store update interval reference
let updateInterval: NodeJS.Timeout | null = null;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get('action') || 'fetch';
  
  try {
    switch (action) {
      case 'fetch':
        // Fetch latest data from TwelveData
        const realtimeData = await marketDataService.fetchRealtimeData(20);
        
        return NextResponse.json({
          success: true,
          data: realtimeData,
          timestamp: new Date().toISOString(),
          source: 'twelvedata',
          count: realtimeData.length
        });
        
      case 'start':
        // Start automatic updates
        if (!updateInterval) {
          marketDataService.startRealtimeUpdates();
          
          // Set up SSE endpoint for real-time updates
          updateInterval = setInterval(async () => {
            const data = await marketDataService.fetchRealtimeData(1);
            // This would normally push to WebSocket or SSE
            console.log('[Realtime API] New data point:', data[0]);
          }, 300000); // 5 minutes
        }
        
        return NextResponse.json({
          success: true,
          message: 'Realtime updates started',
          interval: '5 minutes',
          schedule: 'Every 5 minutes at :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55'
        });
        
      case 'stop':
        // Stop automatic updates
        if (updateInterval) {
          clearInterval(updateInterval);
          updateInterval = null;
          marketDataService.stopRealtimeUpdates();
        }
        
        return NextResponse.json({
          success: true,
          message: 'Realtime updates stopped'
        });
        
      case 'align':
        // Get complete aligned dataset (historical + realtime)
        const alignedData = await marketDataService.getAlignedDataset();
        
        // Save aligned data to MinIO
        if (alignedData.length > 0) {
          await marketDataService.saveToMinIO(alignedData, 'aligned-usdcop-data');
        }
        
        return NextResponse.json({
          success: true,
          data: alignedData,
          meta: {
            total: alignedData.length,
            historical: alignedData.filter(d => d.source === 'minio').length,
            realtime: alignedData.filter(d => d.source === 'twelvedata').length,
            cache: alignedData.filter(d => d.source === 'cache').length,
            startDate: alignedData[0]?.datetime,
            endDate: alignedData[alignedData.length - 1]?.datetime,
            lastUpdate: new Date().toISOString()
          }
        });
        
      case 'cache':
        // Get cached data
        const cachedData = marketDataService.getCachedData();
        
        return NextResponse.json({
          success: true,
          data: cachedData,
          count: cachedData.length,
          source: 'cache'
        });
        
      default:
        return NextResponse.json(
          { success: false, error: 'Invalid action' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('[Realtime API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to process request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

// POST endpoint to manually add data
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { data, save = false } = body;
    
    if (!data || !Array.isArray(data)) {
      return NextResponse.json(
        { success: false, error: 'Invalid data format' },
        { status: 400 }
      );
    }
    
    // Save to MinIO if requested
    if (save) {
      const saved = await marketDataService.saveToMinIO(data);
      if (!saved) {
        throw new Error('Failed to save data to MinIO');
      }
    }
    
    return NextResponse.json({
      success: true,
      message: 'Data processed successfully',
      count: data.length,
      saved: save
    });
    
  } catch (error) {
    console.error('[Realtime API] POST Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to process data',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}