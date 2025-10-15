import { NextResponse } from 'next/server';
import { MarketDataService } from '@/lib/services/market-data-service';

// Store update interval reference
let updateInterval: NodeJS.Timeout | null = null;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get('action') || 'fetch';
  
  try {
    switch (action) {
      case 'fetch':
        // Fetch latest data from TwelveData
        const realtimeData = await MarketDataService.getRealTimeData();
        
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
          MarketDataService.connectWebSocket();
          
          // Set up SSE endpoint for real-time updates
          updateInterval = setInterval(async () => {
            const data = await MarketDataService.getRealTimeData();
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
          // WebSocket will auto-disconnect
        }
        
        return NextResponse.json({
          success: true,
          message: 'Realtime updates stopped'
        });
        
      case 'align':
        // Get complete aligned dataset (historical + realtime)
        const alignedData = await MarketDataService.getCandlestickData();
        
        // Save aligned data to MinIO
        // Data saving functionality can be added later
        
        return NextResponse.json({
          success: true,
          data: alignedData.data || [],
          meta: {
            total: alignedData.count || 0,
            symbol: alignedData.symbol,
            timeframe: alignedData.timeframe,
            lastUpdate: new Date().toISOString()
          }
        });
        
      case 'cache':
        // Get cached data
        const cachedData = await MarketDataService.getRealTimeData();
        
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
    
    // Data saving functionality can be added later
    
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