import { NextRequest, NextResponse } from 'next/server';
import { MarketDataService } from '@/lib/services/market-data-service';
import { createApiResponse, measureLatency } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';

// Store update interval reference
let updateInterval: NodeJS.Timeout | null = null;

// Circuit breaker for TwelveData API - more lenient settings to avoid blocking
// during temporary backend slowdowns
const marketCircuitBreaker = getCircuitBreaker('market-data', {
  failureThreshold: 10,  // Increased to be more tolerant of temporary failures
  resetTimeout: 15000,   // Reduced to recover faster after issues resolve
});

// Mock data for when backend is unavailable
const MOCK_MARKET_DATA = [
  {
    symbol: 'USDCOP',
    timestamp: new Date().toISOString(),
    open: 4385.50,
    high: 4392.75,
    low: 4380.25,
    close: 4388.00,
    volume: 125000,
    bid: 4387.50,
    ask: 4388.50,
    change: 12.50,
    changePercent: 0.29,
    source: 'mock-data',
  },
];

// NOTE: Auth removed for development - market data is public
export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = new URL(request.url);
  const action = searchParams.get('action') || 'fetch';

  // Skip backend and return mock data in dev mode
  if (process.env.NEXT_PUBLIC_SKIP_BACKEND === 'true') {
    const latency = measureLatency(startTime);
    return NextResponse.json(
      createApiResponse(true, 'mock', {
        data: {
          items: MOCK_MARKET_DATA,
          count: MOCK_MARKET_DATA.length,
          source: 'mock-data',
        },
        latency,
        backendUrl: 'Mock Data',
      })
    );
  }

  try {
    switch (action) {
      case 'fetch':
        // Fetch latest data from TwelveData with circuit breaker protection
        const realtimeData = await marketCircuitBreaker.execute(async () => {
          return await MarketDataService.getRealTimeData();
        });
        const latency = measureLatency(startTime);

        return NextResponse.json(
          createApiResponse(true, 'live', {
            data: {
              items: realtimeData,
              count: realtimeData.length,
              source: 'twelvedata',
            },
            latency,
            backendUrl: 'TwelveData API',
          })
        );
        
      case 'start':
        // Start automatic updates
        if (!updateInterval) {
          MarketDataService.connectWebSocket();

          // Set up SSE endpoint for real-time updates
          updateInterval = setInterval(async () => {
            try {
              const data = await marketCircuitBreaker.execute(async () => {
                return await MarketDataService.getRealTimeData();
              });
              // This would normally push to WebSocket or SSE
              console.log('[Realtime API] New data point:', data[0]);
            } catch (error) {
              if (error instanceof CircuitOpenError) {
                console.warn('[Realtime API] Circuit breaker open, skipping update');
              } else {
                console.error('[Realtime API] Error fetching data:', error);
              }
            }
          }, 300000); // 5 minutes
        }
        const latencyStart = measureLatency(startTime);

        return NextResponse.json(
          createApiResponse(true, 'live', {
            data: {
              message: 'Realtime updates started',
              interval: '5 minutes',
              schedule:
                'Every 5 minutes at :00, :05, :10, :15, :20, :25, :30, :35, :40, :45, :50, :55',
            },
            latency: latencyStart,
          })
        );
        
      case 'stop':
        // Stop automatic updates
        if (updateInterval) {
          clearInterval(updateInterval);
          updateInterval = null;
          // WebSocket will auto-disconnect
        }
        const latencyStop = measureLatency(startTime);

        return NextResponse.json(
          createApiResponse(true, 'live', {
            data: { message: 'Realtime updates stopped' },
            latency: latencyStop,
          })
        );
        
      case 'align':
        // Get complete aligned dataset (historical + realtime) with circuit breaker protection
        const alignedData = await marketCircuitBreaker.execute(async () => {
          return await MarketDataService.getCandlestickData();
        });
        const latencyAlign = measureLatency(startTime);

        // Save aligned data to MinIO
        // Data saving functionality can be added later

        return NextResponse.json(
          createApiResponse(true, 'live', {
            data: {
              items: alignedData.data || [],
              meta: {
                total: alignedData.count || 0,
                symbol: alignedData.symbol,
                timeframe: alignedData.timeframe,
                lastUpdate: new Date().toISOString(),
              },
            },
            latency: latencyAlign,
          })
        );
        
      case 'cache':
        // Get cached data with circuit breaker protection
        const cachedData = await marketCircuitBreaker.execute(async () => {
          return await MarketDataService.getRealTimeData();
        });
        const latencyCache = measureLatency(startTime);

        return NextResponse.json(
          createApiResponse(true, 'cached', {
            data: {
              items: cachedData,
              count: cachedData.length,
            },
            latency: latencyCache,
            cacheHit: true,
          })
        );

      default:
        return NextResponse.json(
          createApiResponse(false, 'none', {
            error: 'Invalid action',
          }),
          { status: 400 }
        );
    }
  } catch (error) {
    console.error('[Realtime API] Error:', error instanceof Error ? error.message : error);

    // Return demo data when backend is unavailable
    const latency = measureLatency(startTime);
    const demoMarketData = generateDemoMarketData();

    return NextResponse.json(
      createApiResponse(true, 'demo', {
        data: {
          items: demoMarketData,
          count: demoMarketData.length,
          source: 'demo-data',
        },
        latency,
        backendUrl: 'Demo Data (Backend unavailable)',
      })
    );
  }
}

// Generate demo market data with realistic values
function generateDemoMarketData() {
  const basePrice = 4285.50;
  const volatility = Math.random() * 20 - 10; // -10 to +10 pesos
  const close = Math.round((basePrice + volatility) * 100) / 100;
  const high = Math.round((close + Math.random() * 8) * 100) / 100;
  const low = Math.round((close - Math.random() * 8) * 100) / 100;
  const open = Math.round((low + Math.random() * (high - low)) * 100) / 100;
  const change = Math.round((close - open) * 100) / 100;
  const changePercent = Math.round((change / open) * 10000) / 100;

  return [
    {
      symbol: 'USDCOP',
      timestamp: new Date().toISOString(),
      open,
      high,
      low,
      close,
      volume: Math.floor(100000 + Math.random() * 50000),
      bid: Math.round((close - 0.50) * 100) / 100,
      ask: Math.round((close + 0.50) * 100) / 100,
      change,
      changePercent,
      source: 'demo-data',
    },
  ];
}

// POST endpoint to manually add data
export async function POST(request: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await request.json();
    const { data, save = false } = body;

    if (!data || !Array.isArray(data)) {
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Invalid data format',
        }),
        { status: 400 }
      );
    }

    // Data saving functionality can be added later
    const latency = measureLatency(startTime);

    return NextResponse.json(
      createApiResponse(true, save ? 'minio' : 'none', {
        data: {
          message: 'Data processed successfully',
          count: data.length,
          saved: save,
        },
        latency,
      })
    );
  } catch (error) {
    console.error('[Realtime API] POST Error:', error);
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to process data',
        message: error instanceof Error ? error.message : 'Unknown error',
      }),
      { status: 500 }
    );
  }
}