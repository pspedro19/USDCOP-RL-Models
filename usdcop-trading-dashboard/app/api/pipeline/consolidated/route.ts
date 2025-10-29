import { NextRequest, NextResponse } from 'next/server';

/**
 * Consolidated Pipeline Status API - 100% DYNAMIC
 * NO HARDCODED VALUES - All data from real backend APIs
 *
 * Aggregates status from L0-L6 layers by calling individual status endpoints
 */

const INTERNAL_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:5000';

interface LayerStatus {
  layer: string;
  name: string;
  status: 'pass' | 'fail' | 'warning' | 'unknown';
  pass: boolean;
  quality_metrics: Record<string, any>;
  last_update: string;
  data_shape?: Record<string, any>;
}

export async function GET(request: NextRequest) {
  try {
    // Fetch status from all layers in parallel
    const [l0Res, l1Res, l2Res, l3Res, l4Res, l5Res, l6Res] = await Promise.allSettled([
      fetch(`${INTERNAL_BASE}/api/pipeline/l0/status`, { cache: 'no-store' }),
      fetch(`${INTERNAL_BASE}/api/pipeline/l1/status`, { cache: 'no-store' }),
      fetch(`${INTERNAL_BASE}/api/pipeline/l2/status`, { cache: 'no-store' }),
      fetch(`${INTERNAL_BASE}/api/pipeline/l3/status`, { cache: 'no-store' }),
      fetch(`${INTERNAL_BASE}/api/pipeline/l4/status`, { cache: 'no-store' }),
      fetch(`${INTERNAL_BASE}/api/pipeline/l5/status`, { cache: 'no-store' }),
      fetch(`${INTERNAL_BASE}/api/pipeline/l6/status`, { cache: 'no-store' }),
    ]);

    // Parse responses
    const parseResponse = async (result: PromiseSettledResult<Response>): Promise<LayerStatus | null> => {
      if (result.status === 'fulfilled' && result.value.ok) {
        try {
          return await result.value.json();
        } catch {
          return null;
        }
      }
      return null;
    };

    const [l0Data, l1Data, l2Data, l3Data, l4Data, l5Data, l6Data] = await Promise.all([
      parseResponse(l0Res),
      parseResponse(l1Res),
      parseResponse(l2Res),
      parseResponse(l3Res),
      parseResponse(l4Res),
      parseResponse(l5Res),
      parseResponse(l6Res),
    ]);

    // Calculate system health
    const layers = [l0Data, l1Data, l2Data, l3Data, l4Data, l5Data, l6Data];
    const passingLayers = layers.filter(l => l?.pass).length;
    const totalLayers = 7;
    const healthPercentage = (passingLayers / totalLayers) * 100;

    let overallStatus = 'healthy';
    if (healthPercentage < 60) overallStatus = 'critical';
    else if (healthPercentage < 80) overallStatus = 'degraded';

    // Build response with REAL data (no fallbacks to hardcoded values)
    const response = {
      success: true,
      timestamp: new Date().toISOString(),
      system_health: {
        health_percentage: healthPercentage,
        passing_layers: passingLayers,
        total_layers: totalLayers,
        status: overallStatus
      },
      layers: {
        l0: l0Data || {
          layer: 'L0',
          name: 'Raw Data',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L0 status endpoint not available'
        },
        l1: l1Data || {
          layer: 'L1',
          name: 'Standardized',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L1 status endpoint not available'
        },
        l2: l2Data || {
          layer: 'L2',
          name: 'Prepared',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L2 not executed yet. Run: usdcop_m5__03_l2_prepare'
        },
        l3: l3Data || {
          layer: 'L3',
          name: 'Features',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L3 not executed yet. Run: usdcop_m5__04_l3_feature'
        },
        l4: l4Data || {
          layer: 'L4',
          name: 'RL-Ready',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L4 not executed yet. Run: usdcop_m5__05_l4_rlready'
        },
        l5: l5Data || {
          layer: 'L5',
          name: 'Serving',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L5 not executed yet. Run: usdcop_m5__06_l5_serving'
        },
        l6: l6Data || {
          layer: 'L6',
          name: 'Backtest',
          status: 'unknown',
          pass: false,
          quality_metrics: {},
          last_update: new Date().toISOString(),
          error: 'L6 not executed yet. Run: usdcop_m5__07_l6_backtest_referencia'
        }
      }
    };

    return NextResponse.json(response, {
      headers: {
        'Cache-Control': 'no-store, max-age=0',
      },
    });

  } catch (error: unknown) {
    console.error('[Pipeline Consolidated API] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch pipeline status',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}
