import { NextRequest, NextResponse } from 'next/server';
import { apiConfig } from '@/lib/config/api.config';
import { createApiResponse } from '@/lib/types/api';

/**
 * ML Analytics Health API
 * =======================
 *
 * Connects to REAL ML Analytics backend (Port 8004)
 * NO MOCK DATA - All health metrics from real model monitoring
 */

const ML_ANALYTICS_BASE = apiConfig.mlAnalytics.baseUrl;

interface ModelHealthStatus {
  model_id: string;
  model_name: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  last_prediction_time: string;
  health_score: number;
  alerts: ModelAlert[];
  metrics: {
    prediction_latency: number;
    throughput: number;
    error_rate: number;
    drift_score: number;
    confidence_avg: number;
  };
  resource_usage: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
  };
}

interface ModelAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  type: 'drift' | 'performance' | 'resource' | 'availability';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  threshold_value?: number;
  current_value?: number;
}

interface SystemHealthSummary {
  overall_status: 'healthy' | 'warning' | 'critical';
  total_models: number;
  healthy_models: number;
  models_with_warnings: number;
  critical_models: number;
  offline_models: number;
  total_alerts: number;
  critical_alerts: number;
  last_updated: string;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const modelId = searchParams.get('modelId');

    // Build URL for real backend
    const backendUrl = `${ML_ANALYTICS_BASE}/api/ml-analytics/health`;
    const queryParams = new URLSearchParams();
    if (action) queryParams.set('action', action);
    if (modelId) queryParams.set('modelId', modelId);

    try {
      const response = await fetch(
        `${backendUrl}?${queryParams.toString()}`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(10000),
          cache: 'no-store'
        }
      );

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(
          createApiResponse(
            { source: 'ml_analytics_backend', ...data },
            'live'
          )
        );
      }

      console.warn(`[ML Health API] Backend returned ${response.status}`);

    } catch (backendError) {
      console.warn('[ML Health API] Backend unavailable:', backendError);
    }

    // Return service unavailable with troubleshooting info (NO MOCK DATA)
    return NextResponse.json(
      createApiResponse(
        {
          troubleshooting: {
            backend_url: ML_ANALYTICS_BASE,
            expected_endpoint: '/api/ml-analytics/health',
            required_services: ['ml_analytics_service (port 8004)'],
            how_to_start: 'cd services/ml_analytics_service && python main.py'
          }
        },
        'none',
        'ML Analytics backend unavailable',
        { message: 'The ML Analytics service (port 8004) is not responding. Please ensure the service is running.' }
      ),
      { status: 503 }
    );

  } catch (error) {
    console.error('[ML Health API] Error:', error);
    return NextResponse.json(
      createApiResponse(
        process.env.NODE_ENV === 'development' ? { details: error } : null,
        'none',
        error instanceof Error ? error.message : 'Internal server error'
      ),
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, alert_id, model_id } = body;

    // Forward to real backend
    const backendUrl = `${ML_ANALYTICS_BASE}/api/ml-analytics/health`;

    try {
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000)
      });

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(
          createApiResponse(
            { source: 'ml_analytics_backend', ...data },
            'live'
          )
        );
      }
    } catch (backendError) {
      console.warn('[ML Health API POST] Backend unavailable:', backendError);
    }

    // Return error for critical operations (NO MOCK RESPONSES)
    return NextResponse.json(
      createApiResponse(
        { action_attempted: action },
        'none',
        'ML Analytics backend unavailable',
        { message: 'Cannot perform health operations without backend connection.' }
      ),
      { status: 503 }
    );

  } catch (error) {
    console.error('[ML Health POST API] Error:', error);
    return NextResponse.json(
      createApiResponse(
        null,
        'none',
        error instanceof Error ? error.message : 'Internal server error'
      ),
      { status: 500 }
    );
  }
}
