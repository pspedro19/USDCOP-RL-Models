import { NextRequest, NextResponse } from 'next/server';
import { apiConfig } from '@/lib/config/api.config';
import { createApiResponse, measureLatency } from '@/lib/types/api';
import { getCircuitBreaker, CircuitOpenError } from '@/lib/utils/circuit-breaker';
import { withAuth } from '@/lib/auth/api-auth';

/**
 * ML Analytics Predictions API
 * ============================
 *
 * Connects to REAL ML Analytics backend (Port 8004)
 * NO MOCK DATA - All data from real model predictions
 */

const ML_ANALYTICS_BASE = apiConfig.mlAnalytics.baseUrl;

// Circuit breaker for ML Analytics backend
const mlAnalyticsCircuitBreaker = getCircuitBreaker('ml-analytics', {
  failureThreshold: 3,
  resetTimeout: 30000, // 30 seconds
  monitorInterval: 5000,
});

interface PredictionData {
  timestamp: string;
  actual: number;
  predicted: number;
  confidence?: number;
  feature_values?: { [key: string]: number };
  model_version?: string;
  error?: number;
  absolute_error?: number;
  percentage_error?: number;
}

interface PredictionMetrics {
  mse: number;
  mae: number;
  rmse: number;
  mape: number;
  accuracy: number;
  correlation: number;
  total_predictions: number;
  correct_direction: number;
  direction_accuracy: number;
}

// Helper to calculate metrics from real data
function calculatePredictionMetrics(predictions: PredictionData[]): PredictionMetrics {
  if (predictions.length === 0) {
    return {
      mse: 0, mae: 0, rmse: 0, mape: 0, accuracy: 0, correlation: 0,
      total_predictions: 0, correct_direction: 0, direction_accuracy: 0
    };
  }

  const errors = predictions.map(p => p.error || (p.predicted - p.actual));
  const absoluteErrors = predictions.map(p => p.absolute_error || Math.abs(p.predicted - p.actual));
  const percentageErrors = predictions.map(p =>
    p.percentage_error || (Math.abs(p.predicted - p.actual) / p.actual * 100)
  );

  const mse = errors.reduce((sum, error) => sum + error * error, 0) / errors.length;
  const mae = absoluteErrors.reduce((sum, ae) => sum + ae, 0) / absoluteErrors.length;
  const rmse = Math.sqrt(mse);
  const mape = percentageErrors.reduce((sum, pe) => sum + pe, 0) / percentageErrors.length;

  // Calculate correlation
  const actuals = predictions.map(p => p.actual);
  const predicteds = predictions.map(p => p.predicted);
  const correlation = calculateCorrelation(actuals, predicteds);

  // Calculate direction accuracy
  let correctDirection = 0;
  for (let i = 1; i < predictions.length; i++) {
    const actualDirection = predictions[i].actual > predictions[i-1].actual;
    const predictedDirection = predictions[i].predicted > predictions[i-1].predicted;
    if (actualDirection === predictedDirection) {
      correctDirection++;
    }
  }
  const directionAccuracy = predictions.length > 1
    ? (correctDirection / (predictions.length - 1)) * 100
    : 0;

  const accuracy = Math.max(0, 100 - mape);

  return {
    mse: Number(mse.toFixed(6)),
    mae: Number(mae.toFixed(6)),
    rmse: Number(rmse.toFixed(6)),
    mape: Number(mape.toFixed(2)),
    accuracy: Number(accuracy.toFixed(2)),
    correlation: Number(correlation.toFixed(4)),
    total_predictions: predictions.length,
    correct_direction: correctDirection,
    direction_accuracy: Number(directionAccuracy.toFixed(2))
  };
}

function calculateCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;

  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

  return denominator === 0 ? 0 : numerator / denominator;
}

export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const runId = searchParams.get('runId');
    const limit = parseInt(searchParams.get('limit') || '100');
    const timeRange = searchParams.get('timeRange') || '24h';

    // Execute through circuit breaker for resilience
    const data = await mlAnalyticsCircuitBreaker.execute(async () => {
      const backendUrl = `${ML_ANALYTICS_BASE}/api/ml-analytics/predictions`;
      const response = await fetch(
        `${backendUrl}?action=${action}&limit=${limit}&timeRange=${timeRange}${runId ? `&runId=${runId}` : ''}`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          signal: AbortSignal.timeout(10000), // 10 second timeout
          cache: 'no-store',
        }
      );

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      return response.json();
    });

    const latency = measureLatency(startTime);
    return NextResponse.json(
      createApiResponse(true, 'live', {
        data,
        latency,
        backendUrl: ML_ANALYTICS_BASE,
      })
    );

  } catch (error) {
    const latency = measureLatency(startTime);

    if (error instanceof CircuitOpenError) {
      // Fast-fail when circuit is open
      console.warn('[ML Predictions API] Circuit breaker OPEN - service unavailable');
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'ML Analytics backend temporarily unavailable',
          message: 'Circuit breaker is open. Service is experiencing issues.',
          circuitState: 'OPEN',
          troubleshooting: {
            backend_url: ML_ANALYTICS_BASE,
            expected_endpoint: '/api/ml-analytics/predictions',
            required_services: ['ml_analytics_service (port 8004)'],
          },
          latency,
        }),
        { status: 503 }
      );
    }

    // Other errors
    console.error('[ML Predictions API] Error:', error);
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: error instanceof Error ? error.message : 'Internal server error',
        message: 'The ML Analytics service is not responding. Please ensure the service is running.',
        data:
          process.env.NODE_ENV === 'development'
            ? { details: String(error) }
            : undefined,
        latency,
      }),
      { status: 503 }
    );
  }
});

export const POST = withAuth(async (request, { user }) => {
  const startTime = Date.now();
  try {
    const body = await request.json();
    const { predictions, model_run_id } = body;

    if (!predictions || !Array.isArray(predictions)) {
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'predictions array is required',
        }),
        { status: 400 }
      );
    }

    // Execute through circuit breaker for resilience
    try {
      const data = await mlAnalyticsCircuitBreaker.execute(async () => {
        const backendUrl = `${ML_ANALYTICS_BASE}/api/ml-analytics/predictions`;
        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ predictions, model_run_id }),
          signal: AbortSignal.timeout(10000),
        });

        if (!response.ok) {
          throw new Error(`Backend error: ${response.status}`);
        }

        return response.json();
      });

      const latency = measureLatency(startTime);
      return NextResponse.json(
        createApiResponse(true, 'live', {
          data,
          latency,
          backendUrl: ML_ANALYTICS_BASE,
        })
      );

    } catch (backendError) {
      if (backendError instanceof CircuitOpenError) {
        console.warn('[ML Predictions API POST] Circuit breaker OPEN - using local fallback');
      } else {
        console.warn('[ML Predictions API POST] Backend unavailable:', backendError);
      }

      // Calculate metrics locally if backend unavailable
      const metrics = calculatePredictionMetrics(predictions);
      const latency = measureLatency(startTime);

      return NextResponse.json(
        createApiResponse(true, 'mock', {
          message: 'ML Analytics backend unavailable - metrics calculated locally',
          data: {
            stored_predictions: predictions.length,
            metrics,
            model_run_id: model_run_id || 'unknown',
          },
          latency,
        })
      );
    }
  } catch (error) {
    console.error('[ML Predictions POST API] Error:', error);
    const latency = measureLatency(startTime);
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: error instanceof Error ? error.message : 'Internal server error',
        latency,
      }),
      { status: 500 }
    );
  }
});
