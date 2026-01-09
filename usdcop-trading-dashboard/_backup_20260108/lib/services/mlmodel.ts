/**
 * ML Model Service
 *
 * CRITICAL: This service connects to REAL ML backends only.
 * NO mock data, NO random generation, NO fallbacks.
 * If ML service unavailable, return explicit error.
 *
 * All predictions come from live ML analytics API endpoints.
 * Math.random() is PROHIBITED in this file.
 *
 * @example
 * ```typescript
 * try {
 *   const prediction = await getPrediction('USDCOP');
 *   console.log('Live prediction:', prediction);
 * } catch (error) {
 *   // Handle error - NO fallback to fake data
 *   console.error('ML service unavailable:', error);
 * }
 * ```
 */

import { getCircuitBreaker } from '@/lib/utils/circuit-breaker';

const ML_API_URL = process.env.NEXT_PUBLIC_ML_ANALYTICS_API_URL || 'http://localhost:8004';
const mlCircuitBreaker = getCircuitBreaker('ml-analytics', {
  failureThreshold: 3,
  resetTimeout: 60000,  // 1 minute
  monitorInterval: 10000  // 10 seconds
});

export interface MLPrediction {
  symbol: string;
  timestamp: string | number;
  prediction: number;
  confidence: number;
  action: 'buy' | 'sell' | 'hold';
  signal: 'buy' | 'sell' | 'hold';  // Alias for compatibility
  modelVersion: string;
  dataSource: 'live' | 'none';
}

/**
 * Get real-time ML prediction from live backend
 *
 * @param symbol - Trading pair symbol (default: 'USDCOP')
 * @returns Live ML prediction with confidence scores
 * @throws Error if ML service is unavailable - NO FALLBACK PROVIDED
 */
export async function getPrediction(symbol: string = 'USDCOP'): Promise<MLPrediction> {
  try {
    const result = await mlCircuitBreaker.execute(async () => {
      const response = await fetch(`${ML_API_URL}/api/predictions/${symbol}`, {
        signal: AbortSignal.timeout(5000),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`ML API returned ${response.status}: ${response.statusText}`);
      }

      return response.json();
    });

    // Normalize response to include both 'action' and 'signal' fields
    return {
      symbol: result.symbol || symbol,
      timestamp: result.timestamp || Date.now(),
      prediction: result.prediction,
      confidence: result.confidence,
      action: result.action || result.signal,
      signal: result.signal || result.action,
      modelVersion: result.modelVersion || 'unknown',
      dataSource: 'live',
    };
  } catch (error) {
    // NO FALLBACK TO FAKE DATA - Return error state explicitly
    throw new Error(`ML prediction unavailable: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Get multiple ML predictions for a symbol
 *
 * @returns Array of live ML predictions
 * @throws Error if ML service is unavailable - NO FALLBACK PROVIDED
 */
export async function getMLPredictions(symbol: string = 'USDCOP'): Promise<MLPrediction[]> {
  try {
    const result = await mlCircuitBreaker.execute(async () => {
      const response = await fetch(`${ML_API_URL}/api/predictions`, {
        signal: AbortSignal.timeout(5000),
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`ML API returned ${response.status}: ${response.statusText}`);
      }

      return response.json();
    });

    // Filter by symbol if multiple predictions returned
    const predictions = Array.isArray(result) ? result : [result];
    return predictions
      .filter((pred: any) => pred.symbol === symbol)
      .map((pred: any) => ({
        symbol: pred.symbol || symbol,
        timestamp: pred.timestamp || Date.now(),
        prediction: pred.prediction,
        confidence: pred.confidence,
        action: pred.action || pred.signal,
        signal: pred.signal || pred.action,
        modelVersion: pred.modelVersion || 'unknown',
        dataSource: 'live' as const,
      }));
  } catch (error) {
    // NO FALLBACK TO FAKE DATA - Return error state explicitly
    throw new Error(`ML predictions unavailable: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Get ML model health status
 *
 * @returns Health check response from ML service
 * @throws Error if ML service is unavailable
 */
export async function getMLServiceHealth(): Promise<{
  status: 'healthy' | 'unhealthy';
  modelVersion: string;
  lastUpdate: string;
}> {
  try {
    const response = await fetch(`${ML_API_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    throw new Error(`ML service health check failed: ${error instanceof Error ? error.message : String(error)}`);
  }
}
