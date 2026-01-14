/**
 * Type-Safe API Client
 * ====================
 *
 * Centralized API client with Zod validation for all endpoints.
 * Single source of truth for frontend API calls.
 *
 * @module lib/api/client
 */

import { z } from 'zod';
import {
  RealtimePriceSchema,
  CandlesticksResponseSchema,
  ModelsResponseSchema,
  MetricsResponseSchema,
  EquityCurveResponseSchema,
  SignalsResponseSchema,
  TradesHistoryResponseSchema,
  ReplayLoadResponseSchema,
  ReplayLoadRequestSchema,
  validateWithErrors,
  type RealtimePrice,
  type CandlesticksResponse,
  type ModelsResponse,
  type MetricsResponse,
  type EquityCurveResponse,
  type SignalsResponse,
  type TradesHistoryResponse,
  type ReplayLoadResponse,
  type ReplayLoadRequest,
} from '../../types/schemas';

// =============================================================================
// CONFIGURATION
// =============================================================================

const API_BASE = '';  // Empty for same-origin requests in Next.js
const DEFAULT_TIMEOUT = 30000;  // 30 seconds

interface FetchOptions extends RequestInit {
  timeout?: number;
  validateResponse?: boolean;
}

// =============================================================================
// BASE FETCH WITH ERROR HANDLING
// =============================================================================

class ApiClientError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string,
    public validationErrors?: z.ZodError
  ) {
    super(message);
    this.name = 'ApiClientError';
  }
}

async function fetchWithTimeout<T>(
  url: string,
  options: FetchOptions = {}
): Promise<T> {
  const { timeout = DEFAULT_TIMEOUT, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiClientError(
        errorData.error || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        url
      );
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof ApiClientError) {
      throw error;
    }

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new ApiClientError(`Request timeout after ${timeout}ms`, undefined, url);
      }
      throw new ApiClientError(error.message, undefined, url);
    }

    throw new ApiClientError('Unknown error occurred', undefined, url);
  }
}

/**
 * Validates response data with Zod schema
 */
function validateData<T>(schema: z.ZodType<T>, data: unknown, endpoint: string): T {
  const result = schema.safeParse(data);

  if (!result.success) {
    console.error(`[API Client] Validation failed for ${endpoint}:`, result.error.format());
    throw new ApiClientError(
      `Response validation failed: ${result.error.issues[0]?.message || 'Unknown validation error'}`,
      undefined,
      endpoint,
      result.error
    );
  }

  return result.data;
}

// =============================================================================
// API CLIENT
// =============================================================================

export const apiClient = {
  // ===========================================================================
  // MARKET DATA
  // ===========================================================================

  /**
   * Get real-time USD/COP price
   */
  async getRealtimePrice(): Promise<RealtimePrice> {
    const endpoint = '/api/market/realtime-price';
    const response = await fetchWithTimeout<{ success: boolean; data: unknown }>(
      `${API_BASE}${endpoint}`
    );

    if (!response.success) {
      throw new ApiClientError('Failed to fetch realtime price', undefined, endpoint);
    }

    return validateData(RealtimePriceSchema, response.data, endpoint);
  },

  /**
   * Get OHLCV candlestick data
   */
  async getCandlesticks(params: {
    startDate?: string;
    endDate?: string;
    limit?: number;
  } = {}): Promise<CandlesticksResponse> {
    const endpoint = '/api/market/candlesticks-filtered';
    const searchParams = new URLSearchParams();

    if (params.startDate) searchParams.set('start_date', params.startDate);
    if (params.endDate) searchParams.set('end_date', params.endDate);
    if (params.limit) searchParams.set('limit', String(params.limit));

    const url = `${API_BASE}${endpoint}${searchParams.toString() ? `?${searchParams}` : ''}`;
    const response = await fetchWithTimeout<{ success: boolean; data: unknown }>(url);

    if (!response.success) {
      throw new ApiClientError('Failed to fetch candlesticks', undefined, endpoint);
    }

    return validateData(CandlesticksResponseSchema, response.data, endpoint);
  },

  // ===========================================================================
  // MODELS
  // ===========================================================================

  /**
   * Get all available models
   */
  async getModels(): Promise<ModelsResponse> {
    const endpoint = '/api/models';
    const response = await fetchWithTimeout<unknown>(`${API_BASE}${endpoint}`);
    return validateData(ModelsResponseSchema, response, endpoint);
  },

  /**
   * Get model performance metrics
   */
  async getModelMetrics(
    modelId: string,
    params: { period?: string; from?: string; to?: string } = {}
  ): Promise<MetricsResponse> {
    const endpoint = `/api/models/${modelId}/metrics`;
    const searchParams = new URLSearchParams();

    if (params.period) searchParams.set('period', params.period);
    if (params.from) searchParams.set('from', params.from);
    if (params.to) searchParams.set('to', params.to);

    const url = `${API_BASE}${endpoint}${searchParams.toString() ? `?${searchParams}` : ''}`;
    const response = await fetchWithTimeout<{ success: boolean; data: unknown }>(url);

    if (!response.success) {
      throw new ApiClientError(`Failed to fetch metrics for model ${modelId}`, undefined, endpoint);
    }

    return validateData(MetricsResponseSchema, response.data, endpoint);
  },

  /**
   * Get model equity curve
   */
  async getModelEquityCurve(
    modelId: string,
    params: { days?: number; from?: string; to?: string } = {}
  ): Promise<EquityCurveResponse> {
    const endpoint = `/api/models/${modelId}/equity-curve`;
    const searchParams = new URLSearchParams();

    if (params.days) searchParams.set('days', String(params.days));
    if (params.from) searchParams.set('from', params.from);
    if (params.to) searchParams.set('to', params.to);

    const url = `${API_BASE}${endpoint}${searchParams.toString() ? `?${searchParams}` : ''}`;
    const response = await fetchWithTimeout<{ success: boolean; data: unknown }>(url);

    if (!response.success) {
      throw new ApiClientError(`Failed to fetch equity curve for model ${modelId}`, undefined, endpoint);
    }

    return validateData(EquityCurveResponseSchema, response.data, endpoint);
  },

  // ===========================================================================
  // TRADING
  // ===========================================================================

  /**
   * Get trading signals
   */
  async getSignals(params: { limit?: number; modelId?: string } = {}): Promise<SignalsResponse> {
    const endpoint = '/api/trading/signals';
    const searchParams = new URLSearchParams();

    if (params.limit) searchParams.set('limit', String(params.limit));
    if (params.modelId) searchParams.set('model_id', params.modelId);

    const url = `${API_BASE}${endpoint}${searchParams.toString() ? `?${searchParams}` : ''}`;
    const response = await fetchWithTimeout<{ success: boolean; signals: unknown[]; metadata?: unknown }>(url);

    if (!response.success) {
      throw new ApiClientError('Failed to fetch signals', undefined, endpoint);
    }

    // Transform response to match schema
    const transformed = {
      signals: response.signals,
      total: Array.isArray(response.signals) ? response.signals.length : 0,
    };

    return validateData(SignalsResponseSchema, transformed, endpoint);
  },

  /**
   * Get trades history
   */
  async getTradesHistory(params: {
    limit?: number;
    modelId?: string;
    from?: string;
    to?: string;
  } = {}): Promise<TradesHistoryResponse> {
    const endpoint = '/api/trading/trades/history';
    const searchParams = new URLSearchParams();

    if (params.limit) searchParams.set('limit', String(params.limit));
    if (params.modelId) searchParams.set('model_id', params.modelId);
    if (params.from) searchParams.set('from', params.from);
    if (params.to) searchParams.set('to', params.to);

    const url = `${API_BASE}${endpoint}${searchParams.toString() ? `?${searchParams}` : ''}`;
    const response = await fetchWithTimeout<{ success: boolean; data: { trades: unknown[]; total: number; source: string } }>(url);

    if (!response.success) {
      throw new ApiClientError('Failed to fetch trades history', undefined, endpoint);
    }

    return validateData(TradesHistoryResponseSchema, response.data, endpoint);
  },

  // ===========================================================================
  // REPLAY
  // ===========================================================================

  /**
   * Load trades for replay mode
   */
  async loadReplayTrades(request: ReplayLoadRequest): Promise<ReplayLoadResponse> {
    const endpoint = '/api/replay/load-trades';

    // Validate request
    const validatedRequest = validateData(ReplayLoadRequestSchema, request, endpoint);

    const response = await fetchWithTimeout<{ success: boolean; data: unknown }>(
      `${API_BASE}${endpoint}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validatedRequest),
        timeout: 300000, // 5 minutes for long backtests
      }
    );

    if (!response.success) {
      throw new ApiClientError('Failed to load replay trades', undefined, endpoint);
    }

    return validateData(ReplayLoadResponseSchema, response.data, endpoint);
  },
};

// =============================================================================
// HOOKS HELPERS
// =============================================================================

/**
 * Type-safe fetch wrapper for use with SWR or React Query
 */
export function createFetcher<T>(schema: z.ZodType<T>) {
  return async (url: string): Promise<T> => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new ApiClientError(`HTTP ${response.status}`, response.status, url);
    }
    const data = await response.json();
    return validateData(schema, data, url);
  };
}

// Export error class for error handling
export { ApiClientError };
