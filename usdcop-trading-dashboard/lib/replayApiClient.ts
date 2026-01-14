

import { z } from 'zod';
import {
  ReplayTrade,
  EquityPoint,
  Candlestick,
  ReplayData,
  TradesResponseSchema,
  EquityCurveResponseSchema,
  CandlestickSchema,
  Result,
  MODEL_CONFIG,
} from '@/types/replay';
import { ReplayError, ReplayErrorCode, toReplayError } from '@/utils/replayErrors';

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

export const API_CONFIG = {
  BASE_URL: '/api',
  TIMEOUT_MS: 30000,
  INFERENCE_TIMEOUT_MS: 300000, // 5 minutes for inference/backtest
  MAX_RETRIES: 3,
  RETRY_BASE_DELAY_MS: 1000,
  RETRY_MAX_DELAY_MS: 10000,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

interface FetchOptions {
  timeout?: number;
  retries?: number;
  signal?: AbortSignal;
}

interface DateRangeParams {
  from: string;
  to: string;
  dataType?: 'validation' | 'test' | 'both';
}

// ═══════════════════════════════════════════════════════════════════════════
// RETRY LOGIC
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Calculate delay for exponential backoff
 */
function calculateBackoff(attempt: number): number {
  const delay = API_CONFIG.RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
  const jitter = Math.random() * 1000;
  return Math.min(delay + jitter, API_CONFIG.RETRY_MAX_DELAY_MS);
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Determine if error is retryable
 */
function isRetryableError(error: unknown): boolean {
  if (error instanceof ReplayError) {
    return [
      ReplayErrorCode.API_TIMEOUT,
      ReplayErrorCode.API_ERROR,
      ReplayErrorCode.DATA_LOAD_FAILED,
    ].includes(error.code);
  }

  if (error instanceof Error) {
    return (
      error.name === 'AbortError' ||
      error.message.includes('timeout') ||
      error.message.includes('network') ||
      error.message.includes('fetch')
    );
  }

  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// FETCH WITH TIMEOUT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Fetch with timeout support
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number }
): Promise<Response> {
  const { timeout = API_CONFIG.TIMEOUT_MS, ...fetchOptions } = options;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  // Merge signals if one was provided
  if (options.signal) {
    options.signal.addEventListener('abort', () => controller.abort());
  }

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ReplayError(
        ReplayErrorCode.API_TIMEOUT,
        'Request timed out',
        true
      );
    }
    throw error;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// FETCH WITH RETRY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Fetch with automatic retry and exponential backoff
 */
async function fetchWithRetry<T>(
  url: string,
  schema: z.ZodType<T>,
  options: FetchOptions = {}
): Promise<Result<T, ReplayError>> {
  const { timeout = API_CONFIG.TIMEOUT_MS, retries = API_CONFIG.MAX_RETRIES, signal } = options;

  let lastError: ReplayError | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    // Check if aborted
    if (signal?.aborted) {
      return Result.err(new ReplayError(
        ReplayErrorCode.API_TIMEOUT,
        'Request was aborted',
        false
      ));
    }

    try {
      const response = await fetchWithTimeout(url, { timeout, signal });

      if (!response.ok) {
        throw new ReplayError(
          ReplayErrorCode.API_ERROR,
          `API error: ${response.status} ${response.statusText}`,
          true,
          { status: response.status, url }
        );
      }

      const json = await response.json();

      // Validate response with Zod
      const parseResult = schema.safeParse(json);
      if (!parseResult.success) {
        throw new ReplayError(
          ReplayErrorCode.DATA_VALIDATION_FAILED,
          `Invalid response format: ${parseResult.error.message}`,
          false,
          { zodError: parseResult.error.format() }
        );
      }

      return Result.ok(parseResult.data);
    } catch (error) {
      lastError = toReplayError(error);

      // Don't retry non-retryable errors
      if (!isRetryableError(error) || attempt === retries) {
        break;
      }

      // Wait before retry
      const delay = calculateBackoff(attempt);
      console.log(`[ReplayAPI] Retry ${attempt + 1}/${retries} after ${delay}ms`);
      await sleep(delay);
    }
  }

  return Result.err(lastError || new ReplayError(
    ReplayErrorCode.UNKNOWN_ERROR,
    'Unknown error occurred',
    true
  ));
}

// ═══════════════════════════════════════════════════════════════════════════
// API ENDPOINTS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build URL with query parameters
 */
function buildUrl(endpoint: string, params: Record<string, string | undefined>): string {
  const url = new URL(`${API_CONFIG.BASE_URL}${endpoint}`, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) {
      url.searchParams.set(key, value);
    }
  });
  return url.toString();
}

// Flexible trades response schema that matches the actual API
const FlexibleTradesSchema = z.object({
  success: z.literal(true),
  data: z.object({
    trades: z.array(z.object({
      trade_id: z.number(),
      timestamp: z.string().optional(),
      entry_time: z.string().optional(),
      side: z.string().optional(),
      entry_price: z.number(),
      exit_price: z.number().nullable().optional(),
      pnl: z.number().optional(),
      pnl_usd: z.number().optional(),
      pnl_percent: z.number().optional(),
      pnl_pct: z.number().optional(),
      status: z.string().optional(),
      duration_minutes: z.number().nullable().optional(),
      exit_reason: z.string().nullable().optional(),
      // Additional optional fields
      strategy_code: z.string().optional(),
      strategy_name: z.string().optional(),
      size: z.number().optional(),
      commission: z.number().optional(),
      equity_at_entry: z.number().optional(),
      equity_at_exit: z.number().nullable().optional(),
      entry_confidence: z.number().nullable().optional(),
      exit_confidence: z.number().nullable().optional(),
      model_metadata: z.any().nullable().optional(),
      features_snapshot: z.any().nullable().optional(),
      market_regime: z.string().optional(),
      max_adverse_excursion: z.number().nullable().optional(),
      max_favorable_excursion: z.number().nullable().optional(),
    })),
    total: z.number().optional(),
    timestamp: z.string().optional(),
    source: z.string().optional(),
  }),
});

// Schema for inference-based trades loading response
const InferenceTradesSchema = z.object({
  success: z.literal(true),
  data: z.object({
    trades: z.array(z.object({
      trade_id: z.number(),
      timestamp: z.string(),
      strategy_code: z.string().optional(),
      strategy_name: z.string().optional(),
      side: z.string(),
      entry_price: z.number(),
      exit_price: z.number().nullable().optional(),
      size: z.number().optional(),
      pnl: z.number(),
      pnl_percent: z.number(),
      status: z.string(),
      duration_minutes: z.number().nullable().optional(),
      exit_reason: z.string().nullable().optional(),
      commission: z.number().optional(),
      equity_at_entry: z.number().nullable().optional(),
      equity_at_exit: z.number().nullable().optional(),
      entry_confidence: z.number().nullable().optional(),
      exit_confidence: z.number().nullable().optional(),
      model_metadata: z.any().nullable().optional(),
      features_snapshot: z.any().nullable().optional(),
      market_regime: z.string().optional(),
      max_adverse_excursion: z.number().nullable().optional(),
      max_favorable_excursion: z.number().nullable().optional(),
    })),
    total: z.number(),
    summary: z.object({
      total_trades: z.number(),
      winning_trades: z.number(),
      losing_trades: z.number(),
      win_rate: z.number(),
      total_pnl: z.number(),
      total_return_pct: z.number(),
      max_drawdown_pct: z.number(),
      avg_trade_duration_minutes: z.number().optional(),
    }).nullable().optional(),
    source: z.string(),
    dateRange: z.object({
      start: z.string(),
      end: z.string(),
    }).optional(),
    processingTimeMs: z.number().optional(),
    timestamp: z.string().optional(),
  }),
});

/**
 * Fetch trades using the inference service (generates trades if not cached)
 * This is the main function for replay mode - it will:
 * 1. Check if trades exist in database
 * 2. If not, run PPO model inference to generate them
 * 3. Persist the generated trades
 * 4. Return the trades
 */
interface InferenceOptions extends FetchOptions {
  modelId?: string;
  forceRegenerate?: boolean;
}

export async function fetchTradesWithInference(
  params: DateRangeParams,
  options?: InferenceOptions
): Promise<Result<{ trades: ReplayTrade[]; summary: any; source: string }, ReplayError>> {
  const url = `${API_CONFIG.BASE_URL}/replay/load-trades`;

  // Use longer timeout for inference operations
  const inferenceTimeout = options?.timeout || API_CONFIG.INFERENCE_TIMEOUT_MS;
  const modelId = options?.modelId || process.env.CURRENT_MODEL_ID || 'ppo_v20';
  const forceRegenerate = options?.forceRegenerate ?? false;

  console.log(`[ReplayAPI] Fetching trades: ${params.from} to ${params.to} (model=${modelId}, force=${forceRegenerate})`);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), inferenceTimeout);

    // Merge signals if one was provided
    if (options?.signal) {
      options.signal.addEventListener('abort', () => controller.abort());
    }

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        startDate: params.from,
        endDate: params.to,
        modelId,
        forceRegenerate,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `API error: ${response.status}`;

      try {
        const errorJson = JSON.parse(errorText);
        if (errorJson.error) {
          errorMessage = errorJson.error;
        }
      } catch {
        // Use status text
      }

      // Special handling for service unavailable
      if (response.status === 503) {
        throw new ReplayError(
          ReplayErrorCode.API_ERROR,
          'Inference service unavailable. Start it with: uvicorn services.inference_api.main:app --port 8000',
          false,
          { status: response.status }
        );
      }

      throw new ReplayError(
        ReplayErrorCode.API_ERROR,
        errorMessage,
        true,
        { status: response.status }
      );
    }

    const json = await response.json();

    // Validate response
    const parseResult = InferenceTradesSchema.safeParse(json);
    if (!parseResult.success) {
      throw new ReplayError(
        ReplayErrorCode.DATA_VALIDATION_FAILED,
        `Invalid inference response: ${parseResult.error.message}`,
        false,
        { zodError: parseResult.error.format() }
      );
    }

    // Convert to ReplayTrade format
    const trades: ReplayTrade[] = parseResult.data.data.trades.map((t) => ({
      trade_id: t.trade_id,
      timestamp: t.timestamp,
      entry_time: t.timestamp,
      side: t.side,
      entry_price: t.entry_price,
      exit_price: t.exit_price || null,
      pnl: t.pnl,
      pnl_usd: t.pnl,
      pnl_percent: t.pnl_percent,
      pnl_pct: t.pnl_percent,
      status: t.status,
      duration_minutes: t.duration_minutes,
      exit_reason: t.exit_reason,
      strategy_code: t.strategy_code || 'RL_PPO',
      strategy_name: t.strategy_name || 'PPO RL Agent',
      size: t.size || 1,
      commission: t.commission || 0,
      // Include equity fields for proper equity curve generation
      equity_at_entry: t.equity_at_entry ?? undefined,
      equity_at_exit: t.equity_at_exit ?? undefined,
    }));

    return Result.ok({
      trades,
      summary: parseResult.data.data.summary,
      source: parseResult.data.data.source,
    });

  } catch (error) {
    if (error instanceof ReplayError) {
      return Result.err(error);
    }

    if (error instanceof Error && error.name === 'AbortError') {
      return Result.err(new ReplayError(
        ReplayErrorCode.API_TIMEOUT,
        'Inference request timed out - backtest may be taking too long',
        true
      ));
    }

    return Result.err(toReplayError(error));
  }
}

/**
 * Fetch trades for a date range (legacy - uses direct DB query)
 */
export async function fetchTrades(
  params: DateRangeParams,
  options?: FetchOptions
): Promise<Result<ReplayTrade[], ReplayError>> {
  // Use the correct API endpoint: /api/trading/trades/history
  const url = buildUrl('/trading/trades/history', {
    limit: '500',
    model_id: process.env.CURRENT_MODEL_ID || 'ppo_v20', // Default model
  });

  const result = await fetchWithRetry(url, FlexibleTradesSchema, options);

  if (result.success) {
    // Convert to ReplayTrade format
    const trades: ReplayTrade[] = result.data.data.trades.map((t: any) => ({
      trade_id: t.trade_id,
      timestamp: t.timestamp || t.entry_time,
      entry_time: t.entry_time || t.timestamp,
      side: t.side || 'buy',
      entry_price: t.entry_price,
      exit_price: t.exit_price,
      pnl: t.pnl ?? t.pnl_usd ?? 0,
      pnl_usd: t.pnl ?? t.pnl_usd ?? 0,
      pnl_percent: t.pnl_percent ?? t.pnl_pct ?? 0,
      pnl_pct: t.pnl_percent ?? t.pnl_pct ?? 0,
      status: t.status || 'closed',
      duration_minutes: t.duration_minutes,
      exit_reason: t.exit_reason,
      strategy_code: t.strategy_code,
      strategy_name: t.strategy_name,
      size: t.size || 1,
      commission: t.commission || 0,
    }));

    // Filter trades by date range client-side
    const fromDate = new Date(params.from);
    const toDate = new Date(params.to);
    toDate.setHours(23, 59, 59, 999); // Include full end date

    const filteredTrades = trades.filter((trade: ReplayTrade) => {
      const tradeDate = new Date(trade.timestamp || trade.entry_time || '');
      return tradeDate >= fromDate && tradeDate <= toDate;
    });

    return Result.ok(filteredTrades);
  }

  return result as Result<ReplayTrade[], ReplayError>;
}

/**
 * Fetch equity curve for a date range
 */
export async function fetchEquityCurve(
  params: DateRangeParams,
  options?: FetchOptions & { modelId?: string }
): Promise<Result<EquityPoint[], ReplayError>> {
  // P0-6 FIX: Use dynamic modelId from options, then env, then default
  const modelId = options?.modelId || process.env.CURRENT_MODEL_ID || 'ppo_v20';
  // Use the correct API endpoint: /api/models/{modelId}/equity-curve
  const url = buildUrl(`/models/${modelId}/equity-curve`, {
    days: '90', // Get last 90 days of data
  });

  // The equity curve API returns a different format, create a flexible schema
  const FlexibleEquitySchema = z.object({
    success: z.literal(true),
    data: z.object({
      points: z.array(z.object({
        timestamp: z.string(),
        value: z.number(),
        drawdown_pct: z.number().optional(),
        position: z.string().optional(),
      })).optional(),
      summary: z.any().optional(),
    }),
  });

  const result = await fetchWithRetry(url, FlexibleEquitySchema, options);

  if (result.success) {
    // Convert API response to EquityPoint format
    const points = result.data.data.points || [];
    const equityPoints: EquityPoint[] = points.map((p: any) => ({
      timestamp: p.timestamp,
      equity: p.value,
      drawdown: p.drawdown_pct || 0,
      position: p.position,
    }));

    // Filter by date range
    const fromDate = new Date(params.from);
    const toDate = new Date(params.to);
    toDate.setHours(23, 59, 59, 999);

    const filteredPoints = equityPoints.filter((point) => {
      const pointDate = new Date(point.timestamp);
      return pointDate >= fromDate && pointDate <= toDate;
    });

    return Result.ok(filteredPoints);
  }

  return result as Result<EquityPoint[], ReplayError>;
}

// Candlesticks response schema - flexible to handle actual API response
const FlexibleCandlesticksSchema = z.object({
  success: z.literal(true),
  data: z.array(z.object({
    time: z.string(),
    open: z.number(),
    high: z.number(),
    low: z.number(),
    close: z.number(),
    volume: z.number().optional(),
  })),
  count: z.number().optional(),
});

/**
 * Fetch candlesticks for a date range
 */
export async function fetchCandlesticks(
  params: DateRangeParams,
  options?: FetchOptions
): Promise<Result<Candlestick[], ReplayError>> {
  // Use the correct API endpoint: /api/market/candlesticks-filtered
  const url = buildUrl('/market/candlesticks-filtered', {
    start_date: params.from,
    end_date: params.to,
    limit: '50000', // Request a large number to cover the date range
  });

  const result = await fetchWithRetry(url, FlexibleCandlesticksSchema, options);

  if (result.success) {
    // Convert API response to Candlestick format
    const candlesticks: Candlestick[] = result.data.data.map((c: any) => ({
      time: c.time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
      volume: c.volume || 0,
    }));

    return Result.ok(candlesticks);
  }

  return result as Result<Candlestick[], ReplayError>;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMBINED DATA LOADER
// ═══════════════════════════════════════════════════════════════════════════

interface LoadReplayDataOptions extends FetchOptions {
  modelId?: string;
  forceRegenerate?: boolean;
}

/**
 * Load all replay data for a date range
 * Uses the inference service to generate trades if they don't exist
 */
export async function loadReplayData(
  startDate: Date,
  endDate: Date,
  options?: LoadReplayDataOptions
): Promise<Result<ReplayData, ReplayError>> {
  const startTime = performance.now();

  const params: DateRangeParams = {
    from: startDate.toISOString().split('T')[0],
    to: endDate.toISOString().split('T')[0],
  };

  // Determine data type based on date range
  const testStart = new Date(MODEL_CONFIG.DATES.TEST_START);
  const validationEnd = new Date(MODEL_CONFIG.DATES.VALIDATION_END);

  if (startDate >= testStart) {
    params.dataType = 'test';
  } else if (endDate <= validationEnd) {
    params.dataType = 'validation';
  } else {
    params.dataType = 'both';
  }

  console.log(`[ReplayAPI] Loading data for ${params.from} to ${params.to} (${params.dataType})`);

  // Fetch trades using inference service (generates if not cached)
  // and fetch equity/candlestick data in parallel
  const inferenceOptions: InferenceOptions = {
    ...options,
    modelId: options?.modelId || process.env.CURRENT_MODEL_ID || 'ppo_v20',
    forceRegenerate: options?.forceRegenerate ?? false,
  };

  console.log(`[ReplayAPI] Loading data for ${params.from} to ${params.to} (model=${inferenceOptions.modelId}, force=${inferenceOptions.forceRegenerate})`);

  const [tradesResult, equityResult, candlesResult] = await Promise.all([
    fetchTradesWithInference(params, inferenceOptions),
    fetchEquityCurve(params, options),
    fetchCandlesticks(params, options),
  ]);

  // Check for trade loading errors
  if (!tradesResult.success) {
    console.error('[ReplayAPI] Trade loading failed:', tradesResult.error);
    return Result.err(tradesResult.error);
  }

  const { trades, summary, source } = tradesResult.data;
  console.log(`[ReplayAPI] Loaded ${trades.length} trades from ${source}`);

  // Equity curve is optional - if it fails, generate from trades
  let equityCurve: EquityPoint[] = [];
  if (equityResult.success && equityResult.data.length > 0) {
    equityCurve = equityResult.data;
  } else {
    console.warn('[ReplayAPI] Equity curve failed or empty, generating from trades');
    // Generate equity curve from trades using equity_at_entry and equity_at_exit
    // This creates TWO points per trade: entry and exit
    const initialEquity = 10000;
    let peakEquity = initialEquity;

    // Add initial point
    if (trades.length > 0) {
      const firstTradeTime = trades[0].timestamp || trades[0].entry_time || '';
      equityCurve.push({
        timestamp: firstTradeTime,
        equity: initialEquity,
        drawdown: 0,
        position: 'flat',
      });
    }

    // Add entry and exit points for each trade
    for (const trade of trades) {
      const entryEquity = trade.equity_at_entry ?? initialEquity;
      const exitEquity = trade.equity_at_exit ?? (entryEquity + (trade.pnl || 0));
      const entryTime = trade.timestamp || trade.entry_time || '';

      // Calculate exit time (entry + duration)
      const exitTime = trade.duration_minutes
        ? new Date(new Date(entryTime).getTime() + trade.duration_minutes * 60000).toISOString()
        : entryTime;

      // Update peak for drawdown calculation
      peakEquity = Math.max(peakEquity, entryEquity, exitEquity);

      // Entry point
      equityCurve.push({
        timestamp: entryTime,
        equity: entryEquity,
        drawdown: ((peakEquity - entryEquity) / peakEquity) * 100,
        position: trade.side,
      });

      // Exit point (only if we have duration)
      if (exitTime !== entryTime) {
        equityCurve.push({
          timestamp: exitTime,
          equity: exitEquity,
          drawdown: ((peakEquity - exitEquity) / peakEquity) * 100,
          position: 'flat',
        });
      }
    }

    // Sort by timestamp
    equityCurve.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }

  const candlesticks = candlesResult.success ? candlesResult.data : [];

  // Validate data limits
  if (trades.length > MODEL_CONFIG.LIMITS.MAX_TRADES_PER_LOAD) {
    return Result.err(new ReplayError(
      ReplayErrorCode.TOO_MANY_DATA_POINTS,
      `Too many trades: ${trades.length} exceeds limit of ${MODEL_CONFIG.LIMITS.MAX_TRADES_PER_LOAD}`,
      true,
      { tradeCount: trades.length }
    ));
  }

  if (candlesticks.length > MODEL_CONFIG.LIMITS.MAX_CANDLES_PER_LOAD) {
    return Result.err(new ReplayError(
      ReplayErrorCode.TOO_MANY_DATA_POINTS,
      `Too many candlesticks: ${candlesticks.length} exceeds limit of ${MODEL_CONFIG.LIMITS.MAX_CANDLES_PER_LOAD}`,
      true,
      { candlestickCount: candlesticks.length }
    ));
  }

  // If no trades found, the inference service couldn't generate them
  if (trades.length === 0) {
    return Result.err(new ReplayError(
      ReplayErrorCode.NO_TRADES_IN_RANGE,
      'No trades could be generated for the selected date range. Check if OHLCV data exists.',
      true,
      { from: params.from, to: params.to }
    ));
  }

  const loadTime = performance.now() - startTime;

  return Result.ok({
    trades,
    equityCurve,
    candlesticks: candlesticks.length > 0 ? candlesticks : undefined,
    summary,
    meta: {
      loadTime,
      tradeCount: trades.length,
      equityPointCount: equityCurve.length,
      candlestickCount: candlesticks.length,
      dateRange: { start: params.from, end: params.to },
      modelId: MODEL_CONFIG.VERSION,
      source,
    },
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// ABORT CONTROLLER FACTORY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create an abort controller with optional timeout
 */
export function createAbortController(timeoutMs?: number): {
  controller: AbortController;
  cancel: () => void;
} {
  const controller = new AbortController();
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  if (timeoutMs) {
    timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  }

  const cancel = () => {
    if (timeoutId) clearTimeout(timeoutId);
    controller.abort();
  };

  return { controller, cancel };
}

// ═══════════════════════════════════════════════════════════════════════════
// PRELOAD HINT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Preload data for a date range (returns promise, doesn't wait)
 * Useful for prefetching data the user might need soon
 */
export function preloadReplayData(
  startDate: Date,
  endDate: Date
): { promise: Promise<Result<ReplayData, ReplayError>>; cancel: () => void } {
  const { controller, cancel } = createAbortController();

  const promise = loadReplayData(startDate, endDate, { signal: controller.signal });

  return { promise, cancel };
}
