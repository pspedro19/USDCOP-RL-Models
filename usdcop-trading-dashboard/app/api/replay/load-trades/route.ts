import { NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { withAuth } from '@/lib/auth/api-auth';
import { generateSyntheticTrades, calculateBacktestSummary } from '@/lib/services/synthetic-backtest.service';

/**
 * Replay Load Trades Endpoint
 * ===========================
 * Calls the Python Inference Service to load or generate trades for replay.
 *
 * If trades exist in the database → Returns cached trades
 * If trades don't exist → Runs PPO model inference and persists results
 * If inference service unavailable → Falls back to synthetic trades
 *
 * This endpoint acts as a bridge between the Next.js frontend and the
 * FastAPI inference service running on port 8000.
 */

// Docker: http://usdcop-backtest-api:8000, Local: http://localhost:8003
const INFERENCE_SERVICE_URL = process.env.BACKTEST_SERVICE_URL ||
  process.env.INFERENCE_SERVICE_URL || 'http://localhost:8003';
const INFERENCE_TIMEOUT_MS = 300000; // 5 minutes for long backtests

interface BacktestRequest {
  start_date: string;
  end_date: string;
  model_id?: string;
  force_regenerate?: boolean;
}

interface InferenceServiceTrade {
  trade_id: number;
  model_id: string;
  timestamp: string;
  entry_time: string;
  exit_time: string | null;
  side: string;
  entry_price: number;
  exit_price: number | null;
  pnl: number | null;
  pnl_usd: number | null;
  pnl_percent: number | null;
  pnl_pct: number | null;
  status: string;
  duration_minutes: number | null;
  exit_reason: string | null;
  equity_at_entry: number | null;
  equity_at_exit: number | null;
  entry_confidence: number | null;
  exit_confidence: number | null;
}

interface InferenceServiceResponse {
  success: boolean;
  source: 'database' | 'generated' | 'error';
  trade_count: number;
  trades: InferenceServiceTrade[];
  summary: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    total_return_pct: number;
    max_drawdown_pct: number;
    avg_trade_duration_minutes?: number;
  } | null;
  processing_time_ms: number;
  date_range?: {
    start: string;
    end: string;
  };
}

/**
 * Transform inference service trade to frontend format
 */
function transformTrade(trade: InferenceServiceTrade) {
  return {
    trade_id: trade.trade_id,
    timestamp: trade.entry_time || trade.timestamp,
    strategy_code: 'RL_PPO',
    strategy_name: 'PPO RL Agent',
    side: (trade.side || 'buy').toLowerCase(),
    entry_price: trade.entry_price,
    exit_price: trade.exit_price,
    size: 1.0,
    pnl: trade.pnl_usd ?? trade.pnl ?? 0,
    pnl_percent: trade.pnl_pct ?? trade.pnl_percent ?? 0,
    status: trade.status || (trade.exit_time ? 'closed' : 'open'),
    duration_minutes: trade.duration_minutes,
    commission: 0,
    exit_reason: trade.exit_reason,
    equity_at_entry: trade.equity_at_entry,
    equity_at_exit: trade.equity_at_exit,
    entry_confidence: trade.entry_confidence,
    exit_confidence: trade.exit_confidence,
    model_metadata: { model_id: trade.model_id },
    features_snapshot: null,
    market_regime: 'unknown',
    max_adverse_excursion: null,
    max_favorable_excursion: null,
  };
}

/**
 * Generate synthetic fallback response when inference service is unavailable.
 * Uses the same synthetic-backtest.service with engineered investor demo metrics.
 */
function buildSyntheticFallback(startDate: string, endDate: string, modelId: string, startTime: number) {
  console.log(`[Replay API] Inference unavailable, generating synthetic trades for ${startDate} to ${endDate}`);

  const trades = generateSyntheticTrades({
    startDate,
    endDate,
    modelId,
  });
  const summary = calculateBacktestSummary(trades);

  // Transform to frontend format (same shape as inference service response)
  const frontendTrades = trades.map((t) => transformTrade({
    trade_id: t.trade_id,
    model_id: t.model_id,
    timestamp: t.timestamp,
    entry_time: t.entry_time,
    exit_time: t.exit_time ?? null,
    side: t.side,
    entry_price: t.entry_price,
    exit_price: t.exit_price ?? null,
    pnl: t.pnl,
    pnl_usd: t.pnl_usd,
    pnl_percent: t.pnl_percent,
    pnl_pct: t.pnl_pct,
    status: t.status,
    duration_minutes: t.duration_minutes,
    exit_reason: t.exit_reason ?? null,
    equity_at_entry: t.equity_at_entry ?? null,
    equity_at_exit: t.equity_at_exit ?? null,
    entry_confidence: t.entry_confidence ?? null,
    exit_confidence: t.exit_confidence ?? null,
  }));

  frontendTrades.sort((a, b) =>
    new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  const data = {
    trades: frontendTrades,
    total: trades.length,
    summary,
    source: 'generated' as const,
    dateRange: { start: startDate, end: endDate },
    processingTimeMs: Date.now() - startTime,
    timestamp: new Date().toISOString(),
  };

  const response = createApiResponse(data, 'fallback');
  response.metadata.latency = Date.now() - startTime;
  response.metadata.isRealData = false;

  return NextResponse.json(response, {
    headers: { 'Cache-Control': 'no-store, max-age=0' },
  });
}

export const POST = withAuth(async (request) => {
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { startDate, endDate, modelId = 'ppo_v20', forceRegenerate = false } = body;

    // Validate required parameters
    if (!startDate || !endDate) {
      return NextResponse.json(
        createApiResponse(null, 'error', 'startDate and endDate are required'),
        { status: 400 }
      );
    }

    // Validate date format (YYYY-MM-DD)
    const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
    if (!dateRegex.test(startDate) || !dateRegex.test(endDate)) {
      return NextResponse.json(
        createApiResponse(null, 'error', 'Dates must be in YYYY-MM-DD format'),
        { status: 400 }
      );
    }

    console.log(`[Replay API] Requesting trades: ${startDate} to ${endDate} (model=${modelId}, force=${forceRegenerate})`);

    // Call Python inference service
    const inferenceRequest: BacktestRequest = {
      start_date: startDate,
      end_date: endDate,
      model_id: modelId,
      force_regenerate: forceRegenerate,
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), INFERENCE_TIMEOUT_MS);

    try {
      const response = await fetch(`${INFERENCE_SERVICE_URL}/api/v1/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inferenceRequest),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[Replay API] Inference service error: ${response.status} - ${errorText}`);

        // Fallback to synthetic on server error
        return buildSyntheticFallback(startDate, endDate, modelId, startTime);
      }

      const result: InferenceServiceResponse = await response.json();

      console.log(`[Replay API] Received ${result.trade_count} trades from ${result.source} in ${result.processing_time_ms.toFixed(0)}ms`);

      // Transform trades to frontend format
      const frontendTrades = result.trades.map(transformTrade);

      // Sort by timestamp (ascending for replay)
      frontendTrades.sort((a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );

      const data = {
        trades: frontendTrades,
        total: result.trade_count,
        summary: result.summary,
        source: result.source,
        dateRange: result.date_range || { start: startDate, end: endDate },
        processingTimeMs: result.processing_time_ms,
        timestamp: new Date().toISOString(),
      };

      const successResponse = createApiResponse(data, 'live');
      successResponse.metadata.latency = Date.now() - startTime;
      successResponse.metadata.isRealData = true;

      return NextResponse.json(successResponse, {
        headers: { 'Cache-Control': 'no-store, max-age=0' },
      });

    } catch (fetchError) {
      clearTimeout(timeoutId);

      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.error('[Replay API] Inference service timeout, falling back to synthetic');
        return buildSyntheticFallback(startDate, endDate, modelId, startTime);
      }

      // Connection refused / fetch failed → synthetic fallback
      const msg = fetchError instanceof Error ? fetchError.message : '';
      if (msg.includes('ECONNREFUSED') || msg.includes('fetch failed')) {
        return buildSyntheticFallback(startDate, endDate, modelId, startTime);
      }

      throw fetchError;
    }

  } catch (error) {
    console.error('[Replay API] Error:', error);

    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    return NextResponse.json(
      createApiResponse(null, 'error', `Failed to load trades: ${errorMessage}`),
      { status: 500 }
    );
  }
});

/**
 * GET endpoint for checking backtest status
 */
export const GET = withAuth(async (request) => {
  const { searchParams } = new URL(request.url);
  const startDate = searchParams.get('startDate');
  const endDate = searchParams.get('endDate');
  const modelId = searchParams.get('modelId') || 'ppo_v20';

  if (!startDate || !endDate) {
    return NextResponse.json(
      createApiResponse(null, 'error', 'startDate and endDate query params required'),
      { status: 400 }
    );
  }

  try {
    const response = await fetch(
      `${INFERENCE_SERVICE_URL}/api/v1/backtest/status/${modelId}?start_date=${startDate}&end_date=${endDate}`,
      { method: 'GET' }
    );

    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }

    const status = await response.json();

    return NextResponse.json(createApiResponse(status, 'live'));

  } catch (error) {
    console.error('[Replay API] Status check error:', error);

    return NextResponse.json(
      createApiResponse(null, 'error', 'Could not check backtest status'),
      { status: 500 }
    );
  }
});
