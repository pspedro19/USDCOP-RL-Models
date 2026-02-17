import { NextRequest } from 'next/server';
import { readFileSync } from 'fs';
import path from 'path';
import { createSyntheticSSEStream } from '@/lib/services/synthetic-backtest.service';
import { isForecastStrategy, FORECAST_STRATEGIES, createForecastSSEStream } from '@/lib/services/forecast-backtest.service';
import { REPLAY_SPEEDS, type ReplaySpeed, DEFAULT_REPLAY_SPEED } from '@/lib/contracts/backtest.contract';

const BACKEND_URL = process.env.INFERENCE_API_URL || 'http://localhost:8003';

/**
 * Backtest Stream API Proxy
 * =========================
 * Supports two modes:
 *
 * 1. BACKTEST REPLAY (mode=replay)
 *    - Passes historical data through L1 (features) + L5 (inference) bar-by-bar
 *    - Shows equity curve building dynamically
 *    - Same inference code as production, just with historical data
 *    - Supports speed control (0.5x to 16x)
 *
 * 2. STANDARD BACKTEST (mode=standard)
 *    - Runs complete backtest and streams results
 *    - Faster, used for quick evaluation
 *
 * Both modes use the same L1+L5 pipeline:
 * - L1: Feature computation (log returns, RSI, ATR, macro features)
 * - L5: Model inference (loads model, computes action)
 *
 * The fallback generates synthetic data for demo/testing when backend unavailable.
 * All modes use SSOT config for consistent metrics calculation.
 */

/**
 * Parse and validate replay speed parameter
 */
function parseReplaySpeed(value: string | null): ReplaySpeed {
  if (!value) return DEFAULT_REPLAY_SPEED;
  const parsed = parseFloat(value);
  if (REPLAY_SPEEDS.includes(parsed as ReplaySpeed)) {
    return parsed as ReplaySpeed;
  }
  return DEFAULT_REPLAY_SPEED;
}

/**
 * GET handler for EventSource connections
 * Supports speed parameter for replay control (0.5x to 16x)
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const startDate = searchParams.get('startDate');
  const endDate = searchParams.get('endDate');
  const modelId = searchParams.get('modelId');
  const mode = searchParams.get('mode') || 'replay';
  const replaySpeed = parseReplaySpeed(searchParams.get('speed'));

  if (!startDate || !endDate || !modelId) {
    return new Response('Missing required parameters', { status: 400 });
  }

  // Forecast strategies: serve from pre-computed JSON files
  if (isForecastStrategy(modelId)) {
    try {
      const filePath = path.join(process.cwd(), 'public', FORECAST_STRATEGIES[modelId].file);
      const data = JSON.parse(readFileSync(filePath, 'utf-8'));
      const stream = createForecastSSEStream(data.trades, data.summary, {
        startDate: startDate,
        endDate: endDate,
        replaySpeed,
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    } catch (err) {
      console.error(`[Stream] Failed to load forecast data for ${modelId}:`, err);
      return new Response(`Forecast data not found for ${modelId}. Run: python scripts/backtest_2025_10k.py`, { status: 404 });
    }
  }

  // Try backend first
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    // Use replay endpoint for L1+L5 bar-by-bar simulation
    const endpoint = mode === 'replay'
      ? `${BACKEND_URL}/v1/backtest/replay`
      : `${BACKEND_URL}/v1/backtest/stream`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        start_date: startDate,
        end_date: endDate,
        model_id: modelId,
        mode: mode,
        // For replay mode, emit bar-level events for dynamic equity curve
        emit_bar_events: mode === 'replay',
        // Pass speed to backend for replay control
        replay_speed: replaySpeed,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok && response.body) {
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }
  } catch {
    // Backend unavailable, generate synthetic data
  }

  // Fallback: generate synthetic backtest with fluid, progressive replay
  const stream = createSyntheticSSEStream({
    startDate: startDate,
    endDate: endDate,
    modelId: modelId,
    emitBarEvents: mode === 'replay',
    replaySpeed: replaySpeed,
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}

/**
 * POST handler for traditional backtest requests
 * Supports replay_speed parameter for speed control
 */
export async function POST(request: NextRequest) {
  const body = await request.json();
  const { start_date, end_date, model_id, force_regenerate, replay_speed, emit_bar_events } = body;
  const speed = parseReplaySpeed(String(replay_speed ?? 1));

  // Forecast strategies: serve from pre-computed JSON files
  if (model_id && isForecastStrategy(model_id)) {
    try {
      const filePath = path.join(process.cwd(), 'public', FORECAST_STRATEGIES[model_id].file);
      const data = JSON.parse(readFileSync(filePath, 'utf-8'));
      const stream = createForecastSSEStream(data.trades, data.summary, {
        startDate: start_date,
        endDate: end_date,
        replaySpeed: speed,
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    } catch (err) {
      console.error(`[Stream] Failed to load forecast data for ${model_id}:`, err);
      return new Response(`Forecast data not found for ${model_id}. Run: python scripts/backtest_2025_10k.py`, { status: 404 });
    }
  }

  // Try backend first
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${BACKEND_URL}/v1/backtest/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify({
        start_date,
        end_date,
        model_id,
        force_regenerate,
        replay_speed: speed,
        emit_bar_events: emit_bar_events ?? true,
      }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok && response.body) {
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }
  } catch {
    // Backend unavailable, generate synthetic data
  }

  // Fallback: generate synthetic backtest with fluid, progressive replay
  const stream = createSyntheticSSEStream({
    startDate: start_date,
    endDate: end_date,
    modelId: model_id,
    emitBarEvents: emit_bar_events ?? true,
    replaySpeed: speed,
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}
