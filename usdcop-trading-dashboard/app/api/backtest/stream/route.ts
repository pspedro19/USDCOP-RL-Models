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
 * 3. DEMO (mode=demo) — EXPLICIT caller opt-in ONLY
 *    Streams fabricated numbers from `synthetic-backtest.service`. Labelled with a
 *    first-level `synthetic: true` in the `result` event and `X-Data-Origin: SYNTHETIC`.
 *
 * **FAIL-CLOSED**: when the backend is unreachable or answers !ok, this route responds
 * `503` with the real reason and emits NOT ONE fabricated trade. Fabricating a plausible
 * equity curve on a decision surface (Vote 2 is cast on these numbers) is worse than an
 * empty screen — see `.claude/rules/quant-constitution.md` §6/§7.
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

const SSE_HEADERS = {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache',
  'Connection': 'keep-alive',
} as const;

/**
 * Closed set of client-visible reasons. Adding a member is a contract change; anything
 * outside this union can never reach the browser.
 */
type UnavailableReason =
  | 'backend_unreachable'
  | 'backend_error_response'
  | 'backend_timeout'
  | 'backend_empty_response';

/** Short token that ties the client's 503 to the full server-side log line. */
function newCorrelationId(): string {
  const raw = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(16).slice(2);
  return raw.replace(/-/g, '').slice(0, 8).padEnd(8, '0');
}

/**
 * Fail-closed response for a dead/erroring inference backend.
 *
 * Nothing has been flushed to the client at this point (the upstream stream is piped
 * through only on success), so we can still set a real status: `503`. The body carries
 * EXACTLY ONE SSE frame, of type `error` — never a `result`, never a `trade`.
 *
 * The frame is SANITIZED: closed-set `reason` + `correlation_id`, no upstream URL and no
 * upstream body. `data` is rendered verbatim to the user by `backtest.service.ts`
 * (`String(event.data)`), so leaking the internal host there would put it on screen.
 */
function backendUnavailable(reason: UnavailableReason, serverDetail: string): Response {
  const correlationId = newCorrelationId();
  console.error(
    `[Stream] inference_backend_unavailable ref=${correlationId} reason=${reason} — ${serverDetail}`,
  );
  // `data` is a STRING on purpose: the SSE consumer (`backtest.service.ts`, `case 'error'`)
  // does `String(event.data)`, so an object would surface to the user as "[object Object]".
  // The machine-readable code travels alongside it, at the frame's top level.
  const payload = JSON.stringify({
    type: 'error',
    error: 'inference_backend_unavailable',
    reason,
    correlation_id: correlationId,
    data: `inference_backend_unavailable: ${reason} (ref ${correlationId})`,
  });
  return new Response(
    `data: ${payload}\n\n`,
    {
      status: 503,
      headers: { ...SSE_HEADERS, 'X-Data-Origin': 'ERROR' },
    },
  );
}

/** Synthetic stream, reachable ONLY through the explicit `mode=demo` opt-in. */
function demoStream(config: Parameters<typeof createSyntheticSSEStream>[0]): Response {
  console.warn('[Stream] mode=demo — serving FABRICATED trades (explicit caller opt-in)');
  return new Response(createSyntheticSSEStream(config), {
    headers: { ...SSE_HEADERS, 'X-Data-Origin': 'SYNTHETIC' },
  });
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

  // Explicit demo opt-in: fabricated numbers, clearly labelled as such.
  if (mode === 'demo') {
    return demoStream({
      startDate,
      endDate,
      modelId,
      emitBarEvents: true,
      replaySpeed,
    });
  }

  const endpoint = mode === 'replay'
    ? `${BACKEND_URL}/v1/backtest/replay`
    : `${BACKEND_URL}/v1/backtest/stream`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    let response: Response;
    try {
      // Use replay endpoint for L1+L5 bar-by-bar simulation
      response = await fetch(endpoint, {
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
    } finally {
      clearTimeout(timeoutId);
    }

    if (response.ok && response.body) {
      return new Response(response.body, { headers: { ...SSE_HEADERS } });
    }

    return backendUnavailable(
      response.ok ? 'backend_empty_response' : 'backend_error_response',
      response.ok
        ? `${endpoint} responded ${response.status} with an empty body`
        : `${endpoint} responded ${response.status}`,
    );
  } catch (error) {
    // NEVER swallow this: the empty `catch {}` that used to live here is precisely
    // what allowed fabricated equity curves to be served as real for months.
    const cause = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    const aborted = error instanceof Error && error.name === 'AbortError';
    return backendUnavailable(
      aborted ? 'backend_timeout' : 'backend_unreachable',
      `${endpoint} ${aborted ? 'timed out after 5000ms' : 'unreachable'} (${cause})`,
    );
  }
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

  // Explicit demo opt-in: fabricated numbers, clearly labelled as such.
  if (body.mode === 'demo' || request.nextUrl.searchParams.get('mode') === 'demo') {
    return demoStream({
      startDate: start_date,
      endDate: end_date,
      modelId: model_id,
      emitBarEvents: emit_bar_events ?? true,
      replaySpeed: speed,
    });
  }

  const endpoint = `${BACKEND_URL}/v1/backtest/stream`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    let response: Response;
    try {
      response = await fetch(endpoint, {
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
    } finally {
      clearTimeout(timeoutId);
    }

    if (response.ok && response.body) {
      return new Response(response.body, { headers: { ...SSE_HEADERS } });
    }

    return backendUnavailable(
      response.ok ? 'backend_empty_response' : 'backend_error_response',
      response.ok
        ? `${endpoint} responded ${response.status} with an empty body`
        : `${endpoint} responded ${response.status}`,
    );
  } catch (error) {
    // NEVER swallow this: see the note in GET.
    const cause = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    const aborted = error instanceof Error && error.name === 'AbortError';
    return backendUnavailable(
      aborted ? 'backend_timeout' : 'backend_unreachable',
      `${endpoint} ${aborted ? 'timed out after 5000ms' : 'unreachable'} (${cause})`,
    );
  }
}
