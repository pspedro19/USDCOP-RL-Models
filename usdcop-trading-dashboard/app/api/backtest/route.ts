import { NextRequest, NextResponse } from 'next/server';
import { generateSyntheticTrades, calculateBacktestSummary } from '@/lib/services/synthetic-backtest.service';

const BACKEND_URL = process.env.INFERENCE_API_URL || 'http://localhost:8003';

/**
 * Non-streaming backtest endpoint.
 * =================================
 * Proxies to the inference backend. **FAIL-CLOSED**: if the backend is unreachable or
 * answers !ok, this route responds `503` with the real reason. It NEVER fabricates a
 * backtest and serves it as if it were real — `/dashboard` is a decision surface (Vote 2
 * is cast there) and the repo doctrine is fail-safe (`PreTradeGate`: error ⇒ BLOCK).
 *
 * The synthetic generator is still reachable, but only behind an EXPLICIT caller opt-in
 * (`?mode=demo` or `{ mode: 'demo' }`), and what it returns is labelled as fabricated
 * with a first-level `synthetic: true` plus the `X-Data-Origin: SYNTHETIC` header.
 */

/** Explicit, caller-side opt-in into fabricated numbers. Never inferred from a failure. */
function wantsDemo(request: NextRequest, body: Record<string, unknown>): boolean {
  return request.nextUrl.searchParams.get('mode') === 'demo' || body?.mode === 'demo';
}

/**
 * Closed set of client-visible reasons. Adding a member is a contract change; anything
 * outside this union can never reach the browser.
 */
type UnavailableReason = 'backend_unreachable' | 'backend_error_response' | 'backend_timeout';

/** Short token that ties the client's 503 to the full server-side log line. */
function newCorrelationId(): string {
  const raw = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(16).slice(2);
  return raw.replace(/-/g, '').slice(0, 8).padEnd(8, '0');
}

/**
 * FAIL-CLOSED 503.
 *
 * The client gets a SANITIZED motive: a closed-set `reason` plus a `correlation_id`.
 * The upstream URL, its status line and its response body stay in the server log — they
 * name an internal host/port and can echo back arbitrary upstream text, which is useful
 * to an attacker mapping the network and useless to the browser (nothing renders it; the
 * UI only shows the error code). An operator joins the two halves through `correlation_id`.
 */
function backendUnavailable(reason: UnavailableReason, serverDetail: string) {
  const correlationId = newCorrelationId();
  console.error(
    `[Backtest API] inference_backend_unavailable ref=${correlationId} reason=${reason} — ${serverDetail}`,
  );
  return NextResponse.json(
    { success: false, error: 'inference_backend_unavailable', reason, correlation_id: correlationId },
    { status: 503, headers: { 'Cache-Control': 'no-store, max-age=0' } },
  );
}

function syntheticDemoResponse(body: Record<string, unknown>) {
  const startMs = Date.now();
  const trades = generateSyntheticTrades({
    startDate: body.start_date as string,
    endDate: body.end_date as string,
    modelId: body.model_id as string,
  });
  const summary = calculateBacktestSummary(trades);

  return NextResponse.json(
    {
      // First-level, POSITIVE declaration that these numbers are fabricated.
      synthetic: true,
      data_origin: 'SYNTHETIC',
      success: true,
      source: 'generated',
      trade_count: trades.length,
      trades,
      summary,
      processing_time_ms: Date.now() - startMs,
      date_range: { start: body.start_date, end: body.end_date },
    },
    { headers: { 'X-Data-Origin': 'SYNTHETIC', 'Cache-Control': 'no-store, max-age=0' } },
  );
}

export async function POST(request: NextRequest) {
  const body = await request.json();

  if (wantsDemo(request, body)) {
    return syntheticDemoResponse(body);
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    let response: Response;
    try {
      response = await fetch(`${BACKEND_URL}/v1/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timeoutId);
    }

    if (response.ok) {
      return NextResponse.json(await response.json());
    }

    const errorText = await response.text().catch(() => '');
    return backendUnavailable(
      'backend_error_response',
      `${BACKEND_URL}/v1/backtest responded ${response.status}${errorText ? `: ${errorText.slice(0, 500)}` : ''}`,
    );
  } catch (error) {
    // NEVER swallow this silently: an empty `catch {}` is what let the synthetic
    // fallback masquerade as a real backtest for months.
    const cause = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    const aborted = error instanceof Error && error.name === 'AbortError';
    return backendUnavailable(
      aborted ? 'backend_timeout' : 'backend_unreachable',
      `${BACKEND_URL}/v1/backtest ${aborted ? 'timed out after 30000ms' : 'unreachable'} (${cause})`,
    );
  }
}
