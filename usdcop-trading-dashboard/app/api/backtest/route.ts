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

function backendUnavailable(detail: string) {
  console.error(`[Backtest API] Inference backend unavailable — responding 503: ${detail}`);
  return NextResponse.json(
    { success: false, error: 'inference_backend_unavailable', detail },
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
      `${BACKEND_URL}/v1/backtest responded ${response.status}${errorText ? `: ${errorText.slice(0, 500)}` : ''}`,
    );
  } catch (error) {
    // NEVER swallow this silently: an empty `catch {}` is what let the synthetic
    // fallback masquerade as a real backtest for months.
    const reason = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    return backendUnavailable(`${BACKEND_URL}/v1/backtest unreachable (${reason})`);
  }
}
