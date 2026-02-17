import { NextRequest, NextResponse } from 'next/server';
import { generateSyntheticTrades, calculateBacktestSummary } from '@/lib/services/synthetic-backtest.service';

const BACKEND_URL = process.env.INFERENCE_API_URL || 'http://localhost:8003';

/**
 * Non-streaming backtest endpoint.
 * Proxies to backend or returns synthetic fallback result.
 */
export async function POST(request: NextRequest) {
  const body = await request.json();

  // Try backend
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);
    const response = await fetch(`${BACKEND_URL}/v1/backtest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (response.ok) {
      return NextResponse.json(await response.json());
    }
  } catch {
    // Backend unavailable
  }

  // Fallback: generate synthetic result
  const startMs = Date.now();
  const trades = generateSyntheticTrades({
    startDate: body.start_date,
    endDate: body.end_date,
    modelId: body.model_id,
  });
  const summary = calculateBacktestSummary(trades);

  return NextResponse.json({
    success: true,
    source: 'generated',
    trade_count: trades.length,
    trades,
    summary,
    processing_time_ms: Date.now() - startMs,
    date_range: { start: body.start_date, end: body.end_date },
  });
}
