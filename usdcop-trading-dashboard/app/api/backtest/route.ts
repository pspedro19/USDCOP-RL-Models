import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.INFERENCE_API_URL || 'http://localhost:8003';

/**
 * Non-streaming backtest endpoint.
 * Proxies to backend or returns empty result.
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

  return NextResponse.json({
    success: true,
    source: 'generated',
    trade_count: 0,
    trades: [],
    summary: null,
    processing_time_ms: 0,
    date_range: { start: body.start_date, end: body.end_date },
  });
}
