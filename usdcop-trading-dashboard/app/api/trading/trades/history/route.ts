import { NextRequest, NextResponse } from 'next/server';

/**
 * Trades History API
 * Returns historical trades for a given model.
 * Falls back to empty array when backend/DB is unavailable.
 */
export async function GET(request: NextRequest) {
  const limit = request.nextUrl.searchParams.get('limit') || '50';
  const modelId = request.nextUrl.searchParams.get('model_id') || 'ppo_primary';

  // Try backend first
  const backendUrl = process.env.INFERENCE_API_URL || 'http://localhost:8003';
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(
      `${backendUrl}/api/trading/trades/history?limit=${limit}&model_id=${modelId}`,
      { signal: controller.signal, cache: 'no-store' }
    );
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend unavailable
  }

  // Default empty response
  return NextResponse.json({
    trades: [],
    model_id: modelId,
    total: 0,
    source: 'default',
  });
}
