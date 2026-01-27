import { NextRequest, NextResponse } from 'next/server';

/**
 * Model Metrics API
 * Returns performance metrics for a specific model.
 * Falls back to empty/default metrics when backend is unavailable.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const { modelId } = await params;
  const period = request.nextUrl.searchParams.get('period') || 'all';

  // Try backend first
  const backendUrl = process.env.INFERENCE_API_URL || 'http://localhost:8003';
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(`${backendUrl}/api/models/${modelId}/metrics?period=${period}`, {
      signal: controller.signal,
      cache: 'no-store',
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend unavailable, return defaults
  }

  // Default metrics in expected format: { data: { metrics: { ... } } }
  return NextResponse.json({
    model_id: modelId,
    period,
    data: {
      metrics: {
        sharpe_ratio: 0,
        max_drawdown: 0,
        win_rate: 0,
        total_trades: 0,
        total_return: 0,
        avg_trade_duration: 0,
        profit_factor: 0,
      },
    },
    source: 'default',
  });
}
