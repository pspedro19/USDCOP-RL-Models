import { NextRequest, NextResponse } from 'next/server';

/**
 * Model Equity Curve API
 * Returns equity curve data for a specific model.
 * Falls back to empty data when backend is unavailable.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const { modelId } = await params;
  const days = request.nextUrl.searchParams.get('days') || '90';

  // Try backend first
  const backendUrl = process.env.INFERENCE_API_URL || 'http://localhost:8003';
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(`${backendUrl}/api/models/${modelId}/equity-curve?days=${days}`, {
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

  // Default equity curve in expected format: { data: { points: [], summary: {} } }
  return NextResponse.json({
    model_id: modelId,
    days: parseInt(days),
    data: {
      points: [],
      summary: {
        start_equity: 10000,
        end_equity: 10000,
        total_return: 0,
        total_return_pct: 0,
        max_drawdown_pct: 0,
      },
    },
    source: 'default',
  });
}
