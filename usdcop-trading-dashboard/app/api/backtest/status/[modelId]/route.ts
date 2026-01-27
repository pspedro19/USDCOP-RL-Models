import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.INFERENCE_API_URL || 'http://localhost:8003';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ modelId: string }> }
) {
  const { modelId } = await params;
  const startDate = request.nextUrl.searchParams.get('start_date');
  const endDate = request.nextUrl.searchParams.get('end_date');

  // Try backend
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    const response = await fetch(
      `${BACKEND_URL}/v1/backtest/status/${modelId}?start_date=${startDate}&end_date=${endDate}`,
      { signal: controller.signal }
    );
    clearTimeout(timeoutId);
    if (response.ok) {
      return NextResponse.json(await response.json());
    }
  } catch {
    // Backend unavailable
  }

  return NextResponse.json({ exists: false, model_id: modelId });
}
