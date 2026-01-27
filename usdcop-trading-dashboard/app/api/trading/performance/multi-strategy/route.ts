import { NextRequest, NextResponse } from 'next/server';

/**
 * Multi-Strategy Performance API
 * Returns comparative performance data across strategies.
 * Falls back to empty data when backend is unavailable.
 */
export async function GET(request: NextRequest) {
  // Try backend first
  const backendUrl = process.env.INFERENCE_API_URL || 'http://localhost:8003';
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(`${backendUrl}/api/trading/performance/multi-strategy`, {
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

  // Default empty response
  return NextResponse.json({
    strategies: [],
    source: 'default',
  });
}
