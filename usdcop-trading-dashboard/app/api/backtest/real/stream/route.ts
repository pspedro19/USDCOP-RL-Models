/**
 * Real Backtest SSE Stream Proxy
 * ==============================
 *
 * Proxies SSE stream from Python backtest-api for real-time backtest progress.
 * Emits: progress, trade, complete, error events
 */

import { NextRequest } from 'next/server';

const BACKTEST_API_URL = process.env.BACKTEST_API_URL || 'http://backtest-api:8000';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const proposalId = searchParams.get('proposal_id');
  const startDate = searchParams.get('start_date');
  const endDate = searchParams.get('end_date');

  if (!proposalId || !startDate || !endDate) {
    return new Response(
      JSON.stringify({ error: 'Missing required params: proposal_id, start_date, end_date' }),
      { status: 400, headers: { 'Content-Type': 'application/json' } }
    );
  }

  console.log(`[RealBacktest SSE] Proxying stream for ${proposalId}: ${startDate} to ${endDate}`);

  try {
    const url = `${BACKTEST_API_URL}/api/v1/backtest/real/stream?proposal_id=${encodeURIComponent(proposalId)}&start_date=${startDate}&end_date=${endDate}`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'text/event-stream',
      },
      // @ts-ignore - signal not typed but works
      signal: AbortSignal.timeout(300000), // 5 min timeout
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[RealBacktest SSE] Backend error: ${response.status} - ${errorText}`);
      return new Response(
        JSON.stringify({ error: `Backend error: ${errorText}` }),
        { status: response.status, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Stream the response through
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });

  } catch (error) {
    console.error('[RealBacktest SSE] Error:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Stream failed' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
