/**
 * GET /api/public/market-price?symbol=USD/COP — last close + 24h change.
 *
 * PUBLIC by design: a spot FX quote is public market data, not monetized IP. Exposing it
 * here lets the anonymous /login hero show a live tick without calling the session-gated
 * /api/market or /api/proxy/trading endpoints (which 401 for anon and spam the console).
 * Read-only, short-cached, degrades to {unavailable:true} on any DB hiccup.
 */
import { NextResponse } from 'next/server';

import { query } from '@/lib/db/postgres-client';

const ALLOWED = new Set(['USD/COP', 'USD/MXN', 'USD/BRL']);

export async function GET(req: Request) {
  const raw = new URL(req.url).searchParams.get('symbol') ?? 'USD/COP';
  const symbol = raw.includes('/') ? raw : raw.replace(/^(USD)(COP|MXN|BRL)$/i, '$1/$2').toUpperCase();
  if (!ALLOWED.has(symbol)) {
    return NextResponse.json({ unavailable: true, reason: 'unsupported symbol' }, { status: 200 });
  }
  try {
    // Latest close and the most recent close at least ~24h earlier (for the change).
    const latest = await query(
      `SELECT time, close FROM usdcop_m5_ohlcv WHERE symbol = $1 ORDER BY time DESC LIMIT 1`,
      [symbol],
    );
    if (!latest.rows.length) return NextResponse.json({ unavailable: true }, { status: 200 });
    const last = latest.rows[0];
    const prior = await query(
      `SELECT close FROM usdcop_m5_ohlcv
       WHERE symbol = $1 AND time <= $2::timestamptz - interval '24 hours'
       ORDER BY time DESC LIMIT 1`,
      [symbol, last.time],
    );
    const price = Number(last.close);
    const prevClose = prior.rows.length ? Number(prior.rows[0].close) : null;
    const change = prevClose != null ? price - prevClose : null;
    const changePercent = prevClose ? (change! / prevClose) * 100 : null;

    return NextResponse.json(
      { symbol, price, change, changePercent, timestamp: last.time },
      { headers: { 'Cache-Control': 'public, max-age=60, stale-while-revalidate=300' } },
    );
  } catch {
    return NextResponse.json({ unavailable: true }, { status: 200 });
  }
}
