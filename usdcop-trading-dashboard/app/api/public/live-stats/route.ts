/**
 * GET /api/public/live-stats — marketing-approved LIVE aggregates for the landing
 * (ux-navigation S3: trust bar shows ONLY ● LIVE numbers, never backtest).
 *
 * Contract safety: reads the PUBLISHED bundle (`summary.json` = production forward, P1
 * "un solo número") server-side; exposes only aggregate marketing figures — no signals,
 * no levels, no trades (those stay session-gated). Public by design (RBAC matrix entry).
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const PROD_DIR = path.join(process.cwd(), 'public', 'data', 'production');

interface LiveStats {
  phase: 'live';
  strategy_name: string;
  year: number;
  return_ytd_pct: number | null;
  max_dd_pct: number | null;
  weeks_live: number | null;
  n_trades: number | null;
  /** Sharpe only when N>=20 trades (quant-constitution §6 / ui.contract canShowRatios). */
  sharpe: number | null;
  bundle_date: string | null;
  backtest: { year: number; return_pct: number | null; sharpe: number | null } | null;
}

export const revalidate = 300; // marketing stats: 5-min cache is plenty

export async function GET() {
  try {
    const summary = JSON.parse(
      await fs.readFile(path.join(PROD_DIR, 'summary.json'), 'utf-8'));
    const sid: string = summary.strategy_id ?? 'smart_simple_v11';
    const s = summary.strategies?.[sid] ?? {};
    const nTrades = (s.n_long ?? 0) + (s.n_short ?? 0);
    const freezeDate = new Date('2026-03-18T00:00:00Z'); // v2.0 freeze = clock start
    const weeksLive = Math.max(0, Math.floor((Date.now() - freezeDate.getTime()) / 604_800_000));

    let backtest: LiveStats['backtest'] = null;
    try {
      const bt = JSON.parse(
        await fs.readFile(path.join(PROD_DIR, 'summary_2025.json'), 'utf-8'));
      const bs = bt.strategies?.[bt.strategy_id ?? sid] ?? {};
      backtest = { year: 2025, return_pct: bs.total_return_pct ?? null, sharpe: bs.sharpe ?? null };
    } catch { /* backtest summary optional */ }

    const body: LiveStats = {
      phase: 'live',
      strategy_name: summary.strategy_name ?? 'Smart Simple',
      year: summary.year ?? new Date().getFullYear(),
      return_ytd_pct: s.total_return_pct ?? null,
      max_dd_pct: s.max_dd_pct ?? null,
      weeks_live: weeksLive,
      n_trades: nTrades || null,
      sharpe: nTrades >= 20 ? (s.sharpe ?? null) : null, // N<20 => no ratios, ever
      bundle_date: summary.generated_at?.slice(0, 10) ?? null,
      backtest,
    };
    return NextResponse.json(body, {
      headers: { 'Cache-Control': 'public, max-age=300, stale-while-revalidate=3600' },
    });
  } catch {
    // Fail soft: the landing renders without the trust bar rather than erroring.
    return NextResponse.json({ phase: 'live', unavailable: true }, { status: 200 });
  }
}
