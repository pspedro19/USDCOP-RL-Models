/**
 * GET /api/production/strategies — dynamic list of strategies with PRODUCTION exports
 * (StrategySelector contract, strategy-contract.md). Scans public/data/production for:
 *   summary.json                → the ACTIVE (singleton) strategy — today COP smart_simple
 *   summary_<sid>.json          → additional per-strategy production exports (e.g. BTC paper)
 * (summary_*_2025.json backtest files are excluded.)
 * No hardcoded strategy ids — the selector renders whatever was legitimately exported.
 *
 * SYNTHETIC DEMO BUNDLES get their own door. A real strategy only reaches this list after a
 * human Vote-2, and that gate is not negotiable. A demo fixture has no approval and must
 * never acquire one — writing an APPROVED record for it would forge the very signature the
 * gate exists to record. So a summary carrying `synthetic: true` is listed on the strength
 * of being declared synthetic, never on the strength of an approval, and is flagged
 * `mode: 'demo'` so every consumer can tell the two apart without inspecting the id.
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

import { readApprovalState } from '@/lib/approvals/store';

const PROD_DIR = path.join(process.cwd(), 'public', 'data', 'production');

interface StrategyOption {
  strategy_id: string;
  strategy_name: string;
  status: string;
  year: number | null;
  return_pct: number | null;
  mode: string;
  is_active_default: boolean;
}

async function readJson<T>(p: string): Promise<T | null> {
  try { return JSON.parse(await fs.readFile(p, 'utf-8')) as T; } catch { return null; }
}

export async function GET() {
  const options: StrategyOption[] = [];
  const seen = new Set<string>();

  type Summary = {
    strategy_id?: string; strategy_name?: string; year?: number; mode?: string;
    /** Declared by the bundle itself; the ONLY thing that opens the demo door. */
    synthetic?: boolean;
    strategies?: Record<string, { total_return_pct?: number | null }>;
  };
  const push = async (summaryFile: string, isDefault: boolean) => {
    const s = await readJson<Summary>(path.join(PROD_DIR, summaryFile));
    if (!s?.strategy_id || seen.has(s.strategy_id)) return;
    seen.add(s.strategy_id);
    // Approval state lives OUTSIDE public/ (CXD-057) — only `status` is used here,
    // which IS in the public allowlist; the rest of the document never leaves the server.
    const isSynthetic = s.synthetic === true;
    const a = (await readApprovalState(isDefault ? null : s.strategy_id))?.state ?? null;
    // Production lists ONLY operator-approved strategies; a PENDING per-sid export
    // stays out until the human Vote-2 on /dashboard moves it here (operator directive).
    // A synthetic demo is exempt from the APPROVAL check and from nothing else: it can
    // never be the default, and it is labelled DEMO on the way out.
    if (!isDefault && !isSynthetic && a?.status !== 'APPROVED' && a?.status !== 'LIVE') return;
    if (isSynthetic && isDefault) return; // a fixture must never be the active strategy
    options.push({
      strategy_id: s.strategy_id,
      strategy_name: s.strategy_name ?? s.strategy_id,
      status: isSynthetic ? 'DEMO' : (a?.status ?? 'LIVE'),
      year: s.year ?? null,
      return_pct: s.strategies?.[s.strategy_id]?.total_return_pct ?? null,
      mode: isSynthetic ? 'demo' : (s.mode ?? 'live'),
      is_active_default: isDefault,
    });
  };

  await push('summary.json', true);
  try {
    const files = await fs.readdir(PROD_DIR);
    for (const f of files) {
      const m = f.match(/^summary_([A-Za-z0-9_-]+)\.json$/);
      if (m && !/_\d{4}$/.test(m[1])) await push(f, false); // exclude summary_*_2025.json
    }
  } catch { /* dir unreadable → return what we have */ }

  return NextResponse.json(
    { strategies: options, active_strategy_id: options[0]?.strategy_id ?? null },
    { headers: { 'Cache-Control': 'private, max-age=30' } },
  );
}
