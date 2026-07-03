/**
 * GET /api/strategies/[strategyId]/manifest — Self-describing strategy bundle (CTR-STRAT-REGISTRY-001)
 *
 * Returns the StrategyBundleManifest for one strategy: its versions, per-(version, year) backtests,
 * replayability, and current production/approval pointers. The frontend uses this to build the
 * version dropdown and resolve which immutable artifacts to fetch for KPIs / trades / replay.
 *
 * Source of truth: public/data/strategies/<strategyId>/manifest.json, produced by the Python
 * BundlePublisher (DAG exit contract). This route stays thin — no discovery logic here (DRY).
 *
 * Additive: does NOT touch legacy production/*.json consumptions.
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'public', 'data');

// Guard against path traversal: strategy ids are slugs.
function isSafeId(id: string): boolean {
  return /^[A-Za-z0-9._-]+$/.test(id);
}

export async function GET(
  _req: Request,
  { params }: { params: { strategyId: string } },
) {
  const { strategyId } = params;
  if (!isSafeId(strategyId)) {
    return NextResponse.json({ error: 'invalid strategy id' }, { status: 400 });
  }
  const file = path.join(DATA_DIR, 'strategies', strategyId, 'manifest.json');
  try {
    const manifest = JSON.parse(await fs.readFile(file, 'utf-8'));
    return NextResponse.json(manifest);
  } catch {
    return NextResponse.json(
      { error: `manifest not found for strategy '${strategyId}'` },
      { status: 404 },
    );
  }
}
