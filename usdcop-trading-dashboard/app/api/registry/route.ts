/**
 * GET /api/registry — Dynamic strategy/asset registry (CTR-STRAT-REGISTRY-001)
 *
 * File-based, no database. Returns the registry index the frontend uses to build ALL selectors
 * (Asset -> Strategy -> Version -> Year) with zero hardcoded strategy_id / symbol / year.
 *
 * Source of truth: public/data/registry.json, produced by the Python RegistryBuilder
 * (scripts/build_strategy_registry.py) — invoked by the DAG exit contract (register_bundle).
 * Discovery logic lives in ONE place (Python); this route stays thin (DRY).
 *
 * Fallback: if registry.json is absent, synthesize a minimal index from the legacy
 * strategies.json so the dashboard still renders during migration.
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'public', 'data');
const REGISTRY_FILE = path.join(DATA_DIR, 'registry.json');
const LEGACY_STRATEGIES_FILE = path.join(DATA_DIR, 'production', 'strategies.json');

function chartSymbolFor(symbol: string): string {
  return symbol.replace(/[/ ]/g, '').toUpperCase();
}

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(file, 'utf-8')) as T;
  } catch {
    return null;
  }
}

/** Minimal fallback index built from the legacy strategies.json (migration shim). */
async function synthesizeLegacyIndex() {
  const legacy = await readJson<any>(LEGACY_STRATEGIES_FILE);
  const items: any[] = legacy?.strategies ?? [];
  const strategies = items.map((s) => ({
    strategy_id: s.strategy_id,
    asset_id: s.asset_id ?? 'usdcop',
    status: s.status === 'APPROVED' ? 'production' : 'paper',
    display_name: s.strategy_name ?? s.strategy_id,
    pipeline_type: s.pipeline ?? s.pipeline_type ?? 'ml_forecasting',
    timeframe: s.timeframe ?? 'weekly',
    manifest: `strategies/${s.strategy_id}/manifest.json`,
    backtest_years: s.backtest_year ? [s.backtest_year] : [],
    has_production: true,
    has_replay: false,
  }));
  return {
    schema_version: '1.0.0',
    generated_at: new Date().toISOString(),
    assets: [{ asset_id: 'usdcop', symbol: 'USD/COP', chart_symbol: chartSymbolFor('USD/COP'), display_name: 'USD/COP', asset_class: 'fx' }],
    strategies,
    default: { asset_id: 'usdcop', strategy_id: legacy?.default_strategy ?? strategies[0]?.strategy_id ?? 'smart_simple_v11' },
  };
}

export async function GET() {
  const registry = await readJson<any>(REGISTRY_FILE);
  if (registry) {
    return NextResponse.json(registry);
  }
  return NextResponse.json(await synthesizeLegacyIndex());
}
