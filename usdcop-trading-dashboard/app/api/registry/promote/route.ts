/**
 * POST /api/registry/promote — Promote a model version to "active" for a strategy.
 * (CTR-STRAT-REGISTRY-001 — human promotion step: "la promuevo si me gusta")
 *
 * Body: { strategy_id: string, version: string, status?: string }
 *
 * Mutates ONLY the per-strategy manifest (public/data/strategies/<sid>/manifest.json):
 *   - model_versions[]: the chosen version becomes active:true, the rest active:false
 *   - production pointer: model_version = chosen version
 *   - status: optionally bumped (e.g. experimental -> production)
 * Then best-effort syncs the matching registry.json strategy `status` field so the
 * selectors stay consistent until the next Python register_bundle rebuild.
 *
 * DRY: registry DISCOVERY logic stays in Python (RegistryBuilder). This route does a
 * targeted field sync only — it never re-derives the index.
 *
 * Additive: never touches legacy production/*.json (approval/deploy remain the separate,
 * unchanged 2-vote flow). Promotion is a per-version preference, not an approval override.
 */
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

const DATA_DIR = path.join(process.cwd(), 'public', 'data');
const REGISTRY_FILE = path.join(DATA_DIR, 'registry.json');

function isSafeId(id: string): boolean {
  return typeof id === 'string' && /^[A-Za-z0-9._-]+$/.test(id);
}

function manifestPath(strategyId: string): string {
  return path.join(DATA_DIR, 'strategies', strategyId, 'manifest.json');
}

export async function POST(request: NextRequest) {
  let body: { strategy_id?: string; version?: string; status?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'invalid JSON body' }, { status: 400 });
  }

  const strategyId = body.strategy_id ?? '';
  const version = body.version ?? '';
  if (!isSafeId(strategyId) || !version) {
    return NextResponse.json(
      { error: 'strategy_id and version are required' },
      { status: 400 },
    );
  }

  // Load manifest (404 if the strategy has no bundle yet).
  let manifest: any;
  try {
    manifest = JSON.parse(await fs.readFile(manifestPath(strategyId), 'utf-8'));
  } catch {
    return NextResponse.json(
      { error: `manifest not found for strategy '${strategyId}'` },
      { status: 404 },
    );
  }

  const versions: any[] = manifest.model_versions ?? [];
  if (!versions.some((m) => m.version === version)) {
    return NextResponse.json(
      { error: `version '${version}' not found for strategy '${strategyId}'` },
      { status: 409 },
    );
  }

  try {
    // 1. Flip the active flag: chosen version active, all others inactive.
    for (const m of versions) {
      m.active = m.version === version;
    }
    // 2. Point the production entry at the promoted version.
    manifest.production = { ...(manifest.production ?? {}), model_version: version };
    // 3. Optional lifecycle bump (experimental -> production, etc.).
    if (typeof body.status === 'string' && body.status) {
      manifest.status = body.status;
    }
    manifest.promoted = {
      version,
      status: manifest.status,
      promoted_at: new Date().toISOString(),
      promoted_by: 'dashboard_user',
    };

    await fs.writeFile(
      manifestPath(strategyId),
      JSON.stringify(manifest, null, 2),
      'utf-8',
    );

    // 4. Best-effort registry status sync (targeted field, NOT a rebuild).
    try {
      const registry = JSON.parse(await fs.readFile(REGISTRY_FILE, 'utf-8'));
      const entry = (registry.strategies ?? []).find(
        (s: any) => s.strategy_id === strategyId,
      );
      if (entry) {
        entry.status = manifest.status;
        await fs.writeFile(REGISTRY_FILE, JSON.stringify(registry, null, 2), 'utf-8');
      }
    } catch {
      /* registry sync is optional; the manifest is the source of truth for versions */
    }

    return NextResponse.json({
      success: true,
      strategy_id: strategyId,
      active_version: version,
      status: manifest.status,
      message: `Version ${version} promoted for ${strategyId}`,
    });
  } catch (err) {
    return NextResponse.json(
      { error: `promote failed: ${(err as Error).message}` },
      { status: 500 },
    );
  }
}
