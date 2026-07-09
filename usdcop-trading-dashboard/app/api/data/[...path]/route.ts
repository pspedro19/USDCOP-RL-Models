/**
 * GET /api/data/<path...> — serve a JSON file from public/data via fs (CTR-STRAT-REGISTRY-001).
 *
 * WHY THIS EXISTS: in the `output: 'standalone'` production build, Next only serves the `public/`
 * files that existed when the image was built. Bundle artifacts published AFTER the build (e.g. a
 * newly promoted immutable version `strategies/<sid>/backtests/<v>/{summary,trades}_<year>.json`,
 * created by the Python BundlePublisher into the read/write `public/data` mount) are physically
 * present in the container but 404 through Next's static handler — which silently broke the backtest
 * replay for any freshly-published version. This route reads the file from disk at request time (the
 * same fs pattern as /api/registry and /api/strategies/[id]/manifest), so post-build bundles are
 * always served. The frontend fetches `/api/data/<relPath>` instead of the static `/data/<relPath>`.
 *
 * Read-only, JSON-only, path-traversal-guarded. Additive: legacy static `/data/...` still works for
 * build-time files.
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

import { getEntitlements, isFresherThanAllowed } from '@/lib/auth/entitlements';

const DATA_DIR = path.join(process.cwd(), 'public', 'data');

export async function GET(
  _req: Request,
  { params }: { params: { path: string[] } },
) {
  const segments = params.path ?? [];

  // ── Monetization gate (CTR-RBAC-001 R3). Middleware guarantees a session and stamps
  // x-user-id on the request; here we apply the PLAN: asset scoping + freshness delay
  // for the gated `analysis/**` tree (free ⇒ resumen T+7). Admin/paid plans pass clean.
  if (segments[0] === 'analysis') {
    const userId = _req.headers.get('x-user-id');
    const ent = await getEntitlements(userId);
    const asset = segments.length > 2 ? segments[1] : 'usdcop'; // analysis/<asset>/... or legacy root
    const knownAsset = /^(usdcop|xauusd|btcusdt)$/.test(asset) ? asset : 'usdcop';
    if (!ent.assets.includes(knownAsset)) {
      return NextResponse.json(
        { error: 'asset not in plan', asset: knownAsset, plan: ent.plan, upgrade: true },
        { status: 403 },
      );
    }
    const fileName = segments[segments.length - 1];
    if (isFresherThanAllowed(fileName, ent.analysis_delay_days)) {
      return NextResponse.json(
        { error: 'content delayed for your plan', delay_days: ent.analysis_delay_days,
          plan: ent.plan, upgrade: true },
        { status: 403 },
      );
    }
  }
  // Only JSON is served; reject anything else up front.
  if (!segments.length || !segments[segments.length - 1].endsWith('.json')) {
    return NextResponse.json({ error: 'only .json files are served' }, { status: 400 });
  }
  // Reject traversal / absolute segments before resolving.
  if (segments.some((s) => s.includes('..') || s.includes('\0') || path.isAbsolute(s))) {
    return NextResponse.json({ error: 'invalid path' }, { status: 400 });
  }

  const target = path.resolve(DATA_DIR, ...segments);
  // Defense in depth: the resolved path must stay inside DATA_DIR.
  const rel = path.relative(DATA_DIR, target);
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    return NextResponse.json({ error: 'path escapes data dir' }, { status: 400 });
  }

  try {
    const raw = await fs.readFile(target, 'utf-8');
    // Parse+re-serialize so we return valid JSON (and reject a corrupt file with a 500).
    return NextResponse.json(JSON.parse(raw));
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    if (code === 'ENOENT') {
      return NextResponse.json({ error: 'not found' }, { status: 404 });
    }
    return NextResponse.json({ error: 'failed to read file' }, { status: 500 });
  }
}
