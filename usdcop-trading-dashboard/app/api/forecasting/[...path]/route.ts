/**
 * GET /api/forecasting/<path...> — serve a JSON/CSV file from public/forecasting with the PLAN
 * gate (CTR-RBAC-001 R3). Middleware guarantees a session and stamps x-user-id; here we apply:
 *   • asset scoping   — free ⇒ ['usdcop']; Gold/BTC weekly inference ⇒ 403 { upgrade:true }
 *   • forecast delay  — free ⇒ 168h (T-1 week): current-week entries are stripped from
 *                       weekly_inference_*.json AND rows newer than the cutoff are stripped from
 *                       bi_dashboard_unified.csv (by `inference_date`), so free sees last week's
 *                       forecast for their entitled asset too, paid sees live.
 *
 * Read-only, path-traversal-guarded. Legacy static /forecasting/<asset>/... still exists for
 * build-time files but is asset-agnostic; the app fetches through here so the plan is enforced.
 */
import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

import { getEntitlements } from '@/lib/auth/entitlements';

const FC_DIR = path.join(process.cwd(), 'public', 'forecasting');
const KNOWN_ASSET = /^(usdcop|xauusd|btcusdt)$/;

/** Drop CSV rows whose `inference_date` is newer than (now - delayHours). Header preserved. */
function applyCsvDelay(raw: string, delayHours: number): string {
  if (!delayHours) return raw;
  const cutoff = Date.now() - delayHours * 3_600_000;
  const lines = raw.split(/\r?\n/);
  if (lines.length < 2) return raw;
  const header = lines[0].split(',');
  const dateIdx = header.indexOf('inference_date');
  if (dateIdx < 0) return raw;
  const kept = [lines[0]];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const t = Date.parse(lines[i].split(',')[dateIdx]);
    if (Number.isNaN(t) || t <= cutoff) kept.push(lines[i]);
  }
  return kept.join('\n');
}

/** Drop weekly-inference entries whose week is newer than (now - delayHours). */
function applyDelay(json: unknown, delayHours: number): unknown {
  if (!delayHours || typeof json !== 'object' || json === null) return json;
  const cutoff = Date.now() - delayHours * 3_600_000;
  const doc = json as { strategies?: Array<{ weeks?: Array<{ week_start?: string }> }> };
  if (!Array.isArray(doc.strategies)) return json;
  for (const s of doc.strategies) {
    if (Array.isArray(s.weeks)) {
      s.weeks = s.weeks.filter((w) => {
        const t = w.week_start ? Date.parse(w.week_start) : NaN;
        return Number.isNaN(t) || t <= cutoff;
      });
    }
  }
  return doc;
}

export async function GET(
  req: Request,
  { params }: { params: { path: string[] } },
) {
  const segments = params.path ?? [];
  const leaf = segments.length ? segments[segments.length - 1] : '';
  const isCsv = leaf.endsWith('.csv');
  const isPng = leaf.endsWith('.png');
  if (!segments.length || (!leaf.endsWith('.json') && !isCsv && !isPng)) {
    return NextResponse.json({ error: 'only .json/.csv/.png files are served' }, { status: 400 });
  }
  if (segments.some((s) => s.includes('..') || s.includes('\0') || path.isAbsolute(s))) {
    return NextResponse.json({ error: 'invalid path' }, { status: 400 });
  }

  const ent = await getEntitlements(req.headers.get('x-user-id'));
  // First segment is the asset when it matches a known id (…/<asset>/weekly_inference_YYYY.json).
  const asset = KNOWN_ASSET.test(segments[0]) ? segments[0] : 'usdcop';
  if (!ent.assets.includes(asset)) {
    return NextResponse.json(
      { error: 'asset not in plan', asset, plan: ent.plan, upgrade: true },
      { status: 403 },
    );
  }

  const target = path.resolve(FC_DIR, ...segments);
  const rel = path.relative(FC_DIR, target);
  if (rel.startsWith('..') || path.isAbsolute(rel)) {
    return NextResponse.json({ error: 'path escapes forecasting dir' }, { status: 400 });
  }

  try {
    if (isPng) {
      // Per-file PNG delay (R3 remainder): filenames carry the ISO week (`_YYYY_WNN`).
      // A delayed plan must not fetch a chart newer than its cutoff even by direct URL.
      if (ent.forecast_delay_hours) {
        const m = leaf.match(/_(\d{4})_W(\d{2})/);
        if (m) {
          // Monday of that ISO week vs cutoff
          const yr = Number(m[1]); const wk = Number(m[2]);
          const jan4 = new Date(Date.UTC(yr, 0, 4));
          const monday = new Date(jan4);
          monday.setUTCDate(jan4.getUTCDate() - ((jan4.getUTCDay() + 6) % 7) + (wk - 1) * 7);
          if (monday.getTime() > Date.now() - ent.forecast_delay_hours * 3_600_000) {
            return NextResponse.json(
              { error: 'chart delayed for your plan', upgrade: true }, { status: 403 });
          }
        }
      }
      const buf = await fs.readFile(target);
      return new NextResponse(new Uint8Array(buf), {
        headers: { 'Content-Type': 'image/png', 'Cache-Control': 'private, max-age=300' },
      });
    }
    const raw = await fs.readFile(target, 'utf-8');
    if (isCsv) {
      const body = applyCsvDelay(raw, ent.forecast_delay_hours);
      return new NextResponse(body, {
        headers: { 'Content-Type': 'text/csv; charset=utf-8', 'Cache-Control': 'private, max-age=60' },
      });
    }
    let doc = JSON.parse(raw);
    if (/weekly_inference_\d+\.json$/.test(leaf)) {
      doc = applyDelay(doc, ent.forecast_delay_hours);
    }
    return NextResponse.json(doc, { headers: { 'Cache-Control': 'private, max-age=60' } });
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    if (code === 'ENOENT') return NextResponse.json({ error: 'not found' }, { status: 404 });
    return NextResponse.json({ error: 'failed to read file' }, { status: 500 });
  }
}
