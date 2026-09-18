/**
 * GET /api/analysis/week/[year]/[week]?asset=<asset_id>
 * Returns the full weekly view data for a specific ISO week + asset.
 * Reads public/data/analysis/<asset>/weekly_YYYY_WXX.json (file-based),
 * falling back to the legacy root for the default asset.
 *
 * PLAN-GATED (CTR-RBAC-001 R3): `?asset=` is monetized, so the handler enforces the caller's
 * entitlement — the route matrix only decides whether the ROLE may reach the route, never
 * whether the PLAN covers the asset. Without this a `free` session read Gold/BTC/S&P analysis
 * undelayed by editing one query parameter.
 */

import { NextRequest, NextResponse } from 'next/server';

import type { WeeklyViewData } from '@/lib/contracts/weekly-analysis.contract';
import { readAnalysisJson } from '@/lib/analysis-paths';
import { gateAnalysisAccess } from '@/lib/auth/analysis-gate';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ year: string; week: string }> }
) {
  const { year, week } = await params;
  const asset = request.nextUrl.searchParams.get('asset');

  const yearNum = parseInt(year, 10);
  const weekNum = parseInt(week, 10);

  if (isNaN(yearNum) || isNaN(weekNum) || weekNum < 1 || weekNum > 53) {
    return NextResponse.json(
      { error: 'Invalid year or week parameter' },
      { status: 400 }
    );
  }

  const weekStr = String(weekNum).padStart(2, '0');
  const filename = `weekly_${yearNum}_W${weekStr}.json`;

  // Asset scoping + per-plan freshness delay, before touching the filesystem.
  const denied = await gateAnalysisAccess(request.headers.get('x-user-id'), asset, filename);
  if (denied) return denied;

  const data = await readAnalysisJson<WeeklyViewData>(asset, filename);

  if (!data) {
    return NextResponse.json(
      { error: `No analysis data for ${yearNum}-W${weekStr}` },
      { status: 404 }
    );
  }
  return NextResponse.json(data);
}
