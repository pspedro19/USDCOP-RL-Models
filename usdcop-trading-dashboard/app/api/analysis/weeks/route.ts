/**
 * GET /api/analysis/weeks?asset=<asset_id>
 * Returns the analysis index (list of available weeks) for an asset.
 * Reads public/data/analysis/<asset>/analysis_index.json (file-based),
 * falling back to the legacy root for the default asset.
 *
 * PLAN-GATED (CTR-RBAC-001 R3): the asset must be in the caller's plan, and weeks still
 * inside the plan's delay window are filtered OUT of the index — otherwise the week picker
 * offers a week whose content the reader would be refused, which reads as a broken product
 * rather than as an upsell.
 */

import { NextRequest, NextResponse } from 'next/server';

import type { AnalysisIndex } from '@/lib/contracts/weekly-analysis.contract';
import { readAnalysisJson } from '@/lib/analysis-paths';
import { filterDelayedWeeks, gateAnalysisAccess } from '@/lib/auth/analysis-gate';

const DEFAULT_INDEX: AnalysisIndex = { weeks: [] };

export async function GET(request: NextRequest) {
  const asset = request.nextUrl.searchParams.get('asset');
  const userId = request.headers.get('x-user-id');

  // The index itself carries no single dated artifact, so only the asset is gated here;
  // the delay is applied per entry below.
  const denied = await gateAnalysisAccess(userId, asset);
  if (denied) return denied;

  const index = await readAnalysisJson<AnalysisIndex>(asset, 'analysis_index.json');
  if (!index?.weeks?.length) return NextResponse.json(index ?? DEFAULT_INDEX);

  const weeks = await filterDelayedWeeks(
    userId, index.weeks, (w) => `weekly_${w.year}_W${String(w.week).padStart(2, '0')}.json`,
  );
  return NextResponse.json({ ...index, weeks });
}
