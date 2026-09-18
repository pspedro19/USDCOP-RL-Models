/**
 * GET /api/analysis/calendar?asset=<asset_id>
 * Returns upcoming economic events for the next 7 days for an asset.
 * Reads public/data/analysis/<asset>/upcoming_events.json (file-based),
 * falling back to the legacy root for the default asset.
 *
 * PLAN-GATED (CTR-RBAC-001 R3): `?asset=` is monetized. The event list is undated as a
 * whole, so only the asset scope is enforced here — there is no per-file delay to apply.
 */

import { NextRequest, NextResponse } from 'next/server';

import type { EconomicEvent } from '@/lib/contracts/weekly-analysis.contract';
import { readAnalysisJson } from '@/lib/analysis-paths';
import { gateAnalysisAccess } from '@/lib/auth/analysis-gate';

interface UpcomingEventsResponse {
  events: EconomicEvent[];
  generated_at: string | null;
}

const DEFAULT_RESPONSE: UpcomingEventsResponse = {
  events: [],
  generated_at: null,
};

export async function GET(request: NextRequest) {
  const asset = request.nextUrl.searchParams.get('asset');

  const denied = await gateAnalysisAccess(request.headers.get('x-user-id'), asset);
  if (denied) return denied;

  const data = await readAnalysisJson<UpcomingEventsResponse>(asset, 'upcoming_events.json');
  return NextResponse.json(data ?? DEFAULT_RESPONSE);
}
