/**
 * GET /api/analysis/calendar?asset=<asset_id>
 * Returns upcoming economic events for the next 7 days for an asset.
 * Reads public/data/analysis/<asset>/upcoming_events.json (file-based),
 * falling back to the legacy root for the default asset.
 */

import { NextRequest, NextResponse } from 'next/server';

import type { EconomicEvent } from '@/lib/contracts/weekly-analysis.contract';
import { readAnalysisJson } from '@/lib/analysis-paths';

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
  const data = await readAnalysisJson<UpcomingEventsResponse>(asset, 'upcoming_events.json');
  return NextResponse.json(data ?? DEFAULT_RESPONSE);
}
