/**
 * GET /api/analysis/weeks?asset=<asset_id>
 * Returns the analysis index (list of available weeks) for an asset.
 * Reads public/data/analysis/<asset>/analysis_index.json (file-based),
 * falling back to the legacy root for the default asset.
 */

import { NextRequest, NextResponse } from 'next/server';

import type { AnalysisIndex } from '@/lib/contracts/weekly-analysis.contract';
import { readAnalysisJson } from '@/lib/analysis-paths';

const DEFAULT_INDEX: AnalysisIndex = { weeks: [] };

export async function GET(request: NextRequest) {
  const asset = request.nextUrl.searchParams.get('asset');
  const index = await readAnalysisJson<AnalysisIndex>(asset, 'analysis_index.json');
  return NextResponse.json(index ?? DEFAULT_INDEX);
}
