/**
 * GET /api/analysis/assets
 * Returns the list of assets that have weekly/daily analysis, from the SSOT.
 * Drives the frontend asset selector dynamically — adding an asset to the SSOT
 * makes it appear here with no route/component change.
 */

import { NextResponse } from 'next/server';

import { ANALYSIS_ASSETS, DEFAULT_ANALYSIS_ASSET } from '@/lib/contracts/analysis-assets';

export async function GET() {
  return NextResponse.json({
    assets: ANALYSIS_ASSETS,
    default_asset: DEFAULT_ANALYSIS_ASSET,
  });
}
