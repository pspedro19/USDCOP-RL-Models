/**
 * analysis-paths — server-side resolver for per-asset analysis data files.
 *
 * On-disk layout (namespaced by asset):
 *   public/data/analysis/<asset_id>/analysis_index.json
 *   public/data/analysis/<asset_id>/weekly_YYYY_WXX.json
 *   public/data/analysis/<asset_id>/upcoming_events.json
 *
 * Legacy USD/COP files live un-namespaced at public/data/analysis/*.json. For the
 * default asset we transparently fall back to that root, so nothing breaks while
 * the per-asset directories are populated.
 *
 * All resolution goes through resolveAnalysisAsset() so a caller can never escape
 * the analysis directory (no path traversal): unknown ids collapse to the default.
 */

import fs from 'fs/promises';
import path from 'path';

import { DEFAULT_ANALYSIS_ASSET, resolveAnalysisAsset } from '@/lib/contracts/analysis-assets';

const ANALYSIS_ROOT = path.join(process.cwd(), 'public', 'data', 'analysis');

/** Directory for an asset's analysis files (validated, path-safe). */
export function analysisDir(asset: string | null | undefined): string {
  return path.join(ANALYSIS_ROOT, resolveAnalysisAsset(asset));
}

/** Legacy un-namespaced root — only used as a fallback for the default asset. */
function legacyRoot(): string {
  return ANALYSIS_ROOT;
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Resolve a filename for an asset, preferring the namespaced dir and falling back
 * to the legacy root for the default asset. Returns null if neither exists.
 */
export async function resolveAnalysisFile(
  asset: string | null | undefined,
  filename: string,
): Promise<string | null> {
  const resolved = resolveAnalysisAsset(asset);
  const namespaced = path.join(analysisDir(resolved), filename);
  if (await fileExists(namespaced)) return namespaced;

  if (resolved === DEFAULT_ANALYSIS_ASSET) {
    const legacy = path.join(legacyRoot(), filename);
    if (await fileExists(legacy)) return legacy;
  }
  return null;
}

/** Read + JSON.parse an asset analysis file, or return null if missing/invalid. */
export async function readAnalysisJson<T>(
  asset: string | null | undefined,
  filename: string,
): Promise<T | null> {
  const filepath = await resolveAnalysisFile(asset, filename);
  if (!filepath) return null;
  try {
    const raw = await fs.readFile(filepath, 'utf-8');
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}
