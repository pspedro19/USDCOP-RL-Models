/**
 * GET /api/analysis/news-feed?limit=<n>&asset=<asset_id?>
 *
 * Cross-asset latest-headlines feed for the global news bell. Reads the latest
 * weekly analysis JSON per analysed asset and flattens its real news_intelligence
 * clusters into a merged, most-recent-first headline list. File-based (no DB) and
 * gated by middleware (deny-by-default: authenticated session required, per RBAC).
 */

import { NextRequest, NextResponse } from 'next/server';

import { readAnalysisJson } from '@/lib/analysis-paths';
import { ANALYSIS_ASSETS, getAnalysisAsset } from '@/lib/contracts/analysis-assets';
import type {
  AnalysisIndex,
  NewsArticleRef,
  NewsClusterOutput,
} from '@/lib/contracts/weekly-analysis.contract';
import type {
  NewsFeedAssetSummary,
  NewsFeedItem,
  NewsFeedResponse,
} from '@/lib/contracts/news-feed.contract';

interface WeeklyLite {
  news_intelligence?: {
    clusters?: NewsClusterOutput[];
    top_stories?: NewsArticleRef[];
    avg_sentiment?: number;
  };
  news_context?: { article_count?: number; avg_sentiment?: number | null };
}

/** Latest week file (has_weekly) for an asset, newest-first. */
function latestWeekFile(index: AnalysisIndex | null): { file: string; label: string } | null {
  const weeks = (index?.weeks ?? []).filter((w) => w.has_weekly);
  if (weeks.length === 0) return null;
  const w = [...weeks].sort((a, b) => b.year - a.year || b.week - a.week)[0];
  const wk = String(w.week).padStart(2, '0');
  return { file: `weekly_${w.year}_W${wk}.json`, label: `${w.year}-W${wk}` };
}

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const limit = Math.min(Math.max(Number(sp.get('limit')) || 12, 1), 50);
  const assetFilter = sp.get('asset');
  const assets = assetFilter
    ? [getAnalysisAsset(assetFilter)]
    : ANALYSIS_ASSETS;

  const items: NewsFeedItem[] = [];
  const byAsset: NewsFeedAssetSummary[] = [];

  for (const asset of assets) {
    const index = await readAnalysisJson<AnalysisIndex>(asset.asset_id, 'analysis_index.json');
    const latest = latestWeekFile(index);
    if (!latest) continue;

    const week = await readAnalysisJson<WeeklyLite>(asset.asset_id, latest.file);
    const ni = week?.news_intelligence;
    if (!ni) continue;

    byAsset.push({
      asset_id: asset.asset_id,
      symbol: asset.symbol,
      display_name: asset.display_name,
      week_label: latest.label,
      article_count: week?.news_context?.article_count ?? 0,
      avg_sentiment:
        ni.avg_sentiment ?? week?.news_context?.avg_sentiment ?? null,
    });

    // Prefer cluster articles (carry a theme label); fall back to top_stories.
    const seen = new Set<string>();
    const pushArticle = (a: NewsArticleRef, clusterLabel?: string) => {
      const key = (a.title || '').trim().toLowerCase();
      if (!key || seen.has(key)) return;
      seen.add(key);
      items.push({
        asset_id: asset.asset_id,
        symbol: asset.symbol,
        display_name: asset.display_name,
        title: a.title,
        url: a.url,
        source: a.source,
        date: a.date || '',
        tone: typeof a.tone === 'number' ? a.tone : 0,
        category: a.category,
        cluster_label: clusterLabel,
      });
    };

    if (Array.isArray(ni.clusters) && ni.clusters.length > 0) {
      for (const c of ni.clusters) {
        for (const a of c.articles ?? []) pushArticle(a, c.label);
      }
    }
    for (const a of ni.top_stories ?? []) pushArticle(a);
  }

  // Most-recent-first; break ties by sentiment magnitude (more impactful first).
  items.sort(
    (a, b) =>
      (b.date || '').localeCompare(a.date || '') ||
      Math.abs(b.tone) - Math.abs(a.tone),
  );

  const body: NewsFeedResponse = {
    items: items.slice(0, limit),
    by_asset: byAsset,
    generated_at: new Date().toISOString(),
  };
  return NextResponse.json(body);
}
