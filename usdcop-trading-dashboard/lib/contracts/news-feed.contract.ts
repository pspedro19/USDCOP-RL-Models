/**
 * news-feed.contract — cross-asset latest-headlines feed for the global news bell.
 *
 * Backed by GET /api/analysis/news-feed, which reads the latest weekly analysis
 * JSON per asset (public/data/analysis/<asset>/weekly_*.json) and flattens the
 * real news_intelligence clusters into a merged, most-recent-first headline list.
 * No new data source — it reuses the enriched news blocks (CTR-NEWS-ENRICH-001).
 */

export interface NewsFeedItem {
  asset_id: string;
  symbol: string;
  display_name: string;
  title: string;
  url?: string;
  source: string;
  date: string;        // YYYY-MM-DD
  tone: number;        // -1..+1
  category?: string;
  cluster_label?: string;
}

export interface NewsFeedAssetSummary {
  asset_id: string;
  symbol: string;
  display_name: string;
  week_label: string;          // e.g. "2026-W27"
  article_count: number;
  avg_sentiment: number | null;
}

export interface NewsFeedResponse {
  items: NewsFeedItem[];
  by_asset: NewsFeedAssetSummary[];
  generated_at: string;
}
