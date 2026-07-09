/**
 * News-cluster normalization adapter
 * ==================================
 *
 * SSOT reconciliation at the frontend boundary. The USD/COP news engine emits
 * clusters that already match `NewsClusterOutput`, but the Gold/BTC analysis
 * generator emits a leaner shape (`{ theme, count, articles: [{ title, url }] }`).
 * Rendering that lean shape directly crashes `NewsClusterCard`
 * (`cluster.representative_titles.slice()` / `Object.keys(cluster.bias_distribution)`
 * on `undefined`), which the ErrorBoundary surfaces as "Something went wrong".
 *
 * This pure adapter maps ANY stored cluster shape onto the `NewsClusterOutput`
 * contract, so presentational components can trust the contract (DRY: one place,
 * SRP: normalization only, OCP: components stay unchanged). It is defensive by
 * construction — missing/optional fields degrade to safe defaults, never throw.
 */

import type {
  NewsArticleRef,
  NewsClusterOutput,
} from '@/lib/contracts/weekly-analysis.contract';

/** Loose stored shapes we may receive before normalization. */
type RawArticle = Partial<NewsArticleRef> & { title?: string; url?: string };
type RawCluster = Partial<NewsClusterOutput> & {
  theme?: string;
  count?: number;
  articles?: RawArticle[];
};

function normalizeArticle(raw: RawArticle, fallbackSource: string): NewsArticleRef {
  return {
    title: raw.title ?? '',
    url: raw.url,
    source: raw.source ?? fallbackSource,
    date: raw.date ?? '',
    tone: raw.tone ?? 0,
    category: raw.category,
    bias_label: raw.bias_label,
    factuality: raw.factuality,
  };
}

/** Map a single raw cluster onto the `NewsClusterOutput` contract. */
export function normalizeNewsCluster(raw: RawCluster, index: number): NewsClusterOutput {
  const articles = Array.isArray(raw.articles) ? raw.articles : [];
  const theme = raw.theme ?? '';

  // representative_titles: prefer the field; otherwise derive from article titles.
  const representativeTitles =
    Array.isArray(raw.representative_titles) && raw.representative_titles.length > 0
      ? raw.representative_titles
      : articles
          .map((a) => a.title)
          .filter((t): t is string => Boolean(t))
          .slice(0, 3);

  return {
    cluster_id: raw.cluster_id ?? index,
    label: raw.label ?? theme ?? `Cluster #${index + 1}`,
    article_count: raw.article_count ?? raw.count ?? articles.length,
    avg_sentiment: raw.avg_sentiment ?? 0,
    dominant_category: raw.dominant_category ?? theme,
    bias_distribution: raw.bias_distribution ?? {},
    representative_titles: representativeTitles,
    narrative_summary: raw.narrative_summary ?? '',
    articles: articles.map((a) => normalizeArticle(a, theme)),
  };
}

/** Map a raw cluster array onto contract-conforming clusters (tolerates non-arrays). */
export function normalizeNewsClusters(raw: unknown): NewsClusterOutput[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((c, i) => normalizeNewsCluster(c as RawCluster, i));
}
