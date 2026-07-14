'use client';

import { useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Newspaper, ExternalLink, ChevronDown, ChevronUp, Users } from 'lucide-react';
import type { NewsClusterOutput, NewsArticleRef } from '@/lib/contracts/weekly-analysis.contract';
import { useGmLang, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, type GmTone } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface NewsClusterCardProps {
  clusters: NewsClusterOutput[];
  maxVisible?: number;
}

const sentimentToneOf = (score: number): GmTone =>
  score > 0.15 ? 'pos' : score < -0.15 ? 'neg' : 'neutral';

const themeOf = (c: NewsClusterOutput): string =>
  c.label || c.dominant_category || `#${c.cluster_id + 1}`;

export function NewsClusterCard({ clusters, maxVisible = 2 }: NewsClusterCardProps) {
  const t = useGmT(ANALYSIS_DICT);
  const lang = useGmLang();
  const [expanded, setExpanded] = useState(false);
  const [theme, setTheme] = useState('all');
  const [sent, setSent] = useState('all');

  const L = (es: string, en: string) => (lang === 'es' ? es : en);

  const themeOptions = useMemo(
    () => Array.from(new Set(clusters.map(themeOf))).filter(Boolean),
    [clusters],
  );

  const filtered = useMemo(
    () =>
      clusters.filter((c) => {
        if (theme !== 'all' && themeOf(c) !== theme) return false;
        if (sent !== 'all' && sentimentToneOf(c.avg_sentiment) !== sent) return false;
        return true;
      }),
    [clusters, theme, sent],
  );

  if (clusters.length === 0) return null;

  const visible = expanded ? filtered : filtered.slice(0, maxVisible);
  const hasMore = filtered.length > maxVisible;
  const selectCls = `${GMT.meta} ${GM.textSec} bg-[var(--gm-surface)] border border-[var(--gm-border)] rounded px-2 py-1 ${GM.focus} cursor-pointer`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-5`}
    >
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
          <Newspaper className={`w-4 h-4 ${GM.accent}`} />
          {t('newsClusters')}
        </h3>
        <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
          {clusters.reduce((sum, c) => sum + (c.article_count ?? 0), 0)} {t('articlesIn')} {clusters.length} {t('clustersWord')}
        </span>
      </div>

      {/* Filters — theme + sentiment dropdowns (browse the news) */}
      <div className="flex items-center gap-2 mb-4 flex-wrap">
        <select
          aria-label={L('Filtrar por tema', 'Filter by theme')}
          value={theme}
          onChange={(e) => { setTheme(e.target.value); setExpanded(true); }}
          className={selectCls}
        >
          <option value="all">{L('Todos los temas', 'All themes')}</option>
          {themeOptions.map((th) => (
            <option key={th} value={th}>{th}</option>
          ))}
        </select>
        <select
          aria-label={L('Filtrar por sentimiento', 'Filter by sentiment')}
          value={sent}
          onChange={(e) => { setSent(e.target.value); setExpanded(true); }}
          className={selectCls}
        >
          <option value="all">{L('Todo sentimiento', 'All sentiment')}</option>
          <option value="pos">{t('positive')}</option>
          <option value="neutral">{t('neutral')}</option>
          <option value="neg">{t('negative')}</option>
        </select>
        {(theme !== 'all' || sent !== 'all') && (
          <button
            onClick={() => { setTheme('all'); setSent('all'); }}
            className={`${GMT.meta} ${GM.accent} hover:opacity-80 ${GM.focus} rounded px-1`}
          >
            {L('Limpiar', 'Clear')}
          </button>
        )}
      </div>

      {filtered.length === 0 ? (
        <p className={`${GMT.meta} ${GM.textMuted} py-4 text-center`}>
          {L('Sin noticias para este filtro.', 'No news match this filter.')}
        </p>
      ) : (
        <div className="space-y-3">
          {visible.map((cluster, i) => (
            <ClusterItem key={cluster.cluster_id ?? i} cluster={cluster} />
          ))}
        </div>
      )}

      {/* Expand/collapse */}
      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className={`mt-3 flex items-center gap-1 ${GMT.meta} font-semibold ${GM.accent} hover:opacity-80 transition-opacity duration-[var(--gm-dur-fast)] ${GM.focus} rounded`}
        >
          {expanded ? (
            <>
              <ChevronUp className="w-3 h-3" /> {t('showLess')}
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3" /> {t('showMorePrefix')} {filtered.length - maxVisible} {t('morePlural')}
            </>
          )}
        </button>
      )}
    </motion.div>
  );
}

function ClusterItem({ cluster }: { cluster: NewsClusterOutput }) {
  const t = useGmT(ANALYSIS_DICT);
  const [showArticles, setShowArticles] = useState(false);
  const tone = sentimentToneOf(cluster.avg_sentiment);
  const sentLabel = tone === 'pos' ? t('positive') : tone === 'neg' ? t('negative') : t('neutral');

  return (
    <div className={`${GM.panelSoft} p-3`}>
      {/* Cluster header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`${GMT.body} font-semibold ${GM.textStrong}`}>
            {cluster.label || cluster.dominant_category || `Cluster #${cluster.cluster_id + 1}`}
          </span>
          <GmBadge tone={tone}>{sentLabel}</GmBadge>
        </div>
        <div className={`flex items-center gap-2 ${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
          <Users className="w-3 h-3" />
          {cluster.article_count}
        </div>
      </div>

      {/* Narrative summary */}
      {cluster.narrative_summary && (
        <p className={`${GMT.meta} ${GM.textSec} mb-2 leading-relaxed`}>{cluster.narrative_summary}</p>
      )}

      {/* Representative titles */}
      <div className="space-y-1 mb-2">
        {(cluster.representative_titles ?? []).slice(0, 3).map((title, i) => (
          <div key={i} className={`flex items-start gap-1.5 ${GMT.meta}`}>
            <span className={`${GM.textFaint} mt-0.5 shrink-0`}>•</span>
            <span className={GM.textSec}>{title}</span>
          </div>
        ))}
      </div>

      {/* Source articles (expandable) */}
      {(cluster.articles?.length ?? 0) > 0 && (
        <>
          <button
            onClick={() => setShowArticles(!showArticles)}
            className={`${GMT.micro} ${GM.textMuted} hover:text-[var(--gm-text-sec)] transition-colors duration-[var(--gm-dur-fast)] ${GM.focus} rounded`}
          >
            {showArticles ? t('hide') : t('view')} {cluster.articles.length} {t('sourcesWord')}
          </button>

          <AnimatePresence>
            {showArticles && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1.5 border-t border-[var(--gm-border)] pt-2">
                  {cluster.articles.slice(0, 5).map((article, i) => (
                    <ArticleRow key={i} article={article} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}

      {/* Bias distribution */}
      {Object.keys(cluster.bias_distribution ?? {}).length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {Object.entries(cluster.bias_distribution ?? {})
            .sort(([, a], [, b]) => b - a)
            .slice(0, 3)
            .map(([bias, count]) => (
              <span
                key={bias}
                className={`px-1.5 py-0.5 rounded ${GM.neutralBadge} ${GMT.micro}`}
              >
                {bias}: {count}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}

function ArticleRow({ article }: { article: NewsArticleRef }) {
  return (
    <div className={`flex items-start gap-2 ${GMT.meta}`}>
      <div className="flex-1 min-w-0">
        {article.url ? (
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className={`${GM.text} hover:text-[var(--gm-accent)] transition-colors duration-[var(--gm-dur-fast)] inline-flex items-center gap-1 ${GM.focus} rounded`}
          >
            <span className="truncate">{article.title}</span>
            <ExternalLink className="w-2.5 h-2.5 shrink-0" />
          </a>
        ) : (
          <span className={`${GM.textSec} truncate block`}>{article.title}</span>
        )}
        <div className="flex items-center gap-2 mt-0.5">
          <span className={GM.textMuted}>{article.source}</span>
          <span className={`${GM.textFaint} ${GMT.mono}`}>{article.date}</span>
          {article.bias_label && (
            <span className={`${GM.textMuted} ${GMT.micro}`}>[{article.bias_label}]</span>
          )}
        </div>
      </div>
    </div>
  );
}
