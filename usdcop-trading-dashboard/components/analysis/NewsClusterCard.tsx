'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Newspaper, ExternalLink, ChevronDown, ChevronUp, Users } from 'lucide-react';
import type { NewsClusterOutput, NewsArticleRef } from '@/lib/contracts/weekly-analysis.contract';

interface NewsClusterCardProps {
  clusters: NewsClusterOutput[];
  maxVisible?: number;
}

const SENTIMENT_LABEL = (score: number) =>
  score > 0.15 ? 'Positivo' : score < -0.15 ? 'Negativo' : 'Neutral';

const SENTIMENT_COLOR = (score: number) =>
  score > 0.15
    ? 'text-emerald-400 bg-emerald-500/10'
    : score < -0.15
    ? 'text-red-400 bg-red-500/10'
    : 'text-gray-400 bg-gray-500/10';

export function NewsClusterCard({ clusters, maxVisible = 2 }: NewsClusterCardProps) {
  const [expanded, setExpanded] = useState(false);

  if (clusters.length === 0) return null;

  const visible = expanded ? clusters : clusters.slice(0, maxVisible);
  const hasMore = clusters.length > maxVisible;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Newspaper className="w-4 h-4 text-cyan-400" />
          Clusters de Noticias
        </h3>
        <span className="text-xs text-gray-500">
          {clusters.reduce((sum, c) => sum + c.article_count, 0)} articulos en {clusters.length} clusters
        </span>
      </div>

      <div className="space-y-3">
        {visible.map((cluster, i) => (
          <ClusterItem key={cluster.cluster_id ?? i} cluster={cluster} />
        ))}
      </div>

      {/* Expand/collapse */}
      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-3 flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
        >
          {expanded ? (
            <>
              <ChevronUp className="w-3 h-3" /> Mostrar menos
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3" /> Ver {clusters.length - maxVisible} mas
            </>
          )}
        </button>
      )}
    </motion.div>
  );
}

function ClusterItem({ cluster }: { cluster: NewsClusterOutput }) {
  const [showArticles, setShowArticles] = useState(false);
  const sentColor = SENTIMENT_COLOR(cluster.avg_sentiment);
  const sentLabel = SENTIMENT_LABEL(cluster.avg_sentiment);

  return (
    <div className="bg-gray-800/30 rounded-lg p-3">
      {/* Cluster header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white">
            {cluster.label || cluster.dominant_category || `Cluster #${cluster.cluster_id + 1}`}
          </span>
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${sentColor}`}>
            {sentLabel}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Users className="w-3 h-3" />
          {cluster.article_count}
        </div>
      </div>

      {/* Narrative summary */}
      {cluster.narrative_summary && (
        <p className="text-xs text-gray-400 mb-2 leading-relaxed">{cluster.narrative_summary}</p>
      )}

      {/* Representative titles */}
      <div className="space-y-1 mb-2">
        {cluster.representative_titles.slice(0, 3).map((title, i) => (
          <div key={i} className="flex items-start gap-1.5 text-xs">
            <span className="text-gray-600 mt-0.5 shrink-0">•</span>
            <span className="text-gray-400">{title}</span>
          </div>
        ))}
      </div>

      {/* Source articles (expandable) */}
      {cluster.articles.length > 0 && (
        <>
          <button
            onClick={() => setShowArticles(!showArticles)}
            className="text-[10px] text-gray-500 hover:text-gray-400 transition-colors"
          >
            {showArticles ? 'Ocultar' : 'Ver'} {cluster.articles.length} fuentes
          </button>

          <AnimatePresence>
            {showArticles && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-2 space-y-1.5 border-t border-gray-800/50 pt-2">
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
      {Object.keys(cluster.bias_distribution).length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {Object.entries(cluster.bias_distribution)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 3)
            .map(([bias, count]) => (
              <span
                key={bias}
                className="px-1.5 py-0.5 rounded bg-gray-800/60 text-[10px] text-gray-500"
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
    <div className="flex items-start gap-2 text-xs">
      <div className="flex-1 min-w-0">
        {article.url ? (
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-300 hover:text-cyan-400 transition-colors inline-flex items-center gap-1"
          >
            <span className="truncate">{article.title}</span>
            <ExternalLink className="w-2.5 h-2.5 shrink-0" />
          </a>
        ) : (
          <span className="text-gray-400 truncate block">{article.title}</span>
        )}
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-gray-600">{article.source}</span>
          <span className="text-gray-700">{article.date}</span>
          {article.bias_label && (
            <span className="text-gray-600 text-[10px]">[{article.bias_label}]</span>
          )}
        </div>
      </div>
    </div>
  );
}
