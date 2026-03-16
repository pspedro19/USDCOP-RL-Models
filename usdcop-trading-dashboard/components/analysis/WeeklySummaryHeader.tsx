'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Tag, Award, ChevronDown, ChevronUp } from 'lucide-react';
import { getSentimentColor } from '@/lib/contracts/weekly-analysis.contract';
import type { WeeklySummary, WeeklyTheme } from '@/lib/contracts/weekly-analysis.contract';
import { AnalysisMarkdown } from './AnalysisMarkdown';

interface WeeklySummaryHeaderProps {
  summary: WeeklySummary;
  qualityScore?: number;
  newsArticleCount?: number;
  clusterCount?: number;
  sourceBreakdown?: Record<string, number>;
}

/** Extract first section (up to second ## heading) from markdown. */
function truncateMarkdown(md: string): { preview: string; isTruncated: boolean } {
  if (!md || md.length < 600) return { preview: md, isTruncated: false };

  // Find second ## heading or first --- separator
  const lines = md.split('\n');
  let headingCount = 0;
  let cutoff = -1;

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].match(/^#{1,3}\s/)) {
      headingCount++;
      if (headingCount >= 3) {
        cutoff = i;
        break;
      }
    }
  }

  if (cutoff > 0) {
    const preview = lines.slice(0, cutoff).join('\n').trim();
    return { preview, isTruncated: true };
  }

  // Fallback: cut at ~500 chars on a paragraph boundary
  const idx = md.indexOf('\n\n', 500);
  if (idx > 0 && idx < md.length - 50) {
    return { preview: md.slice(0, idx), isTruncated: true };
  }

  return { preview: md, isTruncated: false };
}

export function WeeklySummaryHeader({ summary, qualityScore, newsArticleCount, clusterCount, sourceBreakdown }: WeeklySummaryHeaderProps) {
  const [expanded, setExpanded] = useState(false);
  const sentimentColor = getSentimentColor(summary.sentiment);
  const ohlcv = summary.ohlcv ?? {};
  const changePct = ohlcv.change_pct ?? 0;
  const isPositive = changePct >= 0;
  const hasOhlcv = ohlcv.open != null;

  const { preview, isTruncated } = truncateMarkdown(summary.markdown);
  const displayMarkdown = expanded ? summary.markdown : preview;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-6"
    >
      {/* Top row: headline + badges */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <h2 className="text-lg font-bold text-white flex-1">{summary.headline}</h2>
        <div className="flex items-center gap-2 shrink-0">
          {qualityScore !== undefined && qualityScore !== null && (
            <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-[10px] font-medium border ${
              qualityScore >= 0.8
                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                : qualityScore >= 0.6
                ? 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                : 'bg-red-500/15 text-red-400 border-red-500/30'
            }`}>
              <Award className="w-3 h-3" />
              {(qualityScore * 100).toFixed(0)}%
            </span>
          )}
          <span className={`px-3 py-1 rounded-full text-xs font-semibold uppercase ${sentimentColor.bg} ${sentimentColor.text} ${sentimentColor.border} border`}>
            {summary.sentiment}
          </span>
        </div>
      </div>

      {/* OHLCV row */}
      {hasOhlcv && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-5">
          <OHLCVCard label="Apertura" value={ohlcv.open} />
          <OHLCVCard label="Maximo" value={ohlcv.high} />
          <OHLCVCard label="Minimo" value={ohlcv.low} />
          <OHLCVCard label="Cierre" value={ohlcv.close} />
          <div className="bg-gray-800/40 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Cambio</p>
            <div className="flex items-center gap-1">
              {isPositive ? (
                <TrendingUp className="w-4 h-4 text-emerald-400" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-400" />
              )}
              <span className={`text-lg font-bold ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                {isPositive ? '+' : ''}{changePct.toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Themes */}
      {summary.themes.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {summary.themes.map((theme, i) => (
            <ThemeBadge key={i} theme={theme} />
          ))}
        </div>
      )}

      {/* News digest */}
      {(newsArticleCount !== undefined && newsArticleCount > 0) && (
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <span className="text-xs text-gray-500">{newsArticleCount} articulos</span>
          {clusterCount !== undefined && clusterCount > 0 && (
            <span className="text-xs text-gray-500">&middot; {clusterCount} clusters</span>
          )}
          {sourceBreakdown && Object.keys(sourceBreakdown).length > 0 && (
            <>
              <span className="text-xs text-gray-700">|</span>
              {Object.entries(sourceBreakdown).map(([src, count]) => (
                <span
                  key={src}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-gray-800/60 text-gray-400 border border-gray-700/40"
                >
                  {src.replace('_', ' ')}: {count}
                </span>
              ))}
            </>
          )}
        </div>
      )}

      {/* Markdown summary (truncated) */}
      <AnalysisMarkdown content={displayMarkdown} />

      {/* Expand/collapse button */}
      {isTruncated && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-3 flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 transition-colors font-medium"
        >
          {expanded ? (
            <>
              <ChevronUp className="w-3.5 h-3.5" />
              Ver menos
            </>
          ) : (
            <>
              <ChevronDown className="w-3.5 h-3.5" />
              Ver informe completo
            </>
          )}
        </button>
      )}
    </motion.div>
  );
}

function OHLCVCard({ label, value }: { label: string; value?: number }) {
  return (
    <div className="bg-gray-800/40 rounded-lg p-3">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className="text-lg font-bold text-white">{value != null ? value.toFixed(2) : '\u2014'}</p>
    </div>
  );
}

function ThemeBadge({ theme }: { theme: WeeklyTheme | string }) {
  // Handle both object format {theme, impact, description} and plain string
  const isObject = typeof theme === 'object' && theme !== null;
  const label = isObject ? theme.theme : String(theme);
  const impact = isObject ? theme.impact : 'neutral';
  const description = isObject ? theme.description : '';

  const impactColors: Record<string, string> = {
    positive: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
    negative: 'bg-red-500/15 text-red-400 border-red-500/30',
    neutral: 'bg-slate-500/15 text-slate-400 border-slate-500/30',
  };
  const color = impactColors[impact] || impactColors.neutral;

  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs border ${color}`} title={description}>
      <Tag className="w-3 h-3" />
      {label}
    </span>
  );
}
