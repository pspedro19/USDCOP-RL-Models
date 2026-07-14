'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Tag, Award, ChevronDown, ChevronUp } from 'lucide-react';
import type { WeeklySummary, WeeklyTheme } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, GM_TONE_TEXT, type GmTone } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT, sentimentTone } from './gm-analysis';
import { AnalysisMarkdown } from './AnalysisMarkdown';

interface WeeklySummaryHeaderProps {
  summary: WeeklySummary;
  qualityScore?: number;
  newsArticleCount?: number;
  clusterCount?: number;
  sourceBreakdown?: Record<string, number>;
  /**
   * Suppress the markdown body + expand button (KPIs/themes/news digest still show).
   * Set when a richer `synthesis_markdown` is rendered separately (SynthesisCard) so
   * the same report isn't shown twice. Defaults to false → legacy behavior preserved.
   */
  hideMarkdown?: boolean;
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

export function WeeklySummaryHeader({ summary, qualityScore, newsArticleCount, clusterCount, sourceBreakdown, hideMarkdown = false }: WeeklySummaryHeaderProps) {
  const t = useGmT(ANALYSIS_DICT);
  const [expanded, setExpanded] = useState(false);
  const tone = sentimentTone(summary.sentiment);
  const ohlcv = summary.ohlcv ?? {};
  const changePct = ohlcv.change_pct ?? 0;
  const isPositive = changePct >= 0;
  const hasOhlcv = ohlcv.open != null;

  const { preview, isTruncated } = truncateMarkdown(summary.markdown);
  const displayMarkdown = expanded ? summary.markdown : preview;

  const sentimentLabel =
    tone === 'pos' ? t('bullish') : tone === 'neg' ? t('bearish') : tone === 'warn' ? t('mixed') : t('neutral');

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-6`}
    >
      {/* Top row: headline + badges */}
      <div className="flex items-start justify-between gap-4 mb-4">
        <h2 className={`${GMT.h2} ${GM.headline} flex-1`}>{summary.headline}</h2>
        <div className="flex items-center gap-2 shrink-0">
          {qualityScore !== undefined && qualityScore !== null && (
            <GmBadge
              tone={qualityScore >= 0.8 ? 'pos' : qualityScore >= 0.6 ? 'warn' : 'neg'}
              className={GMT.mono}
            >
              <Award className="w-3 h-3" />
              {(qualityScore * 100).toFixed(0)}%
            </GmBadge>
          )}
          <GmBadge tone={tone}>{sentimentLabel}</GmBadge>
        </div>
      </div>

      {/* OHLCV row — prototype "Recap semana anterior" (label + 18px mono figure) */}
      {hasOhlcv && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-5">
          <OHLCVCard label={t('open')} value={ohlcv.open} />
          <OHLCVCard label={t('high')} value={ohlcv.high} tone="pos" />
          <OHLCVCard label={t('low')} value={ohlcv.low} tone="neg" />
          <OHLCVCard label={t('close')} value={ohlcv.close} />
          <div className={`${GM.panelSoft} p-3`}>
            <p className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{t('change')}</p>
            <div className="flex items-center gap-1">
              {isPositive ? (
                <TrendingUp className={`w-4 h-4 ${GM.pos}`} />
              ) : (
                <TrendingDown className={`w-4 h-4 ${GM.neg}`} />
              )}
              <span className={`text-lg font-extrabold ${GMT.mono} ${isPositive ? GM.pos : GM.neg}`}>
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
          <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>{newsArticleCount} {t('articles')}</span>
          {clusterCount !== undefined && clusterCount > 0 && (
            <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>&middot; {clusterCount} {t('clustersWord')}</span>
          )}
          {sourceBreakdown && Object.keys(sourceBreakdown).length > 0 && (
            <>
              <span className={`${GMT.meta} ${GM.textFaint}`}>|</span>
              {Object.entries(sourceBreakdown).map(([src, count]) => (
                <GmBadge key={src} tone="neutral" className="normal-case tracking-normal font-semibold">
                  {src.replace('_', ' ')}: {count}
                </GmBadge>
              ))}
            </>
          )}
        </div>
      )}

      {/* Markdown summary (truncated) — suppressed when a richer synthesis renders separately */}
      {!hideMarkdown && <AnalysisMarkdown content={displayMarkdown} />}

      {/* Expand/collapse button */}
      {!hideMarkdown && isTruncated && (
        <button
          onClick={() => setExpanded(!expanded)}
          className={`mt-3 flex items-center gap-1.5 ${GMT.meta} font-semibold ${GM.accent} hover:opacity-80 transition-opacity duration-[var(--gm-dur-fast)] ${GM.focus} rounded`}
        >
          {expanded ? (
            <>
              <ChevronUp className="w-3.5 h-3.5" />
              {t('showLess')}
            </>
          ) : (
            <>
              <ChevronDown className="w-3.5 h-3.5" />
              {t('showFullReport')}
            </>
          )}
        </button>
      )}
    </motion.div>
  );
}

function OHLCVCard({ label, value, tone }: { label: string; value?: number; tone?: GmTone }) {
  return (
    <div className={`${GM.panelSoft} p-3`}>
      <p className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{label}</p>
      <p className={`text-lg font-extrabold ${GMT.mono} ${tone ? GM_TONE_TEXT[tone] : GM.text}`}>
        {value != null ? value.toFixed(2) : '—'}
      </p>
    </div>
  );
}

function ThemeBadge({ theme }: { theme: WeeklyTheme | string }) {
  // Handle both object format {theme, impact, description} and plain string
  const isObject = typeof theme === 'object' && theme !== null;
  const label = isObject ? theme.theme : String(theme);
  const impact = isObject ? theme.impact : 'neutral';
  const description = isObject ? theme.description : '';

  return (
    <GmBadge tone={sentimentTone(impact)} className="normal-case tracking-normal font-semibold">
      <span title={description} className="inline-flex items-center gap-1">
        <Tag className="w-3 h-3" />
        {label}
      </span>
    </GmBadge>
  );
}
