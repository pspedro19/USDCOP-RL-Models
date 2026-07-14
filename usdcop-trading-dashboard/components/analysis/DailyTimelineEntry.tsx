'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Newspaper, CalendarDays, ChevronDown, ChevronUp } from 'lucide-react';
import { DAY_NAMES_ES } from '@/lib/contracts/weekly-analysis.contract';
import type { DailyAnalysisEntry } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT, impactTone, sentimentTone } from './gm-analysis';
import { MacroEventChip } from './MacroEventChip';
import { AnalysisMarkdown } from './AnalysisMarkdown';

interface DailyTimelineEntryProps {
  entry: DailyAnalysisEntry;
  index: number;
  isLast: boolean;
}

const PREVIEW_CHARS = 300;

/** Timeline dot color per sentiment tone (prototype: Lun-Vie dots). */
const DOT_BG: Record<string, string> = {
  pos: 'bg-[var(--gm-pos)]',
  neg: 'bg-[var(--gm-neg)]',
  warn: 'bg-[var(--gm-warn)]',
  neutral: 'bg-[var(--gm-text-faint)]',
};

function getPreview(md: string): { preview: string; isTruncated: boolean } {
  if (!md || md.length <= PREVIEW_CHARS + 50) return { preview: md, isTruncated: false };

  // Cut at first paragraph break after PREVIEW_CHARS
  const idx = md.indexOf('\n\n', PREVIEW_CHARS);
  if (idx > 0 && idx < md.length - 30) {
    return { preview: md.slice(0, idx), isTruncated: true };
  }
  // Fallback: hard cut on a word boundary
  const cut = md.lastIndexOf(' ', PREVIEW_CHARS + 100);
  if (cut > PREVIEW_CHARS - 50) {
    return { preview: md.slice(0, cut) + '...', isTruncated: true };
  }
  return { preview: md, isTruncated: false };
}

export function DailyTimelineEntry({ entry, index, isLast }: DailyTimelineEntryProps) {
  const t = useGmT(ANALYSIS_DICT);
  const [expanded, setExpanded] = useState(false);
  const tone = sentimentTone(entry.sentiment);
  const dayName = DAY_NAMES_ES[entry.day_of_week] || `Dia ${entry.day_of_week}`;
  const isPositive = entry.usdcop_change_pct !== null && entry.usdcop_change_pct >= 0;

  const { preview, isTruncated } = entry.summary_markdown
    ? getPreview(entry.summary_markdown)
    : { preview: '', isTruncated: false };
  const displayMarkdown = expanded ? entry.summary_markdown : preview;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className="flex gap-4"
    >
      {/* Timeline connector */}
      <div className="flex flex-col items-center">
        <div className={`w-3 h-3 rounded-full mt-2 ${DOT_BG[tone]}`} />
        {!isLast && <div className="w-px flex-1 bg-[var(--gm-border)] mt-1" />}
      </div>

      {/* Content card */}
      <div className="flex-1 pb-6">
        <div className={`${GM.panel} gm-contain p-4 hover:border-[rgba(148,163,184,.24)] transition-colors duration-[var(--gm-dur-fast)]`}>
          {/* Day header */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <CalendarDays className={`w-4 h-4 ${GM.textMuted}`} />
              <span className={`${GMT.body} font-bold ${GM.textStrong} ${GMT.mono}`}>{dayName}</span>
              <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>{entry.analysis_date}</span>
            </div>
            {entry.sentiment && (
              <GmBadge tone={tone}>{entry.sentiment}</GmBadge>
            )}
          </div>

          {/* Headline */}
          {entry.headline && (
            <h4 className={`${GMT.body} font-medium ${GM.text} mb-2`}>{entry.headline}</h4>
          )}

          {/* Price bar */}
          {entry.usdcop_close !== null && (
            <div className={`flex items-center gap-3 mb-3 ${GM.panelSoft} px-3 py-2`}>
              <span className={`${GMT.body} ${GM.textSec}`}>USD/COP:</span>
              <span className={`${GMT.body} font-bold ${GM.textStrong} ${GMT.mono}`}>{entry.usdcop_close.toFixed(2)}</span>
              {entry.usdcop_change_pct !== null && (
                <span className={`flex items-center gap-1 ${GMT.meta} font-semibold ${GMT.mono} ${isPositive ? GM.pos : GM.neg}`}>
                  {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {isPositive ? '+' : ''}{entry.usdcop_change_pct.toFixed(2)}%
                </span>
              )}
              {entry.usdcop_high !== null && entry.usdcop_low !== null && (
                <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono} ml-auto`}>
                  {t('range')}: {entry.usdcop_low.toFixed(0)}–{entry.usdcop_high.toFixed(0)}
                </span>
              )}
            </div>
          )}

          {/* Macro publications */}
          {(entry.macro_publications?.length ?? 0) > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {(entry.macro_publications ?? []).map((pub, i) => (
                <MacroEventChip key={i} publication={pub} />
              ))}
            </div>
          )}

          {/* Economic events */}
          {(entry.economic_events?.length ?? 0) > 0 && (
            <div className="mb-3 space-y-1">
              {(entry.economic_events ?? []).slice(0, 3).map((evt, i) => {
                const evtTone = impactTone(evt.impact_level);
                return (
                  <div key={i} className={`flex items-center gap-2 ${GMT.meta}`}>
                    <span className={evtTone === 'neg' ? GM.neg : evtTone === 'warn' ? GM.warn : GM.textMuted}>
                      {evt.impact_level === 'low' ? '○' : '●'}
                    </span>
                    <span className={GM.textSec}>{evt.event}</span>
                    {evt.actual && <span className={`${GM.textStrong} font-semibold ${GMT.mono} ml-auto`}>{evt.actual}</span>}
                  </div>
                );
              })}
            </div>
          )}

          {/* News highlights */}
          {(entry.news_highlights?.length ?? 0) > 0 && (
            <div className="mb-3 space-y-1.5">
              {(entry.news_highlights ?? []).slice(0, 5).map((news, i) => (
                <div key={i} className={`flex items-start gap-2 ${GMT.meta}`}>
                  <Newspaper className={`w-3 h-3 ${GM.textFaint} mt-0.5 shrink-0`} />
                  <div className="flex-1 min-w-0">
                    {news.url ? (
                      <a
                        href={news.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className={`${GM.text} hover:text-[var(--gm-accent)] transition-colors duration-[var(--gm-dur-fast)] underline decoration-[rgba(148,163,184,.3)] hover:decoration-[rgba(34,211,238,.5)] ${GM.focus} rounded`}
                      >
                        {news.title}
                      </a>
                    ) : (
                      <span className={GM.textSec}>{news.title}</span>
                    )}
                  </div>
                  <span className={`${GMT.micro} px-1.5 py-0.5 rounded ${GM.neutralBadge} shrink-0 uppercase tracking-wide`}>
                    {news.source}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* AI Summary (truncated) */}
          {displayMarkdown && (
            <AnalysisMarkdown content={displayMarkdown} className="mt-2" />
          )}

          {/* Expand/collapse */}
          {isTruncated && (
            <button
              onClick={() => setExpanded(!expanded)}
              className={`mt-2 flex items-center gap-1 text-[11px] font-semibold ${GM.accent} hover:opacity-80 transition-opacity duration-[var(--gm-dur-fast)] ${GM.focus} rounded`}
            >
              {expanded ? (
                <><ChevronUp className="w-3 h-3" /> {t('showLess')}</>
              ) : (
                <><ChevronDown className="w-3 h-3" /> {t('showFullAnalysis')}</>
              )}
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}
