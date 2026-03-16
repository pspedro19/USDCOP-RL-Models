'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Newspaper, CalendarDays, ChevronDown, ChevronUp } from 'lucide-react';
import { getSentimentColor, DAY_NAMES_ES } from '@/lib/contracts/weekly-analysis.contract';
import type { DailyAnalysisEntry } from '@/lib/contracts/weekly-analysis.contract';
import { MacroEventChip } from './MacroEventChip';
import { AnalysisMarkdown } from './AnalysisMarkdown';

interface DailyTimelineEntryProps {
  entry: DailyAnalysisEntry;
  index: number;
  isLast: boolean;
}

const PREVIEW_CHARS = 300;

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
  const [expanded, setExpanded] = useState(false);
  const sentimentColor = getSentimentColor(entry.sentiment);
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
        <div className={`w-3 h-3 rounded-full mt-2 ${sentimentColor.bg} border-2 ${sentimentColor.border}`} />
        {!isLast && <div className="w-px flex-1 bg-gray-800/60 mt-1" />}
      </div>

      {/* Content card */}
      <div className="flex-1 pb-6">
        <div className="bg-gray-900/40 backdrop-blur-sm rounded-xl border border-gray-800/40 p-4 hover:border-gray-700/60 transition-all">
          {/* Day header */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <CalendarDays className="w-4 h-4 text-gray-500" />
              <span className="text-sm font-semibold text-white">{dayName}</span>
              <span className="text-xs text-gray-500">{entry.analysis_date}</span>
            </div>
            {entry.sentiment && (
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium uppercase ${sentimentColor.bg} ${sentimentColor.text}`}>
                {entry.sentiment}
              </span>
            )}
          </div>

          {/* Headline */}
          {entry.headline && (
            <h4 className="text-sm font-medium text-gray-200 mb-2">{entry.headline}</h4>
          )}

          {/* Price bar */}
          {entry.usdcop_close !== null && (
            <div className="flex items-center gap-3 mb-3 bg-gray-800/30 rounded-lg px-3 py-2">
              <span className="text-sm text-gray-400">USD/COP:</span>
              <span className="text-sm font-bold text-white">{entry.usdcop_close.toFixed(2)}</span>
              {entry.usdcop_change_pct !== null && (
                <span className={`flex items-center gap-1 text-xs font-medium ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {isPositive ? '+' : ''}{entry.usdcop_change_pct.toFixed(2)}%
                </span>
              )}
              {entry.usdcop_high !== null && entry.usdcop_low !== null && (
                <span className="text-xs text-gray-600 ml-auto">
                  Rango: {entry.usdcop_low.toFixed(0)}-{entry.usdcop_high.toFixed(0)}
                </span>
              )}
            </div>
          )}

          {/* Macro publications */}
          {entry.macro_publications.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {entry.macro_publications.map((pub, i) => (
                <MacroEventChip key={i} publication={pub} />
              ))}
            </div>
          )}

          {/* Economic events */}
          {entry.economic_events.length > 0 && (
            <div className="mb-3 space-y-1">
              {entry.economic_events.slice(0, 3).map((evt, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  <span className={
                    evt.impact_level === 'high' ? 'text-red-400' :
                    evt.impact_level === 'medium' ? 'text-amber-400' : 'text-gray-500'
                  }>
                    {evt.impact_level === 'high' ? '\u25cf' : evt.impact_level === 'medium' ? '\u25cf' : '\u25cb'}
                  </span>
                  <span className="text-gray-400">{evt.event}</span>
                  {evt.actual && <span className="text-white font-medium ml-auto">{evt.actual}</span>}
                </div>
              ))}
            </div>
          )}

          {/* News highlights */}
          {entry.news_highlights.length > 0 && (
            <div className="mb-3 space-y-1.5">
              {entry.news_highlights.slice(0, 5).map((news, i) => (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <Newspaper className="w-3 h-3 text-gray-600 mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    {news.url ? (
                      <a
                        href={news.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-gray-300 hover:text-cyan-400 transition-colors underline decoration-gray-700 hover:decoration-cyan-400/50"
                      >
                        {news.title}
                      </a>
                    ) : (
                      <span className="text-gray-400">{news.title}</span>
                    )}
                  </div>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800/60 text-gray-500 shrink-0 uppercase tracking-wide">
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
              className="mt-2 flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300 transition-colors font-medium"
            >
              {expanded ? (
                <><ChevronUp className="w-3 h-3" /> Ver menos</>
              ) : (
                <><ChevronDown className="w-3 h-3" /> Ver analisis completo</>
              )}
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
}
