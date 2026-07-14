'use client';

import { ChevronLeft, ChevronRight, Calendar } from 'lucide-react';
import { motion } from 'framer-motion';
import type { AnalysisWeekEntry } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT } from './gm-analysis';

interface WeekSelectorProps {
  weeks: AnalysisWeekEntry[];
  currentYear: number;
  currentWeek: number;
  onSelect: (year: number, week: number) => void;
}

export function WeekSelector({ weeks, currentYear, currentWeek, onSelect }: WeekSelectorProps) {
  const t = useGmT(ANALYSIS_DICT);
  const currentIndex = weeks.findIndex(w => w.year === currentYear && w.week === currentWeek);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < weeks.length - 1;

  const goPrev = () => {
    if (hasPrev) {
      const prev = weeks[currentIndex - 1];
      onSelect(prev.year, prev.week);
    }
  };

  const goNext = () => {
    if (hasNext) {
      const next = weeks[currentIndex + 1];
      onSelect(next.year, next.week);
    }
  };

  const currentEntry = weeks.find(w => w.year === currentYear && w.week === currentWeek);

  return (
    <div className="flex items-center justify-between gap-4">
      {/* Prev */}
      <button
        onClick={goPrev}
        disabled={!hasPrev}
        aria-label={t('prevWeek')}
        className={`${GM.ctaGhost} ${GM.focus} p-2.5 disabled:opacity-30 disabled:cursor-not-allowed`}
      >
        <ChevronLeft className={`w-5 h-5 ${GM.text}`} />
      </button>

      {/* Center label */}
      <motion.div
        key={`${currentYear}-${currentWeek}`}
        initial={{ opacity: 0, y: -5 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex-1 text-center"
      >
        <div className="flex items-center justify-center gap-2 mb-1">
          <Calendar className={`w-4 h-4 ${GM.accent}`} />
          <span className={`text-xl font-bold ${GM.textStrong} ${GMT.mono}`}>
            {t('weekWord')} {currentWeek}
          </span>
          <span className={`${GMT.body} ${GM.textMuted} ${GMT.mono}`}>{currentYear}</span>
        </div>
        {currentEntry && (
          <p className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
            {currentEntry.start} — {currentEntry.end}
            {currentEntry.daily_count > 0 && (
              <span className={`ml-2 ${GM.textFaint}`}>({currentEntry.daily_count} {t('days')})</span>
            )}
          </p>
        )}
      </motion.div>

      {/* Next */}
      <button
        onClick={goNext}
        disabled={!hasNext}
        aria-label={t('nextWeek')}
        className={`${GM.ctaGhost} ${GM.focus} p-2.5 disabled:opacity-30 disabled:cursor-not-allowed`}
      >
        <ChevronRight className={`w-5 h-5 ${GM.text}`} />
      </button>

      {/* Jump dropdown */}
      {weeks.length > 3 && (
        <select
          value={`${currentYear}-${currentWeek}`}
          aria-label={t('jumpToWeek')}
          onChange={(e) => {
            const [y, w] = e.target.value.split('-').map(Number);
            onSelect(y, w);
          }}
          className={`ml-2 ${GM.input} ${GM.focus} ${GMT.mono}`}
        >
          {weeks.map(w => (
            <option key={`${w.year}-${w.week}`} value={`${w.year}-${w.week}`}>
              {w.year}-W{String(w.week).padStart(2, '0')}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
