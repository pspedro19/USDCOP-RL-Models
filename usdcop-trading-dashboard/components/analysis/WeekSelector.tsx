'use client';

import { ChevronLeft, ChevronRight, Calendar } from 'lucide-react';
import { motion } from 'framer-motion';
import type { AnalysisWeekEntry } from '@/lib/contracts/weekly-analysis.contract';

interface WeekSelectorProps {
  weeks: AnalysisWeekEntry[];
  currentYear: number;
  currentWeek: number;
  onSelect: (year: number, week: number) => void;
}

export function WeekSelector({ weeks, currentYear, currentWeek, onSelect }: WeekSelectorProps) {
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

  const weekStr = String(currentWeek).padStart(2, '0');
  const currentEntry = weeks.find(w => w.year === currentYear && w.week === currentWeek);

  return (
    <div className="flex items-center justify-between gap-4">
      {/* Prev */}
      <button
        onClick={goPrev}
        disabled={!hasPrev}
        className="p-2 rounded-lg bg-gray-800/50 border border-gray-700/50 hover:bg-gray-700/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
      >
        <ChevronLeft className="w-5 h-5 text-gray-300" />
      </button>

      {/* Center label */}
      <motion.div
        key={`${currentYear}-${currentWeek}`}
        initial={{ opacity: 0, y: -5 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex-1 text-center"
      >
        <div className="flex items-center justify-center gap-2 mb-1">
          <Calendar className="w-4 h-4 text-cyan-400" />
          <span className="text-xl font-bold text-white">
            SEMANA {currentWeek}
          </span>
          <span className="text-sm text-gray-500">{currentYear}</span>
        </div>
        {currentEntry && (
          <p className="text-xs text-gray-500">
            {currentEntry.start} — {currentEntry.end}
            {currentEntry.daily_count > 0 && (
              <span className="ml-2 text-gray-600">({currentEntry.daily_count} dias)</span>
            )}
          </p>
        )}
      </motion.div>

      {/* Next */}
      <button
        onClick={goNext}
        disabled={!hasNext}
        className="p-2 rounded-lg bg-gray-800/50 border border-gray-700/50 hover:bg-gray-700/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
      >
        <ChevronRight className="w-5 h-5 text-gray-300" />
      </button>

      {/* Jump dropdown */}
      {weeks.length > 3 && (
        <select
          value={`${currentYear}-${currentWeek}`}
          onChange={(e) => {
            const [y, w] = e.target.value.split('-').map(Number);
            onSelect(y, w);
          }}
          className="ml-2 bg-gray-800/80 border border-gray-700/50 text-gray-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
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
