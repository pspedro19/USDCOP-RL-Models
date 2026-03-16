'use client';

import { motion } from 'framer-motion';
import { Clock } from 'lucide-react';
import type { DailyAnalysisEntry } from '@/lib/contracts/weekly-analysis.contract';
import { DailyTimelineEntry } from './DailyTimelineEntry';

interface DailyTimelineProps {
  entries: DailyAnalysisEntry[];
}

export function DailyTimeline({ entries }: DailyTimelineProps) {
  if (entries.length === 0) {
    return (
      <div className="bg-gray-900/40 rounded-xl border border-gray-800/40 p-8 text-center">
        <Clock className="w-8 h-8 text-gray-600 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">Sin analisis diarios para esta semana</p>
      </div>
    );
  }

  // Sort by day_of_week (Mon=0 to Fri=4)
  const sorted = [...entries].sort((a, b) => a.day_of_week - b.day_of_week);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-4 h-4 text-cyan-400" />
        <h2 className="text-base font-semibold text-white">Timeline Diario</h2>
        <span className="text-xs text-gray-500 bg-gray-800/50 rounded-full px-2 py-0.5">
          {entries.length} dias
        </span>
      </div>

      <div className="space-y-0">
        {sorted.map((entry, i) => (
          <DailyTimelineEntry
            key={entry.analysis_date}
            entry={entry}
            index={i}
            isLast={i === sorted.length - 1}
          />
        ))}
      </div>
    </motion.div>
  );
}
