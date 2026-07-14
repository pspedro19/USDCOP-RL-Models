'use client';

import { motion } from 'framer-motion';
import { Clock } from 'lucide-react';
import type { DailyAnalysisEntry } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';
import { DailyTimelineEntry } from './DailyTimelineEntry';

interface DailyTimelineProps {
  entries: DailyAnalysisEntry[];
}

export function DailyTimeline({ entries }: DailyTimelineProps) {
  const t = useGmT(ANALYSIS_DICT);

  if (!entries || entries.length === 0) {
    return (
      <div className={`${GM.panel} gm-contain p-8 text-center`}>
        <Clock className={`w-8 h-8 ${GM.textFaint} mx-auto mb-3`} />
        <p className={`${GM.textMuted} ${GMT.body}`}>{t('noDaily')}</p>
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
        <Clock className={`w-4 h-4 ${GM.accent}`} />
        <h2 className={`${GMT.h2} ${GM.textStrong}`}>{t('timelineTitle')}</h2>
        <GmBadge tone="neutral" className={GMT.mono}>
          {entries.length} {t('days')}
        </GmBadge>
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
