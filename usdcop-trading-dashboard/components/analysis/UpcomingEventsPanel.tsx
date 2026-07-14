'use client';

import { motion } from 'framer-motion';
import { CalendarDays } from 'lucide-react';
import type { EconomicEvent } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface UpcomingEventsPanelProps {
  events: EconomicEvent[];
}

export function UpcomingEventsPanel({ events }: UpcomingEventsPanelProps) {
  const t = useGmT(ANALYSIS_DICT);

  if (events.length === 0) {
    return (
      <div className={`${GM.panel} gm-contain p-5 text-center`}>
        <CalendarDays className={`w-6 h-6 ${GM.textFaint} mx-auto mb-2`} />
        <p className={`${GM.textMuted} ${GMT.body}`}>{t('noEvents')}</p>
      </div>
    );
  }

  // Group by date
  const grouped = events.reduce((acc, evt) => {
    if (!acc[evt.date]) acc[evt.date] = [];
    acc[evt.date].push(evt);
    return acc;
  }, {} as Record<string, EconomicEvent[]>);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-5`}
    >
      <div className="flex items-center gap-2 mb-4">
        <CalendarDays className={`w-4 h-4 ${GM.accent}`} />
        <h3 className={`${GMT.panelTitle} ${GM.textStrong}`}>{t('eventsTitle')}</h3>
        <GmBadge tone="neutral" className={GMT.mono}>{events.length}</GmBadge>
      </div>

      <div className="space-y-4">
        {Object.entries(grouped).map(([date, dateEvents]) => (
          <div key={date}>
            <p className={`${GMT.meta} font-semibold ${GM.textMuted} ${GMT.mono} mb-2`}>{formatDate(date)}</p>
            <div className="space-y-1.5">
              {dateEvents.map((evt, i) => (
                <div
                  key={i}
                  className={`flex items-center gap-2 ${GM.panelSoft} px-3 py-2`}
                >
                  <ImpactDot level={evt.impact_level} label={t('impact')} />
                  <span className={`${GMT.body} ${GM.text} flex-1`}>{evt.event}</span>
                  {evt.forecast && (
                    <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
                      {t('forecastAbbr')} {evt.forecast}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function ImpactDot({ level, label }: { level: 'high' | 'medium' | 'low'; label: string }) {
  const colors = {
    high: 'bg-[var(--gm-neg)]',
    medium: 'bg-[var(--gm-warn)]',
    low: 'bg-[var(--gm-text-faint)]',
  };

  return (
    <span className={`w-2 h-2 rounded-full ${colors[level]} shrink-0`} title={`${label}: ${level}`} />
  );
}

function formatDate(dateStr: string): string {
  try {
    const date = new Date(dateStr + 'T00:00:00');
    const days = ['Domingo', 'Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado'];
    const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
    return `${days[date.getDay()]} ${date.getDate()} ${months[date.getMonth()]}`;
  } catch {
    return dateStr;
  }
}
