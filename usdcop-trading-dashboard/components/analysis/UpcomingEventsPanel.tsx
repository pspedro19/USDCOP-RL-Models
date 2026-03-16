'use client';

import { motion } from 'framer-motion';
import { CalendarDays, AlertCircle } from 'lucide-react';
import type { EconomicEvent } from '@/lib/contracts/weekly-analysis.contract';

interface UpcomingEventsPanelProps {
  events: EconomicEvent[];
}

export function UpcomingEventsPanel({ events }: UpcomingEventsPanelProps) {
  if (events.length === 0) {
    return (
      <div className="bg-gray-900/40 rounded-xl border border-gray-800/40 p-5 text-center">
        <CalendarDays className="w-6 h-6 text-gray-600 mx-auto mb-2" />
        <p className="text-gray-500 text-sm">Sin eventos proximos</p>
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
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
    >
      <div className="flex items-center gap-2 mb-4">
        <CalendarDays className="w-4 h-4 text-cyan-400" />
        <h3 className="text-sm font-semibold text-white">Proximos Eventos</h3>
        <span className="text-xs text-gray-500 bg-gray-800/50 rounded-full px-2 py-0.5">
          {events.length}
        </span>
      </div>

      <div className="space-y-4">
        {Object.entries(grouped).map(([date, dateEvents]) => (
          <div key={date}>
            <p className="text-xs font-medium text-gray-500 mb-2">{formatDate(date)}</p>
            <div className="space-y-1.5">
              {dateEvents.map((evt, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 bg-gray-800/30 rounded-lg px-3 py-2"
                >
                  <ImpactDot level={evt.impact_level} />
                  <span className="text-sm text-gray-300 flex-1">{evt.event}</span>
                  {evt.forecast && (
                    <span className="text-xs text-gray-500">
                      Est: {evt.forecast}
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

function ImpactDot({ level }: { level: 'high' | 'medium' | 'low' }) {
  const colors = {
    high: 'bg-red-400',
    medium: 'bg-amber-400',
    low: 'bg-gray-500',
  };

  return (
    <span className={`w-2 h-2 rounded-full ${colors[level]} shrink-0`} title={`Impacto: ${level}`} />
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
