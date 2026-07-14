'use client';

import { motion } from 'framer-motion';
import { ArrowUp, ArrowDown, Minus, Zap, BarChart3 } from 'lucide-react';
import type { SignalSummaries } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, GM_VIOLET } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface SignalSummaryCardsProps {
  signals: SignalSummaries;
}

export function SignalSummaryCards({ signals }: SignalSummaryCardsProps) {
  const t = useGmT(ANALYSIS_DICT);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {/* H5 Weekly Signal */}
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className={`${GM.panel} gm-contain p-5`}
      >
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 rounded-lg bg-[rgba(139,92,246,.14)] border border-[rgba(139,92,246,.3)]">
            <Zap className={`w-4 h-4 ${GM_VIOLET.text}`} />
          </div>
          <h3 className={`${GMT.panelTitle} ${GM.textStrong}`}>{t('h5Title')}</h3>
        </div>

        {signals.h5.direction && signals.h5.direction !== 'HOLD' ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <DirectionBadge direction={signals.h5.direction} />
              {signals.h5.confidence && (
                <GmBadge tone="neutral">{signals.h5.confidence}</GmBadge>
              )}
            </div>
            {signals.h5.predicted_return !== undefined && signals.h5.predicted_return !== null && (
              <p className={`${GMT.body} ${GM.textSec}`}>
                {t('predictedReturn')}:{' '}
                <span className={`font-semibold ${GMT.mono} ${signals.h5.predicted_return >= 0 ? GM.pos : GM.neg}`}>
                  {signals.h5.predicted_return >= 0 ? '+' : ''}{signals.h5.predicted_return.toFixed(3)}%
                </span>
              </p>
            )}
            {signals.h5.leverage !== undefined && signals.h5.leverage !== null && (
              <p className={`${GMT.body} ${GM.textSec}`}>
                {t('leverage')}: <span className={`${GM.textStrong} font-semibold ${GMT.mono}`}>{signals.h5.leverage.toFixed(2)}x</span>
              </p>
            )}
          </div>
        ) : signals.h5.direction === 'HOLD' ? (
          <p className={`${GMT.body} ${GM.textMuted}`}>{t('noTradeWeek')}</p>
        ) : (
          <p className={`${GMT.body} ${GM.textMuted}`}>{t('noH5')}</p>
        )}
      </motion.div>

      {/* H1 Daily Signal */}
      <motion.div
        initial={{ opacity: 0, x: 10 }}
        animate={{ opacity: 1, x: 0 }}
        className={`${GM.panel} gm-contain p-5`}
      >
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 rounded-lg bg-[rgba(34,211,238,.10)] border border-[rgba(34,211,238,.28)]">
            <BarChart3 className={`w-4 h-4 ${GM.accent}`} />
          </div>
          <h3 className={`${GMT.panelTitle} ${GM.textStrong}`}>{t('h1Title')}</h3>
        </div>

        {signals.h1.direction && signals.h1.direction !== 'N/A' ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <DirectionBadge direction={signals.h1.direction} />
              {signals.h1.magnitude !== undefined && signals.h1.magnitude !== null && (
                <span className={`${GMT.meta} ${GM.textSec} ${GMT.mono}`}>
                  {t('magnitude')}: {signals.h1.magnitude.toFixed(4)}
                </span>
              )}
            </div>
            {signals.h1.signals && signals.h1.signals.length > 0 && (
              <div className="space-y-1">
                <p className={`${GMT.meta} ${GM.textMuted}`}>{t('lastSignals')}:</p>
                {signals.h1.signals.slice(-3).map((s, i) => (
                  <div key={i} className={`flex items-center gap-2 ${GMT.meta}`}>
                    <span className={`${GM.textFaint} ${GMT.mono}`}>{s.date.slice(5)}</span>
                    <DirectionBadge direction={s.direction} small />
                    <span className={`${GM.textMuted} ${GMT.mono}`}>{s.signal_strength.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : signals.h1.direction === 'N/A' ? (
          <p className={`${GMT.body} ${GM.textMuted}`}>{t('dataUnavailable')}</p>
        ) : (
          <p className={`${GMT.body} ${GM.textMuted}`}>{t('noH1')}</p>
        )}
      </motion.div>
    </div>
  );
}

function DirectionBadge({ direction, small = false }: { direction: string; small?: boolean }) {
  const isShort = direction.toUpperCase() === 'SHORT';
  const isLong = direction.toUpperCase() === 'LONG';

  const tone = isShort ? 'neg' : isLong ? 'pos' : 'neutral';
  const Icon = isShort ? ArrowDown : isLong ? ArrowUp : Minus;

  return (
    <GmBadge tone={tone} className={small ? '' : 'text-xs px-2.5 py-1'}>
      <Icon className={small ? 'w-2.5 h-2.5' : 'w-3.5 h-3.5'} />
      {direction}
    </GmBadge>
  );
}
