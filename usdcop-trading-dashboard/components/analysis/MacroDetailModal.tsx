'use client';

import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { MacroVariableSnapshot, MacroChartData } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, Z } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT } from './gm-analysis';
import { MacroVariableChart } from './MacroVariableChart';

interface MacroDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  variableKey: string;
  snapshot: MacroVariableSnapshot | null;
  chartData: MacroChartData | null;
}

export function MacroDetailModal({ isOpen, onClose, variableKey, snapshot, chartData }: MacroDetailModalProps) {
  const t = useGmT(ANALYSIS_DICT);

  return (
    <AnimatePresence>
      {isOpen && snapshot && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={`fixed inset-0 ${Z.backdrop} bg-black/70 backdrop-blur-sm`}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className={`fixed inset-4 sm:inset-10 ${Z.modal} ${GM.popover} overflow-auto`}
            role="dialog"
            aria-modal
            aria-label={snapshot.variable_name}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-[var(--gm-border)]">
              <div>
                <h2 className={`${GMT.h2} ${GM.headline}`}>{snapshot.variable_name}</h2>
                <p className={`${GMT.body} ${GM.textMuted} ${GMT.mono}`}>
                  {snapshot.snapshot_date} | {variableKey.toUpperCase()}
                </p>
              </div>
              <button
                onClick={onClose}
                aria-label={t('closeModal')}
                className={`p-2 rounded-lg hover:bg-[rgba(148,163,184,.10)] transition-colors duration-[var(--gm-dur-fast)] ${GM.focus}`}
              >
                <X className={`w-5 h-5 ${GM.textSec}`} />
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Chart */}
              {chartData && chartData.data.length > 0 && (
                <MacroVariableChart
                  data={chartData.data}
                  variableName={snapshot.variable_name}
                  className="h-72"
                />
              )}

              {/* Indicator grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <IndicatorCard label={t('currentValue')} value={snapshot.value} />
                <IndicatorCard label="SMA 5" value={snapshot.sma_5} />
                <IndicatorCard label="SMA 10" value={snapshot.sma_10} />
                <IndicatorCard label="SMA 20" value={snapshot.sma_20} />
                <IndicatorCard label="SMA 50" value={snapshot.sma_50} />
                <IndicatorCard label="RSI 14" value={snapshot.rsi_14} highlight={
                  snapshot.rsi_14 ? (snapshot.rsi_14 > 70 ? 'red' : snapshot.rsi_14 < 30 ? 'green' : undefined) : undefined
                } />
                <IndicatorCard label="BB Upper" value={snapshot.bollinger_upper_20} />
                <IndicatorCard label="BB Lower" value={snapshot.bollinger_lower_20} />
              </div>

              {/* MACD */}
              <div className={`${GM.panelSoft} p-4`}>
                <h3 className={`${GMT.panelTitle} ${GM.textSec} mb-3`}>MACD</h3>
                <div className="grid grid-cols-3 gap-3">
                  <IndicatorCard label="MACD Line" value={snapshot.macd_line} compact />
                  <IndicatorCard label="Signal" value={snapshot.macd_signal} compact />
                  <IndicatorCard label="Histogram" value={snapshot.macd_histogram} compact highlight={
                    snapshot.macd_histogram ? (snapshot.macd_histogram > 0 ? 'green' : 'red') : undefined
                  } />
                </div>
              </div>

              {/* Additional indicators */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <IndicatorCard label="ROC 5" value={snapshot.roc_5} suffix="%" />
                <IndicatorCard label="ROC 20" value={snapshot.roc_20} suffix="%" />
                <IndicatorCard label="Z-Score (20)" value={snapshot.z_score_20} />
                <IndicatorCard label="BB Width" value={snapshot.bollinger_width_20} />
              </div>

              {/* Trend & Signal */}
              <div className="flex gap-3">
                {snapshot.trend && (
                  <div className={`flex items-center gap-2 ${GM.panelSoft} px-4 py-2`}>
                    <span className={`${GMT.meta} ${GM.textMuted}`}>{t('trend')}:</span>
                    <TrendLabel trend={snapshot.trend} />
                  </div>
                )}
                {snapshot.signal && (
                  <div className={`flex items-center gap-2 ${GM.panelSoft} px-4 py-2`}>
                    <span className={`${GMT.meta} ${GM.textMuted}`}>{t('signalWord')}:</span>
                    <span className={`${GMT.body} font-semibold ${GM.textStrong}`}>{snapshot.signal}</span>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

function IndicatorCard({
  label,
  value,
  suffix = '',
  highlight,
  compact = false,
}: {
  label: string;
  value: number | null;
  suffix?: string;
  highlight?: 'red' | 'green';
  compact?: boolean;
}) {
  const textColor = highlight === 'red' ? GM.neg
    : highlight === 'green' ? GM.pos
    : GM.textStrong;

  return (
    <div className={`${GM.panelSoft} ${compact ? 'p-2' : 'p-3'}`}>
      <p className={`${GMT.label} ${GM.textMuted} mb-1`}>{label}</p>
      <p className={`${compact ? 'text-sm' : 'text-base'} font-bold ${GMT.mono} ${value !== null ? textColor : GM.textFaint}`}>
        {value !== null ? `${value.toFixed(2)}${suffix}` : '—'}
      </p>
    </div>
  );
}

function TrendLabel({ trend }: { trend: string }) {
  const t = useGmT(ANALYSIS_DICT);

  const colors: Record<string, string> = {
    golden_cross: GM.pos,
    above: GM.pos,
    death_cross: GM.neg,
    below: GM.neg,
    neutral: GM.textSec,
  };

  const labels: Record<string, string> = {
    golden_cross: t('trendGoldenCross'),
    death_cross: t('trendDeathCross'),
    above: t('trendAbove'),
    below: t('trendBelow'),
    neutral: t('trendNeutral'),
  };

  return (
    <span className={`${GMT.body} font-semibold ${colors[trend] || GM.textSec}`}>
      {labels[trend] || trend}
    </span>
  );
}
