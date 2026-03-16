'use client';

import { X, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { MacroVariableChart } from './MacroVariableChart';
import type { MacroVariableSnapshot, MacroChartData } from '@/lib/contracts/weekly-analysis.contract';

interface MacroDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  variableKey: string;
  snapshot: MacroVariableSnapshot | null;
  chartData: MacroChartData | null;
}

export function MacroDetailModal({ isOpen, onClose, variableKey, snapshot, chartData }: MacroDetailModalProps) {
  return (
    <AnimatePresence>
      {isOpen && snapshot && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-4 sm:inset-10 z-50 bg-gray-900/95 rounded-2xl border border-gray-800 shadow-2xl overflow-auto"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-800/50">
              <div>
                <h2 className="text-xl font-bold text-white">{snapshot.variable_name}</h2>
                <p className="text-sm text-gray-500">
                  {snapshot.snapshot_date} | {variableKey.toUpperCase()}
                </p>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-gray-800 transition-colors"
              >
                <X className="w-5 h-5 text-gray-400" />
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
                <IndicatorCard label="Valor Actual" value={snapshot.value} />
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
              <div className="bg-gray-800/30 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">MACD</h3>
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
                  <div className="flex items-center gap-2 bg-gray-800/40 rounded-lg px-4 py-2">
                    <span className="text-xs text-gray-500">Tendencia:</span>
                    <TrendLabel trend={snapshot.trend} />
                  </div>
                )}
                {snapshot.signal && (
                  <div className="flex items-center gap-2 bg-gray-800/40 rounded-lg px-4 py-2">
                    <span className="text-xs text-gray-500">Señal:</span>
                    <span className="text-sm font-medium text-white">{snapshot.signal}</span>
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
  const textColor = highlight === 'red' ? 'text-red-400'
    : highlight === 'green' ? 'text-emerald-400'
    : 'text-white';

  return (
    <div className={`bg-gray-800/40 rounded-lg ${compact ? 'p-2' : 'p-3'}`}>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className={`${compact ? 'text-sm' : 'text-base'} font-bold ${value !== null ? textColor : 'text-gray-600'}`}>
        {value !== null ? `${value.toFixed(2)}${suffix}` : '—'}
      </p>
    </div>
  );
}

function TrendLabel({ trend }: { trend: string }) {
  const colors: Record<string, string> = {
    golden_cross: 'text-emerald-400',
    above: 'text-emerald-400',
    death_cross: 'text-red-400',
    below: 'text-red-400',
    neutral: 'text-gray-400',
  };

  const labels: Record<string, string> = {
    golden_cross: 'Golden Cross',
    death_cross: 'Death Cross',
    above: 'Por encima SMA20',
    below: 'Por debajo SMA20',
    neutral: 'Neutral',
  };

  return (
    <span className={`text-sm font-medium ${colors[trend] || 'text-gray-400'}`}>
      {labels[trend] || trend}
    </span>
  );
}
