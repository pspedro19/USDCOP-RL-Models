'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Gauge, Eye } from 'lucide-react';
import type { TechnicalAnalysisOutput } from '@/lib/contracts/weekly-analysis.contract';

interface TechnicalAnalysisCardProps {
  ta: TechnicalAnalysisOutput;
}

const BIAS_CONFIG = {
  bullish: { label: 'Alcista', color: 'text-emerald-400', bg: 'bg-emerald-500/20', icon: TrendingUp },
  bearish: { label: 'Bajista', color: 'text-red-400', bg: 'bg-red-500/20', icon: TrendingDown },
  neutral: { label: 'Neutral', color: 'text-gray-400', bg: 'bg-gray-500/20', icon: Minus },
} as const;

const VOLATILITY_CONFIG = {
  low: { label: 'Baja', color: 'text-blue-400' },
  normal: { label: 'Normal', color: 'text-gray-400' },
  high: { label: 'Alta', color: 'text-amber-400' },
} as const;

export function TechnicalAnalysisCard({ ta }: TechnicalAnalysisCardProps) {
  const bias = BIAS_CONFIG[ta.dominant_bias] || BIAS_CONFIG.neutral;
  const vol = VOLATILITY_CONFIG[ta.volatility_regime] || VOLATILITY_CONFIG.normal;
  const BiasIcon = bias.icon;
  const confidencePct = (ta.bias_confidence * 100).toFixed(0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Gauge className="w-4 h-4 text-cyan-400" />
          Analisis Tecnico
        </h3>
        <span className="text-xs text-gray-500">
          USD/COP {ta.current_price.toFixed(2)}
        </span>
      </div>

      {/* Bias gauge */}
      <div className="flex items-center gap-4 mb-4">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${bias.bg}`}>
          <BiasIcon className={`w-5 h-5 ${bias.color}`} />
          <div>
            <p className={`text-sm font-bold ${bias.color}`}>{bias.label}</p>
            <p className="text-xs text-gray-500">Confianza {confidencePct}%</p>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="flex-1">
          <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
            {/* Bearish side (left) / Bullish side (right) */}
            <div className="absolute inset-0 flex">
              <div className="w-1/2 flex justify-end">
                {ta.dominant_bias === 'bearish' && (
                  <div
                    className="h-full bg-red-400/60 rounded-l-full"
                    style={{ width: `${ta.bias_confidence * 100}%` }}
                  />
                )}
              </div>
              <div className="w-px bg-gray-600" />
              <div className="w-1/2">
                {ta.dominant_bias === 'bullish' && (
                  <div
                    className="h-full bg-emerald-400/60 rounded-r-full"
                    style={{ width: `${ta.bias_confidence * 100}%` }}
                  />
                )}
              </div>
            </div>
          </div>
          <div className="flex justify-between text-[10px] text-gray-600 mt-0.5">
            <span>Bajista</span>
            <span>Alcista</span>
          </div>
        </div>
      </div>

      {/* Volatility + ATR */}
      <div className="flex items-center gap-4 mb-4 text-xs">
        <span className="text-gray-500">Volatilidad:</span>
        <span className={`font-medium ${vol.color}`}>{vol.label}</span>
        {ta.atr_pct !== null && (
          <span className="text-gray-500 ml-auto">ATR: {ta.atr_pct.toFixed(2)}%</span>
        )}
        {ta.rsi !== null && ta.rsi !== undefined && (
          <span className={`${ta.rsi > 70 ? 'text-red-400' : ta.rsi < 30 ? 'text-emerald-400' : 'text-gray-400'}`}>
            RSI: {ta.rsi.toFixed(1)}
          </span>
        )}
      </div>

      {/* Signals grid */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Bullish signals */}
        {ta.bullish_signals.length > 0 && (
          <div>
            <p className="text-xs font-medium text-emerald-400 mb-1.5">Alcistas</p>
            <div className="space-y-1">
              {ta.bullish_signals.slice(0, 4).map((sig, i) => (
                <div key={i} className="flex items-center gap-1.5 text-xs text-gray-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400/60" />
                  {sig}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Bearish signals */}
        {ta.bearish_signals.length > 0 && (
          <div>
            <p className="text-xs font-medium text-red-400 mb-1.5">Bajistas</p>
            <div className="space-y-1">
              {ta.bearish_signals.slice(0, 4).map((sig, i) => (
                <div key={i} className="flex items-center gap-1.5 text-xs text-gray-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-400/60" />
                  {sig}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Watch list */}
      {ta.watch_list.length > 0 && (
        <div className="border-t border-gray-800/50 pt-3">
          <p className="text-xs font-medium text-gray-400 flex items-center gap-1 mb-1.5">
            <Eye className="w-3 h-3" /> Vigilar
          </p>
          <div className="flex flex-wrap gap-1.5">
            {ta.watch_list.slice(0, 5).map((item, i) => (
              <span key={i} className="px-2 py-0.5 rounded bg-gray-800/60 text-[10px] text-gray-400">
                {item}
              </span>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
