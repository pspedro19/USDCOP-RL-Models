'use client';

import { motion } from 'framer-motion';
import { ArrowUp, ArrowDown, Minus, Zap, BarChart3 } from 'lucide-react';
import type { SignalSummaries } from '@/lib/contracts/weekly-analysis.contract';

interface SignalSummaryCardsProps {
  signals: SignalSummaries;
}

export function SignalSummaryCards({ signals }: SignalSummaryCardsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {/* H5 Weekly Signal */}
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
      >
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 rounded-lg bg-purple-500/20">
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <h3 className="text-sm font-semibold text-white">H5 Semanal</h3>
        </div>

        {signals.h5.direction && signals.h5.direction !== 'HOLD' ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <DirectionBadge direction={signals.h5.direction} />
              {signals.h5.confidence && (
                <span className="text-xs text-gray-500 bg-gray-800/50 rounded px-2 py-0.5">
                  {signals.h5.confidence}
                </span>
              )}
            </div>
            {signals.h5.predicted_return !== undefined && signals.h5.predicted_return !== null && (
              <p className="text-sm text-gray-400">
                Retorno predicho:{' '}
                <span className={`font-medium ${signals.h5.predicted_return >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {signals.h5.predicted_return >= 0 ? '+' : ''}{signals.h5.predicted_return.toFixed(3)}%
                </span>
              </p>
            )}
            {signals.h5.leverage !== undefined && signals.h5.leverage !== null && (
              <p className="text-sm text-gray-400">
                Apalancamiento: <span className="text-white font-medium">{signals.h5.leverage.toFixed(2)}x</span>
              </p>
            )}
          </div>
        ) : signals.h5.direction === 'HOLD' ? (
          <p className="text-sm text-gray-500">Sin operacion esta semana</p>
        ) : (
          <p className="text-sm text-gray-500">Sin señal H5 activa</p>
        )}
      </motion.div>

      {/* H1 Daily Signal */}
      <motion.div
        initial={{ opacity: 0, x: 10 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
      >
        <div className="flex items-center gap-2 mb-3">
          <div className="p-1.5 rounded-lg bg-cyan-500/20">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
          </div>
          <h3 className="text-sm font-semibold text-white">H1 Diario</h3>
        </div>

        {signals.h1.direction && signals.h1.direction !== 'N/A' ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <DirectionBadge direction={signals.h1.direction} />
              {signals.h1.magnitude !== undefined && signals.h1.magnitude !== null && (
                <span className="text-xs text-gray-400">
                  Magnitud: {signals.h1.magnitude.toFixed(4)}
                </span>
              )}
            </div>
            {signals.h1.signals && signals.h1.signals.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs text-gray-500">Ultimas señales:</p>
                {signals.h1.signals.slice(-3).map((s, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <span className="text-gray-600">{s.date.slice(5)}</span>
                    <DirectionBadge direction={s.direction} small />
                    <span className="text-gray-500">{s.signal_strength.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : signals.h1.direction === 'N/A' ? (
          <p className="text-sm text-gray-500">Datos no disponibles</p>
        ) : (
          <p className="text-sm text-gray-500">Sin señal H1 activa</p>
        )}
      </motion.div>
    </div>
  );
}

function DirectionBadge({ direction, small = false }: { direction: string; small?: boolean }) {
  const isShort = direction.toUpperCase() === 'SHORT';
  const isLong = direction.toUpperCase() === 'LONG';

  const bg = isShort ? 'bg-red-500/20' : isLong ? 'bg-emerald-500/20' : 'bg-gray-500/20';
  const text = isShort ? 'text-red-400' : isLong ? 'text-emerald-400' : 'text-gray-400';
  const Icon = isShort ? ArrowDown : isLong ? ArrowUp : Minus;

  return (
    <span className={`inline-flex items-center gap-1 rounded-full ${bg} ${text} ${
      small ? 'px-1.5 py-0.5 text-[10px]' : 'px-2.5 py-1 text-xs'
    } font-semibold`}>
      <Icon className={small ? 'w-2.5 h-2.5' : 'w-3.5 h-3.5'} />
      {direction}
    </span>
  );
}
