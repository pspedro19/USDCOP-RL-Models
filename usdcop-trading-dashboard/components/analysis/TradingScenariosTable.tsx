'use client';

import { motion } from 'framer-motion';
import { Target, ArrowUp, ArrowDown } from 'lucide-react';
import type { TradingScenario } from '@/lib/contracts/weekly-analysis.contract';

interface TradingScenariosTableProps {
  scenarios: TradingScenario[];
  noTradeZone?: [number, number];
}

const CONFIDENCE_CONFIG = {
  high: 'text-emerald-400 bg-emerald-500/10',
  medium: 'text-amber-400 bg-amber-500/10',
  low: 'text-gray-400 bg-gray-500/10',
} as const;

const PROFILE_LABELS: Record<string, string> = {
  scalp: 'Scalp',
  intraday: 'Intradía',
  swing: 'Swing',
};

export function TradingScenariosTable({ scenarios, noTradeZone }: TradingScenariosTableProps) {
  if (scenarios.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Target className="w-4 h-4 text-cyan-400" />
          Escenarios de Trading
        </h3>
        {noTradeZone && noTradeZone[0] !== noTradeZone[1] && (
          <span className="text-[10px] text-amber-400/80 bg-amber-500/10 px-2 py-0.5 rounded">
            Zona de no-operar: {noTradeZone[0].toFixed(0)}-{noTradeZone[1].toFixed(0)}
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-800/50">
              <th className="text-left py-2 px-2 text-gray-500 font-medium">Dir</th>
              <th className="text-left py-2 px-2 text-gray-500 font-medium">Entrada</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium">Stop</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium">Objetivo(s)</th>
              <th className="text-right py-2 px-2 text-gray-500 font-medium">R:R</th>
              <th className="text-center py-2 px-2 text-gray-500 font-medium">Confianza</th>
              <th className="text-center py-2 px-2 text-gray-500 font-medium">Perfil</th>
            </tr>
          </thead>
          <tbody>
            {scenarios.map((scenario, i) => {
              const isLong = scenario.direction === 'long';
              const confConfig = CONFIDENCE_CONFIG[scenario.confidence] || CONFIDENCE_CONFIG.low;

              return (
                <tr
                  key={i}
                  className="border-b border-gray-800/30 hover:bg-gray-800/20 transition-colors"
                >
                  {/* Direction */}
                  <td className="py-2.5 px-2">
                    <span className={`inline-flex items-center gap-1 font-semibold ${isLong ? 'text-emerald-400' : 'text-red-400'}`}>
                      {isLong ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                      {isLong ? 'LONG' : 'SHORT'}
                    </span>
                  </td>

                  {/* Entry condition */}
                  <td className="py-2.5 px-2 text-gray-300 max-w-[200px]">
                    <div className="truncate" title={scenario.entry_condition}>
                      {scenario.entry_condition}
                    </div>
                    {scenario.entry_price !== null && (
                      <span className="text-gray-500">{scenario.entry_price.toFixed(2)}</span>
                    )}
                  </td>

                  {/* Stop */}
                  <td className="py-2.5 px-2 text-right text-red-400/80">
                    {scenario.stop_loss !== null ? scenario.stop_loss.toFixed(2) : '—'}
                  </td>

                  {/* Targets */}
                  <td className="py-2.5 px-2 text-right text-emerald-400/80">
                    {scenario.targets.length > 0
                      ? scenario.targets.map(t => t.toFixed(0)).join(' / ')
                      : '—'}
                  </td>

                  {/* R:R */}
                  <td className="py-2.5 px-2 text-right text-white font-medium">
                    {scenario.risk_reward !== null ? `${scenario.risk_reward.toFixed(1)}:1` : '—'}
                  </td>

                  {/* Confidence */}
                  <td className="py-2.5 px-2 text-center">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-medium uppercase ${confConfig}`}>
                      {scenario.confidence}
                    </span>
                  </td>

                  {/* Profile */}
                  <td className="py-2.5 px-2 text-center text-gray-400">
                    {PROFILE_LABELS[scenario.profile] || scenario.profile}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}
