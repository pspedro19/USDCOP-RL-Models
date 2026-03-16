'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { MacroVariableSnapshot } from '@/lib/contracts/weekly-analysis.contract';

interface MacroSnapshotBarProps {
  snapshots: Record<string, MacroVariableSnapshot>;
  onVariableClick?: (key: string) => void;
}

const KEY_VARIABLES = ['dxy', 'vix', 'wti', 'embi_col', 'ust10y', 'ibr', 'gold', 'brent'];

const DATA_SOURCES: Record<string, string> = {
  dxy: 'Investing.com',
  vix: 'CBOE',
  wti: 'NYMEX',
  embi_col: 'JP Morgan',
  ust10y: 'FRED',
  ust2y: 'FRED',
  ibr: 'BanRep',
  tpm: 'BanRep',
  gold: 'Investing.com',
  brent: 'ICE',
};

export function MacroSnapshotBar({ snapshots, onVariableClick }: MacroSnapshotBarProps) {
  const displayVars = KEY_VARIABLES.filter(k => snapshots[k]);

  if (displayVars.length === 0) return null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-4 gap-3">
      {displayVars.map((key, i) => (
        <motion.div
          key={key}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
        >
          <MacroCard
            snapshot={snapshots[key]}
            source={DATA_SOURCES[key]}
            onClick={() => onVariableClick?.(key)}
          />
        </motion.div>
      ))}
    </div>
  );
}

function MacroCard({ snapshot, source, onClick }: { snapshot: MacroVariableSnapshot; source?: string; onClick: () => void }) {
  const trend = snapshot.trend;
  const isUp = trend === 'above' || trend === 'golden_cross';
  const isDown = trend === 'below' || trend === 'death_cross';

  const rsiStatus = snapshot.rsi_14
    ? snapshot.rsi_14 > 70 ? 'OB' : snapshot.rsi_14 < 30 ? 'OS' : null
    : null;

  return (
    <button
      onClick={onClick}
      className="w-full bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-4 hover:border-cyan-500/30 transition-all text-left group"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">
          {snapshot.variable_name}
        </span>
        {rsiStatus && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
            rsiStatus === 'OB' ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'
          }`}>
            {rsiStatus}
          </span>
        )}
      </div>

      <div className="flex items-end justify-between">
        <span className="text-xl font-bold text-white">{snapshot.value.toFixed(2)}</span>
        <div className="flex items-center gap-1">
          {isUp && <TrendingUp className="w-4 h-4 text-emerald-400" />}
          {isDown && <TrendingDown className="w-4 h-4 text-red-400" />}
          {!isUp && !isDown && <Minus className="w-4 h-4 text-gray-500" />}
        </div>
      </div>

      {snapshot.sma_20 !== null && (
        <p className="text-xs text-gray-500 mt-1">
          SMA20: {snapshot.sma_20.toFixed(2)}
          {snapshot.rsi_14 !== null && <span className="ml-2">RSI: {snapshot.rsi_14.toFixed(1)}</span>}
        </p>
      )}

      {source && (
        <p className="text-[9px] text-gray-600 mt-1.5 tracking-wide">Fuente: {source}</p>
      )}
    </button>
  );
}
