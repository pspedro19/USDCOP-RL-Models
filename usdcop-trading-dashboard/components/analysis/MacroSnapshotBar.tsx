'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { MacroVariableSnapshot, MacroChartData } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT, GmSpark } from './gm-analysis';

interface MacroSnapshotBarProps {
  snapshots: Record<string, MacroVariableSnapshot>;
  /** Optional weekly series (already fetched with the view) — renders the prototype sparkline. */
  charts?: Record<string, MacroChartData>;
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

export function MacroSnapshotBar({ snapshots, charts, onVariableClick }: MacroSnapshotBarProps) {
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
            sparkValues={charts?.[key]?.data?.map(d => d.value)}
            source={DATA_SOURCES[key]}
            onClick={() => onVariableClick?.(key)}
          />
        </motion.div>
      ))}
    </div>
  );
}

function MacroCard({ snapshot, sparkValues, source, onClick }: {
  snapshot: MacroVariableSnapshot;
  sparkValues?: Array<number | null>;
  source?: string;
  onClick: () => void;
}) {
  const t = useGmT(ANALYSIS_DICT);
  const trend = snapshot.trend;
  const isUp = trend === 'above' || trend === 'golden_cross';
  const isDown = trend === 'below' || trend === 'death_cross';

  const rsiStatus = snapshot.rsi_14
    ? snapshot.rsi_14 > 70 ? 'OB' : snapshot.rsi_14 < 30 ? 'OS' : null
    : null;

  return (
    <button
      onClick={onClick}
      className={`w-full ${GM.panel} gm-contain p-4 hover:border-[rgba(34,211,238,.3)] transition-colors duration-[var(--gm-dur-fast)] text-left group ${GM.focus}`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className={`${GMT.label} ${GM.textMuted}`}>
          {snapshot.variable_name}
        </span>
        <div className="flex items-center gap-1.5">
          {rsiStatus && (
            <span className={`${GMT.micro} px-1.5 py-0.5 rounded font-bold ${rsiStatus === 'OB' ? GM.negBadge : GM.posBadge}`}>
              {rsiStatus}
            </span>
          )}
          {sparkValues && sparkValues.length > 1 && (
            <GmSpark
              values={sparkValues}
              tone={isUp ? 'pos' : isDown ? 'neg' : 'neutral'}
              width={64}
              height={22}
            />
          )}
        </div>
      </div>

      <div className="flex items-end justify-between">
        <span className={`text-xl font-extrabold ${GMT.mono} ${GM.text}`}>{snapshot.value.toFixed(2)}</span>
        <div className="flex items-center gap-1">
          {isUp && <TrendingUp className={`w-4 h-4 ${GM.pos}`} />}
          {isDown && <TrendingDown className={`w-4 h-4 ${GM.neg}`} />}
          {!isUp && !isDown && <Minus className={`w-4 h-4 ${GM.textFaint}`} />}
        </div>
      </div>

      {snapshot.sma_20 !== null && (
        <p className={`${GMT.micro} ${GM.textMuted} ${GMT.mono} mt-1`}>
          SMA20: {snapshot.sma_20.toFixed(2)}
          {snapshot.rsi_14 !== null && <span className="ml-2">RSI: {snapshot.rsi_14.toFixed(1)}</span>}
        </p>
      )}

      {source && (
        <p className={`${GMT.micro} ${GM.textFaint} mt-1.5 tracking-wide`}>{t('source')}: {source}</p>
      )}
    </button>
  );
}
