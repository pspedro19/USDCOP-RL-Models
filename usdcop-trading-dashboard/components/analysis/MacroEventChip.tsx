'use client';

import type { MacroPublication } from '@/lib/contracts/weekly-analysis.contract';

interface MacroEventChipProps {
  publication: MacroPublication;
}

export function MacroEventChip({ publication }: MacroEventChipProps) {
  const changePct = publication.change_pct;
  const isPositive = changePct !== undefined && changePct > 0;
  const isNegative = changePct !== undefined && changePct < 0;

  return (
    <span className="inline-flex items-center gap-1.5 bg-gray-800/60 border border-gray-700/40 rounded-full px-2.5 py-1 text-xs">
      <span className="font-medium text-gray-300">{publication.variable}</span>
      <span className="text-white font-bold">{publication.value.toFixed(2)}</span>
      {changePct !== undefined && (
        <span className={`font-medium ${isPositive ? 'text-emerald-400' : isNegative ? 'text-red-400' : 'text-gray-500'}`}>
          {isPositive ? '+' : ''}{changePct.toFixed(2)}%
        </span>
      )}
    </span>
  );
}
