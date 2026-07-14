'use client';

import type { MacroPublication } from '@/lib/contracts/weekly-analysis.contract';
import { GM, GMT } from '@/lib/ui/gm-tokens';

interface MacroEventChipProps {
  publication: MacroPublication;
}

export function MacroEventChip({ publication }: MacroEventChipProps) {
  const changePct = publication.change_pct;
  const isPositive = changePct !== undefined && changePct > 0;
  const isNegative = changePct !== undefined && changePct < 0;

  return (
    <span className={`inline-flex items-center gap-1.5 ${GM.neutralBadge} rounded-full px-2.5 py-1 ${GMT.meta}`}>
      <span className={`font-medium ${GM.text}`}>{publication.variable}</span>
      <span className={`${GM.textStrong} font-bold ${GMT.mono}`}>{publication.value.toFixed(2)}</span>
      {changePct !== undefined && (
        <span className={`font-semibold ${GMT.mono} ${isPositive ? GM.pos : isNegative ? GM.neg : GM.textMuted}`}>
          {isPositive ? '+' : ''}{changePct.toFixed(2)}%
        </span>
      )}
    </span>
  );
}
