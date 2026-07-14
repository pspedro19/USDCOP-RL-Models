'use client';

import { useEffect, useRef, useState } from 'react';
import { TrendingUp, Coins, Bitcoin, LineChart, ChevronDown, Check } from 'lucide-react';
import type { AnalysisAsset } from '@/lib/contracts/analysis-assets';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, MOTION } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT } from './gm-analysis';

interface AssetSelectorProps {
  assets: AnalysisAsset[];
  selected: string;
  onSelect: (assetId: string) => void;
}

/** Per-asset-class icon (falls back to a generic chart glyph). */
function assetIcon(assetClass: string) {
  switch (assetClass) {
    case 'crypto':
      return Bitcoin;
    case 'commodity':
      return Coins;
    case 'fx':
      return TrendingUp;
    default:
      return LineChart;
  }
}

/**
 * Dynamic asset filter for the /analysis page. A dropdown menu (per operator
 * request) listing every analysed asset from the SSOT (/api/analysis/assets) —
 * USD/COP, Gold, Bitcoin — to switch which asset's weekly/daily news analysis is
 * shown. Style mirrors the /forecasting Modelo/Horizonte GM dropdowns.
 */
export function AssetSelector({ assets, selected, onSelect }: AssetSelectorProps) {
  const t = useGmT(ANALYSIS_DICT);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  if (!assets.length) return null;

  const current = assets.find((a) => a.asset_id === selected) ?? assets[0];
  const CurrentIcon = assetIcon(current.asset_class);

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className={`${GMT.label} ${GM.accent}`}>{t('pair')}</span>
      <div className="relative" ref={ref}>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-haspopup="listbox"
          aria-expanded={open}
          title={current.display_name}
          className={`${GM.input} ${GM.focus} h-10 flex items-center justify-between gap-2.5 font-semibold cursor-pointer ${MOTION.fast}`}
          style={{ minWidth: 190 }}
        >
          <span className="flex items-center gap-2 min-w-0">
            <CurrentIcon className={`w-4 h-4 shrink-0 ${GM.accent}`} aria-hidden />
            <span className="truncate">{current.display_name}</span>
          </span>
          <ChevronDown className={`w-4 h-4 shrink-0 transition-transform ${MOTION.fast} ${open ? 'rotate-180' : ''} ${GM.accent}`} aria-hidden />
        </button>
        {open && (
          <div
            role="listbox"
            className={`absolute top-[46px] left-0 z-40 p-1.5 flex flex-col gap-0.5 max-h-72 overflow-y-auto ${GM.popover}`}
            style={{ minWidth: 220 }}
          >
            {assets.map((asset) => {
              const Icon = assetIcon(asset.asset_class);
              const isActive = asset.asset_id === selected;
              return (
                <button
                  key={asset.asset_id}
                  type="button"
                  role="option"
                  aria-selected={isActive}
                  onClick={() => { onSelect(asset.asset_id); setOpen(false); }}
                  className={`flex items-center gap-2 text-left px-3 py-2 rounded-[9px] text-[12.5px] font-semibold ${GM.focus}
                    ${isActive ? GM.navActive : GM.navIdle}`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isActive ? GM.accent : GM.textMuted}`} aria-hidden />
                  <span className="truncate flex-1">{asset.display_name}</span>
                  {isActive && <Check className={`w-3.5 h-3.5 shrink-0 ${GM.accent}`} aria-hidden />}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
