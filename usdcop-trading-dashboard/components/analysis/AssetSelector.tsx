'use client';

import { motion } from 'framer-motion';
import { TrendingUp, Coins, Bitcoin, LineChart } from 'lucide-react';
import type { AnalysisAsset } from '@/lib/contracts/analysis-assets';

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
 * Dynamic asset filter for the /analysis page. Renders one pill per analysed
 * asset (from the SSOT via /api/analysis/assets) — USD/COP, Gold, Bitcoin — and
 * lets the operator switch which asset's weekly/daily analysis is shown.
 */
export function AssetSelector({ assets, selected, onSelect }: AssetSelectorProps) {
  if (!assets.length) return null;

  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-[11px] font-semibold text-cyan-400/80 uppercase tracking-wider">
        Par
      </span>
      {/* Segmented control — each option is a visible chip so the pair menu reads clearly
          (inactive chips keep a border/background instead of being near-invisible text). */}
      <div className="inline-flex items-center gap-1 rounded-xl border border-gray-700/60 bg-gray-950/60 p-1">
        {assets.map((asset) => {
          const Icon = assetIcon(asset.asset_class);
          const isActive = asset.asset_id === selected;
          return (
            <button
              key={asset.asset_id}
              onClick={() => onSelect(asset.asset_id)}
              aria-pressed={isActive}
              title={asset.display_name}
              className={`relative flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-semibold transition-colors ${
                isActive
                  ? 'text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/60'
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="asset-selector-active"
                  className="absolute inset-0 rounded-lg bg-gradient-to-r from-cyan-500/25 to-blue-500/25 border border-cyan-400/50 shadow-lg shadow-cyan-500/10"
                  transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
              )}
              <Icon className={`w-4 h-4 relative z-10 ${isActive ? 'text-cyan-300' : ''}`} />
              <span className="relative z-10 whitespace-nowrap">{asset.display_name}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
