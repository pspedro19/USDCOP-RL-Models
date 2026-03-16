'use client';

import { motion } from 'framer-motion';
import { MacroVariableChart } from './MacroVariableChart';
import type { MacroChartData } from '@/lib/contracts/weekly-analysis.contract';

interface MacroChartGridProps {
  charts: Record<string, MacroChartData>;
  onChartClick?: (variableKey: string) => void;
}

const DISPLAY_NAMES: Record<string, string> = {
  dxy: 'DXY (Dollar Index)',
  vix: 'VIX (Volatilidad)',
  wti: 'WTI Petroleo',
  embi_col: 'EMBI Colombia',
  ust10y: 'US Treasury 10Y',
  ust2y: 'US Treasury 2Y',
  ibr: 'IBR (Colombia)',
  tpm: 'TPM (BanRep)',
  gold: 'Oro',
  brent: 'Brent',
};

const DATA_SOURCES: Record<string, string> = {
  dxy: 'Investing.com',
  vix: 'CBOE / Investing.com',
  wti: 'NYMEX / Investing.com',
  embi_col: 'JP Morgan / BanRep',
  ust10y: 'FRED (US Treasury)',
  ust2y: 'FRED (US Treasury)',
  ibr: 'BanRep',
  tpm: 'BanRep',
  gold: 'Investing.com',
  brent: 'ICE / Investing.com',
  fedfunds: 'FRED',
  cpi_us: 'BLS / FRED',
  cpi_col: 'DANE',
};

export function MacroChartGrid({ charts, onChartClick }: MacroChartGridProps) {
  const keys = Object.keys(charts);
  if (keys.length === 0) return null;

  // Show top 8 by default (daily macro variables)
  const displayKeys = keys.slice(0, 8);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {displayKeys.map((key, i) => {
        const chart = charts[key];
        const name = DISPLAY_NAMES[key] || key;

        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-4 hover:border-cyan-500/20 transition-all cursor-pointer"
            onClick={() => onChartClick?.(key)}
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-white">{name}</h3>
                {DATA_SOURCES[key] && (
                  <p className="text-[10px] text-gray-600 mt-0.5">Fuente: {DATA_SOURCES[key]}</p>
                )}
              </div>
              {chart.png_url && (
                <span className="text-[10px] text-gray-600 bg-gray-800/50 rounded px-1.5 py-0.5">PNG</span>
              )}
            </div>

            {chart.data.length > 0 ? (
              <MacroVariableChart data={chart.data} variableName={name} />
            ) : chart.png_url ? (
              <img
                src={chart.png_url}
                alt={`${name} chart`}
                className="w-full rounded-lg"
                onError={(e) => {
                  (e.target as HTMLImageElement).closest('div')!.style.display = 'none';
                }}
              />
            ) : (
              <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
                Sin datos
              </div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
