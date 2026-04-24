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

const DATA_SOURCES: Record<string, { name: string; url: string }> = {
  dxy:      { name: 'Investing.com',       url: 'https://www.investing.com/indices/usdollar' },
  vix:      { name: 'CBOE / Investing.com', url: 'https://www.investing.com/indices/volatility-s-p-500' },
  wti:      { name: 'NYMEX / Investing.com', url: 'https://www.investing.com/commodities/crude-oil' },
  embi_col: { name: 'JP Morgan / BanRep',  url: 'https://www.banrep.gov.co/es/estadisticas/spreads-deuda-publica' },
  ust10y:   { name: 'FRED (US Treasury)',   url: 'https://fred.stlouisfed.org/series/DGS10' },
  ust2y:    { name: 'FRED (US Treasury)',   url: 'https://fred.stlouisfed.org/series/DGS2' },
  ibr:      { name: 'BanRep',              url: 'https://www.banrep.gov.co/es/estadisticas/tasas-interes-interbancarias' },
  tpm:      { name: 'BanRep',              url: 'https://www.banrep.gov.co/es/estadisticas/tasas-intervencion-politica-monetaria' },
  gold:     { name: 'Investing.com',       url: 'https://www.investing.com/commodities/gold' },
  brent:    { name: 'ICE / Investing.com', url: 'https://www.investing.com/commodities/brent-oil' },
  fedfunds: { name: 'FRED',               url: 'https://fred.stlouisfed.org/series/FEDFUNDS' },
  cpi_us:   { name: 'BLS / FRED',         url: 'https://fred.stlouisfed.org/series/CPIAUCSL' },
  cpi_col:  { name: 'DANE',               url: 'https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-al-consumidor-ipc' },
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
                  <a
                    href={DATA_SOURCES[key].url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[10px] text-gray-600 mt-0.5 hover:text-cyan-400 transition-colors"
                    onClick={(e) => e.stopPropagation()}
                  >
                    Fuente: {DATA_SOURCES[key].name} ↗
                  </a>
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
