'use client';

import { motion } from 'framer-motion';
import { Gauge, Target, TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown } from 'lucide-react';
import type { AssetTechnicalAnalysis } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, type GmTone } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

/** Adaptive price format: BTC (~77k) → 0 dp, Gold (~4.7k) → 1 dp, small → 2 dp. */
function fmt(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  const dp = a >= 1000 ? 0 : a >= 10 ? 1 : 2;
  return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function trendTone(trend: string): { tone: GmTone; icon: typeof TrendingUp; label: string } {
  if (trend.startsWith('alcista')) return { tone: 'pos', icon: TrendingUp, label: trend };
  if (trend.startsWith('bajista')) return { tone: 'neg', icon: TrendingDown, label: trend };
  return { tone: 'neutral', icon: Minus, label: trend };
}

/**
 * Technical + scenarios panel for the lean Gold/BTC science-stack schema
 * (AssetTechnicalAnalysis) — the USD/COP TechnicalAnalysisCard/TradingScenariosTable
 * expect the richer bias/RSI/watch-list shape and would render empty cells here.
 * Same GM module look as USD/COP; real indicators computed from the asset price.
 */
export function AssetTechnicalCard({ ta, symbol, priceUnit }: {
  ta: AssetTechnicalAnalysis;
  symbol?: string;
  priceUnit?: string;
}) {
  const t = useGmT(ANALYSIS_DICT);
  const tr = trendTone(ta.trend);
  const TrendIcon = tr.icon;
  const rsi = ta.indicators.rsi_14;
  const macdH = ta.indicators.macd_histogram;
  const ntz = ta.support_resistance?.no_trade_zone;

  const sigTone: GmTone = ta.signal === 'sobrecompra' ? 'neg' : ta.signal === 'sobreventa' ? 'pos' : 'neutral';
  const sigLabel = ta.signal === 'sobrecompra' ? t('sigOverbought')
    : ta.signal === 'sobreventa' ? t('sigOversold') : t('sigNeutral');

  const indicators: Array<{ label: string; value: string; tone?: GmTone }> = [
    { label: 'RSI (14)', value: rsi != null ? rsi.toFixed(1) : '—', tone: rsi == null ? undefined : rsi > 70 ? 'neg' : rsi < 30 ? 'pos' : 'neutral' },
    { label: 'SMA 20', value: fmt(ta.indicators.sma_20) },
    { label: 'SMA 50', value: fmt(ta.indicators.sma_50) },
    { label: t('atrLabel'), value: fmt(ta.indicators.atr_14) },
    { label: 'MACD', value: fmt(ta.indicators.macd_line), tone: macdH == null ? undefined : macdH >= 0 ? 'pos' : 'neg' },
    { label: t('supportLabel'), value: fmt(ta.support_resistance?.support), tone: 'pos' },
    { label: t('resistanceLabel'), value: fmt(ta.support_resistance?.resistance), tone: 'neg' },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Technical card */}
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`${GM.panel} gm-contain p-5`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
            <Gauge className={`w-4 h-4 ${GM.accent}`} />
            {t('techTitle')}
          </h3>
          <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
            {symbol ?? ''} {fmt(ta.last_price)}{priceUnit ? ` ${priceUnit}` : ''}
          </span>
        </div>

        {/* Trend + signal chips */}
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <GmBadge tone={tr.tone} className="text-xs px-3 py-1">
            <TrendIcon className="w-3.5 h-3.5" />
            {tr.label}
          </GmBadge>
          <GmBadge tone={sigTone} className="text-xs px-3 py-1">{t('signalWord')}: {sigLabel}</GmBadge>
          {rsi != null && (
            <span className={`${GMT.mono} ${GMT.meta} ${rsi > 70 ? GM.neg : rsi < 30 ? GM.pos : GM.textSec}`}>
              RSI {rsi.toFixed(1)}
            </span>
          )}
        </div>

        {/* Indicators grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
          {indicators.map((ind) => (
            <div key={ind.label} className={`${GM.panelInner} gm-contain px-3 py-2 rounded-lg`}>
              <div className={`${GMT.micro} ${GM.textMuted} mb-0.5`}>{ind.label}</div>
              <div className={`${GMT.mono} font-semibold ${ind.tone && ind.tone !== 'neutral' ? (ind.tone === 'pos' ? GM.pos : GM.neg) : GM.text}`}>
                {ind.value}
              </div>
            </div>
          ))}
        </div>

        {ntz && ntz.low != null && ntz.high != null && (
          <div className="mt-3">
            <span className={`${GMT.micro} ${GM.warnBadge} ${GMT.mono} px-2 py-0.5 rounded`}>
              {t('noTradeZone')}: {fmt(ntz.low)}–{fmt(ntz.high)}
            </span>
          </div>
        )}
      </motion.div>

      {/* Scenarios table */}
      {ta.scenarios?.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`${GM.panel} gm-contain p-5`}>
          <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2 mb-4`}>
            <Target className={`w-4 h-4 ${GM.accent}`} />
            {t('scenTitle')}
          </h3>
          <div className="overflow-x-auto">
            <table className={`w-full ${GMT.meta}`}>
              <thead>
                <tr className="border-b border-[var(--gm-border)]">
                  <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thScenario')}</th>
                  <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thDir')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thTrigger')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thTarget')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thInvalid')}</th>
                </tr>
              </thead>
              <tbody>
                {ta.scenarios.map((s, i) => {
                  const isLong = s.direction === 'long';
                  return (
                    <tr key={i} className={`border-b border-[rgba(148,163,184,.07)] ${GM.rowHover} transition-colors duration-[var(--gm-dur-fast)]`}>
                      <td className={`py-2.5 px-2 ${GM.text}`}>
                        <div>{s.name}</div>
                        {s.rationale && <div className={`${GMT.micro} ${GM.textMuted} max-w-[220px] truncate`} title={s.rationale}>{s.rationale}</div>}
                      </td>
                      <td className="py-2.5 px-2">
                        <span className={`inline-flex items-center gap-1 font-bold ${isLong ? GM.pos : GM.neg}`}>
                          {isLong ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                          {isLong ? 'LONG' : 'SHORT'}
                        </span>
                      </td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.textStrong}`}>{fmt(s.trigger)}</td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.pos}`}>{fmt(s.target)}</td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.neg}`}>{fmt(s.invalidation)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
}
