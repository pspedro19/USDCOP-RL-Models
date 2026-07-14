'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, Gauge, Eye } from 'lucide-react';
import type { TechnicalAnalysisOutput } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, type GmTone } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface TechnicalAnalysisCardProps {
  ta: TechnicalAnalysisOutput;
}

const BIAS_TONE: Record<string, { tone: GmTone; icon: typeof TrendingUp }> = {
  bullish: { tone: 'pos', icon: TrendingUp },
  bearish: { tone: 'neg', icon: TrendingDown },
  neutral: { tone: 'neutral', icon: Minus },
};

export function TechnicalAnalysisCard({ ta }: TechnicalAnalysisCardProps) {
  const t = useGmT(ANALYSIS_DICT);

  // This card renders the rich USD/COP technical schema (bias/RSI/ATR/current_price).
  // Assets with a leaner analysis (Gold/BTC) omit these fields — skip the card rather
  // than crash on `undefined.toFixed()`. Graceful degradation over a broken page.
  if (ta == null || ta.current_price == null || ta.bias_confidence == null) return null;

  const bias = BIAS_TONE[ta.dominant_bias] || BIAS_TONE.neutral;
  const biasLabel = ta.dominant_bias === 'bullish' ? t('bullish') : ta.dominant_bias === 'bearish' ? t('bearish') : t('neutral');
  const volLabel = ta.volatility_regime === 'low' ? t('volLow') : ta.volatility_regime === 'high' ? t('volHigh') : t('volNormal');
  const volClass = ta.volatility_regime === 'low' ? GM.info : ta.volatility_regime === 'high' ? GM.warn : GM.textSec;
  const BiasIcon = bias.icon;
  const confidencePct = (ta.bias_confidence * 100).toFixed(0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-5`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
          <Gauge className={`w-4 h-4 ${GM.accent}`} />
          {t('techTitle')}
        </h3>
        <span className={`${GMT.meta} ${GM.textMuted} ${GMT.mono}`}>
          USD/COP {ta.current_price.toFixed(2)}
        </span>
      </div>

      {/* Bias gauge — prototype: bias chip + "Confianza NN%" */}
      <div className="flex items-center gap-4 mb-4">
        <div className="flex items-center gap-2">
          <GmBadge tone={bias.tone} className="text-xs px-3 py-1">
            <BiasIcon className="w-3.5 h-3.5" />
            {biasLabel}
          </GmBadge>
          <span className={`${GMT.meta} ${GM.textMuted}`}>{t('confidence')} {confidencePct}%</span>
        </div>

        {/* Confidence bar */}
        <div className="flex-1">
          <div className="relative h-2 bg-[rgba(148,163,184,.12)] rounded-full overflow-hidden">
            {/* Bearish side (left) / Bullish side (right) */}
            <div className="absolute inset-0 flex">
              <div className="w-1/2 flex justify-end">
                {ta.dominant_bias === 'bearish' && (
                  <div
                    className="h-full bg-[var(--gm-neg)] opacity-60 rounded-l-full"
                    style={{ width: `${ta.bias_confidence * 100}%` }}
                  />
                )}
              </div>
              <div className="w-px bg-[rgba(148,163,184,.3)]" />
              <div className="w-1/2">
                {ta.dominant_bias === 'bullish' && (
                  <div
                    className="h-full bg-[var(--gm-pos)] opacity-60 rounded-r-full"
                    style={{ width: `${ta.bias_confidence * 100}%` }}
                  />
                )}
              </div>
            </div>
          </div>
          <div className={`flex justify-between ${GMT.micro} ${GM.textFaint} mt-0.5`}>
            <span>{t('bearish')}</span>
            <span>{t('bullish')}</span>
          </div>
        </div>
      </div>

      {/* Volatility + ATR */}
      <div className={`flex items-center gap-4 mb-4 ${GMT.meta}`}>
        <span className={GM.textMuted}>{t('volatility')}:</span>
        <span className={`font-semibold ${volClass}`}>{volLabel}</span>
        {ta.atr_pct !== null && (
          <span className={`${GM.textMuted} ${GMT.mono} ml-auto`}>ATR: {ta.atr_pct.toFixed(2)}%</span>
        )}
        {ta.rsi !== null && ta.rsi !== undefined && (
          <span className={`${GMT.mono} ${ta.rsi > 70 ? GM.neg : ta.rsi < 30 ? GM.pos : GM.textSec}`}>
            RSI: {ta.rsi.toFixed(1)}
          </span>
        )}
      </div>

      {/* Signals grid — prototype tech factors (Alcistas / Bajistas / Vigilar) */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        {/* Bullish signals */}
        {ta.bullish_signals.length > 0 && (
          <div>
            <p className={`${GMT.label} ${GM.pos} mb-1.5`}>{t('bullishGroup')}</p>
            <div className="space-y-1">
              {ta.bullish_signals.slice(0, 4).map((sig, i) => (
                <div key={i} className={`flex items-center gap-1.5 ${GMT.meta} ${GM.textSec}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--gm-pos)] opacity-60 shrink-0" />
                  {sig}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Bearish signals */}
        {ta.bearish_signals.length > 0 && (
          <div>
            <p className={`${GMT.label} ${GM.neg} mb-1.5`}>{t('bearishGroup')}</p>
            <div className="space-y-1">
              {ta.bearish_signals.slice(0, 4).map((sig, i) => (
                <div key={i} className={`flex items-center gap-1.5 ${GMT.meta} ${GM.textSec}`}>
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--gm-neg)] opacity-60 shrink-0" />
                  {sig}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Watch list */}
      {ta.watch_list.length > 0 && (
        <div className="border-t border-[var(--gm-border)] pt-3">
          <p className={`${GMT.label} ${GM.warn} flex items-center gap-1 mb-1.5`}>
            <Eye className="w-3 h-3" /> {t('watch')}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {ta.watch_list.slice(0, 5).map((item, i) => (
              <span key={i} className={`px-2 py-0.5 rounded ${GM.neutralBadge} ${GMT.micro}`}>
                {item}
              </span>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
