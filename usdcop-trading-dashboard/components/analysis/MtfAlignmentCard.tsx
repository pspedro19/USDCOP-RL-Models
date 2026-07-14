'use client';

import { motion } from 'framer-motion';
import { Layers } from 'lucide-react';
import type { MultiTimeframeAnalysis, TechnicalAnalysisOutput } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, type GmTone } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT, sentimentTone } from './gm-analysis';

type Tr = <K extends keyof (typeof ANALYSIS_DICT)['es']>(key: K) => string;

/** Canonical short→long timeframe ordering; unknown keys sort last, stable-ish. */
const TF_ORDER = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'];

function fmtLvl(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—';
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function biasText(t: Tr, bias: string | null | undefined): string {
  const b = (bias || '').toLowerCase();
  if (b === 'bullish') return t('bullish');
  if (b === 'bearish') return t('bearish');
  return t('neutral');
}

function volText(t: Tr, vr: string | null | undefined): string {
  return vr === 'low' ? t('volLow') : vr === 'high' ? t('volHigh') : t('volNormal');
}

/**
 * Multi-timeframe alignment panel (LangGraph `mtf_analysis`): a diverging alignment
 * gauge (score ∈ [-1,1]), per-timeframe bias/RSI/volatility rows, and confluent
 * support/resistance levels.
 *
 * Defensive by design — USD/COP rich weeks only; Gold/BTC and stale weeks omit the
 * field. Every sub-field is optional so a partial payload degrades instead of crashing.
 */
export function MtfAlignmentCard({ mtf }: { mtf?: MultiTimeframeAnalysis | null }) {
  const t = useGmT(ANALYSIS_DICT);

  if (!mtf) return null;

  const reports: Record<string, TechnicalAnalysisOutput> = mtf.reports ?? {};
  const tfKeys = Object.keys(reports);
  const supports = mtf.confluent_supports ?? [];
  const resistances = mtf.confluent_resistances ?? [];
  const hasScore = typeof mtf.alignment_score === 'number' && !Number.isNaN(mtf.alignment_score);

  // Nothing meaningful to show → render nothing (no empty box).
  if (tfKeys.length === 0 && supports.length === 0 && resistances.length === 0 && !hasScore) {
    return null;
  }

  const ordered = [...tfKeys].sort((a, b) => {
    const ia = TF_ORDER.indexOf(a);
    const ib = TF_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });

  const alignTone = sentimentTone(mtf.alignment_label);
  const clamped = hasScore ? Math.max(-1, Math.min(1, mtf.alignment_score)) : 0;
  const fillLeft = clamped < 0 ? `${50 + clamped * 50}%` : '50%';
  const fillWidth = `${Math.abs(clamped) * 50}%`;
  const barColor =
    alignTone === 'pos' ? 'bg-[var(--gm-pos)]' : alignTone === 'neg' ? 'bg-[var(--gm-neg)]' : 'bg-[var(--gm-info)]';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-5`}
    >
      {/* Header: title + alignment label + dominant TF */}
      <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
          <Layers className={`w-4 h-4 ${GM.accent}`} />
          {t('mtfTitle')}
        </h3>
        <div className="flex items-center gap-2">
          {mtf.dominant_timeframe && (
            <span className={`${GMT.meta} ${GM.textMuted}`}>
              {t('mtfDominantTf')}: <span className={`${GM.textSec} ${GMT.mono} font-semibold`}>{mtf.dominant_timeframe}</span>
            </span>
          )}
          {mtf.alignment_label && (
            <GmBadge tone={alignTone} className="text-xs px-3 py-1">{biasText(t, mtf.alignment_label)}</GmBadge>
          )}
        </div>
      </div>

      {/* Alignment gauge (diverging bar centered on 0) */}
      {hasScore && (
        <div className="mb-4">
          <div className={`flex items-center justify-between ${GMT.meta} ${GM.textMuted} mb-1`}>
            <span>{t('mtfAlignment')}</span>
            <span className={`${GMT.mono} font-semibold ${alignTone === 'pos' ? GM.pos : alignTone === 'neg' ? GM.neg : GM.textSec}`}>
              {clamped >= 0 ? '+' : ''}{clamped.toFixed(2)}
            </span>
          </div>
          <div className="relative w-full h-2 rounded-full bg-[rgba(148,163,184,.12)] overflow-hidden">
            <div className="absolute left-1/2 top-0 h-full w-px bg-[var(--gm-border)]" />
            <div
              className={`absolute top-0 h-full rounded-full transition-all duration-[var(--gm-dur-base)] ${barColor}`}
              style={{ left: fillLeft, width: fillWidth }}
            />
          </div>
          <div className={`flex items-center justify-between ${GMT.micro} ${GM.textFaint} mt-1`}>
            <span>-1.0</span>
            <span>0</span>
            <span>+1.0</span>
          </div>
        </div>
      )}

      {/* Per-timeframe rows */}
      {ordered.length > 0 && (
        <div className="overflow-x-auto mb-4">
          <table className={`w-full ${GMT.meta}`}>
            <thead>
              <tr className="border-b border-[var(--gm-border)]">
                <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thTimeframe')}</th>
                <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thBias')}</th>
                <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('confidence')}</th>
                <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thRsi')}</th>
                <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('volatility')}</th>
              </tr>
            </thead>
            <tbody>
              {ordered.map((tf) => {
                const r = reports[tf];
                const tone: GmTone = sentimentTone(r?.dominant_bias);
                const conf = typeof r?.bias_confidence === 'number' ? r.bias_confidence : null;
                const rsi = typeof r?.rsi === 'number' ? r.rsi : null;
                return (
                  <tr
                    key={tf}
                    className={`border-b border-[rgba(148,163,184,.07)] ${GM.rowHover} transition-colors duration-[var(--gm-dur-fast)]`}
                  >
                    <td className={`py-2.5 px-2 ${GM.text} ${GMT.mono} font-bold`}>{tf}</td>
                    <td className="py-2.5 px-2">
                      <GmBadge tone={tone} className="text-[10px] px-2 py-0.5">{biasText(t, r?.dominant_bias)}</GmBadge>
                    </td>
                    <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.textSec}`}>
                      {conf != null ? `${(conf * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td className={`py-2.5 px-2 text-right ${GMT.mono} ${rsi == null ? GM.textMuted : rsi > 70 ? GM.neg : rsi < 30 ? GM.pos : GM.textSec}`}>
                      {rsi != null ? rsi.toFixed(1) : '—'}
                    </td>
                    <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.textSec}`}>{volText(t, r?.volatility_regime)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Confluent support / resistance levels */}
      {(supports.length > 0 || resistances.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {supports.length > 0 && (
            <div>
              <p className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{t('mtfConfSupports')}</p>
              <div className="flex flex-wrap gap-1.5">
                {supports.map((s, i) => (
                  <span key={i} className={`${GMT.mono} ${GMT.micro} ${GM.posBadge} rounded px-2 py-0.5`}>{fmtLvl(s)}</span>
                ))}
              </div>
            </div>
          )}
          {resistances.length > 0 && (
            <div>
              <p className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{t('mtfConfResistances')}</p>
              <div className="flex flex-wrap gap-1.5">
                {resistances.map((r, i) => (
                  <span key={i} className={`${GMT.mono} ${GMT.micro} ${GM.negBadge} rounded px-2 py-0.5`}>{fmtLvl(r)}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
