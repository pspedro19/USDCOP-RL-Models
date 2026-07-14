'use client';

import { motion } from 'framer-motion';
import { Globe, TrendingUp, TrendingDown, Droplet, Landmark, Coins } from 'lucide-react';
import type { FXContextOutput } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, type GmTone, toneOf } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

type Tr = <K extends keyof (typeof ANALYSIS_DICT)['es']>(key: K) => string;

function fmtNum(v: number | null | undefined, dp = 2): string {
  if (v == null || Number.isNaN(v)) return '—';
  return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

function fmtPct(v: number | null | undefined, dp = 2): string {
  if (v == null || Number.isNaN(v)) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(dp)}%`;
}

function carryTone(attractiveness: string | null | undefined): GmTone {
  const a = (attractiveness || '').toLowerCase();
  if (a === 'attractive') return 'pos';
  if (a === 'unattractive') return 'warn';
  return 'neutral';
}

function carryLabel(t: Tr, attractiveness: string | null | undefined): string {
  const a = (attractiveness || '').toLowerCase();
  if (a === 'attractive') return t('carryAttractive');
  if (a === 'unattractive') return t('carryUnattractive');
  return t('carryNeutral');
}

function rateExpMeta(t: Tr, exp: string | null | undefined): { tone: GmTone; label: string } {
  const e = (exp || '').toLowerCase();
  if (e === 'cut') return { tone: 'info', label: t('rateCut') };
  if (e === 'hike') return { tone: 'warn', label: t('rateHike') };
  return { tone: 'neutral', label: t('rateHold') };
}

/** Small labelled metric cell. */
function Metric({ label, value, tone }: { label: string; value: string; tone?: GmTone }) {
  const color = tone === 'pos' ? GM.pos : tone === 'neg' ? GM.neg : tone === 'warn' ? GM.warn : GM.text;
  return (
    <div className={`${GM.panelInner} gm-contain px-3 py-2 rounded-lg`}>
      <div className={`${GMT.micro} ${GM.textMuted} mb-0.5`}>{label}</div>
      <div className={`${GMT.mono} font-semibold ${color}`}>{value}</div>
    </div>
  );
}

/**
 * FX context panel (LangGraph `fx_context`): carry-trade differential, oil impact,
 * BanRep stance, the FX narrative, and the per-driver COP sensitivity table.
 *
 * Defensive by design — USD/COP rich weeks only; Gold/BTC and stale weeks omit the
 * field. Every nested value is optional, so a partial payload degrades gracefully.
 */
export function FxContextCard({ fx }: { fx?: FXContextOutput | null }) {
  const t = useGmT(ANALYSIS_DICT);

  if (!fx) return null;

  const carry = fx.carry_trade;
  const oil = fx.oil_impact;
  const banrep = fx.banrep;
  const sens = fx.sensitivity_impacts ?? {};
  const sensKeys = Object.keys(sens);

  const hasCarry = !!carry && (carry.differential_pct != null || (carry.interpretation ?? '').trim().length > 0);
  const hasOil = !!oil && (oil.wti_current != null || (oil.interpretation ?? '').trim().length > 0);
  const hasBanrep = !!banrep && (banrep.ibr_current != null || banrep.tpm_current != null || banrep.next_meeting != null);
  const hasNarrative = typeof fx.fx_narrative === 'string' && fx.fx_narrative.trim().length > 0;

  // Nothing meaningful → render nothing (no empty box).
  if (!hasCarry && !hasOil && !hasBanrep && !hasNarrative && sensKeys.length === 0 && fx.cop_level == null) {
    return null;
  }

  const rate = rateExpMeta(t, banrep?.rate_expectation);
  const copChange = fx.cop_weekly_change_pct;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-5`}
    >
      {/* Header: title + COP level / weekly change */}
      <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
          <Globe className={`w-4 h-4 ${GM.accent}`} />
          {t('fxTitle')}
        </h3>
        {(fx.cop_level != null || copChange != null) && (
          <div className="flex items-center gap-2">
            {fx.cop_level != null && (
              <span className={`${GMT.meta} ${GM.textMuted}`}>
                {t('fxCopLevel')}: <span className={`${GM.textSec} ${GMT.mono} font-semibold`}>{fmtNum(fx.cop_level)}</span>
              </span>
            )}
            {copChange != null && (
              <span className={`inline-flex items-center gap-1 ${GMT.mono} font-semibold ${copChange >= 0 ? GM.neg : GM.pos}`}>
                {copChange >= 0 ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                {fmtPct(copChange)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Narrative */}
      {hasNarrative && (
        <p className={`${GMT.body} ${GM.textSec} leading-relaxed mb-4`}>{fx.fx_narrative}</p>
      )}

      {/* Sub-panels: carry / oil / banrep */}
      {(hasCarry || hasOil || hasBanrep) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          {hasCarry && (
            <div className={`${GM.panelSoft} p-3`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`${GMT.label} ${GM.textMuted} flex items-center gap-1.5`}>
                  <Coins className="w-3.5 h-3.5" />{t('fxCarry')}
                </span>
                <GmBadge tone={carryTone(carry?.carry_attractiveness)} className="text-[10px] px-2 py-0.5">
                  {carryLabel(t, carry?.carry_attractiveness)}
                </GmBadge>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Metric label={t('fxDifferential')} value={carry?.differential_pct != null ? `${fmtNum(carry.differential_pct)}pp` : '—'} />
                <Metric label={t('fxIbr')} value={fmtNum(carry?.ibr_rate)} />
                <Metric label={t('fxFedFunds')} value={fmtNum(carry?.fed_funds_rate)} />
              </div>
            </div>
          )}

          {hasOil && (
            <div className={`${GM.panelSoft} p-3`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`${GMT.label} ${GM.textMuted} flex items-center gap-1.5`}>
                  <Droplet className="w-3.5 h-3.5" />{t('fxOil')}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Metric label={t('fxWti')} value={fmtNum(oil?.wti_current)} tone={toneOf(oil?.wti_weekly_change_pct)} />
                <Metric label={t('fxWeeklyChange')} value={fmtPct(oil?.wti_weekly_change_pct)} tone={toneOf(oil?.wti_weekly_change_pct)} />
                <Metric label={t('fxCopImpact')} value={fmtPct(oil?.estimated_cop_impact_pct)} tone={toneOf(oil?.estimated_cop_impact_pct)} />
              </div>
            </div>
          )}

          {hasBanrep && (
            <div className={`${GM.panelSoft} p-3`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`${GMT.label} ${GM.textMuted} flex items-center gap-1.5`}>
                  <Landmark className="w-3.5 h-3.5" />{t('fxBanrep')}
                </span>
                <GmBadge tone={rate.tone} className="text-[10px] px-2 py-0.5">{rate.label}</GmBadge>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Metric label={t('fxIbr')} value={fmtNum(banrep?.ibr_current)} />
                <Metric label={t('fxNextMeeting')} value={banrep?.next_meeting ?? '—'} />
              </div>
            </div>
          )}
        </div>
      )}

      {/* COP sensitivity table */}
      {sensKeys.length > 0 && (
        <div>
          <p className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{t('fxSensitivity')}</p>
          <div className="overflow-x-auto">
            <table className={`w-full ${GMT.meta}`}>
              <thead>
                <tr className="border-b border-[var(--gm-border)]">
                  <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('fxVariable')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('currentValue')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('fxWeeklyChange')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('fxSensitivityCoef')}</th>
                  <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('fxCopImpact')}</th>
                </tr>
              </thead>
              <tbody>
                {sensKeys.map((k) => {
                  const s = sens[k];
                  return (
                    <tr
                      key={k}
                      className={`border-b border-[rgba(148,163,184,.07)] ${GM.rowHover} transition-colors duration-[var(--gm-dur-fast)]`}
                    >
                      <td className={`py-2.5 px-2 ${GM.text} ${GMT.mono} font-bold uppercase`}>{k}</td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.textSec}`}>{fmtNum(s?.current)}</td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${s?.weekly_change_pct == null ? GM.textMuted : s.weekly_change_pct >= 0 ? GM.pos : GM.neg}`}>
                        {fmtPct(s?.weekly_change_pct)}
                      </td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${GM.textMuted}`}>{fmtNum(s?.sensitivity)}</td>
                      <td className={`py-2.5 px-2 text-right ${GMT.mono} ${s?.estimated_cop_impact_pct == null ? GM.textMuted : s.estimated_cop_impact_pct >= 0 ? GM.neg : GM.pos}`}>
                        {fmtPct(s?.estimated_cop_impact_pct)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </motion.div>
  );
}
