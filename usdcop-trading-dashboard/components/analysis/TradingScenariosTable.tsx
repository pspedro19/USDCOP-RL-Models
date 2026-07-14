'use client';

import { motion } from 'framer-motion';
import { Target, ArrowUp, ArrowDown } from 'lucide-react';
import type { TradingScenario } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface TradingScenariosTableProps {
  scenarios: TradingScenario[];
  noTradeZone?: [number, number];
}

/** Prototype view-model: conf high → pos, medium → warn, low → neg. */
const CONFIDENCE_TONE = { high: 'pos', medium: 'warn', low: 'neg' } as const;

export function TradingScenariosTable({ scenarios, noTradeZone }: TradingScenariosTableProps) {
  const t = useGmT(ANALYSIS_DICT);
  if (scenarios.length === 0) return null;

  const profileLabels: Record<string, string> = {
    scalp: t('profileScalp'),
    intraday: t('profileIntraday'),
    swing: t('profileSwing'),
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-5`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
          <Target className={`w-4 h-4 ${GM.accent}`} />
          {t('scenTitle')}
        </h3>
        {noTradeZone?.[0] != null && noTradeZone[1] != null && noTradeZone[0] !== noTradeZone[1] && (
          <span className={`${GMT.micro} ${GM.warnBadge} ${GMT.mono} px-2 py-0.5 rounded`}>
            {t('noTradeZone')}: {noTradeZone[0].toFixed(0)}–{noTradeZone[1].toFixed(0)}
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className={`w-full ${GMT.meta}`}>
          <thead>
            <tr className="border-b border-[var(--gm-border)]">
              <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thDir')}</th>
              <th className={`text-left py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thEntry')}</th>
              <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thStop')}</th>
              <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thTargets')}</th>
              <th className={`text-right py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thRR')}</th>
              <th className={`text-center py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thConf')}</th>
              <th className={`text-center py-2 px-2 ${GMT.label} ${GM.textMuted}`}>{t('thProfile')}</th>
            </tr>
          </thead>
          <tbody>
            {scenarios.map((scenario, i) => {
              const isLong = scenario.direction === 'long';
              const confTone = CONFIDENCE_TONE[scenario.confidence] || 'neutral';

              return (
                <tr
                  key={i}
                  className={`border-b border-[rgba(148,163,184,.07)] ${GM.rowHover} transition-colors duration-[var(--gm-dur-fast)]`}
                >
                  {/* Direction */}
                  <td className="py-2.5 px-2">
                    <span className={`inline-flex items-center gap-1 font-bold ${isLong ? GM.pos : GM.neg}`}>
                      {isLong ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
                      {isLong ? 'LONG' : 'SHORT'}
                    </span>
                  </td>

                  {/* Entry condition */}
                  <td className={`py-2.5 px-2 ${GM.text} max-w-[200px]`}>
                    <div className="truncate" title={scenario.entry_condition}>
                      {scenario.entry_condition}
                    </div>
                    {scenario.entry_price != null && (
                      <span className={`${GM.textMuted} ${GMT.mono}`}>{scenario.entry_price.toFixed(2)}</span>
                    )}
                  </td>

                  {/* Stop */}
                  <td className={`py-2.5 px-2 text-right ${GM.neg} ${GMT.mono}`}>
                    {scenario.stop_loss != null ? scenario.stop_loss.toFixed(2) : '—'}
                  </td>

                  {/* Targets */}
                  <td className={`py-2.5 px-2 text-right ${GM.pos} ${GMT.mono}`}>
                    {(scenario.targets?.length ?? 0) > 0
                      ? (scenario.targets ?? []).map(tg => tg.toFixed(0)).join(' / ')
                      : '—'}
                  </td>

                  {/* R:R */}
                  <td className={`py-2.5 px-2 text-right ${GM.textStrong} font-semibold ${GMT.mono}`}>
                    {scenario.risk_reward != null ? `${scenario.risk_reward.toFixed(1)}:1` : '—'}
                  </td>

                  {/* Confidence */}
                  <td className="py-2.5 px-2 text-center">
                    <GmBadge tone={confTone}>{scenario.confidence}</GmBadge>
                  </td>

                  {/* Profile */}
                  <td className={`py-2.5 px-2 text-center ${GM.textSec}`}>
                    {profileLabels[scenario.profile] || scenario.profile}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}
