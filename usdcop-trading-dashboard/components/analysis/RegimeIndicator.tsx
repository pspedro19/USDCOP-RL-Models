'use client';

import { motion } from 'framer-motion';
import { Shield, ShieldAlert, Activity, AlertTriangle } from 'lucide-react';
import type { MacroRegimeOutput } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, type GmTone } from '@/lib/ui/gm-tokens';
import { GmBadge } from '@/components/gm';

import { ANALYSIS_DICT } from './gm-analysis';

interface RegimeIndicatorProps {
  regime: MacroRegimeOutput;
}

const REGIME_CONFIG: Record<string, { tone: GmTone; icon: typeof Shield; dictKey: 'riskOn' | 'riskOff' | 'transition' }> = {
  risk_on: { tone: 'pos', icon: Shield, dictKey: 'riskOn' },
  transition: { tone: 'warn', icon: Activity, dictKey: 'transition' },
  risk_off: { tone: 'neg', icon: ShieldAlert, dictKey: 'riskOff' },
};

export function RegimeIndicator({ regime }: RegimeIndicatorProps) {
  const t = useGmT(ANALYSIS_DICT);

  // Lean-schema assets (Gold/BTC) may omit confidence and the alert/leader arrays —
  // default them so this card degrades gracefully instead of crashing on toFixed()/.length.
  const regimeState = regime.regime;
  const config = REGIME_CONFIG[regimeState?.label] || REGIME_CONFIG.transition;
  const Icon = config.icon;
  const confidence = regimeState?.confidence ?? 0;
  const barColor =
    config.tone === 'pos' ? 'bg-[var(--gm-pos)]' : config.tone === 'neg' ? 'bg-[var(--gm-neg)]' : 'bg-[var(--gm-warn)]';

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`${GM.panel} gm-contain p-4`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong}`}>{t('regimeTitle')}</h3>
        <GmBadge tone={config.tone} className="text-xs px-3 py-1">
          <Icon className="w-3.5 h-3.5" />
          {t(config.dictKey)}
        </GmBadge>
      </div>

      {/* Confidence bar */}
      <div className="mb-3">
        <div className={`flex items-center justify-between ${GMT.meta} ${GM.textMuted} mb-1`}>
          <span>{t('confidence')}</span>
          <span className={GMT.mono}>{(confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full h-1.5 bg-[rgba(148,163,184,.12)] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-[var(--gm-dur-base)] ${barColor}`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Since date */}
      {regimeState.since && (
        <p className={`${GMT.meta} ${GM.textMuted} mb-3`}>
          {t('activeSince')}: <span className={`${GM.textSec} ${GMT.mono}`}>{regimeState.since}</span>
        </p>
      )}

      {/* Z-Score alerts — prototype amber alert chip */}
      {(regime.zscore_alerts?.length ?? 0) > 0 && (
        <div className="space-y-1.5 mb-3">
          <p className={`${GMT.label} ${GM.textMuted}`}>{t('zAlerts')}</p>
          {(regime.zscore_alerts ?? []).slice(0, 3).map((alert, i) => (
            <div key={i} className={`flex items-center gap-2 ${GMT.meta} ${GM.warnBadge} rounded-[10px] px-2.5 py-1.5`}>
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              <span className={alert.direction === 'extreme_high' ? GM.neg : GM.info}>
                {alert.direction === 'extreme_high' ? '▲' : '▼'}
              </span>
              <span className={GM.text}>{alert.variable_name}</span>
              <span className={`${GM.warn} ${GMT.mono} ml-auto font-semibold`}>z={alert.z_score.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Granger leaders — prototype "Variables líderes" mono rows */}
      {(regime.granger_leaders?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <p className={`${GMT.label} ${GM.textMuted}`}>{t('leaders')}</p>
          {(regime.granger_leaders ?? []).slice(0, 3).map((leader, i) => (
            <div key={i} className={`flex items-center gap-2 ${GMT.meta} ${GM.textSec}`}>
              <span className={`${GM.text} font-bold ${GMT.mono}`}>{leader.variable.toUpperCase()}</span>
              <span className={`${GM.textMuted} ${GMT.mono}`}>lag={leader.optimal_lag}d</span>
              <span className={`${GM.textMuted} ${GMT.mono} ml-auto`}>p={leader.p_value.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
