'use client';

import { motion } from 'framer-motion';
import { Shield, ShieldAlert, ShieldOff, Activity } from 'lucide-react';
import type { MacroRegimeOutput } from '@/lib/contracts/weekly-analysis.contract';

interface RegimeIndicatorProps {
  regime: MacroRegimeOutput;
}

const REGIME_CONFIG = {
  risk_on: {
    label: 'Risk-On',
    labelEs: 'Apetito por riesgo',
    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    icon: Shield,
    dotColor: 'bg-emerald-400',
  },
  transition: {
    label: 'Transicion',
    labelEs: 'Transicion de regimen',
    color: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    icon: Activity,
    dotColor: 'bg-amber-400',
  },
  risk_off: {
    label: 'Risk-Off',
    labelEs: 'Aversion al riesgo',
    color: 'bg-red-500/20 text-red-400 border-red-500/30',
    icon: ShieldAlert,
    dotColor: 'bg-red-400',
  },
} as const;

export function RegimeIndicator({ regime }: RegimeIndicatorProps) {
  const regimeState = regime.regime;
  const config = REGIME_CONFIG[regimeState.label] || REGIME_CONFIG.transition;
  const Icon = config.icon;
  const confidence = regimeState.confidence;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-4"
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">Regimen Macro</h3>
        <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${config.color}`}>
          <Icon className="w-3.5 h-3.5" />
          {config.label}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
          <span>Confianza</span>
          <span>{(confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${config.dotColor}`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* Since date */}
      {regimeState.since && (
        <p className="text-xs text-gray-500 mb-3">
          Activo desde: <span className="text-gray-400">{regimeState.since}</span>
        </p>
      )}

      {/* Z-Score alerts */}
      {regime.zscore_alerts.length > 0 && (
        <div className="space-y-1.5 mb-3">
          <p className="text-xs font-medium text-gray-400">Alertas Z-Score</p>
          {regime.zscore_alerts.slice(0, 3).map((alert, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className={alert.direction === 'extreme_high' ? 'text-red-400' : 'text-blue-400'}>
                {alert.direction === 'extreme_high' ? '▲' : '▼'}
              </span>
              <span className="text-gray-300">{alert.variable_name}</span>
              <span className="text-gray-500 ml-auto">z={alert.z_score.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Granger leaders */}
      {regime.granger_leaders.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-gray-400">Variables lideres</p>
          {regime.granger_leaders.slice(0, 3).map((leader, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
              <span className="text-cyan-400 font-medium">{leader.variable.toUpperCase()}</span>
              <span className="text-gray-600">lag={leader.optimal_lag}d</span>
              <span className="text-gray-600 ml-auto">p={leader.p_value.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
