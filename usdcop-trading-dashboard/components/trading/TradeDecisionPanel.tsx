/**
 * Trade Decision Panel Component
 *
 * Displays model metadata during replay:
 * - Confidence meter with visual bar
 * - Action probabilities (HOLD, LONG, SHORT)
 * - Entropy indicator for overfit detection
 * - Features snapshot at trade time
 * - Market regime classification
 * - MAE/MFE (Max Adverse/Favorable Excursion)
 */

'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  AlertTriangle,
  Target,
  BarChart3,
  Clock,
} from 'lucide-react';
import {
  ReplayTrade,
  EnrichedReplayTrade,
  ModelMetadata,
  FeaturesSnapshot,
  MarketRegime,
  isEnrichedTrade,
  selectAnimationType,
} from '@/types/replay';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

interface TradeDecisionPanelProps {
  trade: ReplayTrade | EnrichedReplayTrade | null;
  isVisible?: boolean;
  className?: string;
  compact?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Visual confidence meter with color gradient
 */
function ConfidenceMeter({ value, label = 'Confianza' }: { value: number; label?: string }) {
  const percentage = Math.round(value * 100);
  const color =
    value > 0.8 ? 'bg-green-500' :
    value > 0.6 ? 'bg-blue-500' :
    value > 0.4 ? 'bg-amber-500' :
    'bg-red-500';

  const isOverconfident = value > 0.9;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className={`font-mono ${isOverconfident ? 'text-amber-400' : 'text-slate-300'}`}>
          {percentage}%
          {isOverconfident && <AlertTriangle className="inline w-3 h-3 ml-1" />}
        </span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Action probability bars (HOLD, LONG, SHORT)
 */
function ActionProbabilities({ probs }: { probs: [number, number, number] }) {
  const [hold, long, short] = probs;
  const maxProb = Math.max(hold, long, short);

  const actions = [
    { label: 'HOLD', value: hold, color: 'bg-slate-500', icon: Minus },
    { label: 'LONG', value: long, color: 'bg-green-500', icon: TrendingUp },
    { label: 'SHORT', value: short, color: 'bg-red-500', icon: TrendingDown },
  ];

  return (
    <div className="space-y-2">
      <span className="text-xs text-slate-400">Probabilidades de Acción</span>
      <div className="space-y-1.5">
        {actions.map(({ label, value, color, icon: Icon }) => (
          <div key={label} className="flex items-center gap-2">
            <Icon className="w-3 h-3 text-slate-400" />
            <span className="text-xs w-12 font-mono text-slate-400">{label}</span>
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full ${color} transition-all duration-300 ${value === maxProb ? 'opacity-100' : 'opacity-60'}`}
                style={{ width: `${value * 100}%` }}
              />
            </div>
            <span className={`text-xs font-mono w-10 text-right ${value === maxProb ? 'text-white font-bold' : 'text-slate-500'}`}>
              {(value * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Entropy indicator with overfit warning
 */
function EntropyIndicator({ entropy, confidence }: { entropy: number | null; confidence: number }) {
  if (entropy === null) return null;

  const isLowEntropy = entropy < 0.1;
  const isPossibleOverfit = isLowEntropy && confidence > 0.9;

  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-slate-400 flex items-center gap-1">
        <Activity className="w-3 h-3" />
        Entropy
      </span>
      <div className="flex items-center gap-1">
        <span className={`font-mono ${isPossibleOverfit ? 'text-amber-400' : 'text-slate-300'}`}>
          {entropy.toFixed(3)}
        </span>
        {isPossibleOverfit && (
          <span className="px-1.5 py-0.5 bg-amber-500/20 text-amber-400 rounded text-xs">
            Overfit?
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Features snapshot grid
 */
function FeaturesGrid({ features }: { features: FeaturesSnapshot }) {
  const formatValue = (val: number | undefined) =>
    val === undefined ? '-' : val.toFixed(2);

  const featureItems = [
    { label: 'RSI(14)', value: features.rsi_14, warn: features.rsi_14 > 70 || features.rsi_14 < 30 },
    { label: 'MACD Hist', value: features.macd_histogram },
    { label: 'BB Pos', value: features.bb_position },
    { label: 'Vol Z', value: features.volume_zscore, warn: Math.abs(features.volume_zscore) > 2 },
    { label: 'EMA Cross', value: features.trend_ema_cross },
    { label: 'Hora', value: features.hour_of_day },
  ];

  return (
    <div className="space-y-1.5">
      <span className="text-xs text-slate-400">Features Snapshot</span>
      <div className="grid grid-cols-3 gap-1">
        {featureItems.map(({ label, value, warn }) => (
          <div
            key={label}
            className={`px-2 py-1 rounded text-xs ${warn ? 'bg-amber-500/10 border border-amber-500/20' : 'bg-slate-800'}`}
          >
            <div className="text-slate-500 text-[10px]">{label}</div>
            <div className={`font-mono ${warn ? 'text-amber-400' : 'text-slate-300'}`}>
              {formatValue(value)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Market regime badge
 */
function MarketRegimeBadge({ regime }: { regime: MarketRegime }) {
  const config: Record<MarketRegime, { label: string; color: string; bg: string }> = {
    trending: { label: 'Tendencia', color: 'text-green-400', bg: 'bg-green-500/20' },
    ranging: { label: 'Rango', color: 'text-blue-400', bg: 'bg-blue-500/20' },
    volatile: { label: 'Volátil', color: 'text-amber-400', bg: 'bg-amber-500/20' },
    unknown: { label: 'Desconocido', color: 'text-slate-400', bg: 'bg-slate-500/20' },
  };

  const { label, color, bg } = config[regime];

  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-slate-400">Régimen</span>
      <span className={`px-2 py-0.5 rounded ${bg} ${color}`}>{label}</span>
    </div>
  );
}

/**
 * MAE/MFE excursion display
 */
function ExcursionDisplay({
  mae,
  mfe,
}: {
  mae: number | null;
  mfe: number | null;
}) {
  if (mae === null && mfe === null) return null;

  return (
    <div className="grid grid-cols-2 gap-2 text-xs">
      <div className="px-2 py-1.5 bg-red-500/10 rounded border border-red-500/20">
        <div className="text-red-400/70 text-[10px]">MAE (Max Pérdida)</div>
        <div className="font-mono text-red-400">
          {mae !== null ? `$${Math.abs(mae).toFixed(2)}` : '-'}
        </div>
      </div>
      <div className="px-2 py-1.5 bg-green-500/10 rounded border border-green-500/20">
        <div className="text-green-400/70 text-[10px]">MFE (Max Ganancia)</div>
        <div className="font-mono text-green-400">
          {mfe !== null ? `$${mfe.toFixed(2)}` : '-'}
        </div>
      </div>
    </div>
  );
}

/**
 * Trade info header (basic trade details)
 */
function TradeInfoHeader({ trade }: { trade: ReplayTrade }) {
  const pnl = trade.pnl || trade.pnl_usd || 0;
  const isWin = pnl > 0;
  const animationType = selectAnimationType(trade);

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span
          className={`px-2 py-0.5 rounded text-xs font-medium ${
            trade.side === 'LONG'
              ? 'bg-green-500/20 text-green-400'
              : 'bg-red-500/20 text-red-400'
          }`}
        >
          {trade.side}
        </span>
        <span className="text-xs text-slate-500 font-mono">#{trade.trade_id}</span>
      </div>
      <div className="flex items-center gap-2">
        <span
          className={`text-sm font-mono font-bold ${isWin ? 'text-green-400' : 'text-red-400'}`}
        >
          {isWin ? '+' : ''}${pnl.toFixed(2)}
        </span>
        <span
          className={`px-1.5 py-0.5 rounded text-[10px] ${
            animationType === 'bounce' ? 'bg-green-500/20 text-green-400' :
            animationType === 'fade' ? 'bg-red-500/20 text-red-400' :
            animationType === 'slide' ? 'bg-blue-500/20 text-blue-400' :
            'bg-slate-500/20 text-slate-400'
          }`}
        >
          {animationType}
        </span>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export function TradeDecisionPanel({
  trade,
  isVisible = true,
  className = '',
  compact = false,
}: TradeDecisionPanelProps) {
  if (!trade || !isVisible) {
    return null;
  }

  const enriched = isEnrichedTrade(trade) ? trade : null;
  const metadata = enriched?.model_metadata;
  const features = enriched?.features_snapshot;
  const regime = enriched?.market_regime || 'unknown';
  const mae = enriched?.max_adverse_excursion ?? null;
  const mfe = enriched?.max_favorable_excursion ?? null;

  // Fallback confidence from basic trade
  const confidence = metadata?.confidence ?? trade.confidence ?? 0.5;

  if (compact) {
    return (
      <Card className={`bg-slate-900/90 border-slate-700 backdrop-blur-sm ${className}`}>
        <CardContent className="p-3 space-y-2">
          <TradeInfoHeader trade={trade} />
          <ConfidenceMeter value={confidence} />
          {metadata && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>Entropy: {metadata.entropy?.toFixed(3) ?? '-'}</span>
              <span className="text-slate-600">|</span>
              <MarketRegimeBadge regime={regime} />
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`bg-slate-900/90 border-slate-700 backdrop-blur-sm ${className}`}>
      <CardHeader className="py-2 px-3">
        <CardTitle className="text-sm font-medium text-slate-300 flex items-center gap-2">
          <Brain className="w-4 h-4 text-cyan-400" />
          Decisión del Modelo
        </CardTitle>
      </CardHeader>
      <CardContent className="px-3 pb-3 space-y-3">
        {/* Trade Info */}
        <TradeInfoHeader trade={trade} />

        {/* Confidence Meter */}
        <ConfidenceMeter value={confidence} label="Confianza de Entrada" />

        {/* Action Probabilities */}
        {metadata?.action_probs && (
          <ActionProbabilities probs={metadata.action_probs} />
        )}

        {/* Entropy & Critic Value */}
        {metadata && (
          <div className="space-y-1.5 pt-2 border-t border-slate-700">
            <EntropyIndicator entropy={metadata.entropy} confidence={confidence} />
            {metadata.critic_value !== null && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center gap-1">
                  <Target className="w-3 h-3" />
                  Critic Value
                </span>
                <span className="font-mono text-slate-300">
                  {metadata.critic_value.toFixed(3)}
                </span>
              </div>
            )}
            {metadata.advantage !== undefined && (
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400 flex items-center gap-1">
                  <BarChart3 className="w-3 h-3" />
                  Advantage
                </span>
                <span className={`font-mono ${metadata.advantage > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {metadata.advantage.toFixed(3)}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Market Regime */}
        <div className="pt-2 border-t border-slate-700">
          <MarketRegimeBadge regime={regime} />
        </div>

        {/* Features Snapshot */}
        {features && (
          <div className="pt-2 border-t border-slate-700">
            <FeaturesGrid features={features} />
          </div>
        )}

        {/* MAE/MFE */}
        {(mae !== null || mfe !== null) && (
          <div className="pt-2 border-t border-slate-700">
            <ExcursionDisplay mae={mae} mfe={mfe} />
          </div>
        )}

        {/* Hold Time */}
        {(trade.hold_time_minutes || trade.duration_minutes) && (
          <div className="flex items-center justify-between text-xs pt-2 border-t border-slate-700">
            <span className="text-slate-400 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Duración
            </span>
            <span className="font-mono text-slate-300">
              {trade.hold_time_minutes || trade.duration_minutes} min
            </span>
          </div>
        )}

        {/* No Metadata Warning */}
        {!enriched && (
          <div className="px-2 py-1.5 bg-slate-800 rounded border border-slate-700 text-xs text-slate-500 text-center">
            Metadata del modelo no disponible
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default TradeDecisionPanel;
