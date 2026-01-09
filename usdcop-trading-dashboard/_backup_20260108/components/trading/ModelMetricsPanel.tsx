'use client';

/**
 * ModelMetricsPanel Component
 * ============================
 * Displays key performance metrics for the selected model.
 * Shows live metrics compared to backtest benchmarks.
 */

import React from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import { useModelSignals } from '@/hooks/useModelSignals';
import { useModelMetrics } from '@/hooks/useModelMetrics';
import {
  getSignalBadgeProps,
  formatPnL,
  formatPct,
  getPnLColorClass,
} from '@/lib/config/models.config';
import { cn } from '@/lib/utils';

interface ModelMetricsPanelProps {
  className?: string;
}

export function ModelMetricsPanel({ className }: ModelMetricsPanelProps) {
  const { model, isLoading: isModelLoading } = useSelectedModel();
  const { latestSignal, isConnected, latency } = useModelSignals();
  const { metrics, comparisons, isLoading: isMetricsLoading } = useModelMetrics();

  const isLoading = isModelLoading || isMetricsLoading;

  if (isLoading) {
    return (
      <div className={cn('grid grid-cols-6 gap-4', className)}>
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="h-24 animate-pulse rounded-lg bg-slate-800"
          />
        ))}
      </div>
    );
  }

  if (!model) {
    return (
      <div className={cn('rounded-lg bg-slate-800 p-4 text-center', className)}>
        <p className="text-slate-400">Selecciona un modelo para ver métricas</p>
      </div>
    );
  }

  const signalProps = latestSignal
    ? getSignalBadgeProps(latestSignal.signal)
    : null;

  return (
    <div className={cn('grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6', className)}>
      {/* Price with connection status */}
      <MetricCard
        label="PRECIO"
        value={
          latestSignal?.price
            ? `$${latestSignal.price.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}`
            : '—'
        }
        subValue={isConnected ? `${latency}ms` : 'Desconectado'}
        color={isConnected ? 'green' : 'red'}
        modelColor={model.color}
      />

      {/* Current Signal */}
      <MetricCard
        label="SEÑAL"
        value={
          signalProps ? (
            <span style={{ color: signalProps.color }}>
              {signalProps.icon} {signalProps.label}
            </span>
          ) : (
            '—'
          )
        }
        subValue={
          latestSignal
            ? `${Math.round(latestSignal.confidence * 100)}% conf.`
            : ''
        }
        modelColor={model.color}
      />

      {/* P&L Today */}
      <MetricCard
        label="P&L HOY"
        value={formatPnL(metrics?.live.pnlToday ?? null)}
        subValue={formatPct(metrics?.live.pnlTodayPct ?? null)}
        color={
          metrics?.live.pnlToday
            ? metrics.live.pnlToday > 0
              ? 'green'
              : 'red'
            : undefined
        }
        modelColor={model.color}
      />

      {/* Sharpe Ratio */}
      <MetricCard
        label="SHARPE"
        value={metrics?.live.sharpe?.toFixed(2) ?? '—'}
        subValue={`(${model.backtest?.sharpe?.toFixed(2) ?? '?'} BT)`}
        color={
          comparisons?.sharpe.isBetter === true
            ? 'green'
            : comparisons?.sharpe.isBetter === false
            ? 'amber'
            : undefined
        }
        delta={comparisons?.sharpe.delta}
        modelColor={model.color}
      />

      {/* Max Drawdown */}
      <MetricCard
        label="MAX DD"
        value={
          metrics?.live.maxDrawdown !== null
            ? `${(metrics.live.maxDrawdown * 100).toFixed(2)}%`
            : '—'
        }
        subValue={`(${((model.backtest?.maxDrawdown ?? 0) * 100).toFixed(2)}% BT)`}
        color={
          comparisons?.maxDrawdown.isBetter === true
            ? 'green'
            : comparisons?.maxDrawdown.isBetter === false
            ? 'red'
            : undefined
        }
        modelColor={model.color}
      />

      {/* Win Rate */}
      <MetricCard
        label="WIN RATE"
        value={
          metrics?.live.winRate !== null ? `${metrics.live.winRate}%` : '—'
        }
        subValue={`(${model.backtest?.winRate ?? '?'}% BT)`}
        color={
          comparisons?.winRate.isBetter === true
            ? 'green'
            : comparisons?.winRate.isBetter === false
            ? 'amber'
            : undefined
        }
        modelColor={model.color}
      />
    </div>
  );
}

// ============================================================================
// MetricCard Component
// ============================================================================

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  subValue?: string;
  color?: 'green' | 'red' | 'amber' | 'slate';
  delta?: number | null;
  modelColor?: string;
}

function MetricCard({
  label,
  value,
  subValue,
  color,
  delta,
  modelColor,
}: MetricCardProps) {
  const colorClasses: Record<string, string> = {
    green: 'border-green-500/30 bg-green-500/10',
    red: 'border-red-500/30 bg-red-500/10',
    amber: 'border-amber-500/30 bg-amber-500/10',
    slate: 'border-slate-700 bg-slate-800/50',
  };

  const borderClass = color
    ? colorClasses[color]
    : 'border-slate-700 bg-slate-800/50';

  return (
    <div
      className={cn(
        'rounded-lg border p-4 transition-all',
        borderClass
      )}
    >
      {/* Label with model color accent */}
      <div className="mb-1 flex items-center gap-2">
        {modelColor && (
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: modelColor }}
          />
        )}
        <span className="text-xs font-medium text-slate-400">{label}</span>
      </div>

      {/* Value */}
      <div className="text-xl font-bold text-white">{value}</div>

      {/* Sub value with delta indicator */}
      {(subValue || delta !== null) && (
        <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
          {subValue}
          {delta !== null && delta !== undefined && (
            <span
              className={cn(
                'rounded px-1',
                delta > 0
                  ? 'bg-green-500/20 text-green-400'
                  : delta < 0
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-slate-500/20 text-slate-400'
              )}
            >
              {delta > 0 ? '+' : ''}
              {delta.toFixed(2)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default ModelMetricsPanel;
