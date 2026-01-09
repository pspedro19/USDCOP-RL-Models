'use client';

/**
 * TradingSummaryCard Component
 * ============================
 * Displays a summary of the trading performance for the selected model:
 * - Start date
 * - Initial capital
 * - Current equity
 * - Profit/Loss
 * - Trading days
 */

import React from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import { useModelMetrics } from '@/hooks/useModelMetrics';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { CalendarDays, DollarSign, TrendingUp, Activity } from 'lucide-react';

interface TradingSummaryCardProps {
  className?: string;
}

/**
 * Format date to Colombia Time (COT = UTC-5)
 */
function formatDateCOT(dateStr: string | null | undefined): string {
  if (!dateStr) return '-';
  const date = new Date(dateStr);
  // Convert to COT
  const cotDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  return cotDate.toISOString().split('T')[0];
}

// Project-wide start date (when paper trading began)
const PROJECT_START_DATE = '2025-12-29';

export function TradingSummaryCard({ className = '' }: TradingSummaryCardProps) {
  const { model } = useSelectedModel();
  const { metrics, isLoading } = useModelMetrics();

  const initialCapital = 10000;

  // Get metrics from API response - the hook now normalizes data with live sub-object
  const liveMetrics = (metrics?.live || {}) as any;

  // Equity data - use live metrics which now includes currentEquity and startEquity
  const startEquity = liveMetrics.startEquity || initialCapital;
  const currentEquity = liveMetrics.currentEquity || (startEquity + (liveMetrics.pnlMonth || 0));

  // P&L calculations
  const profit = liveMetrics.pnlMonth ?? (currentEquity - startEquity);
  const profitPct = liveMetrics.pnlMonthPct ?? ((profit / startEquity) * 100);

  // Trading period data
  const totalTrades = liveMetrics.totalTrades ?? 0;
  const tradingDays = liveMetrics.tradingDays ?? Math.max(1, Math.ceil(totalTrades / 8));
  // Use consistent project start date for all models
  const startDateStr = PROJECT_START_DATE;
  const winRate = liveMetrics.winRate ?? 0;

  if (isLoading) {
    return (
      <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm font-medium text-slate-400">
            Trading Summary
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4 space-y-3">
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-slate-700 rounded w-3/4" />
            <div className="h-4 bg-slate-700 rounded w-1/2" />
            <div className="h-4 bg-slate-700 rounded w-2/3" />
            <div className="h-6 bg-slate-700 rounded w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm font-medium text-slate-400 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Trading Summary
          {model && (
            <span
              className="ml-auto text-xs px-2 py-0.5 rounded"
              style={{ backgroundColor: `${model.color}20`, color: model.color }}
            >
              {model.name}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 space-y-3">
        {/* Start Date */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <CalendarDays className="w-3.5 h-3.5" />
            <span className="text-sm">Start Date</span>
          </div>
          <span className="font-mono text-sm text-slate-300">{startDateStr}</span>
        </div>

        {/* Initial Capital */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <DollarSign className="w-3.5 h-3.5" />
            <span className="text-sm">Initial Capital</span>
          </div>
          <span className="font-mono text-sm text-slate-300">
            ${initialCapital.toLocaleString()}
          </span>
        </div>

        {/* Current Equity */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <TrendingUp className="w-3.5 h-3.5" />
            <span className="text-sm">Current Equity</span>
          </div>
          <span className="font-mono text-sm font-bold text-white">
            ${currentEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        {/* Trading Days */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <CalendarDays className="w-3.5 h-3.5" />
            <span className="text-sm">Días de Trading</span>
          </div>
          <span className="font-mono text-sm text-slate-300">{tradingDays}</span>
        </div>

        {/* Total Trades */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <Activity className="w-3.5 h-3.5" />
            <span className="text-sm">Total Operaciones</span>
          </div>
          <span className="font-mono text-sm text-slate-300">{totalTrades}</span>
        </div>

        {/* Win Rate */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <TrendingUp className="w-3.5 h-3.5" />
            <span className="text-sm">Win Rate</span>
          </div>
          <span className={`font-mono text-sm ${winRate >= 50 ? 'text-green-400' : 'text-amber-400'}`}>
            {winRate.toFixed(1)}%
          </span>
        </div>

        {/* Profit/Loss - Highlighted */}
        <div className="border-t border-slate-700 pt-3 mt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-400">Profit/Loss</span>
            <div className="text-right">
              <div
                className={`font-mono text-lg font-bold ${
                  profit >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {profit >= 0 ? '+' : ''}
                ${profit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div
                className={`text-xs font-mono ${
                  profitPct >= 0 ? 'text-green-400/70' : 'text-red-400/70'
                }`}
              >
                {profitPct >= 0 ? '+' : ''}
                {profitPct.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>

        {/* Demo indicator if not real data */}
        {model && !model.isRealData && (
          <div className="mt-2 px-2 py-1.5 rounded bg-amber-500/10 border border-amber-500/20">
            <span className="text-xs text-amber-400">
              Demo data - For illustration purposes only
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default TradingSummaryCard;
