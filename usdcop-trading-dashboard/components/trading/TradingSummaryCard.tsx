'use client';

/**
 * TradingSummaryCard Component
 * ============================
 * Displays a summary of the trading performance for the selected model.
 * Shows zeros when no backtest has been executed.
 *
 * DYNAMIC: All values come from database/API, no hardcoded demo data.
 */

import React from 'react';
import { useSelectedModel } from '@/contexts/ModelContext';
import { useModelMetrics } from '@/hooks/useModelMetrics';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { CalendarDays, DollarSign, TrendingUp, Activity, AlertCircle } from 'lucide-react';

interface TradingSummaryCardProps {
  className?: string;
  isReplayMode?: boolean;
  /** Force empty state (used when clearing dashboard) */
  forceEmpty?: boolean;
  replayVisibleTrades?: Array<{
    pnl?: number;
    pnl_usd?: number;
    pnl_percent?: number;
    pnl_pct?: number;
    duration_minutes?: number;
    hold_time_minutes?: number;
    timestamp?: string;
  }>;
  /** Backtest summary data passed from parent after backtest completes */
  backtestSummary?: {
    totalTrades: number;
    winRate: number;
    totalPnl: number;
    totalPnlPct: number;
    tradingDays: number;
  } | null;
}

// Initial capital for all calculations
const INITIAL_CAPITAL = 10000;

export function TradingSummaryCard({
  className = '',
  isReplayMode = false,
  forceEmpty = false,
  replayVisibleTrades = [],
  backtestSummary = null,
}: TradingSummaryCardProps) {
  const { model } = useSelectedModel();
  const { metrics, isLoading } = useModelMetrics();

  // Check if model is in testing mode (not yet promoted)
  const isTestingMode = model?.dbStatus === 'registered';

  // Calculate replay metrics from visible trades
  const replayData = React.useMemo(() => {
    if (!isReplayMode || replayVisibleTrades.length === 0) return null;

    const trades = replayVisibleTrades;
    const pnls = trades.map(t => t.pnl ?? t.pnl_usd ?? 0);
    const totalPnl = pnls.reduce((sum, p) => sum + p, 0);
    const wins = pnls.filter(p => p > 0).length;

    const uniqueDays = new Set(
      trades
        .filter(t => t.timestamp)
        .map(t => new Date(t.timestamp!).toDateString())
    );

    return {
      currentEquity: INITIAL_CAPITAL + totalPnl,
      profit: totalPnl,
      profitPct: (totalPnl / INITIAL_CAPITAL) * 100,
      totalTrades: trades.length,
      tradingDays: uniqueDays.size,
      winRate: trades.length > 0 ? (wins / trades.length) * 100 : 0,
    };
  }, [isReplayMode, replayVisibleTrades]);

  // Get metrics from API (only for deployed/production models)
  const liveMetrics = (metrics?.live || {}) as any;

  // Determine data source priority:
  // 0. Force empty when dashboard is cleared
  // 1. Replay mode with trades -> use replay data
  // 2. Backtest summary passed from parent -> use backtest data
  // 3. Live metrics from API (only if model is deployed)
  // 4. Default to zeros (no data yet)

  let currentEquity = INITIAL_CAPITAL;
  let profit = 0;
  let profitPct = 0;
  let totalTrades = 0;
  let tradingDays = 0;
  let winRate = 0;
  let startDateStr = '-';
  let dataSource: 'replay' | 'backtest' | 'live' | 'none' = 'none';

  // Force empty state bypasses all data sources
  if (forceEmpty) {
    // Keep all defaults (zeros)
    dataSource = 'none';
  } else if (isReplayMode && replayData) {
    // Replay mode with actual trades
    currentEquity = replayData.currentEquity;
    profit = replayData.profit;
    profitPct = replayData.profitPct;
    totalTrades = replayData.totalTrades;
    tradingDays = replayData.tradingDays;
    winRate = replayData.winRate;
    startDateStr = 'Replay Mode';
    dataSource = 'replay';
  } else if (backtestSummary && backtestSummary.totalTrades > 0) {
    // Backtest completed - use backtest summary
    profit = backtestSummary.totalPnl;
    profitPct = backtestSummary.totalPnlPct;
    totalTrades = backtestSummary.totalTrades;
    tradingDays = backtestSummary.tradingDays;
    winRate = backtestSummary.winRate;
    currentEquity = INITIAL_CAPITAL + profit;
    startDateStr = 'Backtest';
    dataSource = 'backtest';
  } else if (!isTestingMode && liveMetrics.totalTrades > 0) {
    // Live production data (only for deployed models)
    profit = liveMetrics.pnlMonth ?? 0;
    profitPct = liveMetrics.pnlMonthPct ?? 0;
    totalTrades = liveMetrics.totalTrades ?? 0;
    tradingDays = liveMetrics.tradingDays ?? 0;
    winRate = liveMetrics.winRate ?? 0;
    currentEquity = INITIAL_CAPITAL + profit;
    startDateStr = 'Live';
    dataSource = 'live';
  }
  // else: keep defaults (zeros) - no data yet

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
          {isReplayMode && (
            <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 ml-auto">
              REPLAY
            </span>
          )}
          {!isReplayMode && model && (
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
            ${INITIAL_CAPITAL.toLocaleString()}
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

        {/* Status indicator based on data source */}
        {dataSource === 'none' && (
          <div className="mt-2 px-2 py-1.5 rounded bg-slate-700/50 border border-slate-600/30 flex items-center gap-2">
            <AlertCircle className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs text-slate-400">
              Ejecuta el backtest para ver métricas
            </span>
          </div>
        )}
        {dataSource === 'backtest' && (
          <div className="mt-2 px-2 py-1.5 rounded bg-cyan-500/10 border border-cyan-500/20">
            <span className="text-xs text-cyan-400">
              Datos de backtest - Promueve a producción para activar realtime
            </span>
          </div>
        )}
        {dataSource === 'replay' && (
          <div className="mt-2 px-2 py-1.5 rounded bg-purple-500/10 border border-purple-500/20">
            <span className="text-xs text-purple-400">
              Modo Replay - {totalTrades} trades visualizados
            </span>
          </div>
        )}
        {dataSource === 'live' && (
          <div className="mt-2 px-2 py-1.5 rounded bg-emerald-500/10 border border-emerald-500/20">
            <span className="text-xs text-emerald-400">
              Datos en tiempo real - Producción
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default TradingSummaryCard;
