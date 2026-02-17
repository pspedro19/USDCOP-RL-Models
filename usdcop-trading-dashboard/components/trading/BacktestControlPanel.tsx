'use client';

/**
 * Backtest Control Panel - Minimalist Design
 * ==========================================
 * Professional, centered UI for running backtests.
 *
 * Features:
 * - Simplified date presets: Validación, Test, Ambos, Personalizado
 * - Centered, minimalist layout
 * - Real-time progress with SSE
 * - Auto-loads custom dates from validation/test range
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import {
  BacktestResult,
  BacktestSummary,
  BacktestTradeEvent,
  getStatusMessage,
  isBacktestComplete,
  REPLAY_SPEEDS,
  type ReplaySpeed,
  DEFAULT_REPLAY_SPEED,
} from '@/lib/contracts/backtest.contract';
import {
  getDateRangePresets,
  DateRangePreset,
  formatDateRange,
  runRealBacktest,
  RealBacktestResult,
} from '@/lib/services/backtest.service';
import useBacktest from '@/hooks/useBacktest';
import { cn } from '@/lib/utils';
import { Play, X, RotateCcw, Calendar, ChevronDown, ChevronUp, Loader2, Rocket, CheckCircle2, Trash2, Gauge } from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

interface BacktestControlPanelProps {
  models: Array<{ id: string; name: string }>;
  selectedModelId: string;
  onModelChange?: (modelId: string) => void;
  onBacktestComplete?: (result: BacktestResult, startDate: string, endDate: string) => void;
  /** Called for each trade as it's generated - for real-time equity curve updates */
  onTradeGenerated?: (trade: BacktestTradeEvent) => void;
  /** Called when backtest starts - use to clear previous streaming state */
  onBacktestStart?: () => void;
  onClearDashboard?: () => void;
  /** Enable animated replay mode (1 bar/second) */
  enableAnimatedReplay?: boolean;
  expanded?: boolean;
  onToggleExpand?: () => void;
  /** Hide the collapsed header bar - use when toggle is external */
  hideHeader?: boolean;
  /** Proposal ID for pending experiments - uses REAL BacktestEngine when provided */
  proposalId?: string;
}

// ============================================================================
// Sub-components
// ============================================================================

interface PresetChipProps {
  preset: DateRangePreset;
  selected: boolean;
  onClick: () => void;
  showDates?: boolean;
}

function PresetChip({ preset, selected, onClick, showDates = true }: PresetChipProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex flex-col items-center px-2 sm:px-4 py-1.5 sm:py-2 rounded-lg sm:rounded-xl transition-all duration-200",
        "border min-w-[70px] sm:min-w-[100px]",
        selected
          ? "bg-cyan-500/20 border-cyan-500/50"
          : "bg-slate-800/50 border-slate-700 hover:border-slate-600"
      )}
      title={preset.description}
    >
      <span className={cn(
        "text-xs sm:text-sm font-medium",
        selected ? "text-cyan-400" : "text-slate-300"
      )}>
        {preset.label}
      </span>
      {showDates && preset.id !== 'custom' && (
        <span className={cn(
          "text-[8px] sm:text-[10px] mt-0.5",
          selected ? "text-cyan-400/70" : "text-slate-500"
        )}>
          {preset.startDate.slice(5)} → {preset.endDate.slice(5)}
        </span>
      )}
    </button>
  );
}


interface CompactSummaryProps {
  summary: BacktestSummary;
  tradeCount: number;
}

function CompactSummary({ summary, tradeCount }: CompactSummaryProps) {
  const metrics = [
    { label: 'Trades', value: tradeCount, color: 'text-white' },
    { label: 'Win Rate', value: `${summary.win_rate.toFixed(1)}%`, color: summary.win_rate >= 50 ? 'text-emerald-400' : 'text-red-400' },
    { label: 'PnL', value: `$${summary.total_pnl.toFixed(0)}`, color: summary.total_pnl >= 0 ? 'text-emerald-400' : 'text-red-400' },
    { label: 'Return', value: `${summary.total_return_pct.toFixed(1)}%`, color: summary.total_return_pct >= 0 ? 'text-emerald-400' : 'text-red-400' },
    { label: 'Max DD', value: `${summary.max_drawdown_pct.toFixed(1)}%`, color: summary.max_drawdown_pct <= 10 ? 'text-amber-400' : 'text-red-400' },
  ];

  return (
    <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 sm:gap-4 py-2 sm:py-3 px-3 sm:px-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
      {metrics.map((m) => (
        <div key={m.label} className="text-center">
          <div className={cn("text-xs sm:text-sm font-bold", m.color)}>{m.value}</div>
          <div className="text-[9px] sm:text-[10px] text-slate-500 uppercase">{m.label}</div>
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function BacktestControlPanel({
  models,
  selectedModelId,
  onModelChange,
  onBacktestComplete,
  onTradeGenerated,
  onBacktestStart,
  onClearDashboard,
  enableAnimatedReplay = true,
  expanded = false,
  onToggleExpand,
  hideHeader = false,
  proposalId,
}: BacktestControlPanelProps) {
  const datePresets = useMemo(() => getDateRangePresets(), []);

  const [selectedPreset, setSelectedPreset] = useState<string>('backtest_2025');
  const [customStartDate, setCustomStartDate] = useState<string>('2025-01-01');
  const [customEndDate, setCustomEndDate] = useState<string>('2025-12-31');
  const [forceRegenerate, setForceRegenerate] = useState<boolean>(true);
  const [replaySpeed, setReplaySpeed] = useState<ReplaySpeed>(DEFAULT_REPLAY_SPEED);


  // Promote to production state
  const [isPromoting, setIsPromoting] = useState<boolean>(false);
  const [promoteSuccess, setPromoteSuccess] = useState<boolean>(false);
  const [promoteError, setPromoteError] = useState<string | null>(null);

  // SSE backtest progress state
  const [sseProgress, setSseProgress] = useState<number>(0);
  const [sseStatus, setSseStatus] = useState<string>('');
  const [sseTradesCount, setSseTradesCount] = useState<number>(0);


  const currentDateRange = useMemo(() => {
    if (selectedPreset === 'custom') {
      return { startDate: customStartDate, endDate: customEndDate };
    }
    const preset = datePresets.find((p) => p.id === selectedPreset);
    return {
      startDate: preset?.startDate || '',
      endDate: preset?.endDate || '',
    };
  }, [selectedPreset, customStartDate, customEndDate, datePresets]);

  const {
    state,
    startBacktest,
    cancelBacktest,
    resetBacktest,
    isRunning,
    canStart,
    progressPercent,
    elapsedTime,
  } = useBacktest({
    onComplete: (result) => {
      onBacktestComplete?.(result, currentDateRange.startDate, currentDateRange.endDate);
    },
    onTrade: (trade) => {
      // Forward real-time trade events to parent for live equity curve updates
      onTradeGenerated?.(trade);
    },
    onError: (error) => {
      console.error('[BacktestPanel] Error:', error.message);
    },
  });

  const handlePresetSelect = useCallback((presetId: string) => {
    setSelectedPreset(presetId);
    if (presetId !== 'custom') {
      const preset = datePresets.find((p) => p.id === presetId);
      if (preset) {
        setCustomStartDate(preset.startDate);
        setCustomEndDate(preset.endDate);
      }
    }
  }, [datePresets]);

  // State for real backtest loading
  const [isRealBacktestRunning, setIsRealBacktestRunning] = useState(false);

  const handleStartBacktest = useCallback(async () => {
    if (!currentDateRange.startDate || !currentDateRange.endDate) return;

    // Reset promote state when starting new backtest
    setPromoteSuccess(false);
    setPromoteError(null);

    // Notify parent that backtest is starting (clears streaming trades, etc.)
    onBacktestStart?.();

    // Use REAL backtest with SSE streaming when proposalId is provided (pending experiment)
    if (proposalId) {
      console.log(`[BacktestPanel] Using REAL backtest SSE for proposal ${proposalId}`);
      setIsRealBacktestRunning(true);
      setSseProgress(0);
      setSseStatus('Connecting...');
      setSseTradesCount(0);

      // Collect trades as they stream in
      const streamedTrades: BacktestTradeEvent[] = [];
      let modelId = '';

      try {
        const url = `/api/backtest/real/stream?proposal_id=${encodeURIComponent(proposalId)}&start_date=${currentDateRange.startDate}&end_date=${currentDateRange.endDate}`;

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`SSE connection failed: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                switch (data.type) {
                  case 'status':
                    setSseStatus(data.message);
                    console.log(`[BacktestPanel SSE] Status: ${data.message}`);
                    break;

                  case 'progress':
                    setSseProgress(data.percent);
                    setSseTradesCount(data.trades_so_far || 0);
                    break;

                  case 'trade':
                    // Emit trade immediately for real-time equity curve update
                    const trade = data.data;
                    modelId = trade.model_id || modelId;
                    const tradeEvent: BacktestTradeEvent = {
                      ...trade,
                      trade_id: String(trade.trade_id),
                      model_id: modelId,
                      timestamp: trade.entry_time,
                    };
                    streamedTrades.push(tradeEvent);
                    onTradeGenerated?.(tradeEvent);
                    setSseTradesCount(streamedTrades.length);
                    console.log(`[BacktestPanel SSE] Trade #${trade.trade_id}: ${trade.side} @ ${trade.entry_price}`);
                    break;

                  case 'complete':
                    // Build final result from complete event
                    modelId = data.model_id || modelId;
                    const backtestResult: BacktestResult = {
                      success: data.success,
                      source: 'real_backtest_stream',
                      trade_count: data.trade_count,
                      trades: streamedTrades,
                      summary: {
                        total_trades: data.summary.total_trades,
                        winning_trades: data.summary.winning_trades,
                        losing_trades: data.summary.losing_trades,
                        win_rate: data.summary.win_rate,
                        total_pnl: data.summary.total_pnl,
                        total_return_pct: data.summary.total_return_pct,
                        max_drawdown_pct: data.summary.max_drawdown_pct,
                        sharpe_ratio: data.summary.sharpe_ratio,
                        avg_trade_duration_minutes: 0,
                      },
                      date_range: data.date_range,
                    };

                    setSseProgress(100);
                    setSseStatus('Complete!');
                    onBacktestComplete?.(backtestResult, currentDateRange.startDate, currentDateRange.endDate);
                    console.log(`[BacktestPanel SSE] Complete: ${data.trade_count} trades, ${data.summary.win_rate}% WR, Sharpe: ${data.summary.sharpe_ratio}`);
                    break;

                  case 'error':
                    console.error(`[BacktestPanel SSE] Error: ${data.message}`);
                    setSseStatus(`Error: ${data.message}`);
                    break;
                }
              } catch (e) {
                console.warn('[BacktestPanel SSE] Parse error:', e, line);
              }
            }
          }
        }

      } catch (error) {
        console.error('[BacktestPanel SSE] Error:', error);
        setSseStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      } finally {
        setIsRealBacktestRunning(false);
      }

      return;
    }

    // Standard streaming backtest (for non-pending models)
    await startBacktest({
      startDate: currentDateRange.startDate,
      endDate: currentDateRange.endDate,
      modelId: selectedModelId,
      forceRegenerate,
      replaySpeed,
      emitBarEvents: enableAnimatedReplay,
    });
  }, [currentDateRange, selectedModelId, forceRegenerate, replaySpeed, enableAnimatedReplay, startBacktest, onBacktestStart, proposalId, onBacktestComplete, onTradeGenerated]);

  const handlePromoteToProduction = useCallback(async () => {
    setIsPromoting(true);
    setPromoteError(null);

    try {
      const response = await fetch(`/api/models/${selectedModelId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to promote model');
      }

      setPromoteSuccess(true);
      console.log(`[BacktestPanel] Model promoted: ${data.message}`);

      // Refresh the page to reflect the new status
      setTimeout(() => {
        window.location.reload();
      }, 1500);

    } catch (error) {
      console.error('[BacktestPanel] Promote error:', error);
      setPromoteError(error instanceof Error ? error.message : 'Failed to promote model');
    } finally {
      setIsPromoting(false);
    }
  }, [selectedModelId]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const isValidDateRange = Boolean(currentDateRange.startDate && currentDateRange.endDate);
  const isAnyBacktestRunning = isRunning || isRealBacktestRunning;
  const canStartNow = (canStart || proposalId) && isValidDateRange && !isRealBacktestRunning;
  const showSummary = isBacktestComplete(state.status) && state.result?.summary;

  return (
    <div className="w-full flex justify-center">
      <div className="w-full max-w-2xl">
        {/* Collapsed Header - Hidden when external toggle is used */}
        {!hideHeader && (
          <div
            className={cn(
              "flex items-center justify-between px-4 py-3 rounded-xl cursor-pointer transition-all",
              "bg-slate-900/80 backdrop-blur border border-slate-700/50",
              expanded && "rounded-b-none border-b-0"
            )}
            onClick={onToggleExpand}
          >
            <div className="flex items-center gap-3">
              <div className="p-1.5 rounded-lg bg-cyan-500/10">
                <Calendar className="w-4 h-4 text-cyan-400" />
              </div>
              <span className="text-sm font-medium text-white">Backtest</span>
              {isAnyBacktestRunning && (
                <div className="flex items-center gap-2">
                  <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />
                  <span className="text-xs text-cyan-400">{progressPercent}%</span>
                </div>
              )}
              {showSummary && (
                <span className="text-xs text-emerald-400 font-medium">
                  ✓ {state.result!.trade_count} trades
                </span>
              )}
            </div>
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            )}
          </div>
        )}

        {/* Expanded Content */}
        {(expanded || hideHeader) && (
          <div className={cn(
            "px-3 sm:px-6 py-3 sm:py-5 space-y-3 sm:space-y-5",
            "bg-slate-900/80 backdrop-blur border border-slate-700/50",
            hideHeader ? "rounded-xl" : "rounded-b-xl border-t-0"
          )}>
            {/* Date Presets - Centered, mobile grid layout */}
            <div className="flex flex-col items-center gap-2 sm:gap-3">
              <span className="text-[10px] sm:text-xs text-slate-500 uppercase tracking-wider">Seleccionar Periodo</span>
              <div className="grid grid-cols-2 sm:flex sm:flex-wrap sm:justify-center gap-2 sm:gap-3 w-full sm:w-auto">
                {datePresets.map((preset) => (
                  <PresetChip
                    key={preset.id}
                    preset={preset}
                    selected={selectedPreset === preset.id}
                    onClick={() => handlePresetSelect(preset.id)}
                    showDates={true}
                  />
                ))}
              </div>
            </div>

            {/* Selected Date Range Display - Responsive */}
            {selectedPreset !== 'custom' && (
              <div className="flex justify-center">
                <div className="inline-flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                  <div className="flex items-center gap-1.5">
                    <Calendar className="w-3 h-3 sm:w-4 sm:h-4 text-cyan-400" />
                    <span className="text-xs sm:text-sm text-slate-300">Rango:</span>
                  </div>
                  <span className="font-mono text-xs sm:text-sm text-white font-medium">
                    {currentDateRange.startDate} → {currentDateRange.endDate}
                  </span>
                </div>
              </div>
            )}

            {/* Custom Date Inputs - Stack on mobile */}
            {selectedPreset === 'custom' && (
              <div className="flex flex-col items-center gap-2 sm:gap-3">
                <div className="flex flex-col sm:flex-row justify-center gap-2 sm:gap-4 w-full sm:w-auto">
                  <div className="flex flex-col items-center gap-1">
                    <label className="text-[10px] text-slate-500 uppercase">Desde</label>
                    <input
                      type="date"
                      value={customStartDate}
                      onChange={(e) => setCustomStartDate(e.target.value)}
                      min="2020-01-01"
                      className="px-2 sm:px-3 py-1 sm:py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs sm:text-sm text-white text-center w-full sm:w-auto"
                    />
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <label className="text-[10px] text-slate-500 uppercase">Hasta</label>
                    <input
                      type="date"
                      value={customEndDate}
                      onChange={(e) => setCustomEndDate(e.target.value)}
                      max={new Date().toISOString().split('T')[0]}
                      className="px-2 sm:px-3 py-1 sm:py-1.5 bg-slate-800 border border-slate-700 rounded-lg text-xs sm:text-sm text-white text-center w-full sm:w-auto"
                    />
                  </div>
                </div>
                <div className="inline-flex flex-col sm:flex-row items-center gap-1 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                  <div className="flex items-center gap-1.5">
                    <Calendar className="w-3 h-3 sm:w-4 sm:h-4 text-cyan-400" />
                    <span className="text-xs sm:text-sm text-slate-300">Rango:</span>
                  </div>
                  <span className="font-mono text-xs sm:text-sm text-white font-medium">
                    {customStartDate} → {customEndDate}
                  </span>
                </div>
              </div>
            )}

            {/* Progress Bar - Streaming backtest */}
            {isRunning && state.progress && (
              <div className="space-y-2">
                <div className="relative h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-slate-500">
                  <span>{state.progress.trades_generated} trades generados</span>
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400/70">{replaySpeed}x</span>
                    <span>{formatTime(elapsedTime)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Progress Bar - Real backtest with SSE streaming */}
            {isRealBacktestRunning && (
              <div className="space-y-2">
                {/* Progress bar - shows actual percentage from SSE */}
                <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-300 ease-out"
                    style={{ width: `${Math.max(sseProgress, 2)}%` }}
                  />
                  {sseProgress < 100 && sseProgress > 0 && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse" />
                  )}
                </div>
                <div className="flex justify-between text-xs">
                  <span className="flex items-center gap-2 text-slate-400">
                    <Loader2 className="w-3 h-3 animate-spin text-emerald-400" />
                    {sseStatus || 'Iniciando...'}
                  </span>
                  <div className="flex items-center gap-3">
                    {sseTradesCount > 0 && (
                      <span className="text-cyan-400">
                        {sseTradesCount} trades
                      </span>
                    )}
                    <span className="text-emerald-400 font-medium">
                      {sseProgress.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Summary */}
            {showSummary && (
              <CompactSummary
                summary={state.result!.summary}
                tradeCount={state.result!.trade_count}
              />
            )}

            {/* Promote to Production Button - Shows after successful backtest */}
            {showSummary && !promoteSuccess && (
              <div className="flex flex-col items-center gap-1.5 sm:gap-2">
                <button
                  onClick={handlePromoteToProduction}
                  disabled={isPromoting}
                  className={cn(
                    "flex items-center gap-1.5 sm:gap-2 px-4 sm:px-6 py-2 sm:py-2.5 rounded-full font-medium text-xs sm:text-sm transition-all",
                    "bg-gradient-to-r from-emerald-500 to-green-500 text-white",
                    "hover:shadow-lg hover:shadow-emerald-500/25",
                    isPromoting && "opacity-50 cursor-not-allowed"
                  )}
                >
                  {isPromoting ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
                      <span className="hidden sm:inline">Promoviendo...</span>
                      <span className="sm:hidden">...</span>
                    </>
                  ) : (
                    <>
                      <Rocket className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                      <span className="hidden sm:inline">Pasar a Producción</span>
                      <span className="sm:hidden">Producción</span>
                    </>
                  )}
                </button>
                <p className="text-[9px] sm:text-[10px] text-slate-500 text-center max-w-[200px] sm:max-w-xs">
                  Persistir trades y marcar deployed
                </p>
                {promoteError && (
                  <p className="text-[10px] sm:text-xs text-red-400">{promoteError}</p>
                )}
              </div>
            )}

            {/* Promote Success Message */}
            {promoteSuccess && (
              <div className="flex items-center justify-center gap-1.5 sm:gap-2 py-2 sm:py-3 px-3 sm:px-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                <CheckCircle2 className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-400" />
                <p className="text-xs sm:text-sm text-emerald-400 font-medium">
                  Promovido. Recargando...
                </p>
              </div>
            )}

            {/* Error */}
            {state.status === 'error' && state.error && (
              <div className="text-center py-1.5 sm:py-2 px-3 sm:px-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-[10px] sm:text-xs text-red-400">{state.error}</p>
              </div>
            )}

            {/* Speed Control - Dynamic bar-by-bar replay */}
            {enableAnimatedReplay && (
              <div className="flex flex-col items-center gap-2">
                <div className="flex items-center gap-2">
                  <Gauge className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="text-[10px] sm:text-xs text-slate-500 uppercase tracking-wider">Velocidad de Replay</span>
                </div>
                <div className="flex flex-wrap justify-center gap-1.5 sm:gap-2">
                  {REPLAY_SPEEDS.map((speed) => (
                    <button
                      key={speed}
                      onClick={() => setReplaySpeed(speed)}
                      disabled={isAnyBacktestRunning}
                      className={cn(
                        "px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg text-[10px] sm:text-xs font-medium transition-all",
                        "border min-w-[40px] sm:min-w-[50px]",
                        replaySpeed === speed
                          ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400"
                          : "bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-300",
                        isAnyBacktestRunning && "opacity-50 cursor-not-allowed"
                      )}
                    >
                      {speed}x
                    </button>
                  ))}
                </div>
                <p className="text-[9px] text-slate-600 text-center">
                  {replaySpeed < 1 ? 'Cámara lenta' : replaySpeed === 1 ? 'Tiempo real' : `${replaySpeed}x más rápido`}
                </p>
              </div>
            )}

            {/* Action Buttons - Mobile-first centered */}
            <div className="flex flex-col items-center gap-2 sm:gap-3">
              <div className="flex flex-wrap justify-center gap-2 sm:gap-3">
                {/* Clear Dashboard Button */}
                {onClearDashboard && (
                  <button
                    onClick={onClearDashboard}
                    disabled={isAnyBacktestRunning}
                    className={cn(
                      "flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-full font-medium text-xs sm:text-sm transition-all",
                      "bg-slate-800 text-slate-400 border border-slate-700 hover:text-white hover:border-slate-600",
                      isAnyBacktestRunning && "opacity-50 cursor-not-allowed"
                    )}
                    title="Limpiar todos los datos del dashboard"
                  >
                    <Trash2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    <span className="hidden sm:inline">Limpiar</span>
                  </button>
                )}

                {(canStart || (proposalId && !isRealBacktestRunning)) ? (
                  <button
                    onClick={handleStartBacktest}
                    disabled={!canStartNow}
                    className={cn(
                      "flex items-center gap-1.5 sm:gap-2 px-4 sm:px-8 py-2 sm:py-2.5 rounded-full font-medium text-xs sm:text-sm transition-all",
                      canStartNow
                        ? proposalId
                          ? "bg-gradient-to-r from-emerald-500 to-cyan-500 text-white hover:shadow-lg hover:shadow-emerald-500/25"
                          : "bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:shadow-lg hover:shadow-cyan-500/25"
                        : "bg-slate-800 text-slate-500 cursor-not-allowed"
                    )}
                  >
                    <Play className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    {proposalId ? 'Backtest Real' : 'Iniciar'}
                  </button>
                ) : isAnyBacktestRunning ? (
                  <button
                    onClick={proposalId ? undefined : cancelBacktest}
                    disabled={!!proposalId}
                    className={cn(
                      "flex items-center gap-1.5 sm:gap-2 px-4 sm:px-8 py-2 sm:py-2.5 rounded-full font-medium text-xs sm:text-sm transition-all",
                      proposalId
                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                        : "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30"
                    )}
                  >
                    {proposalId ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin" />
                        Ejecutando...
                      </>
                    ) : (
                      <>
                        <X className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                        Cancelar
                      </>
                    )}
                  </button>
                ) : null}

                {(showSummary || state.status === 'error') && (
                  <button
                    onClick={resetBacktest}
                    className="p-2 sm:p-2.5 rounded-full bg-slate-800 text-slate-400 hover:text-white transition-colors"
                    title="Reset"
                  >
                    <RotateCcw className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  </button>
                )}
              </div>
            </div>

            {/* Options */}
            <div className="flex justify-center">
              <label className="flex items-center gap-1.5 sm:gap-2 text-[10px] sm:text-xs text-slate-500 cursor-pointer">
                <input
                  type="checkbox"
                  checked={forceRegenerate}
                  onChange={(e) => setForceRegenerate(e.target.checked)}
                  disabled={isAnyBacktestRunning}
                  className="w-3 h-3 sm:w-3.5 sm:h-3.5 rounded bg-slate-800 border-slate-600"
                />
                Regenerar trades
              </label>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default BacktestControlPanel;
