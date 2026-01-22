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
} from '@/lib/contracts/backtest.contract';
import {
  getDateRangePresets,
  DateRangePreset,
  BACKTEST_DATE_RANGES,
  formatDateRange,
  fetchPipelineDates,
  PipelineDates,
} from '@/lib/services/backtest.service';
import useBacktest from '@/hooks/useBacktest';
import { cn } from '@/lib/utils';
import { Play, X, RotateCcw, Calendar, ChevronDown, ChevronUp, Loader2, Info, BookOpen, FlaskConical, TestTube2, Rocket, CheckCircle2, Trash2 } from 'lucide-react';

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

/**
 * Info panel showing model training/validation/test periods
 * Uses dynamic dates from API when available
 */
interface DateRangeInfoProps {
  pipelineDates: PipelineDates | null;
  isLoading: boolean;
}

function DateRangeInfo({ pipelineDates, isLoading }: DateRangeInfoProps) {
  const today = new Date().toISOString().split('T')[0];

  // Use API dates if available, otherwise fallback to defaults
  const dates = pipelineDates?.dates || {
    training_start: BACKTEST_DATE_RANGES.TRAINING_START,
    training_end: BACKTEST_DATE_RANGES.TRAINING_END,
    validation_start: BACKTEST_DATE_RANGES.VALIDATION_START,
    validation_end: BACKTEST_DATE_RANGES.VALIDATION_END,
    test_start: BACKTEST_DATE_RANGES.TEST_START,
    test_end: today,
    data_start: BACKTEST_DATE_RANGES.DATA_START,
    data_end: today,
  };

  const periods = [
    {
      icon: <BookOpen className="w-3.5 h-3.5" />,
      label: 'Entrenamiento',
      dates: `${dates.training_start} → ${dates.training_end}`,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/30',
      description: 'Datos usados para entrenar el modelo (no usar para backtest)',
    },
    {
      icon: <FlaskConical className="w-3.5 h-3.5" />,
      label: 'Validación',
      dates: `${dates.validation_start} → ${dates.validation_end}`,
      color: 'text-amber-400',
      bgColor: 'bg-amber-500/10',
      borderColor: 'border-amber-500/30',
      description: 'Período de tuning - resultados pueden tener sesgo',
    },
    {
      icon: <TestTube2 className="w-3.5 h-3.5" />,
      label: 'Test',
      dates: `${dates.test_start} → ${dates.test_end}`,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10',
      borderColor: 'border-emerald-500/30',
      description: 'Out-of-sample - resultados más realistas',
    },
  ];

  const configSource = pipelineDates?.metadata?.config_source || 'defaults';
  const modelVersion = pipelineDates?.model_version || 'v20';

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Info className="w-3.5 h-3.5" />
          <span>Períodos del Modelo ({modelVersion})</span>
        </div>
        {isLoading ? (
          <Loader2 className="w-3 h-3 text-slate-500 animate-spin" />
        ) : (
          <span className="text-[9px] text-slate-600 font-mono">
            {configSource.includes('v20_config') ? '✓ config oficial' : 'defaults'}
          </span>
        )}
      </div>
      <div className="grid grid-cols-3 gap-1.5 sm:gap-2">
        {periods.map((period) => (
          <div
            key={period.label}
            className={cn(
              "flex flex-col items-center p-1.5 sm:p-2 rounded-lg border",
              period.bgColor,
              period.borderColor
            )}
            title={period.description}
          >
            <div className={cn("flex items-center gap-1 sm:gap-1.5", period.color)}>
              {period.icon}
              <span className="text-[10px] sm:text-xs font-medium">{period.label}</span>
            </div>
            <span className="text-[8px] sm:text-[10px] text-slate-400 mt-0.5 sm:mt-1 font-mono truncate max-w-full">
              {period.dates}
            </span>
          </div>
        ))}
      </div>

      {/* Data availability info */}
      <div className="text-center text-[10px] text-slate-600 pt-1">
        Datos disponibles: <span className="font-mono text-slate-500">{dates.data_start} → {dates.data_end}</span>
      </div>
    </div>
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
}: BacktestControlPanelProps) {
  const datePresets = useMemo(() => getDateRangePresets(), []);

  const [selectedPreset, setSelectedPreset] = useState<string>('validation');
  const [customStartDate, setCustomStartDate] = useState<string>(BACKTEST_DATE_RANGES.VALIDATION_START);
  const [customEndDate, setCustomEndDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [forceRegenerate, setForceRegenerate] = useState<boolean>(true);

  // Dynamic pipeline dates from API
  const [pipelineDates, setPipelineDates] = useState<PipelineDates | null>(null);
  const [isLoadingDates, setIsLoadingDates] = useState<boolean>(true);

  // Promote to production state
  const [isPromoting, setIsPromoting] = useState<boolean>(false);
  const [promoteSuccess, setPromoteSuccess] = useState<boolean>(false);
  const [promoteError, setPromoteError] = useState<string | null>(null);

  // Fetch pipeline dates on mount
  useEffect(() => {
    let mounted = true;

    async function loadDates() {
      setIsLoadingDates(true);
      const dates = await fetchPipelineDates();
      if (mounted) {
        setPipelineDates(dates);
        setIsLoadingDates(false);
      }
    }

    loadDates();

    return () => {
      mounted = false;
    };
  }, []);

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

  const handleStartBacktest = useCallback(async () => {
    if (!currentDateRange.startDate || !currentDateRange.endDate) return;

    // Reset promote state when starting new backtest
    setPromoteSuccess(false);
    setPromoteError(null);

    // Notify parent that backtest is starting (clears streaming trades, etc.)
    onBacktestStart?.();

    await startBacktest({
      startDate: currentDateRange.startDate,
      endDate: currentDateRange.endDate,
      modelId: selectedModelId,
      forceRegenerate,
    });
  }, [currentDateRange, selectedModelId, forceRegenerate, startBacktest, onBacktestStart]);

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
  const canStartNow = canStart && isValidDateRange;
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
              {isRunning && (
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
            {/* Model Periods Info - Dynamic from API */}
            <DateRangeInfo pipelineDates={pipelineDates} isLoading={isLoadingDates} />

            {/* Divider */}
            <div className="border-t border-slate-700/50" />

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
                      min={BACKTEST_DATE_RANGES.VALIDATION_START}
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

            {/* Progress Bar */}
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
                  <span>{formatTime(elapsedTime)}</span>
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

            {/* Action Buttons - Mobile-first centered */}
            <div className="flex flex-col items-center gap-2 sm:gap-3">
              <div className="flex flex-wrap justify-center gap-2 sm:gap-3">
                {/* Clear Dashboard Button */}
                {onClearDashboard && (
                  <button
                    onClick={onClearDashboard}
                    disabled={isRunning}
                    className={cn(
                      "flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-full font-medium text-xs sm:text-sm transition-all",
                      "bg-slate-800 text-slate-400 border border-slate-700 hover:text-white hover:border-slate-600",
                      isRunning && "opacity-50 cursor-not-allowed"
                    )}
                    title="Limpiar todos los datos del dashboard"
                  >
                    <Trash2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    <span className="hidden sm:inline">Limpiar</span>
                  </button>
                )}

                {canStart ? (
                  <button
                    onClick={handleStartBacktest}
                    disabled={!canStartNow}
                    className={cn(
                      "flex items-center gap-1.5 sm:gap-2 px-4 sm:px-8 py-2 sm:py-2.5 rounded-full font-medium text-xs sm:text-sm transition-all",
                      canStartNow
                        ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:shadow-lg hover:shadow-cyan-500/25"
                        : "bg-slate-800 text-slate-500 cursor-not-allowed"
                    )}
                  >
                    <Play className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    Iniciar
                  </button>
                ) : (
                  <button
                    onClick={cancelBacktest}
                    className="flex items-center gap-1.5 sm:gap-2 px-4 sm:px-8 py-2 sm:py-2.5 rounded-full bg-red-500/20 text-red-400 border border-red-500/30 font-medium text-xs sm:text-sm hover:bg-red-500/30 transition-all"
                  >
                    <X className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    Cancelar
                  </button>
                )}

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
                  disabled={isRunning}
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
