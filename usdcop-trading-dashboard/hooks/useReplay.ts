/**
 * Main Replay Hook
 *
 * Orchestrates all replay functionality by combining:
 * - State machine (useReplayStateMachine)
 * - Animation (useReplayAnimation)
 * - Keyboard shortcuts (useReplayKeyboard)
 * - Trade highlighting (useTradeHighlight)
 * - API data loading (replayApiClient)
 * - Metrics calculation (replayMetrics)
 */

'use client';

import { useEffect, useCallback, useMemo, useState } from 'react';
import { useReplayStateMachine } from './useReplayStateMachine';
import { useReplayAnimation, useTradeHighlight, useVisibilityPause } from './useReplayAnimation';
import { useReplayKeyboard, getNextMode } from './useReplayKeyboard';
import { useReplayError } from '@/utils/replayErrors';
import { loadReplayData, createAbortController } from '@/lib/replayApiClient';
import { IncrementalMetricsCalculator, calculateQuickStats, QuickStats } from '@/utils/replayMetrics';
import { ReplayData, ReplayTrade, ReplayMetrics, EMPTY_METRICS, ReplayMode, ReplaySpeed, Candlestick, GeneratedTimeline, TradeCluster } from '@/types/replay';
import { Result } from '@/types/replay';
import { useReplayTimeline, TimelineNavigationState } from './useReplayTimeline';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface UseReplayOptions {
  initialMode?: ReplayMode;
  autoLoad?: boolean;
  onTradeAppear?: (trade: ReplayTrade) => void;
  onComplete?: () => void;
  // Hybrid replay options
  enableTimeline?: boolean;
  candlesticks?: Candlestick[];
  // Model selection
  modelId?: string;
  // Force regeneration of trades (bypass cache)
  forceRegenerate?: boolean;
}

export interface UseReplayReturn {
  // State
  state: ReturnType<typeof useReplayStateMachine>['state'];
  replayData: ReplayData | null;
  metrics: ReplayMetrics;
  quickStats: QuickStats;
  visibleTrades: ReplayTrade[];
  highlightedTradeIds: Set<string>;

  // Animation
  fps: number;
  qualityLevel: 'high' | 'medium' | 'low';

  // Actions
  load: () => Promise<void>;
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  stop: () => void;
  reset: () => void;
  setSpeed: (speed: ReplaySpeed) => void;
  cycleSpeed: () => void;
  setMode: (mode: ReplayMode) => void;
  cycleMode: () => void;
  seek: (progress: number) => void;
  setDateRange: (startDate: Date, endDate: Date) => void;

  // Error
  error: ReturnType<typeof useReplayError>['error'];
  hasError: boolean;
  clearError: () => void;

  // Status
  isLoading: boolean;
  isPlaying: boolean;
  isReady: boolean;
  canPlay: boolean;

  // Hybrid replay (timeline-based)
  timeline: GeneratedTimeline | null;
  timelineReady: boolean;
  estimatedDuration: string;
  tradeCount: number;
  groupCount: number;
  navigationState: TimelineNavigationState;
  highlightedTrade: ReplayTrade | null;
  isPausedOnTrade: boolean;
  currentCluster: TradeCluster | null;
  goToNextTrade: () => void;
  goToPrevTrade: () => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════

export function useReplay(options: UseReplayOptions = {}): UseReplayReturn {
  const {
    initialMode = 'validation',
    autoLoad = false,
    onTradeAppear,
    onComplete,
    enableTimeline = false,
    candlesticks = [],
    modelId = 'ppo_primary',
    forceRegenerate = false,
  } = options;

  // Core state machine
  const stateMachine = useReplayStateMachine(initialMode);
  const { state, tick, play, pause, togglePlayPause, stop, reset, setSpeed, cycleSpeed, setMode, seek, setDateRange, startLoading, loadSuccess, loadError } = stateMachine;

  // Error handling
  const errorHandler = useReplayError();
  const { error, hasError, handleError, clearError } = errorHandler;

  // Trade highlighting
  const { highlightedTradeIds, highlightTrade, clearHighlights } = useTradeHighlight(2000);

  // Data state
  const [replayData, setReplayData] = useState<ReplayData | null>(null);
  const [metricsCalculator] = useState(() => new IncrementalMetricsCalculator());

  // Abort controller for cancelling requests
  const [abortController, setAbortController] = useState<{
    controller: AbortController;
    cancel: () => void;
  } | null>(null);

  // Calculate visible trades up to current date
  const visibleTrades = useMemo(() => {
    if (!replayData?.trades) return [];
    return replayData.trades.filter((trade) => {
      const tradeTime = new Date(trade.timestamp || trade.entry_time || '');
      return tradeTime <= state.currentDate;
    });
  }, [replayData?.trades, state.currentDate]);

  // Calculate metrics for visible trades
  const metrics = useMemo(() => {
    if (visibleTrades.length === 0) return EMPTY_METRICS;
    const visibleEquity = replayData?.equityCurve?.filter((point) => {
      const pointTime = new Date(point.timestamp);
      return pointTime <= state.currentDate;
    }) || [];
    metricsCalculator.setTrades(visibleTrades);
    metricsCalculator.setEquityPoints(visibleEquity);
    return metricsCalculator.getMetrics();
  }, [visibleTrades, replayData?.equityCurve, state.currentDate, metricsCalculator]);

  // Calculate quick stats
  const quickStats = useMemo(() => {
    if (visibleTrades.length === 0) {
      return {
        totalTrades: 0,
        winRate: 0,
        totalPnL: 0,
        lastTradePnL: 0,
        currentDrawdown: 0,
      };
    }
    const visibleEquity = replayData?.equityCurve?.filter((point) => {
      const pointTime = new Date(point.timestamp);
      return pointTime <= state.currentDate;
    }) || [];
    const currentEquity = visibleEquity.length > 0 ? visibleEquity[visibleEquity.length - 1].equity : 10000;
    const peakEquity = Math.max(...visibleEquity.map((p) => p.equity), 10000);
    return calculateQuickStats(visibleTrades, currentEquity, peakEquity);
  }, [visibleTrades, replayData?.equityCurve, state.currentDate]);

  // Track which trades have appeared (for highlighting new ones)
  const [previousTradeCount, setPreviousTradeCount] = useState(0);

  useEffect(() => {
    if (visibleTrades.length > previousTradeCount && previousTradeCount > 0) {
      // New trades appeared - highlight them
      const newTrades = visibleTrades.slice(previousTradeCount);
      newTrades.forEach((trade) => {
        highlightTrade(String(trade.trade_id));
        onTradeAppear?.(trade);
      });
    }
    setPreviousTradeCount(visibleTrades.length);
  }, [visibleTrades, previousTradeCount, highlightTrade, onTradeAppear]);

  // Hybrid replay timeline
  const timelineHook = useReplayTimeline({
    trades: replayData?.trades || [],
    candlesticks: replayData?.candlesticks || candlesticks,
    speed: state.speed,
    enabled: enableTimeline,
  });

  // Get navigation state at current progress
  const navigationState = useMemo(() => {
    return timelineHook.getNavigationState(state.progress);
  }, [timelineHook, state.progress]);

  // Trade navigation functions for hybrid replay
  const goToNextTrade = useCallback(() => {
    if (!timelineHook.timeline) return;

    // Find the next trade tick after current progress
    const currentTimeMs = (state.progress / 100) * timelineHook.timeline.totalDurationMs;
    let accumulatedTime = 0;

    for (const tick of timelineHook.timeline.ticks) {
      if (accumulatedTime > currentTimeMs && (tick.type === 'TRADE_ENTER' || tick.type === 'GROUP_ENTER')) {
        // Found next trade - seek to it
        const nextProgress = (accumulatedTime / timelineHook.timeline.totalDurationMs) * 100;
        seek(nextProgress);
        return;
      }
      accumulatedTime += tick.duration;
    }
  }, [timelineHook.timeline, state.progress, seek]);

  const goToPrevTrade = useCallback(() => {
    if (!timelineHook.timeline) return;

    // Find the previous trade tick before current progress
    const currentTimeMs = (state.progress / 100) * timelineHook.timeline.totalDurationMs;
    let accumulatedTime = 0;
    let lastTradeTime = 0;

    for (const tick of timelineHook.timeline.ticks) {
      if (accumulatedTime >= currentTimeMs) break;
      if (tick.type === 'TRADE_ENTER' || tick.type === 'GROUP_ENTER') {
        lastTradeTime = accumulatedTime;
      }
      accumulatedTime += tick.duration;
    }

    if (lastTradeTime > 0) {
      const prevProgress = (lastTradeTime / timelineHook.timeline.totalDurationMs) * 100;
      seek(Math.max(0, prevProgress - 1)); // Go slightly before the trade
    }
  }, [timelineHook.timeline, state.progress, seek]);

  // Animation
  const { fps, qualityLevel } = useReplayAnimation({
    isPlaying: state.isPlaying,
    speed: state.speed,
    currentDate: state.currentDate,
    startDate: state.startDate,
    endDate: state.endDate,
    onTick: tick,
    onComplete: () => {
      onComplete?.();
    },
  });

  // Visibility pause
  useVisibilityPause(state.isPlaying, pause, play);

  // Keyboard shortcuts
  const keyboardHandlers = useMemo(() => ({
    onPlayPause: togglePlayPause,
    onStop: stop,
    onSeekForward: (amount: number) => seek(state.progress + amount),
    onSeekBackward: (amount: number) => seek(state.progress - amount),
    onSetSpeed: setSpeed,
    onCycleMode: () => setMode(getNextMode(state.mode)),
    onReset: reset,
    // Hybrid replay navigation
    onNextTrade: goToNextTrade,
    onPrevTrade: goToPrevTrade,
  }), [togglePlayPause, stop, seek, state.progress, setSpeed, setMode, state.mode, reset, goToNextTrade, goToPrevTrade]);

  useReplayKeyboard(keyboardHandlers, { enabled: true });

  // Load data function
  const load = useCallback(async () => {
    // Cancel any previous request
    abortController?.cancel();

    // Create new abort controller
    const newController = createAbortController();
    setAbortController(newController);

    // Start loading state
    startLoading();
    clearError();
    clearHighlights();
    setPreviousTradeCount(0);

    console.log(`[useReplay] Loading data: ${state.startDate.toISOString()} to ${state.endDate.toISOString()} (model=${modelId}, force=${forceRegenerate})`);

    try {
      const result = await loadReplayData(state.startDate, state.endDate, {
        signal: newController.controller.signal,
        modelId,
        forceRegenerate,
      });

      if (Result.isOk(result)) {
        setReplayData(result.data);
        loadSuccess();
      } else {
        handleError(result.error);
        loadError(result.error.message);
      }
    } catch (err) {
      handleError(err);
      loadError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [state.startDate, state.endDate, abortController, startLoading, loadSuccess, loadError, handleError, clearError, clearHighlights, modelId, forceRegenerate]);

  // Auto-load on mount if enabled
  useEffect(() => {
    if (autoLoad) {
      load();
    }
  }, [autoLoad]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      abortController?.cancel();
    };
  }, [abortController]);

  // Cycle mode helper
  const cycleMode = useCallback(() => {
    setMode(getNextMode(state.mode));
  }, [setMode, state.mode]);

  return {
    // State
    state,
    replayData,
    metrics,
    quickStats,
    visibleTrades,
    highlightedTradeIds,

    // Animation
    fps,
    qualityLevel,

    // Actions
    load,
    play,
    pause,
    togglePlayPause,
    stop,
    reset,
    setSpeed,
    cycleSpeed,
    setMode,
    cycleMode,
    seek,
    setDateRange,

    // Error
    error,
    hasError,
    clearError,

    // Status
    isLoading: state.status === 'loading',
    isPlaying: state.isPlaying,
    isReady: state.status === 'ready',
    canPlay: stateMachine.canPlay,

    // Hybrid replay (timeline-based)
    timeline: timelineHook.timeline,
    timelineReady: timelineHook.isReady,
    estimatedDuration: timelineHook.estimatedDuration,
    tradeCount: timelineHook.tradeCount,
    groupCount: timelineHook.groupCount,
    navigationState,
    highlightedTrade: navigationState.highlightedTrade,
    isPausedOnTrade: navigationState.isPausedOnTrade,
    currentCluster: navigationState.currentCluster,
    goToNextTrade,
    goToPrevTrade,
  };
}

export default useReplay;
