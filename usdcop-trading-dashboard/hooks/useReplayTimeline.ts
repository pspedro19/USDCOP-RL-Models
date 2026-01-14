/**
 * Replay Timeline Hook
 *
 * Provides reactive timeline generation and navigation for the hybrid replay system.
 * Regenerates timeline automatically when trades, candlesticks, or speed changes.
 *
 * Features:
 * - Memoized timeline generation
 * - Progress-based navigation helpers
 * - Visible trade tracking
 * - Current tick and animation state
 */

'use client';

import { useMemo, useCallback } from 'react';
import {
  ReplayTrade,
  Candlestick,
  ReplaySpeed,
  GeneratedTimeline,
  ReplayTick,
  TradeCluster,
} from '@/types/replay';
import {
  generateTimeline,
  getTickAtProgress,
  getCandleIndexAtProgress,
  getVisibleTradesAtProgress,
  getCurrentHighlightedTrade,
  isPausedOnTrade,
  getTimelineStats,
  applyEasing,
} from '@/utils/replayTimeline';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface UseReplayTimelineOptions {
  trades: ReplayTrade[];
  candlesticks: Candlestick[];
  speed: ReplaySpeed;
  enabled?: boolean;
}

export interface TimelineNavigationState {
  tick: ReplayTick | null;
  tickIndex: number;
  tickProgress: number;
  candleIndex: number;
  visibleTrades: ReplayTrade[];
  highlightedTrade: ReplayTrade | null;
  isPausedOnTrade: boolean;
  currentCluster: TradeCluster | null;
}

export interface UseReplayTimelineReturn {
  // Timeline data
  timeline: GeneratedTimeline | null;
  isReady: boolean;

  // Navigation functions
  getNavigationState: (progress: number) => TimelineNavigationState;
  getTickAtProgress: (progress: number) => { tick: ReplayTick; tickProgress: number; tickIndex: number } | null;
  getCandleAtProgress: (progress: number) => number;
  getVisibleTradesAtProgress: (progress: number) => ReplayTrade[];

  // Timeline info
  estimatedDuration: string;
  tradeCount: number;
  groupCount: number;
  stats: {
    totalDuration: string;
    transitionTime: number;
    tradeTime: number;
    transitionPercent: number;
    avgPausePerTrade: number;
  } | null;
}

// ═══════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════

export function useReplayTimeline(
  options: UseReplayTimelineOptions
): UseReplayTimelineReturn {
  const { trades, candlesticks, speed, enabled = true } = options;

  // Generate timeline (memoized - only regenerates when inputs change)
  const timeline = useMemo(() => {
    if (!enabled || trades.length === 0 || candlesticks.length === 0) {
      return null;
    }
    return generateTimeline(trades, candlesticks, speed);
  }, [trades, candlesticks, speed, enabled]);

  // Timeline statistics (memoized)
  const stats = useMemo(() => {
    if (!timeline) return null;
    return getTimelineStats(timeline);
  }, [timeline]);

  // Get complete navigation state at a given progress
  const getNavigationState = useCallback((progress: number): TimelineNavigationState => {
    if (!timeline) {
      return {
        tick: null,
        tickIndex: -1,
        tickProgress: 0,
        candleIndex: 0,
        visibleTrades: [],
        highlightedTrade: null,
        isPausedOnTrade: false,
        currentCluster: null,
      };
    }

    const tickResult = getTickAtProgress(timeline, progress);
    const visibleTrades = getVisibleTradesAtProgress(timeline, progress);
    const highlightedTrade = getCurrentHighlightedTrade(timeline, progress);
    const paused = isPausedOnTrade(timeline, progress);

    // Determine current cluster (if in a group tick)
    let currentCluster: TradeCluster | null = null;
    if (tickResult && (tickResult.tick.type === 'GROUP_ENTER' || tickResult.tick.type === 'GROUP_PAUSE')) {
      const groupTick = tickResult.tick;
      currentCluster = {
        id: groupTick.groupId,
        trades: groupTick.trades,
        startTime: groupTick.timestamp,
        endTime: groupTick.timestamp, // Simplified - could calculate from trades
        startCandleIndex: groupTick.candleIndex,
        endCandleIndex: groupTick.candleIndex,
        totalPnL: groupTick.trades.reduce((sum, t) => sum + (t.pnl || t.pnl_usd || 0), 0),
        winCount: groupTick.trades.filter(t => (t.pnl || t.pnl_usd || 0) > 0).length,
        lossCount: groupTick.trades.filter(t => (t.pnl || t.pnl_usd || 0) < 0).length,
      };
    }

    // Calculate candle index with easing for transitions
    let candleIndex = 0;
    if (tickResult) {
      if (tickResult.tick.type === 'TRANSITION') {
        const easedProgress = applyEasing(tickResult.tickProgress, tickResult.tick.easing);
        candleIndex = Math.round(
          tickResult.tick.fromCandleIndex +
          (tickResult.tick.toCandleIndex - tickResult.tick.fromCandleIndex) * easedProgress
        );
      } else {
        candleIndex = tickResult.tick.candleIndex;
      }
    }

    return {
      tick: tickResult?.tick || null,
      tickIndex: tickResult?.tickIndex ?? -1,
      tickProgress: tickResult?.tickProgress ?? 0,
      candleIndex,
      visibleTrades,
      highlightedTrade,
      isPausedOnTrade: paused,
      currentCluster,
    };
  }, [timeline]);

  // Wrapper for getTickAtProgress
  const getTickAtProgressWrapper = useCallback((progress: number) => {
    if (!timeline) return null;
    return getTickAtProgress(timeline, progress);
  }, [timeline]);

  // Wrapper for getCandleAtProgress
  const getCandleAtProgress = useCallback((progress: number): number => {
    if (!timeline) return 0;
    return getCandleIndexAtProgress(timeline, progress);
  }, [timeline]);

  // Wrapper for getVisibleTradesAtProgress
  const getVisibleTradesAtProgressWrapper = useCallback((progress: number): ReplayTrade[] => {
    if (!timeline) return [];
    return getVisibleTradesAtProgress(timeline, progress);
  }, [timeline]);

  // Format estimated duration
  const estimatedDuration = useMemo(() => {
    if (!timeline) return '--:--';
    const seconds = Math.round(timeline.totalDurationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }, [timeline]);

  return {
    // Timeline data
    timeline,
    isReady: timeline !== null,

    // Navigation functions
    getNavigationState,
    getTickAtProgress: getTickAtProgressWrapper,
    getCandleAtProgress,
    getVisibleTradesAtProgress: getVisibleTradesAtProgressWrapper,

    // Timeline info
    estimatedDuration,
    tradeCount: timeline?.tradeCount ?? 0,
    groupCount: timeline?.groupCount ?? 0,
    stats,
  };
}

export default useReplayTimeline;
