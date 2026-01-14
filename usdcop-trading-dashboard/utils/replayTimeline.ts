/**
 * Replay Timeline Generator
 *
 * Generates a sequence of ticks for the hybrid replay system:
 * - TRANSITION: Fast scroll between trades
 * - TRADE_ENTER: Trade appears with animation
 * - TRADE_PAUSE: Pause to view the trade
 * - GROUP_ENTER/PAUSE: Multiple nearby trades grouped together
 *
 * The timeline generator:
 * 1. Clusters trades that are close in time
 * 2. Creates transition ticks between clusters
 * 3. Creates entry/pause ticks for each trade or cluster
 * 4. Calculates durations based on speed config
 */

import {
  ReplayTrade,
  Candlestick,
  ReplaySpeed,
  SpeedConfig,
  SPEED_CONFIGS,
  ReplayTick,
  TransitionTick,
  TradeTick,
  GroupTick,
  TradeCluster,
  GeneratedTimeline,
  EasingType,
  TradeAnimationType,
  selectAnimationType,
} from '@/types/replay';

// ═══════════════════════════════════════════════════════════════════════════
// ID GENERATION
// ═══════════════════════════════════════════════════════════════════════════

let tickIdCounter = 0;

function generateTickId(): string {
  tickIdCounter++;
  return `tick_${Date.now()}_${tickIdCounter}`;
}

function generateClusterId(): string {
  return `cluster_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// ═══════════════════════════════════════════════════════════════════════════
// CANDLE INDEX HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Find the candle index for a given timestamp
 */
export function findCandleIndex(candlesticks: Candlestick[], timestamp: Date): number {
  if (candlesticks.length === 0) return 0;

  const timeInSeconds = timestamp.getTime() / 1000;

  // Binary search for efficiency
  let left = 0;
  let right = candlesticks.length - 1;

  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (candlesticks[mid].time < timeInSeconds) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  return Math.max(0, Math.min(left, candlesticks.length - 1));
}

/**
 * Get timestamp from candle index
 */
export function getCandleTimestamp(candlesticks: Candlestick[], index: number): Date {
  if (candlesticks.length === 0) return new Date();
  const safeIndex = Math.max(0, Math.min(index, candlesticks.length - 1));
  return new Date(candlesticks[safeIndex].time * 1000);
}

// ═══════════════════════════════════════════════════════════════════════════
// TRADE CLUSTERING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Cluster trades that are close together in time
 */
export function clusterTrades(
  trades: ReplayTrade[],
  candlesticks: Candlestick[],
  thresholdMinutes: number
): TradeCluster[] {
  if (trades.length === 0) return [];

  // Sort trades by timestamp
  const sortedTrades = [...trades].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  const clusters: TradeCluster[] = [];
  let currentClusterTrades: ReplayTrade[] = [sortedTrades[0]];

  for (let i = 1; i < sortedTrades.length; i++) {
    const prevTrade = currentClusterTrades[currentClusterTrades.length - 1];
    const currentTrade = sortedTrades[i];

    const timeDiffMinutes =
      (new Date(currentTrade.timestamp).getTime() -
       new Date(prevTrade.timestamp).getTime()) / 60000;

    if (timeDiffMinutes <= thresholdMinutes) {
      // Add to current cluster
      currentClusterTrades.push(currentTrade);
    } else {
      // Finalize current cluster and start new one
      clusters.push(createCluster(currentClusterTrades, candlesticks));
      currentClusterTrades = [currentTrade];
    }
  }

  // Don't forget the last cluster
  if (currentClusterTrades.length > 0) {
    clusters.push(createCluster(currentClusterTrades, candlesticks));
  }

  return clusters;
}

/**
 * Create a cluster from a list of trades
 */
function createCluster(trades: ReplayTrade[], candlesticks: Candlestick[]): TradeCluster {
  const startTime = new Date(trades[0].timestamp);
  const endTime = new Date(trades[trades.length - 1].timestamp);

  const totalPnL = trades.reduce((sum, t) => sum + (t.pnl || t.pnl_usd || 0), 0);
  const winCount = trades.filter(t => (t.pnl || t.pnl_usd || 0) > 0).length;
  const lossCount = trades.filter(t => (t.pnl || t.pnl_usd || 0) < 0).length;

  return {
    id: generateClusterId(),
    trades,
    startTime,
    endTime,
    startCandleIndex: findCandleIndex(candlesticks, startTime),
    endCandleIndex: findCandleIndex(candlesticks, endTime),
    totalPnL,
    winCount,
    lossCount,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// TIMELINE GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Generate the complete timeline for replay
 */
export function generateTimeline(
  trades: ReplayTrade[],
  candlesticks: Candlestick[],
  speed: ReplaySpeed
): GeneratedTimeline {
  const config = SPEED_CONFIGS[speed];

  // Handle empty cases
  if (trades.length === 0 || candlesticks.length === 0) {
    return {
      ticks: [],
      totalDurationMs: 0,
      tradeCount: 0,
      groupCount: 0,
      candleRange: { start: 0, end: 0 },
    };
  }

  // Cluster trades based on speed config
  const clusters = clusterTrades(trades, candlesticks, config.groupingThresholdMinutes);

  const ticks: ReplayTick[] = [];
  let currentCandleIndex = 0;
  let totalDuration = 0;

  for (const cluster of clusters) {
    // 1. Transition to cluster (if needed)
    if (cluster.startCandleIndex > currentCandleIndex) {
      const transitionTick = createTransitionTick(
        currentCandleIndex,
        cluster.startCandleIndex,
        candlesticks,
        config
      );
      ticks.push(transitionTick);
      totalDuration += transitionTick.duration;
    }

    // 2. Create ticks for the cluster
    if (cluster.trades.length === 1) {
      // Single trade - create enter + pause ticks
      const trade = cluster.trades[0];
      const enterTick = createTradeEnterTick(trade, cluster.startCandleIndex, config);
      const pauseTick = createTradePauseTick(trade, cluster.startCandleIndex, config);

      ticks.push(enterTick);
      ticks.push(pauseTick);
      totalDuration += enterTick.duration + pauseTick.duration;
    } else {
      // Multiple trades - create group enter + pause ticks
      const enterTick = createGroupEnterTick(cluster, config);
      const pauseTick = createGroupPauseTick(cluster, config);

      ticks.push(enterTick);
      ticks.push(pauseTick);
      totalDuration += enterTick.duration + pauseTick.duration;
    }

    currentCandleIndex = cluster.endCandleIndex;
  }

  // Count groups (clusters with more than 1 trade)
  const groupCount = clusters.filter(c => c.trades.length > 1).length;

  return {
    ticks,
    totalDurationMs: totalDuration,
    tradeCount: trades.length,
    groupCount,
    candleRange: {
      start: 0,
      end: candlesticks.length - 1,
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// TICK CREATION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create a transition tick for scrolling between trades
 */
function createTransitionTick(
  fromIndex: number,
  toIndex: number,
  candlesticks: Candlestick[],
  config: SpeedConfig
): TransitionTick {
  const candleCount = toIndex - fromIndex;

  // Calculate duration based on candle count and speed
  const rawDuration = (candleCount / config.candlesPerSecond) * 1000;
  const duration = Math.max(
    config.transitionMinDuration,
    Math.min(config.transitionMaxDuration, rawDuration)
  );

  // Choose easing based on distance
  const easing: EasingType = candleCount > 100 ? 'easeInOut' : 'easeOut';

  return {
    id: generateTickId(),
    type: 'TRANSITION',
    duration,
    timestamp: getCandleTimestamp(candlesticks, fromIndex),
    candleIndex: fromIndex,
    fromCandleIndex: fromIndex,
    toCandleIndex: toIndex,
    candleCount,
    easing,
  };
}

/**
 * Create a trade enter tick (animation phase)
 */
function createTradeEnterTick(
  trade: ReplayTrade,
  candleIndex: number,
  config: SpeedConfig
): TradeTick {
  return {
    id: generateTickId(),
    type: 'TRADE_ENTER',
    duration: config.tradeAnimationDuration,
    timestamp: new Date(trade.timestamp),
    candleIndex,
    trade,
    animationType: selectAnimationType(trade),
  };
}

/**
 * Create a trade pause tick (viewing phase)
 */
function createTradePauseTick(
  trade: ReplayTrade,
  candleIndex: number,
  config: SpeedConfig
): TradeTick {
  // Pause duration is total pause minus animation duration
  const pauseDuration = Math.max(0, config.tradePauseDuration - config.tradeAnimationDuration);

  return {
    id: generateTickId(),
    type: 'TRADE_PAUSE',
    duration: pauseDuration,
    timestamp: new Date(trade.timestamp),
    candleIndex,
    trade,
    animationType: selectAnimationType(trade),
  };
}

/**
 * Create a group enter tick (multiple trades animation)
 */
function createGroupEnterTick(cluster: TradeCluster, config: SpeedConfig): GroupTick {
  return {
    id: generateTickId(),
    type: 'GROUP_ENTER',
    duration: config.tradeAnimationDuration,
    timestamp: cluster.startTime,
    candleIndex: cluster.startCandleIndex,
    trades: cluster.trades,
    groupId: cluster.id,
  };
}

/**
 * Create a group pause tick (viewing multiple trades)
 */
function createGroupPauseTick(cluster: TradeCluster, config: SpeedConfig): GroupTick {
  return {
    id: generateTickId(),
    type: 'GROUP_PAUSE',
    duration: config.groupPauseDuration,
    timestamp: cluster.startTime,
    candleIndex: cluster.startCandleIndex,
    trades: cluster.trades,
    groupId: cluster.id,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// EASING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Apply easing to a progress value (0-1)
 */
export function applyEasing(t: number, type: EasingType): number {
  // Clamp input
  const clamped = Math.max(0, Math.min(1, t));

  switch (type) {
    case 'linear':
      return clamped;
    case 'easeOut':
      return 1 - Math.pow(1 - clamped, 3);
    case 'easeInOut':
      return clamped < 0.5
        ? 4 * clamped * clamped * clamped
        : 1 - Math.pow(-2 * clamped + 2, 3) / 2;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// TIMELINE NAVIGATION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get the tick and progress at a given overall progress (0-100)
 */
export function getTickAtProgress(
  timeline: GeneratedTimeline,
  progress: number
): { tick: ReplayTick; tickProgress: number; tickIndex: number } | null {
  if (timeline.ticks.length === 0) return null;

  const targetTime = (progress / 100) * timeline.totalDurationMs;
  let accumulatedTime = 0;

  for (let i = 0; i < timeline.ticks.length; i++) {
    const tick = timeline.ticks[i];
    if (accumulatedTime + tick.duration >= targetTime) {
      const tickProgress = tick.duration > 0
        ? (targetTime - accumulatedTime) / tick.duration
        : 1;
      return { tick, tickProgress, tickIndex: i };
    }
    accumulatedTime += tick.duration;
  }

  // Return last tick at 100%
  const lastTick = timeline.ticks[timeline.ticks.length - 1];
  return { tick: lastTick, tickProgress: 1, tickIndex: timeline.ticks.length - 1 };
}

/**
 * Get the candle index at a given progress, applying easing for transitions
 */
export function getCandleIndexAtProgress(
  timeline: GeneratedTimeline,
  progress: number
): number {
  const result = getTickAtProgress(timeline, progress);
  if (!result) return 0;

  const { tick, tickProgress } = result;

  if (tick.type === 'TRANSITION') {
    const easedProgress = applyEasing(tickProgress, tick.easing);
    return Math.round(
      tick.fromCandleIndex +
      (tick.toCandleIndex - tick.fromCandleIndex) * easedProgress
    );
  }

  return tick.candleIndex;
}

/**
 * Get all visible trades up to a given progress
 */
export function getVisibleTradesAtProgress(
  timeline: GeneratedTimeline,
  progress: number
): ReplayTrade[] {
  if (timeline.ticks.length === 0) return [];

  const targetTime = (progress / 100) * timeline.totalDurationMs;
  let accumulatedTime = 0;
  const visibleTrades: ReplayTrade[] = [];
  const seenTradeIds = new Set<string>();

  for (const tick of timeline.ticks) {
    if (accumulatedTime > targetTime) break;

    // Add trades from TRADE_ENTER or GROUP_ENTER ticks
    if (tick.type === 'TRADE_ENTER') {
      if (!seenTradeIds.has(String(tick.trade.trade_id))) {
        visibleTrades.push(tick.trade);
        seenTradeIds.add(String(tick.trade.trade_id));
      }
    } else if (tick.type === 'GROUP_ENTER') {
      for (const trade of tick.trades) {
        if (!seenTradeIds.has(String(trade.trade_id))) {
          visibleTrades.push(trade);
          seenTradeIds.add(String(trade.trade_id));
        }
      }
    }

    accumulatedTime += tick.duration;
  }

  return visibleTrades;
}

/**
 * Get the currently highlighted trade (if in a trade or group tick)
 */
export function getCurrentHighlightedTrade(
  timeline: GeneratedTimeline,
  progress: number
): ReplayTrade | null {
  const result = getTickAtProgress(timeline, progress);
  if (!result) return null;

  const { tick } = result;

  if (tick.type === 'TRADE_ENTER' || tick.type === 'TRADE_PAUSE') {
    return tick.trade;
  }

  if (tick.type === 'GROUP_ENTER' || tick.type === 'GROUP_PAUSE') {
    // Return the first trade in the group
    return tick.trades[0] || null;
  }

  return null;
}

/**
 * Check if currently paused on a trade
 */
export function isPausedOnTrade(timeline: GeneratedTimeline, progress: number): boolean {
  const result = getTickAtProgress(timeline, progress);
  if (!result) return false;

  return result.tick.type === 'TRADE_PAUSE' || result.tick.type === 'GROUP_PAUSE';
}

// ═══════════════════════════════════════════════════════════════════════════
// TIMELINE STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Calculate statistics about the timeline
 */
export function getTimelineStats(timeline: GeneratedTimeline): {
  totalDuration: string;
  transitionTime: number;
  tradeTime: number;
  transitionPercent: number;
  avgPausePerTrade: number;
} {
  let transitionTime = 0;
  let tradeTime = 0;

  for (const tick of timeline.ticks) {
    if (tick.type === 'TRANSITION') {
      transitionTime += tick.duration;
    } else {
      tradeTime += tick.duration;
    }
  }

  const totalMs = timeline.totalDurationMs;
  const minutes = Math.floor(totalMs / 60000);
  const seconds = Math.round((totalMs % 60000) / 1000);

  return {
    totalDuration: `${minutes}:${seconds.toString().padStart(2, '0')}`,
    transitionTime,
    tradeTime,
    transitionPercent: totalMs > 0 ? (transitionTime / totalMs) * 100 : 0,
    avgPausePerTrade: timeline.tradeCount > 0 ? tradeTime / timeline.tradeCount : 0,
  };
}
