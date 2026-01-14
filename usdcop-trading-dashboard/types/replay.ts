/**
 * Replay System Types - Production Ready
 *
 * This file contains all type definitions, Zod schemas, and branded types
 * for the replay system. It serves as the single source of truth for
 * data contracts across the replay feature.
 */

import { z } from 'zod';

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTES DEL MODELO - Single Source of Truth
// ═══════════════════════════════════════════════════════════════════════════

export const MODEL_CONFIG = {
  VERSION: 'v20',
  DATES: {
    TRAINING_START: '2020-01-01',
    TRAINING_END: '2024-12-31',
    VALIDATION_START: '2025-01-01',
    VALIDATION_END: '2025-06-30',
    TEST_START: '2025-07-01',
    TEST_END: '2026-01-08',
  },
  LIMITS: {
    MAX_TRADES_PER_LOAD: 10_000,
    MAX_CANDLES_PER_LOAD: 50_000,
    MAX_EQUITY_POINTS: 10_000,
    MAX_REPLAY_SPEED: 8,
    MIN_TICK_INTERVAL_MS: 16, // ~60fps cap
    MAX_DATE_RANGE_DAYS: 365,
  },
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// BRANDED TYPES - Previenen errores de tipos primitivos
// ═══════════════════════════════════════════════════════════════════════════

declare const __brand: unique symbol;
type Brand<T, B> = T & { [__brand]: B };

export type ISODateString = Brand<string, 'ISODateString'>;
export type TradeId = Brand<string, 'TradeId'>;
export type ModelId = Brand<string, 'ModelId'>;
export type Percentage = Brand<number, 'Percentage'>; // 0-100
export type Ratio = Brand<number, 'Ratio'>;           // 0-1

// Constructores seguros para Branded Types
export const ISODateString = {
  parse: (value: string): ISODateString => {
    if (!/^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?)?$/.test(value)) {
      throw new Error(`Invalid ISO date string: ${value}`);
    }
    return value as ISODateString;
  },
  fromDate: (date: Date): ISODateString => date.toISOString() as ISODateString,
  unsafe: (value: string): ISODateString => value as ISODateString,
};

export const Percentage = {
  fromRatio: (ratio: number): Percentage => {
    const pct = Math.min(100, Math.max(0, ratio * 100));
    return pct as Percentage;
  },
  clamp: (value: number): Percentage => Math.min(100, Math.max(0, value)) as Percentage,
};

// ═══════════════════════════════════════════════════════════════════════════
// ZOD SCHEMAS - Validación en runtime
// ═══════════════════════════════════════════════════════════════════════════

// Replay Speed (literal union for type safety) - Extended for hybrid replay
export const ReplaySpeedSchema = z.union([
  z.literal(0.5),
  z.literal(1),
  z.literal(2),
  z.literal(4),
  z.literal(8),
  z.literal(16),
]);
export type ReplaySpeed = z.infer<typeof ReplaySpeedSchema>;

// Replay Mode
export const ReplayModeSchema = z.enum(['validation', 'test', 'both']);
export type ReplayMode = z.infer<typeof ReplayModeSchema>;

// Replay Status (state machine states)
export const ReplayStatusSchema = z.enum([
  'idle',
  'loading',
  'ready',
  'playing',
  'paused',
  'completed',
  'error',
]);
export type ReplayStatus = z.infer<typeof ReplayStatusSchema>;

// Replay State (main state object)
export const ReplayStateSchema = z.object({
  status: ReplayStatusSchema,
  startDate: z.date(),
  endDate: z.date(),
  currentDate: z.date(),
  isPlaying: z.boolean(),
  speed: ReplaySpeedSchema,
  mode: ReplayModeSchema,
  progress: z.number().min(0).max(100),
  error: z.string().nullable(),
}).refine(
  (data) => data.startDate <= data.endDate,
  { message: 'startDate must be before or equal to endDate' }
).refine(
  (data) => data.currentDate >= data.startDate && data.currentDate <= data.endDate,
  { message: 'currentDate must be within date range' }
);
export type ReplayState = z.infer<typeof ReplayStateSchema>;

// Trade Side
export const TradeSideSchema = z.enum(['LONG', 'SHORT']);
export type TradeSide = z.infer<typeof TradeSideSchema>;

// Trade Status
export const TradeStatusSchema = z.enum(['OPEN', 'CLOSED', 'CANCELLED']);
export type TradeStatus = z.infer<typeof TradeStatusSchema>;

// Trade Schema (for API responses)
export const ReplayTradeSchema = z.object({
  trade_id: z.union([z.string(), z.number()]).transform(v => String(v)),
  timestamp: z.string(),
  entry_time: z.string().optional(),
  exit_time: z.string().optional(),
  side: TradeSideSchema,
  entry_price: z.number(),
  exit_price: z.number().optional().nullable(),
  pnl: z.number().optional().default(0),
  pnl_usd: z.number().optional(),
  pnl_percent: z.number().optional().default(0),
  pnl_pct: z.number().optional(),
  hold_time_minutes: z.number().optional().default(0),
  duration_bars: z.number().optional(),
  duration_minutes: z.number().optional(),
  status: TradeStatusSchema.optional().default('CLOSED'),
  confidence: z.number().min(0).max(1).optional(),
  model_id: z.string().optional(),
  data_type: z.string().optional(),
  exit_reason: z.string().optional(),
  // Equity tracking for live equity curve updates during replay
  equity_at_entry: z.number().optional(),
  equity_at_exit: z.number().optional(),
  _meta: z.object({
    model_version: z.string().optional(),
    signal_strength: z.number().optional(),
  }).optional(),
});
export type ReplayTrade = z.infer<typeof ReplayTradeSchema>;

// Equity Point Schema
export const EquityPointSchema = z.object({
  timestamp: z.string(),
  value: z.number().optional(),
  equity: z.number().optional(),
  drawdown: z.number().optional().default(0),
  drawdown_pct: z.number().optional().default(0),
  cumulative_pnl: z.number().optional().default(0),
  position: z.string().optional(),
  price: z.number().optional(),
}).transform(data => ({
  timestamp: data.timestamp,
  equity: data.equity ?? data.value ?? 0,
  drawdown: data.drawdown ?? data.drawdown_pct ?? 0,
  cumulative_pnl: data.cumulative_pnl ?? 0,
  position: data.position,
  price: data.price,
}));
export type EquityPoint = z.infer<typeof EquityPointSchema>;

// Candlestick Schema
export const CandlestickSchema = z.object({
  time: z.number(),
  open: z.number(),
  high: z.number(),
  low: z.number(),
  close: z.number(),
  volume: z.number().optional().default(0),
});
export type Candlestick = z.infer<typeof CandlestickSchema>;

// Replay Metrics Schema
export const ReplayMetricsSchema = z.object({
  sharpe_ratio: z.number(),
  max_drawdown: z.number(),
  win_rate: z.number().min(0).max(100),
  avg_hold_time_minutes: z.number().nonnegative(),
  total_trades: z.number().int().nonnegative(),
  winning_trades: z.number().int().nonnegative(),
  losing_trades: z.number().int().nonnegative(),
  total_pnl: z.number(),
  profit_factor: z.number().nonnegative(),
  avg_win: z.number().optional(),
  avg_loss: z.number().optional(),
  largest_win: z.number().optional(),
  largest_loss: z.number().optional(),
  consecutive_wins: z.number().int().nonnegative().optional(),
  consecutive_losses: z.number().int().nonnegative().optional(),
});
export type ReplayMetrics = z.infer<typeof ReplayMetricsSchema>;

// Empty metrics default
export const EMPTY_METRICS: ReplayMetrics = {
  sharpe_ratio: 0,
  max_drawdown: 0,
  win_rate: 0,
  avg_hold_time_minutes: 0,
  total_trades: 0,
  winning_trades: 0,
  losing_trades: 0,
  total_pnl: 0,
  profit_factor: 0,
  avg_win: 0,
  avg_loss: 0,
  largest_win: 0,
  largest_loss: 0,
  consecutive_wins: 0,
  consecutive_losses: 0,
};

// ═══════════════════════════════════════════════════════════════════════════
// API RESPONSE SCHEMAS
// ═══════════════════════════════════════════════════════════════════════════

export const TradesResponseSchema = z.object({
  success: z.literal(true),
  data: z.object({
    trades: z.array(ReplayTradeSchema),
    summary: z.object({
      total_trades: z.number().optional(),
      winning: z.number().optional(),
      losing: z.number().optional(),
      win_rate: z.number().optional(),
    }).optional(),
  }),
});
export type TradesResponse = z.infer<typeof TradesResponseSchema>;

export const EquityCurveResponseSchema = z.object({
  success: z.literal(true),
  data: z.object({
    points: z.array(EquityPointSchema),
    summary: z.object({
      start_equity: z.number(),
      end_equity: z.number(),
      total_return: z.number().optional(),
      total_return_pct: z.number().optional(),
      max_drawdown_pct: z.number().optional(),
      total_points: z.number().optional(),
    }).optional(),
  }),
});
export type EquityCurveResponse = z.infer<typeof EquityCurveResponseSchema>;

// ═══════════════════════════════════════════════════════════════════════════
// RESULT TYPE - Manejo explícito de errores
// ═══════════════════════════════════════════════════════════════════════════

export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export const Result = {
  ok: <T>(data: T): Result<T, never> => ({ success: true, data }),
  err: <E>(error: E): Result<never, E> => ({ success: false, error }),

  map: <T, U, E>(result: Result<T, E>, fn: (data: T) => U): Result<U, E> => {
    if (result.success) return Result.ok(fn(result.data));
    return result;
  },

  flatMap: <T, U, E>(result: Result<T, E>, fn: (data: T) => Result<U, E>): Result<U, E> => {
    if (result.success) return fn(result.data);
    return result;
  },

  unwrapOr: <T, E>(result: Result<T, E>, defaultValue: T): T => {
    if (result.success) return result.data;
    return defaultValue;
  },

  isOk: <T, E>(result: Result<T, E>): result is { success: true; data: T } => {
    return result.success;
  },

  isErr: <T, E>(result: Result<T, E>): result is { success: false; error: E } => {
    return !result.success;
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// REPLAY EVENTS - Para logging/analytics
// ═══════════════════════════════════════════════════════════════════════════

export const ReplayEventTypeSchema = z.enum([
  'REPLAY_STARTED',
  'REPLAY_PAUSED',
  'REPLAY_RESUMED',
  'REPLAY_COMPLETED',
  'REPLAY_RESET',
  'TRADE_HIGHLIGHTED',
  'SPEED_CHANGED',
  'DATE_RANGE_CHANGED',
  'MODE_CHANGED',
  'REPLAY_ERROR',
]);
export type ReplayEventType = z.infer<typeof ReplayEventTypeSchema>;

export interface ReplayEvent {
  type: ReplayEventType;
  payload: Record<string, unknown>;
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// REPLAY DATA - Combined data for replay
// ═══════════════════════════════════════════════════════════════════════════

export interface ReplayData {
  trades: ReplayTrade[];
  equityCurve: EquityPoint[];
  candlesticks?: Candlestick[];
  summary?: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    total_return_pct: number;
    max_drawdown_pct: number;
    avg_trade_duration_minutes?: number;
  } | null;
  meta: {
    loadTime: number;
    tradeCount: number;
    equityPointCount: number;
    candlestickCount?: number;
    dateRange: { start: string; end: string };
    modelId: string;
    source?: string; // 'database' | 'generated'
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get the date range for a specific replay mode
 */
export function getModeRange(mode: ReplayMode): { start: Date; end: Date } {
  const dates = MODEL_CONFIG.DATES;
  switch (mode) {
    case 'validation':
      return {
        start: new Date(dates.VALIDATION_START),
        end: new Date(dates.VALIDATION_END),
      };
    case 'test':
      return {
        start: new Date(dates.TEST_START),
        end: new Date(dates.TEST_END),
      };
    case 'both':
      return {
        start: new Date(dates.VALIDATION_START),
        end: new Date(dates.TEST_END),
      };
  }
}

/**
 * Determine the mode based on a date range
 */
export function getModeFromDateRange(startDate: Date, endDate: Date): ReplayMode {
  const testStart = new Date(MODEL_CONFIG.DATES.TEST_START);
  const validationEnd = new Date(MODEL_CONFIG.DATES.VALIDATION_END);

  if (startDate >= testStart) {
    return 'test';
  } else if (endDate <= validationEnd) {
    return 'validation';
  }
  return 'both';
}

/**
 * Calculate progress percentage
 */
export function calculateProgress(current: Date, start: Date, end: Date): number {
  const totalMs = end.getTime() - start.getTime();
  if (totalMs === 0) return 100;
  const elapsedMs = current.getTime() - start.getTime();
  return Math.min(100, Math.max(0, (elapsedMs / totalMs) * 100));
}

/**
 * Format date for display
 */
export function formatReplayDate(date: Date): string {
  return date.toLocaleDateString('es-CO', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

/**
 * Check if a date is within the valid replay range
 */
export function isValidReplayDate(date: Date): boolean {
  const minDate = new Date(MODEL_CONFIG.DATES.VALIDATION_START);
  const maxDate = new Date(MODEL_CONFIG.DATES.TEST_END);
  return date >= minDate && date <= maxDate;
}

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID REPLAY SYSTEM - Speed Configurations
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Configuration for each replay speed level
 */
export interface SpeedConfig {
  tradePauseDuration: number;       // ms de pausa en cada trade
  tradeAnimationDuration: number;   // ms de animación de entrada
  transitionMinDuration: number;    // ms mínimo entre trades
  transitionMaxDuration: number;    // ms máximo entre trades
  candlesPerSecond: number;         // velocidad de scroll del chart
  groupingThresholdMinutes: number; // agrupar trades si < N minutos
  groupPauseDuration: number;       // ms de pausa para grupos
  showDecisionPanel: boolean;       // mostrar panel de decisión
  enableAnimations: boolean;        // animaciones de trades
}

/**
 * Speed configurations for all replay speeds
 */
export const SPEED_CONFIGS: Record<ReplaySpeed, SpeedConfig> = {
  0.5: {
    tradePauseDuration: 4000,
    tradeAnimationDuration: 1000,
    transitionMinDuration: 500,
    transitionMaxDuration: 3000,
    candlesPerSecond: 10,
    groupingThresholdMinutes: 15,
    groupPauseDuration: 2000,
    showDecisionPanel: true,
    enableAnimations: true,
  },
  1: {
    tradePauseDuration: 2000,
    tradeAnimationDuration: 600,
    transitionMinDuration: 200,
    transitionMaxDuration: 2000,
    candlesPerSecond: 25,
    groupingThresholdMinutes: 10,
    groupPauseDuration: 1000,
    showDecisionPanel: true,
    enableAnimations: true,
  },
  2: { // DEFAULT
    tradePauseDuration: 1000,
    tradeAnimationDuration: 400,
    transitionMinDuration: 100,
    transitionMaxDuration: 1500,
    candlesPerSecond: 50,
    groupingThresholdMinutes: 5,
    groupPauseDuration: 600,
    showDecisionPanel: true,
    enableAnimations: true,
  },
  4: {
    tradePauseDuration: 500,
    tradeAnimationDuration: 250,
    transitionMinDuration: 50,
    transitionMaxDuration: 800,
    candlesPerSecond: 100,
    groupingThresholdMinutes: 3,
    groupPauseDuration: 300,
    showDecisionPanel: true,
    enableAnimations: true,
  },
  8: {
    tradePauseDuration: 250,
    tradeAnimationDuration: 150,
    transitionMinDuration: 30,
    transitionMaxDuration: 400,
    candlesPerSecond: 200,
    groupingThresholdMinutes: 2,
    groupPauseDuration: 150,
    showDecisionPanel: false,
    enableAnimations: false,
  },
  16: {
    tradePauseDuration: 100,
    tradeAnimationDuration: 80,
    transitionMinDuration: 20,
    transitionMaxDuration: 200,
    candlesPerSecond: 400,
    groupingThresholdMinutes: 1,
    groupPauseDuration: 50,
    showDecisionPanel: false,
    enableAnimations: false,
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID REPLAY SYSTEM - Timeline Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Types of ticks in the replay timeline
 */
export type ReplayTickType =
  | 'TRANSITION'      // Scroll rápido entre trades
  | 'TRADE_ENTER'     // Trade aparece con animación
  | 'TRADE_PAUSE'     // Pausa para ver el trade
  | 'GROUP_ENTER'     // Múltiples trades entran
  | 'GROUP_PAUSE';    // Pausa para grupo

/**
 * Animation types for trades
 */
export type TradeAnimationType = 'bounce' | 'fade' | 'scale' | 'slide';

/**
 * Easing types for transitions
 */
export type EasingType = 'linear' | 'easeOut' | 'easeInOut';

/**
 * Base tick interface
 */
export interface BaseTick {
  id: string;
  type: ReplayTickType;
  duration: number;
  timestamp: Date;
  candleIndex: number;
}

/**
 * Transition tick - scroll between trades
 */
export interface TransitionTick extends BaseTick {
  type: 'TRANSITION';
  fromCandleIndex: number;
  toCandleIndex: number;
  candleCount: number;
  easing: EasingType;
}

/**
 * Trade tick - single trade enter/pause
 */
export interface TradeTick extends BaseTick {
  type: 'TRADE_ENTER' | 'TRADE_PAUSE';
  trade: ReplayTrade;
  animationType: TradeAnimationType;
}

/**
 * Group tick - multiple trades enter/pause
 */
export interface GroupTick extends BaseTick {
  type: 'GROUP_ENTER' | 'GROUP_PAUSE';
  trades: ReplayTrade[];
  groupId: string;
}

/**
 * Union type for all tick types
 */
export type ReplayTick = TransitionTick | TradeTick | GroupTick;

/**
 * Generated timeline with all ticks
 */
export interface GeneratedTimeline {
  ticks: ReplayTick[];
  totalDurationMs: number;
  tradeCount: number;
  groupCount: number;
  candleRange: { start: number; end: number };
}

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID REPLAY SYSTEM - Trade Clustering
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A cluster of trades that are close together in time
 */
export interface TradeCluster {
  id: string;
  trades: ReplayTrade[];
  startTime: Date;
  endTime: Date;
  startCandleIndex: number;
  endCandleIndex: number;
  totalPnL: number;
  winCount: number;
  lossCount: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID REPLAY SYSTEM - Model Metadata
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Model decision metadata stored with each trade
 */
export interface ModelMetadata {
  confidence: number;
  action_probs: [number, number, number]; // [HOLD, LONG, SHORT]
  critic_value: number | null;
  entropy: number | null;
  advantage?: number;
}

/**
 * Snapshot of key features at the time of trade
 */
export interface FeaturesSnapshot {
  rsi_14: number;
  macd_histogram: number;
  bb_position: number;
  volume_zscore: number;
  trend_ema_cross: number;
  returns_5m?: number;
  volatility_zscore?: number;
  hour_of_day: number;
  day_of_week: number;
  position_value?: number;
}

/**
 * Market regime classification
 */
export type MarketRegime = 'trending' | 'ranging' | 'volatile' | 'unknown';

/**
 * Extended trade with model metadata (from enriched database)
 */
export interface EnrichedReplayTrade extends ReplayTrade {
  entry_confidence: number | null;
  exit_confidence?: number | null;
  model_metadata: ModelMetadata | null;
  features_snapshot: FeaturesSnapshot | null;
  market_regime: MarketRegime;
  max_adverse_excursion: number | null;
  max_favorable_excursion: number | null;
}

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID REPLAY SYSTEM - Playback State
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Extended playback state for hybrid replay
 */
export interface HybridPlaybackState {
  currentTickIndex: number;
  currentTick: ReplayTick | null;
  tickProgress: number; // 0-1 progress within current tick
  visibleTrades: ReplayTrade[];
  highlightedTradeId: string | null;
  isPausedOnTrade: boolean;
  currentCluster: TradeCluster | null;
}

/**
 * Timeline playback mode
 */
export type TimelinePlaybackMode = 'linear' | 'trade-focused';

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS FOR HYBRID REPLAY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get the default speed config
 */
export function getSpeedConfig(speed: ReplaySpeed): SpeedConfig {
  return SPEED_CONFIGS[speed];
}

/**
 * Check if a trade has enriched metadata
 */
export function isEnrichedTrade(trade: ReplayTrade): trade is EnrichedReplayTrade {
  return 'model_metadata' in trade && trade.model_metadata !== undefined;
}

/**
 * Select animation type based on trade result
 */
export function selectAnimationType(trade: ReplayTrade): TradeAnimationType {
  const pnl = trade.pnl || trade.pnl_usd || 0;
  const holdTime = trade.hold_time_minutes || trade.duration_minutes || 0;

  if (pnl > 0) {
    return 'bounce'; // Winner: energetic
  } else if (pnl < 0) {
    return 'fade';   // Loser: fade out
  } else if (holdTime < 30) {
    return 'slide';  // Scalp: quick
  }
  return 'scale';    // Default
}

/**
 * Format duration for display (mm:ss)
 */
export function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}
