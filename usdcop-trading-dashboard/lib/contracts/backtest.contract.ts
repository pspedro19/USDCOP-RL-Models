/**
 * Backtest Contract
 * =================
 * Single Source of Truth for backtest types.
 *
 * SOLID Principles Applied:
 * - Single Responsibility: Only backtest-related types
 * - Interface Segregation: Separate interfaces for different use cases
 * - Open/Closed: Extensible via generics
 */

import { z } from 'zod';

// ============================================================================
// Backtest Status (State Machine States)
// ============================================================================

export const BacktestStatusSchema = z.enum([
  'idle',           // Not started
  'connecting',     // Establishing SSE connection
  'loading',        // Loading historical data
  'running',        // Running simulation
  'saving',         // Persisting trades
  'completed',      // Successfully finished
  'error',          // Failed
  'cancelled',      // User cancelled
]);
export type BacktestStatus = z.infer<typeof BacktestStatusSchema>;

// ============================================================================
// Backtest Progress Event (from SSE stream)
// ============================================================================

export const BacktestProgressSchema = z.object({
  progress: z.number().min(0).max(1),
  current_bar: z.number().int().nonnegative(),
  total_bars: z.number().int().nonnegative(),
  trades_generated: z.number().int().nonnegative(),
  status: z.string(),
  message: z.string(),
});
export type BacktestProgress = z.infer<typeof BacktestProgressSchema>;

// ============================================================================
// Backtest Summary (from completion)
// ============================================================================

export const BacktestSummarySchema = z.object({
  total_trades: z.number().int().nonnegative(),
  winning_trades: z.number().int().nonnegative(),
  losing_trades: z.number().int().nonnegative(),
  win_rate: z.number().min(0).max(100),
  total_pnl: z.number(),
  total_return_pct: z.number(),
  max_drawdown_pct: z.number().min(0),
  sharpe_ratio: z.number().nullish(),  // Allow null from backend
  avg_trade_duration_minutes: z.number().optional(),
});
export type BacktestSummary = z.infer<typeof BacktestSummarySchema>;

// ============================================================================
// Backtest Trade (simplified for UI)
// ============================================================================

export const BacktestTradeSchema = z.object({
  trade_id: z.number(),
  model_id: z.string(),
  timestamp: z.string(),
  entry_time: z.string(),
  exit_time: z.string().nullable().optional(),
  side: z.string(),
  entry_price: z.number(),
  exit_price: z.number().nullable().optional(),
  pnl: z.number(),
  pnl_usd: z.number(),
  pnl_percent: z.number(),
  pnl_pct: z.number(),
  status: z.string(),
  duration_minutes: z.number(),
  exit_reason: z.string().nullable().optional(),
  equity_at_entry: z.number().nullable().optional(),
  equity_at_exit: z.number().nullable().optional(),
  entry_confidence: z.number().nullable().optional(),
  exit_confidence: z.number().nullable().optional(),
});
export type BacktestTrade = z.infer<typeof BacktestTradeSchema>;

// ============================================================================
// Backtest Result (complete response)
// ============================================================================

export const BacktestResultSchema = z.object({
  success: z.boolean(),
  source: z.enum(['database', 'generated', 'error']),
  trade_count: z.number().int().nonnegative(),
  trades: z.array(BacktestTradeSchema),
  summary: BacktestSummarySchema.nullable(),
  processing_time_ms: z.number(),
  date_range: z.object({
    start: z.string(),
    end: z.string(),
  }).nullish(),  // Allow null or undefined from backend
});
export type BacktestResult = z.infer<typeof BacktestResultSchema>;

// ============================================================================
// Replay Speed Control
// ============================================================================

/** Available replay speeds for bar-by-bar simulation */
export const REPLAY_SPEEDS = [0.5, 1, 2, 4, 8, 16] as const;
export type ReplaySpeed = typeof REPLAY_SPEEDS[number];

/** Default replay speed */
export const DEFAULT_REPLAY_SPEED: ReplaySpeed = 1;

/** Get base delay for a given speed (ms between events) */
export function getReplayDelay(speed: ReplaySpeed): number {
  // Base delay at 1x speed is 300ms between trades
  const BASE_DELAY_MS = 300;
  return BASE_DELAY_MS / speed;
}

// ============================================================================
// Backtest Request Parameters
// ============================================================================

export interface BacktestRequest {
  startDate: string;    // YYYY-MM-DD
  endDate: string;      // YYYY-MM-DD
  modelId: string;
  forceRegenerate?: boolean;
  /** Replay speed multiplier (0.5x to 16x) */
  replaySpeed?: ReplaySpeed;
  /** Emit bar-level events for dynamic equity curve */
  emitBarEvents?: boolean;
}

// ============================================================================
// Backtest State (for hook)
// ============================================================================

export interface BacktestState {
  status: BacktestStatus;
  progress: BacktestProgress | null;
  result: BacktestResult | null;
  error: string | null;
  startedAt: Date | null;
  completedAt: Date | null;
}

// ============================================================================
// SSE Event Types
// ============================================================================

export type BacktestSSEEventType = 'progress' | 'result' | 'error' | 'trade';

export interface BacktestSSEEvent<T = unknown> {
  type: BacktestSSEEventType;
  data: T;
}

// Trade event data (for real-time equity curve updates)
export interface BacktestTradeEvent {
  trade_id: string;
  model_id: string;
  timestamp: string;
  entry_time: string;
  exit_time?: string;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price?: number;
  pnl?: number;
  pnl_usd?: number;
  pnl_percent?: number;
  pnl_pct?: number;
  status: string;
  current_equity: number;  // For equity curve update
}

// ============================================================================
// State Machine - Valid Transitions
// ============================================================================

const VALID_TRANSITIONS: Record<BacktestStatus, BacktestStatus[]> = {
  idle:       ['connecting'],
  connecting: ['loading', 'error', 'cancelled'],
  loading:    ['running', 'error', 'cancelled'],
  running:    ['saving', 'error', 'cancelled'],
  saving:     ['completed', 'error'],
  completed:  ['idle'],
  error:      ['idle'],
  cancelled:  ['idle'],
};

export function transitionBacktestStatus(
  current: BacktestStatus,
  next: BacktestStatus
): BacktestStatus {
  const allowed = VALID_TRANSITIONS[current];
  if (!allowed.includes(next)) {
    console.warn(`[StateMachine] Invalid: ${current} -> ${next}`);
    return current;
  }
  return next;
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create initial backtest state
 */
export function createInitialBacktestState(): BacktestState {
  return {
    status: 'idle',
    progress: null,
    result: null,
    error: null,
    startedAt: null,
    completedAt: null,
  };
}

/**
 * Create empty progress
 */
export function createEmptyProgress(): BacktestProgress {
  return {
    progress: 0,
    current_bar: 0,
    total_bars: 0,
    trades_generated: 0,
    status: 'idle',
    message: '',
  };
}

/**
 * Calculate ETA from progress
 */
export function calculateETA(progress: BacktestProgress, startedAt: Date): string {
  if (progress.progress <= 0 || progress.progress >= 1) {
    return '--:--';
  }

  const elapsed = Date.now() - startedAt.getTime();
  const totalEstimated = elapsed / progress.progress;
  const remaining = totalEstimated - elapsed;

  const seconds = Math.floor(remaining / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
}

/**
 * Format progress percentage
 */
export function formatProgress(progress: number): string {
  return `${Math.round(progress * 100)}%`;
}

// ============================================================================
// Status Helpers
// ============================================================================

export function isBacktestRunning(status: BacktestStatus): boolean {
  return ['connecting', 'loading', 'running', 'saving'].includes(status);
}

export function isBacktestComplete(status: BacktestStatus): boolean {
  return status === 'completed';
}

export function canStartBacktest(status: BacktestStatus): boolean {
  return ['idle', 'completed', 'error', 'cancelled'].includes(status);
}

export function getStatusMessage(status: BacktestStatus): string {
  const messages: Record<BacktestStatus, string> = {
    idle: 'Listo para iniciar',
    connecting: 'Conectando...',
    loading: 'Cargando datos históricos...',
    running: 'Ejecutando simulación...',
    saving: 'Guardando trades...',
    completed: 'Backtest completado',
    error: 'Error en backtest',
    cancelled: 'Backtest cancelado',
  };
  return messages[status];
}

export function getStatusColor(status: BacktestStatus): string {
  const colors: Record<BacktestStatus, string> = {
    idle: 'text-gray-400',
    connecting: 'text-yellow-500',
    loading: 'text-blue-500',
    running: 'text-blue-500',
    saving: 'text-blue-500',
    completed: 'text-green-500',
    error: 'text-red-500',
    cancelled: 'text-orange-500',
  };
  return colors[status];
}
