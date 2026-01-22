/**
 * Signal Bridge Contract - Frontend TypeScript Types
 * ===================================================
 *
 * SSOT for Signal Bridge integration with the dashboard.
 * Aligns with backend contracts in:
 * - services/signalbridge_api/app/contracts/signal_bridge.py
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { ExchangeTypeSchema, type SupportedExchange } from './exchange.contract';
import { OrderStatusSchema, type OrderStatus } from './execution.contract';

// ============================================================================
// TRADING MODE (SSOT - from src/config/trading_flags.py)
// ============================================================================

/**
 * Trading modes in priority order (highest to lowest)
 */
export const TRADING_MODES = [
  'KILLED',     // Emergency stop - highest priority
  'DISABLED',   // Trading disabled
  'SHADOW',     // Signal logging only, no execution
  'PAPER',      // Simulated trading
  'STAGING',    // Pre-production with real data
  'LIVE',       // Full production trading
] as const;
export type TradingMode = typeof TRADING_MODES[number];

export const TradingModeSchema = z.enum(TRADING_MODES);

export const TRADING_MODE_COLORS: Record<TradingMode, string> = {
  KILLED: 'text-red-600 bg-red-100',
  DISABLED: 'text-gray-600 bg-gray-100',
  SHADOW: 'text-purple-600 bg-purple-100',
  PAPER: 'text-yellow-600 bg-yellow-100',
  STAGING: 'text-blue-600 bg-blue-100',
  LIVE: 'text-green-600 bg-green-100',
};

export const TRADING_MODE_LABELS: Record<TradingMode, string> = {
  KILLED: 'Emergency Stop',
  DISABLED: 'Disabled',
  SHADOW: 'Shadow Mode',
  PAPER: 'Paper Trading',
  STAGING: 'Staging',
  LIVE: 'Live Trading',
};

// ============================================================================
// INFERENCE ACTION (SSOT - model output format)
// ============================================================================

/**
 * Inference model action values
 * Note: Different from SignalAction (1/2/3/0) in other contracts
 */
export const INFERENCE_ACTIONS = {
  SELL: 0,
  HOLD: 1,
  BUY: 2,
} as const;
export type InferenceAction = typeof INFERENCE_ACTIONS[keyof typeof INFERENCE_ACTIONS];

export const InferenceActionSchema = z.number().int().min(0).max(2);

export const INFERENCE_ACTION_LABELS: Record<InferenceAction, string> = {
  0: 'SELL',
  1: 'HOLD',
  2: 'BUY',
};

export const INFERENCE_ACTION_COLORS: Record<InferenceAction, string> = {
  0: 'text-red-500',   // SELL
  1: 'text-gray-500',  // HOLD
  2: 'text-green-500', // BUY
};

// ============================================================================
// RISK DECISION (SSOT - from src/trading/risk_enforcer.py)
// ============================================================================

export const RISK_DECISIONS = ['ALLOW', 'BLOCK', 'REDUCE'] as const;
export type RiskDecision = typeof RISK_DECISIONS[number];

export const RiskDecisionSchema = z.enum(RISK_DECISIONS);

export const RISK_REASONS = [
  'approved',
  'kill_switch_active',
  'daily_loss_limit',
  'trade_limit_reached',
  'cooldown_active',
  'max_position_exceeded',
  'low_confidence',
  'exposure_limit',
  'market_closed',
  'short_disabled',
  'user_limit_exceeded',
] as const;
export type RiskReason = typeof RISK_REASONS[number];

export const RiskReasonSchema = z.enum(RISK_REASONS);

export const RISK_REASON_MESSAGES: Record<RiskReason, string> = {
  approved: 'Approved',
  kill_switch_active: 'Kill switch is active',
  daily_loss_limit: 'Daily loss limit reached',
  trade_limit_reached: 'Maximum trades per day reached',
  cooldown_active: 'Cooldown period active',
  max_position_exceeded: 'Maximum position size exceeded',
  low_confidence: 'Confidence below threshold',
  exposure_limit: 'Exposure limit reached',
  market_closed: 'Market is closed',
  short_disabled: 'Short trading disabled',
  user_limit_exceeded: 'User-specific limit exceeded',
};

// ============================================================================
// BRIDGE STATUS SCHEMA
// ============================================================================

/**
 * Signal Bridge status response
 */
export const BridgeStatusSchema = z.object({
  is_active: z.boolean(),
  kill_switch_active: z.boolean(),
  kill_switch_reason: z.string().nullable().optional(),
  trading_mode: TradingModeSchema,
  connected_exchanges: z.array(ExchangeTypeSchema),
  pending_executions: z.number().int().nonnegative(),
  last_signal_at: z.string().datetime().nullable().optional(),
  last_execution_at: z.string().datetime().nullable().optional(),
  inference_ws_connected: z.boolean(),
  uptime_seconds: z.number().nonnegative(),
  stats: z.object({
    signals_received: z.number().int().nonnegative(),
    executions_total: z.number().int().nonnegative(),
    executions_success: z.number().int().nonnegative(),
    executions_failed: z.number().int().nonnegative(),
    blocked_by_risk: z.number().int().nonnegative(),
  }).optional(),
});
export type BridgeStatus = z.infer<typeof BridgeStatusSchema>;

// ============================================================================
// BRIDGE HEALTH CHECK
// ============================================================================

export const BridgeHealthCheckSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  database: z.boolean(),
  redis: z.boolean(),
  vault: z.boolean(),
  inference_ws: z.boolean(),
  exchanges: z.record(z.string(), z.boolean()),
  errors: z.array(z.string()),
});
export type BridgeHealthCheck = z.infer<typeof BridgeHealthCheckSchema>;

// ============================================================================
// RISK CHECK RESULT
// ============================================================================

export const RiskCheckResultSchema = z.object({
  decision: RiskDecisionSchema,
  reason: RiskReasonSchema,
  message: z.string(),
  adjusted_size: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type RiskCheckResult = z.infer<typeof RiskCheckResultSchema>;

// ============================================================================
// EXECUTION RESULT (BRIDGE)
// ============================================================================

export const BridgeExecutionResultSchema = z.object({
  success: z.boolean(),
  execution_id: z.string().uuid().optional(),
  signal_id: z.string().uuid(),
  status: OrderStatusSchema,
  exchange: ExchangeTypeSchema.optional(),
  symbol: z.string().optional(),
  side: z.enum(['buy', 'sell']).optional(),
  requested_quantity: z.number().nonnegative(),
  filled_quantity: z.number().nonnegative(),
  filled_price: z.number().nonnegative(),
  commission: z.number().nonnegative(),
  risk_check: RiskCheckResultSchema.optional(),
  error_message: z.string().optional(),
  executed_at: z.string().datetime().optional(),
  processing_time_ms: z.number().nonnegative().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type BridgeExecutionResult = z.infer<typeof BridgeExecutionResultSchema>;

// ============================================================================
// USER RISK LIMITS
// ============================================================================

export const UserRiskLimitsSchema = z.object({
  id: z.string().uuid().optional(),
  user_id: z.string().uuid(),
  max_daily_loss_pct: z.number().min(0).max(100).default(2.0),
  max_trades_per_day: z.number().int().min(1).max(1000).default(10),
  max_position_size_usd: z.number().nonnegative().default(1000),
  cooldown_minutes: z.number().int().min(0).max(1440).default(15),
  enable_short: z.boolean().default(false),
  created_at: z.string().datetime().optional(),
  updated_at: z.string().datetime().optional(),
});
export type UserRiskLimits = z.infer<typeof UserRiskLimitsSchema>;

export const UserRiskLimitsUpdateSchema = z.object({
  max_daily_loss_pct: z.number().min(0).max(100).optional(),
  max_trades_per_day: z.number().int().min(1).max(1000).optional(),
  max_position_size_usd: z.number().nonnegative().optional(),
  cooldown_minutes: z.number().int().min(0).max(1440).optional(),
  enable_short: z.boolean().optional(),
});
export type UserRiskLimitsUpdate = z.infer<typeof UserRiskLimitsUpdateSchema>;

// ============================================================================
// SIGNAL CREATE (MANUAL)
// ============================================================================

export const ManualSignalCreateSchema = z.object({
  model_id: z.string().min(1).max(100).default('manual'),
  action: InferenceActionSchema,
  confidence: z.number().min(0).max(1).default(1.0),
  symbol: z.string().min(2).max(20),
  credential_id: z.string().uuid(),
  quantity: z.number().positive().optional(),
  stop_loss: z.number().nonnegative().optional(),
  take_profit: z.number().nonnegative().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});
export type ManualSignalCreate = z.infer<typeof ManualSignalCreateSchema>;

// ============================================================================
// KILL SWITCH REQUEST
// ============================================================================

export const KillSwitchRequestSchema = z.object({
  activate: z.boolean(),
  reason: z.string().min(1).max(500),
  confirm: z.boolean().default(false),
});
export type KillSwitchRequest = z.infer<typeof KillSwitchRequestSchema>;

// ============================================================================
// BRIDGE STATISTICS
// ============================================================================

export const BridgeStatisticsSchema = z.object({
  total_signals_received: z.number().int().nonnegative(),
  total_executions: z.number().int().nonnegative(),
  successful_executions: z.number().int().nonnegative(),
  failed_executions: z.number().int().nonnegative(),
  blocked_by_risk: z.number().int().nonnegative(),
  total_volume_usd: z.number().nonnegative(),
  total_pnl_usd: z.number(),
  avg_execution_time_ms: z.number().nonnegative(),
  period_start: z.string().datetime(),
  period_end: z.string().datetime(),
  by_exchange: z.record(z.string(), z.number().int()).optional(),
  by_model: z.record(z.string(), z.number().int()).optional(),
});
export type BridgeStatistics = z.infer<typeof BridgeStatisticsSchema>;

// ============================================================================
// WEBSOCKET MESSAGE TYPES
// ============================================================================

export const WS_MESSAGE_TYPES = [
  'connected',
  'heartbeat',
  'execution_created',
  'execution_updated',
  'execution_filled',
  'execution_failed',
  'risk_alert',
  'kill_switch',
  'trading_mode_changed',
  'subscribed',
  'unsubscribed',
  'pong',
  'error',
] as const;
export type WSMessageType = typeof WS_MESSAGE_TYPES[number];

export const WebSocketMessageSchema = z.object({
  type: z.enum(WS_MESSAGE_TYPES),
  timestamp: z.string().datetime(),
  data: z.record(z.string(), z.unknown()),
});
export type WebSocketMessage = z.infer<typeof WebSocketMessageSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

export const validateBridgeStatus = (data: unknown) =>
  BridgeStatusSchema.safeParse(data);

export const validateBridgeHealth = (data: unknown) =>
  BridgeHealthCheckSchema.safeParse(data);

export const validateRiskCheckResult = (data: unknown) =>
  RiskCheckResultSchema.safeParse(data);

export const validateBridgeExecutionResult = (data: unknown) =>
  BridgeExecutionResultSchema.safeParse(data);

export const validateUserRiskLimits = (data: unknown) =>
  UserRiskLimitsSchema.safeParse(data);

export const validateWebSocketMessage = (data: unknown) =>
  WebSocketMessageSchema.safeParse(data);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if bridge is operational (not killed or disabled)
 */
export function isBridgeOperational(status: BridgeStatus): boolean {
  return status.is_active && !status.kill_switch_active;
}

/**
 * Check if trading mode allows execution
 */
export function canExecuteTrades(mode: TradingMode): boolean {
  return !['KILLED', 'DISABLED', 'SHADOW'].includes(mode);
}

/**
 * Get trading mode badge color classes
 */
export function getTradingModeBadge(mode: TradingMode): string {
  return TRADING_MODE_COLORS[mode] ?? 'text-gray-500 bg-gray-100';
}

/**
 * Get inference action label
 */
export function getActionLabel(action: InferenceAction): string {
  return INFERENCE_ACTION_LABELS[action] ?? 'UNKNOWN';
}

/**
 * Get risk reason message
 */
export function getRiskReasonMessage(reason: RiskReason): string {
  return RISK_REASON_MESSAGES[reason] ?? reason;
}

/**
 * Check if risk decision allows execution
 */
export function isRiskAllowed(result: RiskCheckResult): boolean {
  return result.decision === 'ALLOW' || result.decision === 'REDUCE';
}

/**
 * Format uptime in human-readable format
 */
export function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}
