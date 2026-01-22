/**
 * Trading Config Contract - SignalBridge Integration
 * ===================================================
 *
 * SSOT for trading configuration in the execution module.
 * Uses exchange and risk constants from other contracts.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { SUPPORTED_EXCHANGES, ExchangeTypeSchema } from './exchange.contract';

// ============================================================================
// POSITION STATE
// ============================================================================

export const POSITION_STATES = ['long', 'short', 'flat'] as const;
export type PositionState = typeof POSITION_STATES[number];

export const PositionStateSchema = z.enum(POSITION_STATES);

export const POSITION_STATE_COLORS: Record<PositionState, string> = {
  long: 'text-green-500',
  short: 'text-red-500',
  flat: 'text-gray-500',
};

// ============================================================================
// TRADING CONFIG SCHEMAS
// ============================================================================

/**
 * Trading configuration
 */
export const TradingConfigSchema = z.object({
  symbol: z.string().default('USD/COP'),
  is_enabled: z.boolean().default(false),
  stop_loss_pct: z.number().min(0.001).max(0.1).default(0.02),
  take_profit_pct: z.number().min(0.001).max(0.5).default(0.05),
  min_confidence: z.number().min(0.5).max(1).default(0.7),
  execute_on_exchanges: z.array(ExchangeTypeSchema).default(['mexc', 'binance']),
});
export type TradingConfig = z.infer<typeof TradingConfigSchema>;

/**
 * Partial update for trading config
 */
export const TradingConfigUpdateSchema = TradingConfigSchema.partial();
export type TradingConfigUpdate = z.infer<typeof TradingConfigUpdateSchema>;

/**
 * Trading status
 */
export const TradingStatusSchema = z.object({
  is_enabled: z.boolean(),
  last_signal_at: z.string().datetime().nullable(),
  today_trades_count: z.number().int().nonnegative(),
  max_daily_trades: z.number().int().positive(),
  current_position: PositionStateSchema.nullable(),
});
export type TradingStatus = z.infer<typeof TradingStatusSchema>;

/**
 * Combined config and status
 */
export const TradingStateSchema = z.object({
  config: TradingConfigSchema,
  status: TradingStatusSchema,
});
export type TradingState = z.infer<typeof TradingStateSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

export const validateTradingConfig = (data: unknown) =>
  TradingConfigSchema.safeParse(data);

export const validateTradingConfigUpdate = (data: unknown) =>
  TradingConfigUpdateSchema.safeParse(data);

export const validateTradingStatus = (data: unknown) =>
  TradingStatusSchema.safeParse(data);

export const validateTradingState = (data: unknown) =>
  TradingStateSchema.safeParse(data);

/**
 * Check if trading can occur based on status
 */
export function canTrade(status: TradingStatus): boolean {
  return status.is_enabled && status.today_trades_count < status.max_daily_trades;
}

/**
 * Get remaining trades for the day
 */
export function getRemainingTrades(status: TradingStatus): number {
  return Math.max(0, status.max_daily_trades - status.today_trades_count);
}

/**
 * Get position state color
 */
export function getPositionStateColor(position: PositionState | null): string {
  if (!position) return 'text-gray-500';
  return POSITION_STATE_COLORS[position];
}

/**
 * Create default trading config
 */
export function createDefaultConfig(): TradingConfig {
  return TradingConfigSchema.parse({});
}
