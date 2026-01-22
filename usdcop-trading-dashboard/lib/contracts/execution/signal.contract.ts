/**
 * Signal Contract - SignalBridge Integration
 * ===========================================
 *
 * SSOT for trading signals in the execution module.
 * Integrates with dashboard SSOT for action consistency.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { Action, ACTION_NAMES, VALID_ACTIONS } from '../ssot.contract';

// ============================================================================
// TRADING ACTION (from SSOT)
// ============================================================================

/**
 * Trading action schema - uses SSOT values
 */
export const TradingActionSchema = z.union([
  z.literal(Action.SELL),
  z.literal(Action.HOLD),
  z.literal(Action.BUY),
]);
export type TradingAction = z.infer<typeof TradingActionSchema>;

/**
 * Action name type
 */
export const ActionNameSchema = z.enum(['SELL', 'HOLD', 'BUY']);
export type ActionName = z.infer<typeof ActionNameSchema>;

// ============================================================================
// SIGNAL SCHEMAS
// ============================================================================

/**
 * Base signal schema
 */
export const SignalSchema = z.object({
  signal_id: z.string(),
  timestamp: z.string().datetime(),
  symbol: z.string(),
  action: TradingActionSchema,
  action_name: ActionNameSchema,
  confidence: z.number().min(0).max(1),
  model_version: z.string(),
  execution_count: z.number().int().nonnegative(),
});
export type Signal = z.infer<typeof SignalSchema>;

/**
 * Signal detail with additional data
 */
export const SignalDetailSchema = SignalSchema.extend({
  features: z.record(z.number()).optional(),
  raw_prediction: z.array(z.number()).optional(),
  processing_time_ms: z.number().nonnegative().optional(),
});
export type SignalDetail = z.infer<typeof SignalDetailSchema>;

/**
 * Signal list response with pagination
 */
export const SignalListResponseSchema = z.object({
  data: z.array(SignalSchema),
  pagination: z.object({
    page: z.number().int().positive(),
    limit: z.number().int().positive(),
    total: z.number().int().nonnegative(),
  }),
});
export type SignalListResponse = z.infer<typeof SignalListResponseSchema>;

/**
 * Signal statistics
 */
export const SignalStatsSchema = z.object({
  total: z.number().int().nonnegative(),
  buy_count: z.number().int().nonnegative(),
  sell_count: z.number().int().nonnegative(),
  hold_count: z.number().int().nonnegative(),
  executed_count: z.number().int().nonnegative(),
  skipped_count: z.number().int().nonnegative(),
  avg_confidence: z.number().min(0).max(1),
});
export type SignalStats = z.infer<typeof SignalStatsSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

export const validateSignal = (data: unknown) =>
  SignalSchema.safeParse(data);

export const validateSignalDetail = (data: unknown) =>
  SignalDetailSchema.safeParse(data);

export const validateSignalList = (data: unknown) =>
  SignalListResponseSchema.safeParse(data);

export const validateSignalStats = (data: unknown) =>
  SignalStatsSchema.safeParse(data);

/**
 * Get action name from action value
 */
export function getSignalActionName(action: TradingAction): string {
  return ACTION_NAMES[action] ?? 'UNKNOWN';
}

/**
 * Check if signal should execute based on confidence
 */
export function shouldExecuteSignal(signal: Signal, minConfidence: number = 0.6): boolean {
  return signal.action !== Action.HOLD && signal.confidence >= minConfidence;
}

/**
 * Get signal color based on action
 */
export function getSignalColor(action: TradingAction): string {
  switch (action) {
    case Action.BUY:
      return 'text-green-500';
    case Action.SELL:
      return 'text-red-500';
    default:
      return 'text-gray-500';
  }
}
