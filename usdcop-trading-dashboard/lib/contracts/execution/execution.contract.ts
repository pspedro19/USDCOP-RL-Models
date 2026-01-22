/**
 * Execution Contract - SignalBridge Integration
 * ==============================================
 *
 * SSOT for trade executions in the execution module.
 * Integrates with dashboard SSOT for consistency.
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';
import { TRADE_SIDES } from '../ssot.contract';
import { ExchangeTypeSchema, type SupportedExchange } from './exchange.contract';

// ============================================================================
// ORDER STATUS (SSOT)
// ============================================================================

export const ORDER_STATUSES = [
  'pending',
  'submitted',
  'partial',
  'filled',
  'cancelled',
  'rejected',
  'failed',
] as const;
export type OrderStatus = typeof ORDER_STATUSES[number];

export const OrderStatusSchema = z.enum(ORDER_STATUSES);

export const ORDER_STATUS_COLORS: Record<OrderStatus, string> = {
  pending: 'text-yellow-500',
  submitted: 'text-blue-500',
  partial: 'text-cyan-500',
  filled: 'text-green-500',
  cancelled: 'text-gray-500',
  rejected: 'text-red-500',
  failed: 'text-red-600',
};

export const ORDER_STATUS_NAMES: Record<OrderStatus, string> = {
  pending: 'Pending',
  submitted: 'Submitted',
  partial: 'Partial Fill',
  filled: 'Filled',
  cancelled: 'Cancelled',
  rejected: 'Rejected',
  failed: 'Failed',
};

// ============================================================================
// ORDER SIDE (from SSOT)
// ============================================================================

/**
 * Order side - uses TRADE_SIDES from SSOT for consistency
 */
export const OrderSideSchema = z.enum(TRADE_SIDES);
export type OrderSide = z.infer<typeof OrderSideSchema>;

export const ORDER_SIDE_COLORS: Record<OrderSide, string> = {
  BUY: 'text-green-500',
  SELL: 'text-red-500',
};

// ============================================================================
// EXECUTION SCHEMAS
// ============================================================================

/**
 * Base execution record
 */
export const ExecutionSchema = z.object({
  request_id: z.string(),
  signal_id: z.string(),
  exchange: ExchangeTypeSchema,
  symbol: z.string(),
  side: OrderSideSchema,
  status: OrderStatusSchema,
  requested_quantity: z.number().nonnegative(),
  filled_quantity: z.number().nonnegative(),
  filled_price: z.number().nonnegative(),
  fees: z.number().nonnegative(),
  fees_currency: z.string(),
  pnl: z.number().optional(),
  created_at: z.string().datetime(),
  executed_at: z.string().datetime().optional(),
  error_message: z.string().optional(),
});
export type Execution = z.infer<typeof ExecutionSchema>;

/**
 * Execution detail with signal info
 */
export const ExecutionDetailSchema = ExecutionSchema.extend({
  signal: z.object({
    action: z.number().min(0).max(2),
    action_name: z.string(),
    confidence: z.number().min(0).max(1),
  }).optional(),
  order_id: z.string().optional(),
  avg_price: z.number().nonnegative().optional(),
});
export type ExecutionDetail = z.infer<typeof ExecutionDetailSchema>;

/**
 * Paginated execution list response
 */
export const ExecutionListResponseSchema = z.object({
  data: z.array(ExecutionSchema),
  pagination: z.object({
    page: z.number().int().positive(),
    limit: z.number().int().positive(),
    total: z.number().int().nonnegative(),
  }),
});
export type ExecutionListResponse = z.infer<typeof ExecutionListResponseSchema>;

/**
 * Execution statistics
 */
export const ExecutionStatsSchema = z.object({
  total: z.number().int().nonnegative(),
  filled: z.number().int().nonnegative(),
  rejected: z.number().int().nonnegative(),
  failed: z.number().int().nonnegative(),
  total_fees: z.number().nonnegative(),
  total_pnl: z.number(),
  win_rate: z.number().min(0).max(1).optional(),
});
export type ExecutionStats = z.infer<typeof ExecutionStatsSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

export const validateExecution = (data: unknown) =>
  ExecutionSchema.safeParse(data);

export const validateExecutionDetail = (data: unknown) =>
  ExecutionDetailSchema.safeParse(data);

export const validateExecutionList = (data: unknown) =>
  ExecutionListResponseSchema.safeParse(data);

export const validateExecutionStats = (data: unknown) =>
  ExecutionStatsSchema.safeParse(data);

/**
 * Get status display name
 */
export function getOrderStatusName(status: OrderStatus): string {
  return ORDER_STATUS_NAMES[status] ?? status;
}

/**
 * Check if status is terminal (final state)
 */
export function isTerminalStatus(status: OrderStatus): boolean {
  return ['filled', 'cancelled', 'rejected', 'failed'].includes(status);
}

/**
 * Check if execution was successful
 */
export function isSuccessfulExecution(execution: Execution): boolean {
  return execution.status === 'filled' && execution.filled_quantity > 0;
}
