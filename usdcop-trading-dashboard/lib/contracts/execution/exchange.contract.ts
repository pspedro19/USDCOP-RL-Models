/**
 * Exchange Contract - SignalBridge Integration
 * =============================================
 *
 * SSOT for exchange connections in the execution module.
 * Mirrors backend: services/signalbridge_api/app/contracts/exchange.py
 *
 * @version 1.0.0
 * @lastSync 2026-01-22
 */

import { z } from 'zod';

// ============================================================================
// SUPPORTED EXCHANGES (SSOT)
// ============================================================================

/**
 * Supported exchanges - lowercase for API compatibility
 */
export const SUPPORTED_EXCHANGES = ['binance', 'mexc'] as const;
export type SupportedExchange = typeof SUPPORTED_EXCHANGES[number];

export const ExchangeTypeSchema = z.enum(SUPPORTED_EXCHANGES);

/**
 * Exchange metadata for UI display
 */
export const EXCHANGE_METADATA: Record<SupportedExchange, {
  name: string;
  displayName: string;
  logo: string;
  color: string;
  docsUrl: string;
  description: string;
}> = {
  binance: {
    name: 'binance',
    displayName: 'Binance',
    logo: '🔶',
    color: '#F0B90B',
    docsUrl: 'https://binance-docs.github.io/apidocs/',
    description: 'World\'s largest crypto exchange',
  },
  mexc: {
    name: 'mexc',
    displayName: 'MEXC',
    logo: '🟢',
    color: '#00B897',
    docsUrl: 'https://mexcdevelop.github.io/apidocs/',
    description: 'Global crypto exchange',
  },
};

/**
 * Get display name for exchange
 */
export function getExchangeDisplayName(exchange: SupportedExchange): string {
  return EXCHANGE_METADATA[exchange]?.displayName ?? exchange.toUpperCase();
}

// ============================================================================
// EXCHANGE STATUS
// ============================================================================

export const ExchangeStatusSchema = z.enum([
  'connected',
  'disconnected',
  'error',
  'validating',
]);
export type ExchangeStatus = z.infer<typeof ExchangeStatusSchema>;

export const EXCHANGE_STATUS_COLORS: Record<ExchangeStatus, string> = {
  connected: 'text-green-500',
  disconnected: 'text-gray-500',
  error: 'text-red-500',
  validating: 'text-yellow-500',
};

// ============================================================================
// SCHEMAS
// ============================================================================

/**
 * Request to connect an exchange
 */
export const ConnectExchangeRequestSchema = z.object({
  api_key: z.string().min(10, 'API key must be at least 10 characters'),
  api_secret: z.string().min(10, 'API secret must be at least 10 characters'),
});
export type ConnectExchangeRequest = z.infer<typeof ConnectExchangeRequestSchema>;

/**
 * Connected exchange record (from database)
 */
export const ConnectedExchangeSchema = z.object({
  id: z.string().uuid(),
  exchange: ExchangeTypeSchema,
  is_valid: z.boolean(),
  connected_at: z.string().datetime(),
  last_used_at: z.string().datetime().nullable(),
  key_fingerprint: z.string(),
});
export type ConnectedExchange = z.infer<typeof ConnectedExchangeSchema>;

/**
 * Connection validation result
 */
export const ValidationResultSchema = z.object({
  is_valid: z.boolean(),
  exchange: ExchangeTypeSchema,
  permissions: z.array(z.string()),
  can_trade_spot: z.boolean(),
  has_withdraw_permission: z.boolean(),
  balance_check: z.record(z.number()).optional(),
  error_message: z.string().optional(),
});
export type ValidationResult = z.infer<typeof ValidationResultSchema>;

/**
 * Single asset balance
 */
export const BalanceSchema = z.object({
  asset: z.string(),
  free: z.number().nonnegative(),
  locked: z.number().nonnegative(),
  total: z.number().nonnegative(),
});
export type Balance = z.infer<typeof BalanceSchema>;

/**
 * Exchange balances response
 */
export const ExchangeBalancesSchema = z.object({
  exchange: ExchangeTypeSchema,
  balances: z.array(BalanceSchema),
  total_usd: z.number().nonnegative(),
  updated_at: z.string().datetime(),
});
export type ExchangeBalances = z.infer<typeof ExchangeBalancesSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

export const validateConnectRequest = (data: unknown) =>
  ConnectExchangeRequestSchema.safeParse(data);

export const validateConnectedExchange = (data: unknown) =>
  ConnectedExchangeSchema.safeParse(data);

export const validateExchangeBalances = (data: unknown) =>
  ExchangeBalancesSchema.safeParse(data);

/**
 * Check if exchange is supported
 */
export function isSupportedExchange(exchange: string): exchange is SupportedExchange {
  return SUPPORTED_EXCHANGES.includes(exchange as SupportedExchange);
}
