/**
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 *
 * Generated from Pydantic schemas in shared/schemas/
 * Run: python -m shared.schemas.codegen
 *
 * Contract: CTR-SHARED-CODEGEN-001
 * Generated: 2026-01-12T23:19:07.092950
 */

import { z } from 'zod';

// =============================================================================
// ENUMS
// =============================================================================

export enum SignalType {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD',
}

export const SignalTypeSchema = z.enum(['BUY', 'SELL', 'HOLD']);

export enum TradeSide {
  LONG = 'long',
  SHORT = 'short',
}

export const TradeSideSchema = z.enum(['long', 'short']);

export enum TradeStatus {
  OPEN = 'open',
  CLOSED = 'closed',
  PENDING = 'pending',
  CANCELLED = 'cancelled',
}

export const TradeStatusSchema = z.enum(['open', 'closed', 'pending', 'cancelled']);

export enum OrderSide {
  LONG = 'long',
  SHORT = 'short',
  FLAT = 'flat',
}

export const OrderSideSchema = z.enum(['long', 'short', 'flat']);

export enum MarketStatus {
  OPEN = 'open',
  CLOSED = 'closed',
  PRE_MARKET = 'pre_market',
  POST_MARKET = 'post_market',
}

export const MarketStatusSchema = z.enum(['open', 'closed', 'pre_market', 'post_market']);

export enum DataSource {
  LIVE = 'live',
  POSTGRES = 'postgres',
  MINIO = 'minio',
  CACHED = 'cached',
  MOCK = 'mock',
  DEMO = 'demo',
  FALLBACK = 'fallback',
  NONE = 'none',
}

export const DataSourceSchema = z.enum(['live', 'postgres', 'minio', 'cached', 'mock', 'demo', 'fallback', 'none']);

// =============================================================================
// CONSTANTS
// =============================================================================

export const OBSERVATION_DIM_V20 = 15;
export const FEATURE_ORDER_V20 = ['log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'rsi_9', 'atr_pct', 'adx_14', 'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z', 'brent_change_1d', 'rate_spread', 'usdmxn_change_1d', 'position', 'time_normalized'] as const;
