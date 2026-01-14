/**
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 *
 * Generated from Pydantic schemas in shared/schemas/
 * Run: python -m shared.schemas.codegen
 *
 * Contract: CTR-SHARED-CODEGEN-001
 * Generated: 2026-01-12T23:19:07.093528
 */

import { z } from 'zod';

// =============================================================================
// FEATURE SCHEMAS
// =============================================================================

export const NamedFeaturesV20Schema = z.object({
  log_ret_5m: z.number(),
  log_ret_1h: z.number(),
  log_ret_4h: z.number(),
  rsi_9: z.number().min(0).max(100),
  atr_pct: z.number().min(0),
  adx_14: z.number().min(0).max(100),
  dxy_z: z.number(),
  dxy_change_1d: z.number(),
  vix_z: z.number(),
  embi_z: z.number(),
  brent_change_1d: z.number(),
  rate_spread: z.number(),
  usdmxn_change_1d: z.number(),
  position: z.number().min(-1).max(1),
  time_normalized: z.number().min(0).max(1),
}});

export const ObservationSchemaSchema = z.object({
  values: z.unknown(),
  contract_version: z.string(),
}});

export const NormalizationStatsSchemaSchema = z.object({
  mean: z.number(),
  std: z.number().gt(0),
  clip_min: z.number(),
  clip_max: z.number(),
}});

// =============================================================================
// TRADING SCHEMAS
// =============================================================================

export const CandlestickSchemaSchema = z.object({
  time: z.number().int(),
  open: z.number().min(0),
  high: z.number().min(0),
  low: z.number().min(0),
  close: z.number().min(0),
  volume: z.unknown(),
}});

export const TradeMetadataSchemaSchema = z.object({
  confidence: z.number().min(0).max(1),
  action_probs: z.unknown(),
  critic_value: z.unknown(),
  entropy: z.unknown(),
  advantage: z.unknown(),
  model_version: z.string(),
  norm_stats_version: z.string(),
  model_hash: z.unknown(),
}});

export const TradeSchemaSchema = z.object({
  created_at: z.unknown(),
  updated_at: z.unknown(),
  trade_id: z.string(),
  model_id: z.string(),
  timestamp: z.string(),
  entry_time: z.string(),
  exit_time: z.unknown(),
  side: z.unknown(),
  entry_price: z.number().gt(0),
  exit_price: z.unknown(),
  size: z.number().gt(0),
  pnl: z.number(),
  pnl_usd: z.number(),
  pnl_percent: z.number(),
  pnl_pct: z.number(),
  status: z.unknown(),
  duration_minutes: z.unknown(),
  exit_reason: z.unknown(),
  equity_at_entry: z.unknown(),
  equity_at_exit: z.unknown(),
  entry_confidence: z.unknown(),
  exit_confidence: z.unknown(),
  commission: z.unknown(),
  market_regime: z.unknown(),
  max_adverse_excursion: z.unknown(),
  max_favorable_excursion: z.unknown(),
  features_snapshot: z.unknown(),
  model_metadata: z.unknown(),
}});

export const TradeSummarySchemaSchema = z.object({
  total_trades: z.number().int().min(0),
  winning_trades: z.number().int().min(0),
  losing_trades: z.number().int().min(0),
  win_rate: z.number().min(0).max(1),
  total_pnl: z.number(),
  total_pnl_usd: z.number(),
  total_return_pct: z.number(),
  max_drawdown_pct: z.number().min(0),
  sharpe_ratio: z.unknown(),
  sortino_ratio: z.unknown(),
  profit_factor: z.unknown(),
  avg_win: z.unknown(),
  avg_loss: z.unknown(),
  largest_win: z.unknown(),
  largest_loss: z.unknown(),
  avg_trade_duration_minutes: z.unknown(),
}});

// =============================================================================
// API SCHEMAS
// =============================================================================

export const BacktestRequestSchemaSchema = z.object({
  start_date: z.string(),
  end_date: z.string(),
  model_id: z.string(),
  force_regenerate: z.boolean(),
}});

export const InferenceRequestSchemaSchema = z.object({
  observation: z.unknown(),
  model_id: z.string(),
}});

export const BacktestResponseSchemaSchema = z.object({
  success: z.boolean(),
  source: z.string(),
  trade_count: z.number().int().min(0),
  trades: z.unknown(),
  summary: z.unknown(),
  processing_time_ms: z.unknown(),
  date_range: z.unknown(),
}});

export const HealthResponseSchemaSchema = z.object({
  status: z.string(),
  version: z.string(),
  model_loaded: z.boolean(),
  database_connected: z.boolean(),
  timestamp: z.string(),
}});

export const ErrorResponseSchemaSchema = z.object({
  success: z.boolean(),
  error: z.string(),
  error_code: z.unknown(),
  details: z.unknown(),
}});

