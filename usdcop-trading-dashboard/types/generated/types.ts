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
// FEATURE TYPES
// =============================================================================

export interface NamedFeaturesV20 {
  /** 5-minute log return */
  log_ret_5m: number;
  /** 1-hour log return */
  log_ret_1h: number;
  /** 4-hour log return */
  log_ret_4h: number;
  /** RSI period 9 */
  rsi_9: number;
  /** ATR as percentage */
  atr_pct: number;
  /** ADX period 14 */
  adx_14: number;
  /** DXY z-score */
  dxy_z: number;
  /** DXY daily change */
  dxy_change_1d: number;
  /** VIX z-score */
  vix_z: number;
  /** EMBI z-score */
  embi_z: number;
  /** Brent daily change */
  brent_change_1d: number;
  /** UST 10Y-2Y spread */
  rate_spread: number;
  /** USDMXN daily change */
  usdmxn_change_1d: number;
  /** Current position */
  position: number;
  /** Normalized session time */
  time_normalized: number;
}

export interface ObservationSchema {
  /** Observation vector (15 dimensions) */
  values: List;
  /** Contract version */
  contract_version?: string;
}

export interface NormalizationStatsSchema {
  /** Feature mean from training period */
  mean: number;
  /** Feature standard deviation */
  std: number;
  /** Minimum clip value */
  clip_min?: number;
  /** Maximum clip value */
  clip_max?: number;
}

export interface FeatureSnapshotSchema {
  /** Raw observation vector (15 dims) */
  observation: List;
  /** Named features */
  features: NamedFeaturesV20;
  /** Market context */
  market_context?: Optional;
  /** Contract version */
  contract_version?: string;
}

// =============================================================================
// TRADING TYPES
// =============================================================================

export interface CandlestickSchema {
  /** Unix timestamp (seconds) */
  time: number;
  /** Open price */
  open: number;
  /** High price */
  high: number;
  /** Low price */
  low: number;
  /** Close price */
  close: number;
  /** Volume (often 0 for FX) */
  volume?: Optional;
}

export interface SignalSchema {
  /** Unique signal ID */
  id: string;
  /** Signal timestamp (ISO format) */
  timestamp: string;
  /** Signal type (BUY/SELL/HOLD) */
  type: SignalType;
  /** Model confidence (0-1) */
  confidence: number;
  /** Price at signal time */
  price: number;
  /** Stop loss price */
  stop_loss?: Optional;
  /** Take profit price */
  take_profit?: Optional;
  /** Signal reasoning/factors */
  reasoning?: List;
  /** Risk score (0-1) */
  risk_score?: Optional;
  /** Expected return percentage */
  expected_return?: Optional;
  /** Expected time horizon */
  time_horizon?: Optional;
  /** Model identifier */
  model_source: string;
  /** Specific model ID */
  model_id?: Optional;
  /** Inference latency (ms) */
  latency?: Optional;
  technical_indicators?: Optional;
  /** Data type: backtest, out_of_sample, live */
  data_type?: Optional;
}

export interface TradeMetadataSchema {
  /** Entry confidence */
  confidence: number;
  /** Action probabilities [HOLD, LONG, SHORT] */
  action_probs?: Optional;
  /** Critic value */
  critic_value?: Optional;
  /** Policy entropy */
  entropy?: Optional;
  /** Advantage estimate */
  advantage?: Optional;
  /** Model version used */
  model_version: string;
  /** Normalization stats version */
  norm_stats_version: string;
  /** Model file hash */
  model_hash?: Optional;
}

export interface TradeSchema {
  /** Creation timestamp (UTC) */
  created_at?: Optional;
  /** Last update timestamp (UTC) */
  updated_at?: Optional;
  /** Unique trade ID */
  trade_id: string;
  /** Model that generated the trade */
  model_id: string;
  /** Entry timestamp (ISO format) */
  timestamp: string;
  /** Entry time (ISO format) */
  entry_time: string;
  /** Exit time */
  exit_time?: Optional;
  /** Trade direction */
  side: TradeSide;
  /** Entry price */
  entry_price: number;
  /** Exit price */
  exit_price?: Optional;
  /** Position size */
  size?: number;
  /** P&L in pips/points */
  pnl?: number;
  /** P&L in USD */
  pnl_usd?: number;
  /** P&L percentage */
  pnl_percent?: number;
  /** P&L percentage (alias) */
  pnl_pct?: number;
  status?: TradeStatus;
  /** Trade duration in minutes */
  duration_minutes?: Optional;
  /** Exit reason */
  exit_reason?: Optional;
  /** Equity at entry */
  equity_at_entry?: Optional;
  /** Equity at exit */
  equity_at_exit?: Optional;
  /** Entry confidence */
  entry_confidence?: Optional;
  /** Exit confidence */
  exit_confidence?: Optional;
  commission?: Optional;
  market_regime?: Optional;
  /** Maximum adverse excursion */
  max_adverse_excursion?: Optional;
  /** Maximum favorable excursion */
  max_favorable_excursion?: Optional;
  /** Features at trade entry */
  features_snapshot?: Optional;
  /** Model inference metadata */
  model_metadata?: Optional;
}

export interface TradeSummarySchema {
  /** Total number of trades */
  total_trades: number;
  /** Number of winning trades */
  winning_trades: number;
  /** Number of losing trades */
  losing_trades: number;
  /** Win rate (0-1) */
  win_rate: number;
  /** Total P&L */
  total_pnl: number;
  /** Total P&L in USD */
  total_pnl_usd?: number;
  /** Total return percentage */
  total_return_pct: number;
  /** Maximum drawdown % */
  max_drawdown_pct: number;
  /** Sharpe ratio */
  sharpe_ratio?: Optional;
  /** Sortino ratio */
  sortino_ratio?: Optional;
  /** Profit factor */
  profit_factor?: Optional;
  /** Average winning trade */
  avg_win?: Optional;
  /** Average losing trade */
  avg_loss?: Optional;
  /** Largest win */
  largest_win?: Optional;
  /** Largest loss */
  largest_loss?: Optional;
  /** Average trade duration */
  avg_trade_duration_minutes?: Optional;
}

// =============================================================================
// API TYPES
// =============================================================================

export interface BacktestRequestSchema {
  /** Start date for backtest (YYYY-MM-DD) */
  start_date: string;
  /** End date for backtest (YYYY-MM-DD) */
  end_date: string;
  /** Model ID to use for inference */
  model_id?: string;
  /** Force regeneration even if cached trades exist */
  force_regenerate?: boolean;
}

export interface InferenceRequestSchema {
  /** Observation vector (15 dimensions) */
  observation: List;
  /** Model ID to use */
  model_id?: string;
}

export interface ApiMetadataSchema {
  /** Source of the data */
  data_source: DataSource;
  /** Response timestamp (UTC) */
  timestamp?: string;
  /** Whether data is real or mock */
  is_real_data: boolean;
  /** Request latency in milliseconds */
  latency_ms?: Optional;
  /** Whether cache was hit */
  cache_hit?: Optional;
  /** Request tracking ID */
  request_id?: Optional;
}

export interface BacktestResponseSchema {
  /** Whether backtest succeeded */
  success?: boolean;
  /** 'database' if cached, 'generated' if newly computed */
  source: string;
  /** Number of trades generated */
  trade_count: number;
  /** List of trades */
  trades: List;
  /** Trade summary statistics */
  summary?: Optional;
  /** Processing time in milliseconds */
  processing_time_ms?: Optional;
  /** Actual date range of data */
  date_range?: Optional;
}

export interface HealthResponseSchema {
  /** Service health status */
  status?: string;
  /** Service version */
  version: string;
  /** Whether model is loaded */
  model_loaded: boolean;
  /** Whether database is connected */
  database_connected: boolean;
  /** Health check timestamp */
  timestamp?: string;
}

export interface ErrorResponseSchema {
  success?: boolean;
  /** Error message */
  error: string;
  /** Machine-readable error code */
  error_code?: Optional;
  /** Additional error details */
  details?: Optional;
}

export interface ModelInfoSchema {
  /** Model identifier */
  model_id: string;
  /** Display name */
  display_name: string;
  /** Model version */
  version?: Optional;
  /** Model status */
  status: string;
  /** Observation dimension */
  observation_dim?: number;
  /** Model description */
  description?: Optional;
}

export interface ModelsResponseSchema {
  /** Available models */
  models: List;
  /** Default model ID */
  default_model: string;
  /** Total model count */
  total: number;
}

