/**
 * API Schemas with Zod Validation
 * ================================
 *
 * Single Source of Truth for all API contracts.
 * Used for runtime validation of API requests/responses.
 *
 * @module types/schemas
 */

import { z } from 'zod';

// =============================================================================
// ENUMS & CONSTANTS
// =============================================================================

/**
 * Signal types - Frontend canonical (BUY/SELL/HOLD)
 * Backend may use LONG/SHORT/HOLD - see BackendActionEnum
 */
export const SignalTypeEnum = z.enum(['BUY', 'SELL', 'HOLD']);
export type SignalType = z.infer<typeof SignalTypeEnum>;

/**
 * Backend action types from inference service
 * Maps: LONG -> BUY, SHORT -> SELL, HOLD -> HOLD
 */
export const BackendActionEnum = z.enum(['LONG', 'SHORT', 'HOLD']);
export type BackendAction = z.infer<typeof BackendActionEnum>;

/** Order side for positions */
export const OrderSideEnum = z.enum(['long', 'short', 'flat']);
export type OrderSide = z.infer<typeof OrderSideEnum>;

/** Trade side (lowercase for DB compatibility) */
export const TradeSideEnum = z.enum(['buy', 'sell']);
export type TradeSide = z.infer<typeof TradeSideEnum>;

/** Market status */
export const MarketStatusEnum = z.enum(['open', 'closed', 'pre_market', 'post_market']);
export type MarketStatus = z.infer<typeof MarketStatusEnum>;

/** Data source for API responses */
export const DataSourceEnum = z.enum(['live', 'postgres', 'minio', 'cached', 'mock', 'demo', 'fallback', 'none']);
export type DataSource = z.infer<typeof DataSourceEnum>;

/**
 * Model status - Combined from all sources:
 * - model_registry table: registered, deployed, retired
 * - config.models table: active, inactive, training, deprecated, testing
 * - Frontend display: production, backtest, archived
 */
export const ModelStatusEnum = z.enum([
  // model_registry status values
  'registered',
  'deployed',
  'retired',
  // config.models status values
  'active',
  'inactive',
  'training',
  'deprecated',
  'testing',
  // Frontend display values
  'production',
  'backtest',
  'archived',
]);
export type ModelStatus = z.infer<typeof ModelStatusEnum>;

/** Model registry status (database specific) */
export const ModelRegistryStatusEnum = z.enum(['registered', 'deployed', 'retired']);
export type ModelRegistryStatus = z.infer<typeof ModelRegistryStatusEnum>;

/** Trade status */
export const TradeStatusEnum = z.enum(['open', 'closed', 'pending', 'cancelled']);
export type TradeStatus = z.infer<typeof TradeStatusEnum>;

// =============================================================================
// FEATURE CONTRACT SCHEMAS (Aligned with ARCHITECTURE_CONTRACTS.md)
// =============================================================================

/**
 * V20 Feature order - MUST match Python FeatureBuilder exactly
 * Reference: ARCHITECTURE_CONTRACTS.md section "FEATURE_CONTRACT_V20"
 */
export const FEATURE_ORDER_V20 = [
  'log_ret_5m',      // 0: 5-minute log return
  'log_ret_1h',      // 1: 1-hour log return
  'log_ret_4h',      // 2: 4-hour log return
  'rsi_9',           // 3: RSI period 9
  'atr_pct',         // 4: ATR as % of close (period 10)
  'adx_14',          // 5: ADX period 14
  'dxy_z',           // 6: DXY z-score
  'dxy_change_1d',   // 7: DXY daily change
  'vix_z',           // 8: VIX z-score
  'embi_z',          // 9: EMBI spread z-score
  'brent_change_1d', // 10: Brent daily change
  'rate_spread',     // 11: Rate spread proxy
  'usdmxn_change_1d',// 12: USDMXN daily change
  'position',        // 13: Current position (-1 to 1)
  'time_normalized', // 14: Normalized session time (0 to 1)
] as const;

export const OBSERVATION_DIM_V20 = 15;

/**
 * Market context for trade execution audit
 * Reference: IMPLEMENTATION_PLAN.md P1-2 (FE-03)
 */
export const MarketContextSchema = z.object({
  bidAskSpreadBps: z.number(),
  estimatedSlippageBps: z.number(),
  executionPrice: z.number().optional(),
  timestampUtc: z.string(),
});
export type MarketContext = z.infer<typeof MarketContextSchema>;

/**
 * Named features dictionary for audit clarity
 * Reference: IMPLEMENTATION_PLAN.md P1-2, ARCHITECTURE_CONTRACTS.md
 */
export const NamedFeaturesSchema = z.object({
  log_ret_5m: z.number(),
  log_ret_1h: z.number(),
  log_ret_4h: z.number(),
  rsi_9: z.number(),
  atr_pct: z.number(),
  adx_14: z.number(),
  dxy_z: z.number(),
  dxy_change_1d: z.number(),
  vix_z: z.number(),
  embi_z: z.number(),
  brent_change_1d: z.number(),
  rate_spread: z.number(),
  usdmxn_change_1d: z.number(),
  position: z.number(),
  time_normalized: z.number(),
});
export type NamedFeatures = z.infer<typeof NamedFeaturesSchema>;

/**
 * Complete feature snapshot for trade traceability
 * Reference: IMPLEMENTATION_PLAN.md P0-8, P1-2
 * Stored in: trades_history.features_snapshot (JSONB)
 */
export const FeatureSnapshotSchema = z.object({
  // Raw observation vector (15 dims for v20)
  observation: z.array(z.number()),
  // Named features for audit clarity
  features: NamedFeaturesSchema,
  // Market context at execution time
  marketContext: MarketContextSchema.optional(),
  // Contract version for reproducibility
  contractVersion: z.string(),
});
export type FeatureSnapshot = z.infer<typeof FeatureSnapshotSchema>;

// =============================================================================
// MODEL REGISTRY SCHEMAS (Aligned with IMPLEMENTATION_PLAN.md P1-11)
// =============================================================================

/**
 * Model registry entry from database
 * Reference: database/migrations/005_model_registry.sql
 */
export const ModelRegistrySchema = z.object({
  id: z.number(),
  modelId: z.string(),
  modelVersion: z.string(),
  modelPath: z.string(),
  modelHash: z.string(),
  normStatsHash: z.string().optional(),
  configHash: z.string().optional(),
  observationDim: z.number().default(15),
  actionSpace: z.number().default(3),
  featureOrder: z.array(z.string()).optional(),
  // Backtest metrics
  backtestSharpe: z.number().optional(),
  backtestMaxDrawdown: z.number().optional(),
  backtestWinRate: z.number().optional(),
  backtestTotalTrades: z.number().optional(),
  backtestHoldPercent: z.number().optional(),
  backtestTestPeriod: z.string().optional(),
  // Status and timestamps
  status: ModelRegistryStatusEnum,
  createdAt: z.string().optional(),
  deployedAt: z.string().optional(),
  retiredAt: z.string().optional(),
  // Training info
  trainingDatasetId: z.number().optional(),
  trainingStartDate: z.string().optional(),
  trainingEndDate: z.string().optional(),
});
export type ModelRegistry = z.infer<typeof ModelRegistrySchema>;

/**
 * Model metadata stored with each trade for traceability
 * Reference: IMPLEMENTATION_PLAN.md P1-2
 */
export const TradeModelMetadataSchema = z.object({
  confidence: z.number().min(0).max(1),
  actionProbs: z.array(z.number()).optional(),
  criticValue: z.number().optional(),
  entropy: z.number().optional(),
  advantage: z.number().optional(),
  modelVersion: z.string(),
  normStatsVersion: z.string(),
  modelHash: z.string().optional(),
});
export type TradeModelMetadata = z.infer<typeof TradeModelMetadataSchema>;

// =============================================================================
// API RESPONSE WRAPPER (Canonical Definition)
// =============================================================================

/** API Metadata schema */
export const ApiMetadataSchema = z.object({
  dataSource: DataSourceEnum,
  timestamp: z.string().datetime(),
  isRealData: z.boolean(),
  latency: z.number().optional(),
  cacheHit: z.boolean().optional(),
  requestId: z.string().uuid().optional(),
});
export type ApiMetadata = z.infer<typeof ApiMetadataSchema>;

/** Generic API Response wrapper */
export const createApiResponseSchema = <T extends z.ZodType>(dataSchema: T) =>
  z.object({
    success: z.boolean(),
    data: dataSchema.nullable().optional(),
    error: z.string().optional(),
    message: z.string().optional(),
    metadata: ApiMetadataSchema,
  });

/** API Error response */
export const ApiErrorSchema = z.object({
  success: z.literal(false),
  error: z.string(),
  message: z.string().optional(),
  details: z.record(z.string(), z.unknown()).optional(),
  metadata: ApiMetadataSchema,
});
export type ApiError = z.infer<typeof ApiErrorSchema>;

// =============================================================================
// MARKET DATA SCHEMAS
// =============================================================================

/** Real-time price data */
export const RealtimePriceSchema = z.object({
  price: z.number(),
  change: z.number(),
  changePct: z.number(),
  source: z.string(),
  lastUpdate: z.string(),
  marketStatus: MarketStatusEnum,
  dayHigh: z.number().optional(),
  dayLow: z.number().optional(),
  week52High: z.number().optional(),
  week52Low: z.number().optional(),
});
export type RealtimePrice = z.infer<typeof RealtimePriceSchema>;

/** OHLCV Candlestick data */
export const CandlestickSchema = z.object({
  time: z.number(),
  open: z.number(),
  high: z.number(),
  low: z.number(),
  close: z.number(),
  volume: z.number().optional(),
});
export type Candlestick = z.infer<typeof CandlestickSchema>;

/** Candlesticks response */
export const CandlesticksResponseSchema = z.object({
  symbol: z.string(),
  timeframe: z.string(),
  startDate: z.string(),
  endDate: z.string(),
  count: z.number(),
  data: z.array(CandlestickSchema),
  metadata: z.object({
    source: z.string(),
    marketHoursOnly: z.boolean(),
  }).optional(),
});
export type CandlesticksResponse = z.infer<typeof CandlesticksResponseSchema>;

// =============================================================================
// MODEL SCHEMAS
// =============================================================================

/** Model backtest metrics */
export const BacktestMetricsSchema = z.object({
  sharpe: z.number(),
  maxDrawdown: z.number(),
  winRate: z.number(),
  holdPercent: z.number().optional(),
  totalTrades: z.number(),
  testPeriod: z.string().optional(),
});
export type BacktestMetrics = z.infer<typeof BacktestMetricsSchema>;

/** Model definition */
export const ModelSchema = z.object({
  id: z.string(),
  name: z.string(),
  algorithm: z.string(),
  version: z.string(),
  status: ModelStatusEnum,
  type: z.string(),
  color: z.string().optional(),
  description: z.string().optional(),
  isRealData: z.boolean(),
  dbModelId: z.string().optional(),
  backtest: BacktestMetricsSchema.optional(),
});
export type Model = z.infer<typeof ModelSchema>;

/** Models list response */
export const ModelsResponseSchema = z.object({
  models: z.array(ModelSchema),
  source: z.string(),
});
export type ModelsResponse = z.infer<typeof ModelsResponseSchema>;

/** Model performance metrics */
export const ModelMetricsSchema = z.object({
  pnlMonth: z.number(),
  pnlMonthPct: z.number(),
  pnlWeek: z.number(),
  pnlWeekPct: z.number(),
  pnlToday: z.number(),
  pnlTodayPct: z.number(),
  sharpe: z.number(),
  sortino: z.number().optional(),
  calmar: z.number().optional(),
  maxDrawdown: z.number(),
  maxDrawdownPct: z.number(),
  winRate: z.number(),
  totalTrades: z.number(),
  avgWin: z.number().optional(),
  avgLoss: z.number().optional(),
  profitFactor: z.number().optional(),
  currentEquity: z.number(),
  initialEquity: z.number(),
});
export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;

/** Metrics API response */
export const MetricsResponseSchema = z.object({
  modelId: z.string(),
  dbModelId: z.string().optional(),
  period: z.string(),
  metrics: ModelMetricsSchema,
  live: z.boolean(),
});
export type MetricsResponse = z.infer<typeof MetricsResponseSchema>;

/** Equity curve point */
export const EquityPointSchema = z.object({
  timestamp: z.string(),
  value: z.number(),
  drawdownPct: z.number().optional(),
  position: OrderSideEnum.optional(),
  price: z.number().optional(),
});
export type EquityPoint = z.infer<typeof EquityPointSchema>;

/** Equity curve response */
export const EquityCurveResponseSchema = z.object({
  modelId: z.string(),
  dbModelId: z.string().optional(),
  days: z.number(),
  points: z.array(EquityPointSchema),
  summary: z.object({
    startValue: z.number(),
    endValue: z.number(),
    totalReturn: z.number(),
    totalReturnPct: z.number(),
    maxDrawdown: z.number(),
    maxDrawdownPct: z.number(),
  }).optional(),
});
export type EquityCurveResponse = z.infer<typeof EquityCurveResponseSchema>;

// =============================================================================
// TRADING SIGNAL SCHEMAS
// =============================================================================

/** Technical indicators (optional enrichment) */
export const TechnicalIndicatorsSchema = z.object({
  rsi: z.number().optional(),
  macd: z.number().optional(),
  macdSignal: z.number().optional(),
  ema20: z.number().optional(),
  ema50: z.number().optional(),
  atr: z.number().optional(),
  bollingerUpper: z.number().optional(),
  bollingerLower: z.number().optional(),
}).passthrough();
export type TechnicalIndicators = z.infer<typeof TechnicalIndicatorsSchema>;

/** Trading signal */
export const TradingSignalSchema = z.object({
  id: z.union([z.string(), z.number()]),
  timestamp: z.string(),
  type: SignalTypeEnum,
  confidence: z.number().min(0).max(1),
  price: z.number(),
  stopLoss: z.number().optional(),
  takeProfit: z.number().optional(),
  reasoning: z.array(z.string()),
  riskScore: z.number().min(0).max(1).optional(),
  expectedReturn: z.number().optional(),
  timeHorizon: z.string().optional(),
  modelSource: z.string(),
  modelId: z.string().optional(),
  latency: z.number().optional(),
  technicalIndicators: TechnicalIndicatorsSchema.optional(),
  dataType: z.enum(['backtest', 'out_of_sample', 'live']).optional(),
});
export type TradingSignal = z.infer<typeof TradingSignalSchema>;

/** Signals API response */
export const SignalsResponseSchema = z.object({
  signals: z.array(TradingSignalSchema),
  total: z.number().optional(),
  marketPrice: z.number().optional(),
  marketStatus: MarketStatusEnum.optional(),
});
export type SignalsResponse = z.infer<typeof SignalsResponseSchema>;

// =============================================================================
// TRADE SCHEMAS
// =============================================================================

/** Trade record */
export const TradeSchema = z.object({
  tradeId: z.union([z.string(), z.number()]),
  timestamp: z.string(),
  strategyCode: z.string(),
  strategyName: z.string().optional(),
  side: TradeSideEnum,
  entryPrice: z.number(),
  exitPrice: z.number().nullable().optional(),
  size: z.number(),
  pnl: z.number(),
  pnlPercent: z.number().optional(),
  status: TradeStatusEnum,
  durationMinutes: z.number().optional(),
  commission: z.number().optional(),
  exitReason: z.string().optional(),
  equityAtEntry: z.number().optional(),
  equityAtExit: z.number().optional(),
  entryConfidence: z.number().optional(),
  exitConfidence: z.number().optional(),
  marketRegime: z.string().optional(),
  maxAdverseExcursion: z.number().optional(),
  maxFavorableExcursion: z.number().optional(),
});
export type Trade = z.infer<typeof TradeSchema>;

/** Trades history response */
export const TradesHistoryResponseSchema = z.object({
  trades: z.array(TradeSchema),
  total: z.number(),
  source: z.string(),
});
export type TradesHistoryResponse = z.infer<typeof TradesHistoryResponseSchema>;

// =============================================================================
// REPLAY SCHEMAS
// =============================================================================

/**
 * Replay trade (extended for replay mode with full traceability)
 * Reference: IMPLEMENTATION_PLAN.md P0-8, ARCHITECTURE_CONTRACTS.md
 */
export const ReplayTradeSchema = TradeSchema.extend({
  // Legacy simple features format (backwards compatible)
  features: z.record(z.string(), z.number()).optional(),
  // Full feature snapshot (preferred - P1-2 format)
  featuresSnapshot: FeatureSnapshotSchema.optional(),
  // Model metadata for traceability
  modelMetadata: TradeModelMetadataSchema.optional(),
  // Simplified model metadata (legacy)
  modelVersion: z.string().optional(),
  modelHash: z.string().optional(),
});
export type ReplayTrade = z.infer<typeof ReplayTradeSchema>;

/** Replay load request */
export const ReplayLoadRequestSchema = z.object({
  startDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  endDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  modelId: z.string(),
  forceRegenerate: z.boolean().optional(),
});
export type ReplayLoadRequest = z.infer<typeof ReplayLoadRequestSchema>;

/** Replay load response */
export const ReplayLoadResponseSchema = z.object({
  trades: z.array(ReplayTradeSchema),
  total: z.number(),
  summary: z.object({
    totalPnl: z.number(),
    winRate: z.number(),
    sharpe: z.number(),
    maxDrawdown: z.number(),
  }).optional(),
  source: z.string(),
  dateRange: z.object({
    start: z.string(),
    end: z.string(),
  }),
  processingTimeMs: z.number().optional(),
});
export type ReplayLoadResponse = z.infer<typeof ReplayLoadResponseSchema>;

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

/**
 * Validate API response data with schema
 * @throws ZodError if validation fails
 */
export function validateResponse<T>(schema: z.ZodType<T>, data: unknown): T {
  return schema.parse(data);
}

/**
 * Safe validation that returns null on failure
 */
export function safeValidateResponse<T>(schema: z.ZodType<T>, data: unknown): T | null {
  const result = schema.safeParse(data);
  return result.success ? result.data : null;
}

/**
 * Validate and transform API response with error details
 */
export function validateWithErrors<T>(
  schema: z.ZodType<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: z.ZodError } {
  const result = schema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, errors: result.error };
}

// =============================================================================
// TYPE GUARDS
// =============================================================================

export function isSignalType(value: unknown): value is SignalType {
  return SignalTypeEnum.safeParse(value).success;
}

export function isTradeSide(value: unknown): value is TradeSide {
  return TradeSideEnum.safeParse(value).success;
}

export function isMarketStatus(value: unknown): value is MarketStatus {
  return MarketStatusEnum.safeParse(value).success;
}

// =============================================================================
// SIGNAL MAPPING HELPERS (Backend LONG/SHORT -> Frontend BUY/SELL)
// =============================================================================

/**
 * Maps backend action (LONG/SHORT/HOLD) to frontend signal (BUY/SELL/HOLD)
 * Reference: ARCHITECTURE_CONTRACTS.md - inference service returns LONG/SHORT
 */
export function mapBackendAction(action: BackendAction): SignalType {
  const mapping: Record<BackendAction, SignalType> = {
    LONG: 'BUY',
    SHORT: 'SELL',
    HOLD: 'HOLD',
  };
  return mapping[action];
}

/**
 * Maps frontend signal to backend action
 */
export function mapToBackendAction(signal: SignalType): BackendAction {
  const mapping: Record<SignalType, BackendAction> = {
    BUY: 'LONG',
    SELL: 'SHORT',
    HOLD: 'HOLD',
  };
  return mapping[signal];
}

/**
 * Validates that an observation vector matches V20 contract
 */
export function validateObservationV20(observation: number[]): boolean {
  if (observation.length !== OBSERVATION_DIM_V20) {
    return false;
  }
  // All values should be finite and within reasonable bounds (-5 to 5 after normalization)
  return observation.every(v => Number.isFinite(v) && Math.abs(v) <= 5);
}

/**
 * Creates a NamedFeatures object from observation vector
 */
export function observationToNamedFeatures(observation: number[]): NamedFeatures | null {
  if (observation.length !== OBSERVATION_DIM_V20) {
    return null;
  }

  return {
    log_ret_5m: observation[0],
    log_ret_1h: observation[1],
    log_ret_4h: observation[2],
    rsi_9: observation[3],
    atr_pct: observation[4],
    adx_14: observation[5],
    dxy_z: observation[6],
    dxy_change_1d: observation[7],
    vix_z: observation[8],
    embi_z: observation[9],
    brent_change_1d: observation[10],
    rate_spread: observation[11],
    usdmxn_change_1d: observation[12],
    position: observation[13],
    time_normalized: observation[14],
  };
}
