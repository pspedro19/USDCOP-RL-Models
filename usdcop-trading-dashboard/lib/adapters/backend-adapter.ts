/**
 * Backend Adapter
 * ================
 * Transforms data between backend API format and frontend expected format.
 *
 * KEY MAPPINGS:
 * - model_id (backend: ppo_primary) <-> strategy_code (frontend: RL_PPO)
 * - signal: LONG/SHORT (backend) <-> BUY/SELL (frontend)
 * - Equity curves: by-strategy (backend) <-> pivoted columns (frontend)
 */

// ============================================================================
// STRATEGY/MODEL MAPPING
// ============================================================================

/**
 * Maps frontend strategy_code to backend model_id
 */
export const STRATEGY_TO_MODEL: Record<string, string> = {
  // Production models
  RL_PPO: 'ppo_primary',           // Primary production model
  RL_PPO_SECONDARY: 'ppo_secondary', // Secondary model
  ppo_primary: 'ppo_primary',       // Direct model ID
  ppo_secondary: 'ppo_secondary',   // Direct model ID
  // Demo/placeholder models
  RL_SAC: 'sac_baseline',
  RL_TD3: 'td3_baseline',
  RL_A2C: 'a2c_baseline',
  ML_XGB: 'xgb_primary',
  ML_LGBM: 'lgbm_primary',
  sac_demo: 'sac_baseline',
  llm_claude_demo: 'llm_claude',
};

/**
 * Maps backend model_id to frontend strategy_code
 */
export const MODEL_TO_STRATEGY: Record<string, string> = {
  // Production
  ppo_primary: 'RL_PPO',
  ppo_secondary: 'RL_PPO_SECONDARY',
  // Demo
  sac_baseline: 'RL_SAC',
  td3_baseline: 'RL_TD3',
  a2c_baseline: 'RL_A2C',
  xgb_primary: 'ML_XGB',
  lgbm_primary: 'ML_LGBM',
  llm_claude: 'LLM_CLAUDE',
};

/**
 * Default model ID for production
 */
export const DEFAULT_MODEL_ID = 'ppo_primary';
export const DEFAULT_STRATEGY_CODE = 'RL_PPO';

/**
 * Initial capital for equity calculations
 */
export const INITIAL_CAPITAL = 10000;

/**
 * Get backend model_id from frontend strategy_code
 */
export function getModelId(strategyCode: string): string {
  return STRATEGY_TO_MODEL[strategyCode] || strategyCode.toLowerCase().replace('_', '_v');
}

/**
 * Get frontend strategy_code from backend model_id
 */
export function getStrategyCode(modelId: string): string {
  return MODEL_TO_STRATEGY[modelId] || modelId.toUpperCase().replace('_V', '_');
}

/**
 * Get strategy type from strategy_code
 */
export function getStrategyType(strategyCode: string): 'RL' | 'ML' | 'LLM' {
  if (strategyCode.startsWith('RL_')) return 'RL';
  if (strategyCode.startsWith('ML_')) return 'ML';
  if (strategyCode.startsWith('LLM_')) return 'LLM';
  if (strategyCode === 'ENSEMBLE') return 'ML';
  return 'ML';
}

// ============================================================================
// SIGNAL MAPPING
// ============================================================================

/**
 * Backend signal types
 */
export type BackendSignal = 'LONG' | 'SHORT' | 'HOLD' | 'CLOSE';

/**
 * Frontend signal types (for chart overlay)
 */
export type FrontendChartSignal = 'BUY' | 'SELL';

/**
 * Frontend signal types (for model cards)
 */
export type FrontendModelSignal = 'long' | 'short' | 'flat' | 'close';

/**
 * Maps backend signal to frontend chart signal (BUY/SELL)
 * Returns null if signal should be ignored (HOLD/CLOSE)
 */
export function toChartSignal(backendSignal: BackendSignal): FrontendChartSignal | null {
  switch (backendSignal) {
    case 'LONG':
      return 'BUY';
    case 'SHORT':
      return 'SELL';
    case 'HOLD':
    case 'CLOSE':
      return null; // Ignore for chart overlay
  }
}

/**
 * Maps backend signal to frontend model signal (lowercase)
 */
export function toModelSignal(backendSignal: BackendSignal): FrontendModelSignal {
  switch (backendSignal) {
    case 'LONG':
      return 'long';
    case 'SHORT':
      return 'short';
    case 'HOLD':
      return 'flat';
    case 'CLOSE':
      return 'close';
  }
}

/**
 * Maps backend signal to trade side
 */
export function toTradeSide(backendSignal: BackendSignal): 'buy' | 'sell' | 'hold' {
  switch (backendSignal) {
    case 'LONG':
      return 'buy';
    case 'SHORT':
      return 'sell';
    case 'HOLD':
    case 'CLOSE':
      return 'hold';
  }
}

// ============================================================================
// BACKEND RESPONSE TYPES (Confirmados por Backend)
// ============================================================================

/**
 * Backend inference/signal response
 */
export interface BackendInference {
  inference_id: number;
  timestamp_utc: string;
  model_id: string;
  action_raw: number; // -1 to +1
  action_discretized: BackendSignal;
  confidence: number; // 0-1
  price_at_inference: number;
  features_snapshot?: Record<string, number>;
}

/**
 * Backend trade response - Confirmado: GET /api/models/{model_id}/trades
 */
export interface BackendTrade {
  trade_id: number;
  model_id?: string;
  open_time: string;
  close_time: string | null;
  signal: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number | null;
  position_size?: number;
  pnl: number | null;
  pnl_pct: number | null;
  duration_minutes: number | null;
  status: 'open' | 'closed' | 'cancelled' | 'stopped' | 'expired';
  transaction_costs?: number;
  spread_cost?: number;
  slippage?: number;
  confidence: number;
}

/**
 * Backend trades response with summary - Confirmado
 */
export interface BackendTradesResponse {
  model_id: string;
  period: string;
  trades: BackendTrade[];
  summary: {
    total: number;
    wins: number;
    losses: number;
    holds: number;
    win_rate: number;
    pnl_total: number;
    streak: number;
  };
}

/**
 * Backend equity curve point - Confirmado
 */
export interface BackendEquityPoint {
  timestamp: string;
  equity_value: number;
  return_pct: number;
  drawdown_pct: number;
}

/**
 * Backend equity curve response (by strategy) - Confirmado: GET /api/models/equity-curves
 */
export interface BackendEquityCurve {
  strategy_code: string;
  strategy_name: string;
  model_id?: string; // Optional for backwards compatibility
  data: BackendEquityPoint[];
  summary: {
    starting_equity: number;
    ending_equity: number;
    total_return_pct: number;
  };
}

/**
 * Backend equity curves response - Confirmado
 */
export interface BackendEquityCurvesResponse {
  start_date: string;
  end_date: string;
  resolution: string;
  curves: BackendEquityCurve[];
}

/**
 * Backend metrics response - Confirmado: GET /api/models/{model_id}/metrics
 */
export interface BackendMetrics {
  model_id: string;
  period: string;
  live: {
    sharpe: number;
    sortino?: number;
    max_drawdown: number;
    win_rate: number;
    hold_percent: number;
    total_trades: number;
    pnl_today: number;
    pnl_today_pct: number;
    pnl_month: number;
    pnl_month_pct: number;
  };
  backtest: {
    sharpe: number;
    sortino?: number;
    max_drawdown: number;
    win_rate: number;
    hold_percent: number;
  };
}

/**
 * Backend position response - Confirmado: GET /api/models/positions/current
 */
export interface BackendPosition {
  strategy_code: string;
  strategy_name: string;
  side: 'long' | 'short' | 'flat';
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  entry_time: string;
  duration_minutes: number;
}

/**
 * Backend positions response - Confirmado
 */
export interface BackendPositionsResponse {
  timestamp: string;
  positions: BackendPosition[];
}

/**
 * Backend performance metrics (legacy format)
 */
export interface BackendPerformance {
  model_id: string;
  model_name?: string;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  max_drawdown_pct: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  total_trades: number;
  total_return_pct: number | null;
  current_equity: number;
  calmar_ratio?: number | null;
  winning_trades?: number;
  losing_trades?: number;
}

// ============================================================================
// FRONTEND EXPECTED TYPES
// ============================================================================

/**
 * Frontend signal for chart overlay
 */
export interface FrontendSignalData {
  timestamp: string;
  type: 'BUY' | 'SELL';
  confidence: number;
  price: number;
}

/**
 * Frontend trade for TradeHistory
 */
export interface FrontendTrade {
  trade_id: number;
  timestamp: string;
  strategy_code: string;
  strategy_name: string;
  side: 'buy' | 'sell';
  entry_price: number;
  exit_price: number | null;
  size: number;
  pnl: number;
  pnl_percent: number;
  status: 'open' | 'closed' | 'pending';
  duration_minutes: number | null;
  commission: number;
}

/**
 * Frontend equity curve point (pivoted)
 */
export interface FrontendEquityCurvePoint {
  timestamp: string;
  RL_PPO: number;
  ML_LGBM: number;
  ML_XGB: number | null;
  LLM_CLAUDE: number | null;
  PORTFOLIO: number;
  CAPITAL_INICIAL: number;
}

/**
 * Frontend model signal for cards
 */
export interface FrontendModelCardSignal {
  signal_id: number;
  timestamp_utc: string;
  strategy_code: string;
  strategy_name: string;
  strategy_type: 'RL' | 'ML' | 'LLM';
  signal: 'long' | 'short' | 'flat' | 'close';
  side: 'buy' | 'sell' | 'hold';
  size: number;
  confidence: number;
  entry_price: number;
  stop_loss: number | null;
  take_profit: number | null;
  risk_usd: number;
  notional_usd: number;
  reasoning: string;
  features_snapshot: Record<string, number> | null;
}

/**
 * Frontend strategy performance
 */
export interface FrontendStrategyPerformance {
  strategy_code: string;
  strategy_name: string;
  strategy_type: 'RL' | 'ML' | 'LLM';
  current_equity: number;
  cash_balance: number;
  open_positions: number;
  total_return_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  today_return_pct: number | null;
  today_trades: number | null;
}

/**
 * Frontend metrics for model dashboard - Confirmado
 */
export interface FrontendModelMetrics {
  model_id: string;
  strategy_code: string;
  period: string;
  live: {
    sharpe: number;
    sortino: number;
    max_drawdown: number;
    win_rate: number;
    hold_percent: number;
    total_trades: number;
    pnl_today: number;
    pnl_today_pct: number;
    pnl_month: number;
    pnl_month_pct: number;
  };
  backtest: {
    sharpe: number;
    sortino: number;
    max_drawdown: number;
    win_rate: number;
    hold_percent: number;
  };
}

/**
 * Frontend position - Confirmado
 */
export interface FrontendPosition {
  strategy_code: string;
  strategy_name: string;
  side: 'long' | 'short' | 'flat';
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  entry_time: string;
  duration_minutes: number;
}

/**
 * Frontend trades summary - Confirmado
 */
export interface FrontendTradesSummary {
  total: number;
  wins: number;
  losses: number;
  holds: number;
  win_rate: number;
  pnl_total: number;
  streak: number;
}

// ============================================================================
// TRANSFORMERS
// ============================================================================

/**
 * Transform backend inference to frontend chart signal
 */
export function transformToChartSignal(inference: BackendInference): FrontendSignalData | null {
  const chartType = toChartSignal(inference.action_discretized);
  if (!chartType) return null;

  return {
    timestamp: inference.timestamp_utc,
    type: chartType,
    confidence: inference.confidence,
    price: inference.price_at_inference,
  };
}

/**
 * Transform array of backend inferences to frontend chart signals
 */
export function transformToChartSignals(inferences: BackendInference[]): FrontendSignalData[] {
  return inferences
    .map(transformToChartSignal)
    .filter((s): s is FrontendSignalData => s !== null);
}

/**
 * Transform backend inference to frontend model card signal
 */
export function transformToModelCardSignal(
  inference: BackendInference,
  strategyName?: string
): FrontendModelCardSignal {
  const strategyCode = getStrategyCode(inference.model_id);

  return {
    signal_id: inference.inference_id,
    timestamp_utc: inference.timestamp_utc,
    strategy_code: strategyCode,
    strategy_name: strategyName || strategyCode,
    strategy_type: getStrategyType(strategyCode),
    signal: toModelSignal(inference.action_discretized),
    side: toTradeSide(inference.action_discretized),
    size: Math.abs(inference.action_raw), // Position size from action magnitude
    confidence: inference.confidence,
    entry_price: inference.price_at_inference,
    stop_loss: null, // Not provided by backend
    take_profit: null, // Not provided by backend
    risk_usd: 0, // Calculate if needed
    notional_usd: 0, // Calculate if needed
    reasoning: `Action: ${inference.action_discretized}, Confidence: ${(inference.confidence * 100).toFixed(1)}%`,
    features_snapshot: inference.features_snapshot || null,
  };
}

/**
 * Transform backend trade to frontend trade
 */
export function transformToFrontendTrade(backendTrade: BackendTrade): FrontendTrade {
  const strategyCode = getStrategyCode(backendTrade.model_id || 'ppo_v1');

  // Map backend status to frontend status
  let frontendStatus: 'open' | 'closed' | 'pending';
  switch (backendTrade.status) {
    case 'open':
      frontendStatus = 'open';
      break;
    case 'closed':
    case 'stopped':
    case 'expired':
      frontendStatus = 'closed';
      break;
    case 'cancelled':
      frontendStatus = 'pending';
      break;
    default:
      frontendStatus = 'closed';
  }

  // Calculate commission from cost components (with null safety)
  const transactionCosts = backendTrade.transaction_costs ?? 0;
  const spreadCost = backendTrade.spread_cost ?? 0;
  const slippage = backendTrade.slippage ?? 0;
  const commission = transactionCosts + spreadCost + slippage;

  return {
    trade_id: backendTrade.trade_id,
    timestamp: backendTrade.open_time,
    strategy_code: strategyCode,
    strategy_name: strategyCode, // Could be enhanced with lookup
    side: backendTrade.signal === 'LONG' ? 'buy' : 'sell',
    entry_price: backendTrade.entry_price,
    exit_price: backendTrade.exit_price,
    size: backendTrade.position_size ?? 0.1,
    pnl: backendTrade.pnl ?? 0,
    pnl_percent: backendTrade.pnl_pct ?? 0,
    status: frontendStatus,
    duration_minutes: backendTrade.duration_minutes,
    commission: commission,
  };
}

/**
 * Transform array of backend trades to frontend trades
 */
export function transformToFrontendTrades(backendTrades: BackendTrade[]): FrontendTrade[] {
  return backendTrades.map(transformToFrontendTrade);
}

/**
 * Transform backend equity curves to frontend pivoted format
 */
export function transformToFrontendEquityCurves(
  backendCurves: BackendEquityCurve[],
  initialCapital: number = 10000
): FrontendEquityCurvePoint[] {
  // Collect all unique timestamps
  const allTimestamps = new Set<string>();
  const curvesByStrategy: Record<string, Map<string, BackendEquityPoint>> = {};

  backendCurves.forEach((curve) => {
    const strategyCode = curve.strategy_code || getStrategyCode(curve.model_id || '');
    curvesByStrategy[strategyCode] = new Map();

    curve.data.forEach((point) => {
      allTimestamps.add(point.timestamp);
      curvesByStrategy[strategyCode].set(point.timestamp, point);
    });
  });

  // Sort timestamps
  const sortedTimestamps = Array.from(allTimestamps).sort();

  // Build pivoted data
  return sortedTimestamps.map((timestamp) => {
    const rl_ppo = curvesByStrategy['RL_PPO']?.get(timestamp)?.equity_value ?? initialCapital;
    const ml_lgbm = curvesByStrategy['ML_LGBM']?.get(timestamp)?.equity_value ?? initialCapital;
    const ml_xgb = curvesByStrategy['ML_XGB']?.get(timestamp)?.equity_value ?? null;
    const llm_claude = curvesByStrategy['LLM_CLAUDE']?.get(timestamp)?.equity_value ?? null;

    // Calculate portfolio as sum of all strategies
    const portfolio =
      rl_ppo +
      ml_lgbm +
      (ml_xgb ?? initialCapital) +
      (llm_claude ?? initialCapital);

    return {
      timestamp,
      RL_PPO: rl_ppo,
      ML_LGBM: ml_lgbm,
      ML_XGB: ml_xgb,
      LLM_CLAUDE: llm_claude,
      PORTFOLIO: portfolio,
      CAPITAL_INICIAL: initialCapital,
    };
  });
}

/**
 * Transform backend performance to frontend strategy performance
 */
export function transformToFrontendPerformance(
  backendPerf: BackendPerformance
): FrontendStrategyPerformance {
  const strategyCode = getStrategyCode(backendPerf.model_id);

  return {
    strategy_code: strategyCode,
    strategy_name: backendPerf.model_name || strategyCode,
    strategy_type: getStrategyType(strategyCode),
    current_equity: backendPerf.current_equity,
    cash_balance: backendPerf.current_equity, // Approximation
    open_positions: 0, // Not provided
    total_return_pct: backendPerf.total_return_pct || 0,
    sharpe_ratio: backendPerf.sharpe_ratio || 0,
    sortino_ratio: backendPerf.sortino_ratio || 0,
    max_drawdown_pct: backendPerf.max_drawdown_pct || 0,
    total_trades: backendPerf.total_trades,
    win_rate: backendPerf.win_rate || 0,
    profit_factor: backendPerf.profit_factor || 0,
    today_return_pct: null, // Calculate separately if needed
    today_trades: null, // Calculate separately if needed
  };
}

/**
 * Transform array of backend performances
 */
export function transformToFrontendPerformances(
  backendPerfs: BackendPerformance[]
): FrontendStrategyPerformance[] {
  return backendPerfs.map(transformToFrontendPerformance);
}

/**
 * Transform backend metrics to frontend metrics - Confirmado
 */
export function transformToFrontendMetrics(
  backendMetrics: BackendMetrics
): FrontendModelMetrics {
  return {
    model_id: backendMetrics.model_id,
    strategy_code: getStrategyCode(backendMetrics.model_id),
    period: backendMetrics.period,
    live: {
      sharpe: backendMetrics.live.sharpe,
      sortino: backendMetrics.live.sortino ?? 0,
      max_drawdown: backendMetrics.live.max_drawdown,
      win_rate: backendMetrics.live.win_rate,
      hold_percent: backendMetrics.live.hold_percent,
      total_trades: backendMetrics.live.total_trades,
      pnl_today: backendMetrics.live.pnl_today,
      pnl_today_pct: backendMetrics.live.pnl_today_pct,
      pnl_month: backendMetrics.live.pnl_month,
      pnl_month_pct: backendMetrics.live.pnl_month_pct,
    },
    backtest: {
      sharpe: backendMetrics.backtest.sharpe,
      sortino: backendMetrics.backtest.sortino ?? 0,
      max_drawdown: backendMetrics.backtest.max_drawdown,
      win_rate: backendMetrics.backtest.win_rate,
      hold_percent: backendMetrics.backtest.hold_percent,
    },
  };
}

/**
 * Transform backend position to frontend position - Confirmado
 */
export function transformToFrontendPosition(
  backendPosition: BackendPosition
): FrontendPosition {
  return {
    strategy_code: backendPosition.strategy_code,
    strategy_name: backendPosition.strategy_name,
    side: backendPosition.side,
    entry_price: backendPosition.entry_price,
    current_price: backendPosition.current_price,
    unrealized_pnl: backendPosition.unrealized_pnl,
    unrealized_pnl_pct: backendPosition.unrealized_pnl_pct,
    entry_time: backendPosition.entry_time,
    duration_minutes: backendPosition.duration_minutes,
  };
}

/**
 * Transform backend positions response to frontend positions
 */
export function transformToFrontendPositions(
  response: BackendPositionsResponse
): FrontendPosition[] {
  return response.positions.map(transformToFrontendPosition);
}

// ============================================================================
// API URL HELPERS (Confirmados por Backend)
// ============================================================================

const MULTI_MODEL_API_PORT = 8006;
const MULTI_MODEL_API_BASE =
  process.env.NEXT_PUBLIC_MULTI_MODEL_API_URL || `http://localhost:${MULTI_MODEL_API_PORT}`;

/**
 * Backend endpoint mapping - Confirmados
 *
 * Endpoints activos:
 * - GET /api/models/{model_id}/metrics       → Métricas (live + backtest)
 * - GET /api/models/equity-curves            → Equity curves con filtros
 * - GET /api/models/{model_id}/trades        → Trades con summary
 * - GET /api/models/signals/latest           → Señales más recientes
 * - GET /api/models/positions/current        → Posiciones abiertas
 *
 * SSE Streams:
 * - GET /api/stream/metrics/{model_id}       → Updates cada 30s
 * - GET /api/stream/equity-curves            → Updates cada 5m
 * - GET /api/stream/signals/{model_id}       → Updates cada 5m
 */
export const BACKEND_ENDPOINTS = {
  // Métricas
  metricsByModel: (modelId: string) =>
    `${MULTI_MODEL_API_BASE}/api/models/${modelId}/metrics`,

  // Signals
  signalsLatest: `${MULTI_MODEL_API_BASE}/api/models/signals/latest`,
  signalsByModel: (modelId: string) =>
    `${MULTI_MODEL_API_BASE}/api/models/${modelId}/signals`,

  // Trades
  tradesByModel: (modelId: string, params?: { period?: string; status?: string; limit?: number }) => {
    const url = new URL(`${MULTI_MODEL_API_BASE}/api/models/${modelId}/trades`);
    if (params?.period) url.searchParams.set('period', params.period);
    if (params?.status) url.searchParams.set('status', params.status);
    if (params?.limit) url.searchParams.set('limit', params.limit.toString());
    return url.toString();
  },

  // Equity Curves
  equityCurves: (params?: { hours?: number; strategies?: string; resolution?: string }) => {
    const url = new URL(`${MULTI_MODEL_API_BASE}/api/models/equity-curves`);
    if (params?.hours) url.searchParams.set('hours', params.hours.toString());
    if (params?.strategies) url.searchParams.set('strategies', params.strategies);
    if (params?.resolution) url.searchParams.set('resolution', params.resolution || '5m');
    return url.toString();
  },

  // Positions
  positionsCurrent: (strategy?: string) => {
    const url = new URL(`${MULTI_MODEL_API_BASE}/api/models/positions/current`);
    if (strategy) url.searchParams.set('strategy', strategy);
    return url.toString();
  },

  // Performance (comparación multi-modelo)
  performanceComparison: `${MULTI_MODEL_API_BASE}/api/models/performance/comparison`,

  // SSE Streams - Updates en tiempo real
  streamMetrics: (modelId: string) =>
    `${MULTI_MODEL_API_BASE}/api/stream/metrics/${modelId}`,
  streamEquityCurves: `${MULTI_MODEL_API_BASE}/api/stream/equity-curves`,
  streamSignals: (modelId: string) =>
    `${MULTI_MODEL_API_BASE}/api/stream/signals/${modelId}`,

  // WebSocket (parcialmente implementado - solo heartbeats)
  websocket: `ws://localhost:${MULTI_MODEL_API_PORT}/ws/trading-signals`,
};

/**
 * Helper to fetch with timeout
 */
export async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = 5000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

export default {
  // Constants
  DEFAULT_MODEL_ID,
  DEFAULT_STRATEGY_CODE,
  INITIAL_CAPITAL,

  // Mappings
  STRATEGY_TO_MODEL,
  MODEL_TO_STRATEGY,
  getModelId,
  getStrategyCode,
  getStrategyType,

  // Signal helpers
  toChartSignal,
  toModelSignal,
  toTradeSide,

  // Transformers
  transformToChartSignal,
  transformToChartSignals,
  transformToModelCardSignal,
  transformToFrontendTrade,
  transformToFrontendTrades,
  transformToFrontendEquityCurves,
  transformToFrontendPerformance,
  transformToFrontendPerformances,
  transformToFrontendMetrics,
  transformToFrontendPosition,
  transformToFrontendPositions,

  // Endpoints
  BACKEND_ENDPOINTS,
  fetchWithTimeout,
};
