/**
 * Production Monitor Contract
 * ===========================
 * Types and interfaces for real-time production model monitoring.
 */

// ============================================================================
// Production Model Info (from ProductionContract)
// ============================================================================

export interface ProductionModelInfo {
  modelId: string;
  experimentName: string;
  stage: 'production' | 'staging' | 'archived';
  promotedAt: string;
  approvedBy: string;
  configHash: string;
  featureOrderHash: string;
  modelHash: string;
  metrics: BacktestMetrics;
  lineage: ModelLineage;
}

export interface BacktestMetrics {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeReturn: number;
}

export interface ModelLineage {
  datasetHash: string;
  normStatsHash: string;
  rewardConfigHash: string;
  trainingStart: string;
  trainingEnd: string;
  validationStart: string;
  validationEnd: string;
}

// ============================================================================
// Real-Time Inference State
// ============================================================================

export interface LiveInferenceState {
  lastSignal: TradingSignal | null;
  currentPosition: 'LONG' | 'SHORT' | 'NEUTRAL';
  positionEntryPrice: number | null;
  positionEntryTime: string | null;
  unrealizedPnL: number;
  todayPnL: number;
  todayTrades: number;
  lastUpdateTime: string;
  isMarketOpen: boolean;
}

export interface TradingSignal {
  timestamp: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  modelId: string;
}

// ============================================================================
// Session Equity Curve
// ============================================================================

export interface EquityPoint {
  timestamp: string;
  equity: number;
  position: 'LONG' | 'SHORT' | 'NEUTRAL';
  signal?: 'BUY' | 'SELL';
}

export interface SessionEquityCurve {
  sessionDate: string;
  startEquity: number;
  currentEquity: number;
  points: EquityPoint[];
  highWaterMark: number;
  maxDrawdown: number;
}

// ============================================================================
// Pending Experiments Summary
// ============================================================================

export interface PendingExperimentsSummary {
  count: number;
  experiments: PendingExperimentPreview[];
}

export interface PendingExperimentPreview {
  proposalId: string;
  modelId: string;
  experimentName: string;
  recommendation: 'PROMOTE' | 'REJECT' | 'REVIEW';
  confidence: number;
  hoursUntilExpiry: number | null;
  createdAt: string;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ProductionMonitorResponse {
  productionModel: ProductionModelInfo | null;
  liveState: LiveInferenceState | null;
  equityCurve: SessionEquityCurve | null;
  pendingSummary: PendingExperimentsSummary;
  error?: string;
}

// ============================================================================
// Helpers
// ============================================================================

export function formatPnL(pnl: number): string {
  const sign = pnl >= 0 ? '+' : '';
  return `${sign}$${pnl.toFixed(2)}`;
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function getSignalColor(action: 'BUY' | 'SELL' | 'HOLD'): string {
  switch (action) {
    case 'BUY': return 'text-green-400';
    case 'SELL': return 'text-red-400';
    case 'HOLD': return 'text-gray-400';
  }
}

export function getPositionColor(position: 'LONG' | 'SHORT' | 'NEUTRAL'): string {
  switch (position) {
    case 'LONG': return 'text-green-400';
    case 'SHORT': return 'text-red-400';
    case 'NEUTRAL': return 'text-gray-400';
  }
}

export function isMarketHours(): boolean {
  const now = new Date();
  const hour = now.getUTCHours();
  // USDCOP market: ~13:00 - 21:00 UTC (8am - 4pm Colombia)
  return hour >= 13 && hour < 21;
}

// ============================================================================
// Live Production Monitoring (H5 DB-backed)
// ============================================================================

export interface CurrentSignal {
  signal_date: string;
  direction: number;             // +1 LONG, -1 SHORT
  confidence_tier: string | null; // HIGH / MEDIUM / LOW
  adjusted_leverage: number | null;
  hard_stop_pct: number | null;
  take_profit_pct: number | null;
  ensemble_return: number;
  skip_trade: boolean;
}

export interface ActivePosition {
  execution_id: number;
  entry_price: number;
  entry_timestamp: string;
  direction: number;
  leverage: number;
  current_price: number;
  current_bar_time: string;
  unrealized_pnl_pct: number;
  distance_to_tp_pct: number | null;
  distance_to_hs_pct: number | null;
  bar_count: number;
  peak_price: number | null;
  status: string;
}

export interface EquityCurvePoint {
  date: string;
  equity: number;
  pnl_pct: number;
}

export interface LiveEquityCurve {
  initial_capital: number;
  points: EquityCurvePoint[];
  current_equity: number;
}

export interface LiveStats {
  total_return_pct: number;
  sharpe: number | null;
  max_dd_pct: number | null;
  win_rate_pct: number | null;
  profit_factor: number | null;
  n_trades: number;
  n_long: number;
  n_short: number;
  exit_reasons: Record<string, number>;
}

export interface Guardrails {
  cumulative_pnl_pct: number | null;
  consecutive_losses: number;
  rolling_sharpe_16w: number | null;
  rolling_da_short_16w: number | null;
  circuit_breaker_active: boolean;
  alerts: string[];
}

export interface MarketState {
  is_open: boolean;
  current_price: number | null;
  last_bar_time: string | null;
}

export type LiveDataSource = 'db' | 'file' | 'unavailable';

export interface LiveProductionResponse {
  strategy_id: string;
  strategy_name: string;
  current_signal: CurrentSignal | null;
  active_position: ActivePosition | null;
  trades: LiveTrade[];
  equity_curve: LiveEquityCurve;
  stats: LiveStats;
  guardrails: Guardrails | null;
  market: MarketState;
  generated_at: string;
  data_source: LiveDataSource;
  partial_errors?: string[];
}

export interface LiveTrade {
  trade_id: number;
  timestamp: string;
  exit_timestamp: string | null;
  side: 'LONG' | 'SHORT';
  entry_price: number;
  exit_price: number;
  pnl_usd: number;
  pnl_pct: number;
  exit_reason: string;
  equity_at_entry: number;
  equity_at_exit: number;
  leverage: number;
  confidence_tier: string | null;
  hard_stop_pct: number | null;
  take_profit_pct: number | null;
}
