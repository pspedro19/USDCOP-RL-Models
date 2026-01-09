/**
 * Technical Indicators Types
 * ========================
 *
 * Professional-grade type definitions for institutional trading indicators
 */

export interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorValue {
  timestamp: number;
  value: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

export interface MultiLineIndicator {
  timestamp: number;
  [key: string]: number;
}

export interface BollingerBands extends IndicatorValue {
  upper: number;
  middle: number;
  lower: number;
  width: number;
  percentB: number;
}

export interface MACD extends IndicatorValue {
  macd: number;
  signal: number;
  histogram: number;
  divergence?: 'BULLISH' | 'BEARISH' | null;
}

export interface Stochastic extends IndicatorValue {
  k: number;
  d: number;
  crossover?: 'BULLISH' | 'BEARISH' | null;
}

export interface VolumeProfile {
  levels: VolumeProfileLevel[];
  poc: number; // Point of Control
  valueAreaHigh: number;
  valueAreaLow: number;
  valueAreaVolume: number;
  totalVolume: number;
  profiles: {
    session: VolumeProfileLevel[];
    composite: VolumeProfileLevel[];
  };
}

export interface VolumeProfileLevel {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  percentOfTotal: number;
  delta: number; // buy - sell volume
}

export interface OrderFlow {
  timestamp: number;
  imbalance: number;
  buyPressure: number;
  sellPressure: number;
  netVolume: number;
  vwap: number;
  marketImpact: number;
}

export interface MarketMicrostructure {
  timestamp: number;
  spread: number;
  depth: number;
  liquidity: number;
  volatility: number;
  efficiency: number;
  toxicity: number;
}

export interface CorrelationMatrix {
  assets: string[];
  matrix: number[][];
  eigenvalues: number[];
  principalComponents: number[][];
  clustered: {
    groups: string[][];
    distances: number[][];
  };
}

export interface PerformanceMetrics {
  returns: {
    total: number;
    annualized: number;
    compound: number;
    volatility: number;
  };
  risk: {
    sharpe: number;
    sortino: number;
    calmar: number;
    maxDrawdown: number;
    var95: number;
    cvar95: number;
  };
  ratios: {
    informationRatio: number;
    treynorRatio: number;
    jensenAlpha: number;
    beta: number;
  };
  periods: {
    winRate: number;
    profitFactor: number;
    averageWin: number;
    averageLoss: number;
    consecutiveWins: number;
    consecutiveLosses: number;
  };
}

export interface BacktestResult {
  strategy: string;
  timeframe: string;
  startDate: number;
  endDate: number;
  initialCapital: number;
  finalCapital: number;
  trades: Trade[];
  performance: PerformanceMetrics;
  equity: EquityCurve[];
  drawdowns: DrawdownPeriod[];
}

export interface Trade {
  id: string;
  timestamp: number;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  pnl: number;
  commission: number;
  duration: number;
  indicators: { [key: string]: number };
}

export interface EquityCurve {
  timestamp: number;
  equity: number;
  drawdown: number;
  returns: number;
}

export interface DrawdownPeriod {
  start: number;
  end: number;
  peak: number;
  trough: number;
  drawdown: number;
  duration: number;
  recovery: number;
}

export interface IndicatorConfig {
  name: string;
  type: IndicatorType;
  parameters: { [key: string]: number | string | boolean };
  period?: number;
  source?: 'open' | 'high' | 'low' | 'close' | 'volume' | 'hlc3' | 'ohlc4';
  smooth?: boolean;
  normalize?: boolean;
}

export type IndicatorType =
  | 'sma' | 'ema' | 'wma' | 'dema' | 'tema' | 'kama' | 'hull'
  | 'rsi' | 'stochastic' | 'williams' | 'cci' | 'roc' | 'momentum'
  | 'macd' | 'ppo' | 'trix' | 'adx' | 'aroon' | 'psar'
  | 'bollinger' | 'keltner' | 'donchian' | 'envelope'
  | 'atr' | 'tr' | 'adl' | 'obv' | 'cmf' | 'mfi' | 'vwap'
  | 'ichimoku' | 'fibonacci' | 'pivot' | 'support_resistance'
  | 'volume_profile' | 'order_flow' | 'market_structure'
  | 'custom';

export interface WorkerMessage {
  id: string;
  type: 'CALCULATE_INDICATOR' | 'CALCULATE_CORRELATION' | 'BACKTEST' | 'VOLUME_PROFILE';
  payload: {
    data: CandleData[];
    config: IndicatorConfig | IndicatorConfig[];
    options?: { [key: string]: any };
  };
}

export interface WorkerResponse {
  id: string;
  type: 'SUCCESS' | 'ERROR';
  result?: any;
  error?: string;
  executionTime?: number;
}

export interface ChartVisualization {
  type: 'line' | 'area' | 'histogram' | 'scatter' | 'heatmap' | 'volume_profile';
  data: any[];
  config: {
    colors: string[];
    opacity?: number;
    thickness?: number;
    smooth?: boolean;
    fill?: boolean;
  };
  overlays?: {
    levels: number[];
    zones: { min: number; max: number; color: string; label?: string }[];
    annotations: { x: number; y: number; text: string }[];
  };
}

export interface CustomIndicator {
  id: string;
  name: string;
  description: string;
  formula: string;
  inputs: {
    name: string;
    type: 'number' | 'select' | 'boolean';
    default: any;
    options?: any[];
  }[];
  outputs: {
    name: string;
    type: 'line' | 'histogram' | 'area';
    color: string;
  }[];
  code: string;
  validation: {
    minPeriods: number;
    requiresVolume: boolean;
    outputs: string[];
  };
}

export interface IndicatorAlert {
  id: string;
  indicator: string;
  condition: 'crosses_above' | 'crosses_below' | 'greater_than' | 'less_than' | 'divergence';
  value: number;
  timeframe: string;
  enabled: boolean;
  lastTriggered?: number;
}

export interface OptimizationResult {
  parameters: { [key: string]: number };
  performance: PerformanceMetrics;
  score: number;
  iterations: number;
  converged: boolean;
}