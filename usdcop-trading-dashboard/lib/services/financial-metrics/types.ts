/**
 * Financial Metrics Types
 * Type definitions for financial calculations
 */

// Core trade data structures
export interface Trade {
  id: string;
  timestamp: number;
  symbol: string;
  side: 'buy' | 'sell' | 'long' | 'short';
  quantity: number;
  entryPrice: number;
  exitPrice?: number;
  entryTime: number;
  exitTime?: number;
  pnl: number;
  pnlPercent: number;
  commission: number;
  status: 'open' | 'closed';
  duration?: number; // in minutes
  strategy?: string;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  openTime: number;
  duration: number; // in minutes
  strategy?: string;
}

export interface Signal {
  id: string;
  timestamp: number;
  symbol: string;
  type: 'buy' | 'sell' | 'neutral';
  strength: number; // 0-1
  price: number;
  strategy?: string;
}

// Equity curve point
export interface EquityPoint {
  timestamp: number;
  value: number;
  cumReturn: number;
  drawdown: number;
  drawdownPercent: number;
}

// Drawdown information
export interface DrawdownInfo {
  start: number;
  end?: number;
  peak: number;
  trough: number;
  value: number;
  percent: number;
  duration: number; // in minutes
  recovered: boolean;
}

// Comprehensive financial metrics
export interface FinancialMetrics {
  // P&L Metrics
  totalPnL: number;
  realizedPnL: number;
  unrealizedPnL: number;
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;

  // Returns
  totalReturn: number;
  dailyReturn: number;
  weeklyReturn: number;
  monthlyReturn: number;
  annualizedReturn: number;

  // Performance Ratios
  sharpeRatio: number;       // (Return - RiskFreeRate) / StdDev
  sortinoRatio: number;      // (Return - RiskFreeRate) / DownsideStdDev
  calmarRatio: number;       // AnnualReturn / MaxDrawdown
  profitFactor: number;      // GrossProfit / GrossLoss

  // Trade Statistics
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;           // Winning / Total
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgTradeDuration: number;  // minutes
  expectancy: number;        // Expected value per trade
  payoffRatio: number;       // AvgWin / AvgLoss

  // Risk Metrics
  maxDrawdown: number;
  maxDrawdownPercent: number;
  currentDrawdown: number;
  currentDrawdownPercent: number;
  valueAtRisk95: number;     // 95% VaR
  expectedShortfall: number; // CVaR (Conditional VaR)
  volatility: number;        // Annualized standard deviation
  downsideVolatility: number;// Downside deviation

  // Position Metrics
  openPositions: number;
  avgPositionSize: number;
  largestPosition: number;
  exposure: number;

  // Equity Curve
  equityCurve: EquityPoint[];
  drawdownCurve: DrawdownInfo[];

  // Time-based metrics
  firstTradeTime: number | null;
  lastTradeTime: number | null;
  tradingDays: number;

  // Additional metrics
  profitablePercentOfTime: number;
  avgDailyReturn: number;
  returnStdDev: number;
  informationRatio: number;
  kellyFraction: number;
}

// Calculation options
export interface MetricsOptions {
  riskFreeRate?: number;      // Annual risk-free rate (default: 0.03)
  confidenceLevel?: number;   // For VaR/CVaR (default: 0.95)
  tradingDaysPerYear?: number;// Default: 252
  initialCapital?: number;    // For equity curve calculations
  includeOpenPositions?: boolean; // Include unrealized P&L
}

// Time period for filtering
export interface TimePeriod {
  start: number;
  end: number;
}

// Grouped metrics by strategy or time period
export interface GroupedMetrics {
  [key: string]: FinancialMetrics;
}

// Performance summary for display
export interface PerformanceSummary {
  metrics: FinancialMetrics;
  topTrades: Trade[];
  worstTrades: Trade[];
  recentActivity: {
    last24h: number;
    last7d: number;
    last30d: number;
  };
  riskIndicators: {
    isHighRisk: boolean;
    riskLevel: 'low' | 'medium' | 'high';
    warnings: string[];
  };
}

// Real-time metric updates
export interface MetricUpdate {
  timestamp: number;
  metric: keyof FinancialMetrics;
  value: number;
  previousValue: number;
  change: number;
  changePercent: number;
}

// Historical metric snapshot
export interface MetricSnapshot {
  timestamp: number;
  metrics: FinancialMetrics;
}

// Error handling
export class MetricsCalculationError extends Error {
  constructor(message: string, public details?: any) {
    super(message);
    this.name = 'MetricsCalculationError';
  }
}
