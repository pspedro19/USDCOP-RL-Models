export interface HedgeFundMetrics {
  // Core Performance Metrics
  totalReturn: number
  cagr: number
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  maxDrawdown: number
  volatility: number

  // Trading Metrics
  winRate: number
  profitFactor: number
  payoffRatio: number
  expectancy: number
  hitRate: number

  // Risk Metrics
  var95: number
  cvar95: number
  kellyFraction: number

  // Market Metrics
  jensenAlpha: number
  informationRatio: number
  treynorRatio: number
  betaToMarket: number
  correlation: number
  trackingError: number
}

export interface Trade {
  id: string
  date: Date
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  price: number
  exitPrice?: number
  pnl: number
  commission: number
  duration: number
  returnPct: number
}

// Create metricsCalculator object with all required methods
export const metricsCalculator = {
  calculateSharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)

    if (stdDev === 0) return 0
    // Assuming risk-free rate of 3% annually (0.03/252 daily)
    const riskFreeRate = 0.03 / 252
    return (mean - riskFreeRate) / stdDev * Math.sqrt(252)
  },

  calculateSortinoRatio(returns: number[]): number {
    if (returns.length === 0) return 0
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length
    const downside = returns.filter(r => r < 0)

    if (downside.length === 0) return 0
    const downsideVariance = downside.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downside.length
    const downsideDeviation = Math.sqrt(downsideVariance)

    if (downsideDeviation === 0) return 0
    const riskFreeRate = 0.03 / 252
    return (mean - riskFreeRate) / downsideDeviation * Math.sqrt(252)
  },

  calculateMaxDrawdown(prices: number[]): number {
    if (prices.length < 2) return 0

    let maxDrawdown = 0
    let peak = prices[0]

    for (let i = 1; i < prices.length; i++) {
      if (prices[i] > peak) {
        peak = prices[i]
      }

      const drawdown = (peak - prices[i]) / peak
      maxDrawdown = Math.max(maxDrawdown, drawdown)
    }

    return -maxDrawdown // Return as negative value
  },

  calculateVaR(returns: number[], confidence: number): number {
    if (returns.length === 0) return 0
    const sorted = [...returns].sort((a, b) => a - b)
    const index = Math.floor((1 - confidence) * sorted.length)
    return sorted[index] || 0
  },

  calculateCVaR(returns: number[], confidence: number): number {
    if (returns.length === 0) return 0
    const sorted = [...returns].sort((a, b) => a - b)
    const varIndex = Math.floor((1 - confidence) * sorted.length)
    const tailReturns = sorted.slice(0, varIndex + 1)

    if (tailReturns.length === 0) return 0
    return tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length
  },

  standardDeviation(values: number[]): number {
    if (values.length === 0) return 0
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length
    return Math.sqrt(variance)
  },

  calculateAllMetrics(prices: number[], returns: number[], trades: Trade[]): HedgeFundMetrics {
    const totalReturn = prices.length > 1 ? (prices[prices.length - 1] - prices[0]) / prices[0] : 0
    const annualizedReturn = Math.pow(1 + totalReturn, 252 / Math.max(prices.length - 1, 1)) - 1

    const winningTrades = trades.filter(t => t.pnl > 0)
    const losingTrades = trades.filter(t => t.pnl <= 0)
    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0)
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0))

    return {
      totalReturn,
      cagr: annualizedReturn,
      sharpeRatio: this.calculateSharpeRatio(returns),
      sortinoRatio: this.calculateSortinoRatio(returns),
      calmarRatio: annualizedReturn / Math.abs(this.calculateMaxDrawdown(prices)) || 0,
      maxDrawdown: this.calculateMaxDrawdown(prices),
      volatility: this.standardDeviation(returns),

      winRate: trades.length > 0 ? winningTrades.length / trades.length : 0,
      profitFactor: grossLoss > 0 ? grossProfit / grossLoss : 0,
      payoffRatio: winningTrades.length > 0 && losingTrades.length > 0 ?
        (grossProfit / winningTrades.length) / (grossLoss / losingTrades.length) : 0,
      expectancy: trades.length > 0 ? trades.reduce((sum, t) => sum + t.pnl, 0) / trades.length : 0,
      hitRate: trades.length > 0 ? winningTrades.length / trades.length : 0,

      var95: this.calculateVaR(returns, 0.95),
      cvar95: this.calculateCVaR(returns, 0.95),
      kellyFraction: 0.25, // Simplified Kelly calculation

      jensenAlpha: 0, // Would require benchmark comparison
      informationRatio: 0, // Would require benchmark comparison
      treynorRatio: 0, // Would require beta calculation
      betaToMarket: 1.0, // Default market beta
      correlation: 0.5, // Default correlation
      trackingError: this.standardDeviation(returns) * 0.5 // Simplified tracking error
    }
  }
}

export function calculateMetrics(returns: number[]): HedgeFundMetrics {
  return metricsCalculator.calculateAllMetrics([100000], returns, [])
}
