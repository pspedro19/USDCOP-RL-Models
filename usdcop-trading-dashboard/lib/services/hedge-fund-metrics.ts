export interface HedgeFundMetrics {
  sharpeRatio: number
  calmarRatio: number
  maxDrawdown: number
  volatility: number
  alpha: number
  beta: number
}

export function calculateMetrics(returns: number[]): HedgeFundMetrics {
  return {
    sharpeRatio: 1.2,
    calmarRatio: 0.8,
    maxDrawdown: -0.15,
    volatility: 0.12,
    alpha: 0.05,
    beta: 1.1
  }
}
