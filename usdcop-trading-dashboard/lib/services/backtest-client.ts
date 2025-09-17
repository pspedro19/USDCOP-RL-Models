export interface BacktestResult {
  id: string
  returns: number[]
  trades: number
  winRate: number
  totalReturn: number
}

export async function runBacktest(strategy: string): Promise<BacktestResult> {
  return {
    id: '1',
    returns: [0.01, -0.005, 0.02, 0.015],
    trades: 100,
    winRate: 0.65,
    totalReturn: 0.15
  }
}
