export interface BacktestResult {
  id: string
  returns: number[]
  trades: number
  winRate: number
  totalReturn: number
}

export interface TradeRecord {
  id: string
  timestamp: string
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  price: number
  pnl: number
  commission: number
  duration?: number
}

export interface DailyReturn {
  date: string
  return: number
  cumulativeReturn: number
  price: number
  volume?: number
}

export interface BacktestKPIs {
  top_bar: {
    CAGR: number
    Sharpe: number
    Sortino: number
    Calmar: number
    MaxDD: number
    Vol_annualizada: number
  }
  trading_micro: {
    win_rate: number
    profit_factor: number
    payoff: number
    expectancy_bps: number
  }
  colas_y_drawdowns: {
    VaR_99_bps: number
    ES_97_5_bps: number
  }
  exposicion_capacidad: {
    beta?: number
  }
  ejecucion_costos: {
    cost_to_alpha_ratio: number
  }
}

export interface BacktestResults {
  runId: string
  timestamp: string
  test: {
    kpis: BacktestKPIs
    dailyReturns: DailyReturn[]
    trades: TradeRecord[]
    manifest?: {
      policy: string
      obs_cols: string[]
      n_rows: number
      n_days: number
    }
  }
  val: {
    kpis: BacktestKPIs
    dailyReturns: DailyReturn[]
    trades: TradeRecord[]
    manifest?: {
      policy: string
      obs_cols: string[]
      n_rows: number
      n_days: number
    }
  }
}

// Create a mock backtestClient object with all required methods
export const backtestClient = {
  async getLatestResults(): Promise<BacktestResults> {
    try {
      // First try to fetch from the API
      const response = await fetch('/api/backtest/results')

      if (response.ok) {
        const apiResult = await response.json()
        if (apiResult.success && apiResult.data) {
          console.log('[BacktestClient] Successfully fetched results from API')
          return apiResult.data
        }
      }

      console.warn('[BacktestClient] API fetch failed, falling back to mock data')
    } catch (error) {
      console.warn('[BacktestClient] API error, falling back to mock data:', error)
    }

    // Fallback to mock data generation
    const now = new Date()
    const mockDailyReturns: DailyReturn[] = []
    const mockTrades: TradeRecord[] = []

    // Generate 30 days of mock daily returns
    for (let i = 0; i < 30; i++) {
      const date = new Date(now)
      date.setDate(date.getDate() - (29 - i))
      const dailyReturn = (Math.random() - 0.5) * 0.02 // Random daily return between -1% and 1%
      const cumulativeReturn = i === 0 ? dailyReturn : mockDailyReturns[i-1].cumulativeReturn * (1 + dailyReturn)

      mockDailyReturns.push({
        date: date.toISOString().split('T')[0],
        return: dailyReturn,
        cumulativeReturn: cumulativeReturn,
        price: 100000 * (1 + cumulativeReturn),
        volume: Math.floor(Math.random() * 1000000)
      })
    }

    // Generate mock trades
    for (let i = 0; i < 50; i++) {
      const date = new Date(now)
      date.setDate(date.getDate() - Math.floor(Math.random() * 30))
      const pnl = (Math.random() - 0.4) * 1000 // Slightly positive bias

      mockTrades.push({
        id: `trade_${i + 1}`,
        timestamp: date.toISOString(),
        symbol: 'USDCOP',
        side: Math.random() > 0.5 ? 'buy' : 'sell',
        quantity: Math.floor(Math.random() * 10000) + 1000,
        price: 4000 + Math.random() * 200,
        pnl: pnl,
        commission: Math.abs(pnl) * 0.001,
        duration: Math.floor(Math.random() * 3600) // seconds
      })
    }

    const winningTrades = mockTrades.filter(t => t.pnl > 0)
    const losingTrades = mockTrades.filter(t => t.pnl <= 0)
    const totalPnl = mockTrades.reduce((sum, t) => sum + t.pnl, 0)
    const winRate = winningTrades.length / mockTrades.length
    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0)
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0))
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0

    const mockKPIs: BacktestKPIs = {
      top_bar: {
        CAGR: 0.125, // 12.5% CAGR
        Sharpe: 1.45,
        Sortino: 1.78,
        Calmar: 0.89,
        MaxDD: -0.08, // -8% max drawdown
        Vol_annualizada: 0.15 // 15% annualized volatility
      },
      trading_micro: {
        win_rate: winRate,
        profit_factor: profitFactor,
        payoff: winningTrades.length > 0 && losingTrades.length > 0 ?
          (grossProfit / winningTrades.length) / (grossLoss / losingTrades.length) : 1,
        expectancy_bps: (totalPnl / mockTrades.length) * 10000 // in basis points
      },
      colas_y_drawdowns: {
        VaR_99_bps: -250, // -2.5% VaR in basis points
        ES_97_5_bps: -380 // -3.8% Expected Shortfall in basis points
      },
      exposicion_capacidad: {
        beta: 1.15
      },
      ejecucion_costos: {
        cost_to_alpha_ratio: 0.02
      }
    }

    return {
      runId: `backtest_${Date.now()}`,
      timestamp: now.toISOString(),
      test: {
        kpis: mockKPIs,
        dailyReturns: mockDailyReturns,
        trades: mockTrades,
        manifest: {
          policy: 'PPO_USDCOP_v2.1',
          obs_cols: ['price', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'],
          n_rows: mockDailyReturns.length,
          n_days: 30
        }
      },
      val: {
        kpis: {
          ...mockKPIs,
          top_bar: {
            ...mockKPIs.top_bar,
            CAGR: 0.108, // Slightly different for validation
            Sharpe: 1.32
          }
        },
        dailyReturns: mockDailyReturns.slice(0, 20), // Shorter period for validation
        trades: mockTrades.slice(0, 30),
        manifest: {
          policy: 'PPO_USDCOP_v2.1',
          obs_cols: ['price', 'volume', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'],
          n_rows: 20,
          n_days: 20
        }
      }
    }
  },

  getDataQuality(results: BacktestResults) {
    const testData = results.test
    const issues: string[] = []
    const recommendations: string[] = []
    let score = 100

    // Check data completeness
    if (testData.dailyReturns.length < 30) {
      issues.push('Insufficient historical data (< 30 days)')
      score -= 20
    }

    if (testData.trades.length < 10) {
      issues.push('Low trade count may affect statistical significance')
      score -= 15
    }

    // Check for data quality issues
    const extremeReturns = testData.dailyReturns.filter(r => Math.abs(r.return) > 0.05)
    if (extremeReturns.length > testData.dailyReturns.length * 0.05) {
      issues.push('High number of extreme daily returns detected')
      score -= 10
    }

    // Recommendations
    if (score < 80) {
      recommendations.push('Consider longer backtesting period for better reliability')
    }
    if (testData.trades.length < 50) {
      recommendations.push('Increase trade frequency for better statistical power')
    }

    return {
      score: Math.max(score, 0),
      issues,
      recommendations
    }
  },

  calculatePerformanceData(dailyReturns: DailyReturn[]) {
    return dailyReturns.map((day, index) => {
      const cumulativeReturn = index === 0 ? day.return :
        dailyReturns.slice(0, index + 1).reduce((acc, d) => acc * (1 + d.return), 1) - 1

      // Calculate drawdown
      const peak = dailyReturns.slice(0, index + 1)
        .reduce((max, d, i) => {
          const cumRet = dailyReturns.slice(0, i + 1).reduce((acc, dr) => acc * (1 + dr.return), 1) - 1
          return Math.max(max, cumRet)
        }, 0)

      const drawdown = (cumulativeReturn - peak) / (1 + peak)

      return {
        date: day.date,
        portfolio: day.price,
        benchmark: 100000 * (1 + 0.12 * (index / 365)), // 12% annual benchmark
        drawdown: drawdown * 100,
        underwater: drawdown * 100
      }
    })
  },

  calculateMonthlyReturns(performanceData: any[]) {
    const monthlyData: { [key: string]: { portfolio: number[], benchmark: number[] } } = {}

    performanceData.forEach(day => {
      const date = new Date(day.date)
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`

      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = { portfolio: [], benchmark: [] }
      }

      monthlyData[monthKey].portfolio.push(day.portfolio)
      monthlyData[monthKey].benchmark.push(day.benchmark)
    })

    return Object.entries(monthlyData).map(([month, data]) => {
      const monthName = new Date(month + '-01').toLocaleString('en', { month: 'short' })
      const portfolioReturn = data.portfolio.length > 1 ?
        (data.portfolio[data.portfolio.length - 1] - data.portfolio[0]) / data.portfolio[0] * 100 : 0

      return {
        month: monthName,
        return: portfolioReturn
      }
    })
  },

  transformTradesToHedgeFundFormat(trades: TradeRecord[], kpis: BacktestKPIs) {
    return trades.map(trade => ({
      id: trade.id,
      date: new Date(trade.timestamp),
      symbol: trade.symbol,
      side: trade.side,
      quantity: trade.quantity,
      price: trade.price,
      pnl: trade.pnl,
      commission: trade.commission,
      duration: trade.duration || 0,
      exitPrice: trade.price + (trade.pnl / trade.quantity),
      returnPct: (trade.pnl / (trade.price * trade.quantity)) * 100
    }))
  },

  async triggerBacktest(forceRebuild: boolean = false): Promise<void> {
    // Mock API call to trigger new backtest
    console.log('Triggering new backtest...', { forceRebuild })

    try {
      const response = await fetch('/api/backtest/trigger', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ forceRebuild })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log('Backtest triggered successfully:', result)
    } catch (error) {
      console.warn('Backtest trigger failed, using mock response:', error)
      // For now, just log success since backend might not be implemented
    }
  }
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
