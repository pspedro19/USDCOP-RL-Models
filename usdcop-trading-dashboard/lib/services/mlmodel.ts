export interface MLPrediction {
  symbol: string
  signal: 'buy' | 'sell' | 'hold'
  confidence: number
  timestamp: number
}

export async function getMLPredictions(): Promise<MLPrediction[]> {
  return [
    { symbol: 'USDCOP', signal: 'buy', confidence: 0.85, timestamp: Date.now() }
  ]
}
