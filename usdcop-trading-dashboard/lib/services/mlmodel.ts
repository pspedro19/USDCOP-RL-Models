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

export async function getPrediction(features: any): Promise<MLPrediction> {
  // Mock ML prediction based on features
  const confidence = Math.random() * 0.4 + 0.6; // 0.6 to 1.0
  const signals: Array<'buy' | 'sell' | 'hold'> = ['buy', 'sell', 'hold'];
  const signal = signals[Math.floor(Math.random() * signals.length)];

  return {
    symbol: 'USDCOP',
    signal,
    confidence,
    timestamp: Date.now()
  };
}
