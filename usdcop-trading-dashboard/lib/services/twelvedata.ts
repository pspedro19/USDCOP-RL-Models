export interface TwelveDataResponse {
  symbol: string
  price: number
  timestamp: string
}

export async function fetchTwelveData(symbol: string): Promise<TwelveDataResponse> {
  return {
    symbol,
    price: Math.random() * 100,
    timestamp: new Date().toISOString()
  }
}

export async function fetchRealTimeQuote(symbol: string): Promise<TwelveDataResponse> {
  return {
    symbol,
    price: Math.random() * 4200,
    timestamp: new Date().toISOString()
  }
}

export async function fetchTimeSeries(symbol: string): Promise<TwelveDataResponse[]> {
  return Array.from({ length: 100 }, (_, i) => ({
    symbol,
    price: Math.random() * 4200,
    timestamp: new Date(Date.now() - i * 60000).toISOString()
  }))
}

export async function fetchTechnicalIndicators(symbol: string) {
  return {
    rsi: { rsi: 30 + Math.random() * 40 },
    macd: { macd: (Math.random() - 0.5) * 10 },
    sma: { sma: Math.random() * 4200 },
    ema: { ema: Math.random() * 4200 },
    bbands: {
      upper_band: Math.random() * 4300,
      middle_band: Math.random() * 4200,
      lower_band: Math.random() * 4100
    },
    stoch: {
      slow_k: Math.random() * 100,
      slow_d: Math.random() * 100
    }
  }
}

export interface PriceData {
  symbol: string
  price: number
  timestamp: string
  volume?: number
}

export const wsClient = {
  connect: (symbol: string, callback: (data: PriceData) => void) => {
    const interval = setInterval(() => {
      callback({
        symbol,
        price: Math.random() * 4200,
        timestamp: new Date().toISOString(),
        volume: Math.random() * 1000000
      })
    }, 1000)

    return () => clearInterval(interval)
  }
}
