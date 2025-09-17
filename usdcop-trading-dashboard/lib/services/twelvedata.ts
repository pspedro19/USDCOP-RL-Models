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
