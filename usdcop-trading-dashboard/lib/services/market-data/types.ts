/**
 * Type Definitions for Market Data Services
 * =========================================
 *
 * Shared type definitions for all market data modules
 */

export interface MarketDataPoint {
  symbol: string
  price: number
  timestamp: number
  volume: number
  bid?: number
  ask?: number
  source?: string
}

export interface CandlestickData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface TechnicalIndicators {
  ema_20?: number
  ema_50?: number
  ema_200?: number
  bb_upper?: number
  bb_middle?: number
  bb_lower?: number
  rsi?: number
}

export interface CandlestickResponse {
  symbol: string
  timeframe: string
  start_date: string
  end_date: string
  count: number
  data: (CandlestickData & { indicators?: TechnicalIndicators })[]
}

export interface SymbolStats {
  symbol: string
  price: number
  open_24h: number
  high_24h: number
  low_24h: number
  volume_24h: number
  change_24h: number
  change_percent_24h: number
  spread: number
  timestamp: string
  source: string
}

export interface APIHealthResponse {
  status: string
  database?: string
  total_records?: number
  market_status?: {
    is_open: boolean
  }
  message?: string
}
