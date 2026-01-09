/**
 * Market Data Slice
 * ==================
 *
 * Responsible for:
 * - Real-time market data (price, bid, ask, spread, volume)
 * - Connection status
 * - Last update timestamp
 */

import { StateCreator } from 'zustand'

// ============================================
// TYPES
// ============================================

export interface MarketData {
  symbol: string
  price: number
  bid: number
  ask: number
  spread: number
  volume24h: number
  change24h: number
  changePercent24h: number
  high24h: number
  low24h: number
  timestamp: number
}

export interface MarketDataSlice {
  // State
  marketData: MarketData
  isConnected: boolean
  lastUpdate: number

  // Actions
  updateMarketData: (data: Partial<MarketData>) => void
  setConnectionStatus: (connected: boolean) => void

  // Computed
  getLatestPrice: () => number
}

// ============================================
// INITIAL STATE
// ============================================

export const initialMarketData: MarketData = {
  symbol: 'USDCOP',
  price: 0,
  bid: 0,
  ask: 0,
  spread: 0,
  volume24h: 0,
  change24h: 0,
  changePercent24h: 0,
  high24h: 0,
  low24h: 0,
  timestamp: Date.now(),
}

// ============================================
// SLICE CREATOR
// ============================================

export const createMarketDataSlice: StateCreator<MarketDataSlice> = (set, get) => ({
  // State
  marketData: initialMarketData,
  isConnected: false,
  lastUpdate: Date.now(),

  // Actions
  updateMarketData: (data) =>
    set(
      (state) => ({
        marketData: { ...state.marketData, ...data },
        lastUpdate: Date.now(),
      }),
      false,
      'updateMarketData'
    ),

  setConnectionStatus: (connected) =>
    set({ isConnected: connected }, false, 'setConnectionStatus'),

  // Computed
  getLatestPrice: () => get().marketData.price,
})

// ============================================
// SELECTORS
// ============================================

export const selectMarketData = (state: MarketDataSlice) => state.marketData
export const selectIsConnected = (state: MarketDataSlice) => state.isConnected
export const selectLatestPrice = (state: MarketDataSlice) => state.marketData.price
export const selectSpread = (state: MarketDataSlice) => state.marketData.spread
