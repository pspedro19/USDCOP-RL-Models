/**
 * Trading Store (Zustand)
 * =======================
 *
 * Refactored to follow Interface Segregation Principle.
 * State is now split into focused slices with single responsibilities.
 *
 * Architecture:
 * - Market Data: Real-time price data and connection status
 * - Signals: Trading signals from PPO-LSTM model
 * - Positions: Current positions and PnL tracking
 * - Trades: Trade history and pending orders
 * - UI: User preferences and display settings
 *
 * Usage:
 * - Use specific slice hooks for optimal performance: useMarketDataSlice(), useSignalsSlice(), etc.
 * - Use useTradingStore() for backwards compatibility or when needing multiple slices
 */

import { create } from 'zustand'
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware'

// Import slice creators and types
import {
  createMarketDataSlice,
  MarketDataSlice,
  selectMarketData as selectMarketDataFromSlice,
  selectIsConnected as selectIsConnectedFromSlice,
} from './slices/marketDataSlice'

import {
  createSignalsSlice,
  SignalsSlice,
  selectLatestSignal as selectLatestSignalFromSlice,
} from './slices/signalsSlice'

import {
  createPositionsSlice,
  PositionsSlice,
  selectActivePosition as selectActivePositionFromSlice,
} from './slices/positionsSlice'

import {
  createTradesSlice,
  TradesSlice,
  selectTodayTrades as selectTodayTradesFromSlice,
} from './slices/tradesSlice'

import {
  createUISlice,
  UISlice,
} from './slices/uiSlice'

// ============================================
// COMBINED STORE TYPE
// ============================================

type TradingStore = MarketDataSlice & SignalsSlice & PositionsSlice & TradesSlice & UISlice

// ============================================
// STORE CREATION
// ============================================

export const useTradingStore = create<TradingStore>()(
  devtools(
    subscribeWithSelector(
      persist(
        (...a) => ({
          ...createMarketDataSlice(...a),
          ...createSignalsSlice(...a),
          ...createPositionsSlice(...a),
          ...createTradesSlice(...a),
          ...createUISlice(...a),
        }),
        {
          name: 'trading-store',
          partialize: (state) => ({
            // Only persist UI preferences
            selectedTimeframe: state.selectedTimeframe,
            showVolume: state.showVolume,
            showIndicators: state.showIndicators,
            indicatorsEnabled: state.indicatorsEnabled,
            theme: state.theme,
            chartHeight: state.chartHeight,
          }),
        }
      )
    ),
    { name: 'TradingStore' }
  )
)

// ============================================
// SLICE-SPECIFIC HOOKS (Recommended)
// ============================================

/**
 * Market Data Slice Hook
 * Use this when you only need market data to avoid unnecessary re-renders
 */
export const useMarketDataSlice = () =>
  useTradingStore((state) => ({
    marketData: state.marketData,
    isConnected: state.isConnected,
    lastUpdate: state.lastUpdate,
    updateMarketData: state.updateMarketData,
    setConnectionStatus: state.setConnectionStatus,
    getLatestPrice: state.getLatestPrice,
  }))

/**
 * Signals Slice Hook
 * Use this when you only need trading signals
 */
export const useSignalsSlice = () =>
  useTradingStore((state) => ({
    signals: state.signals,
    latestSignal: state.latestSignal,
    isLoading: state.isLoading,
    error: state.error,
    addSignal: state.addSignal,
    setSignals: state.setSignals,
    clearSignals: state.clearSignals,
    setSignalLoading: state.setSignalLoading,
    setSignalError: state.setSignalError,
  }))

/**
 * Positions Slice Hook
 * Use this when you only need position data
 */
export const usePositionsSlice = () =>
  useTradingStore((state) => ({
    positions: state.positions,
    activePosition: state.activePosition,
    totalPnl: state.totalPnl,
    totalUnrealizedPnl: state.totalUnrealizedPnl,
    setPositions: state.setPositions,
    updatePosition: state.updatePosition,
    setActivePosition: state.setActivePosition,
    closePosition: state.closePosition,
    getPositionSide: state.getPositionSide,
    getTotalPnl: state.getTotalPnl,
  }))

/**
 * Trades Slice Hook
 * Use this when you only need trade history
 */
export const useTradesSlice = () =>
  useTradingStore((state) => ({
    trades: state.trades,
    todayTrades: state.todayTrades,
    pendingOrders: state.pendingOrders,
    addTrade: state.addTrade,
    setTrades: state.setTrades,
  }))

/**
 * UI Slice Hook
 * Use this when you only need UI preferences
 */
export const useUISlice = () =>
  useTradingStore((state) => ({
    selectedTimeframe: state.selectedTimeframe,
    showVolume: state.showVolume,
    showIndicators: state.showIndicators,
    indicatorsEnabled: state.indicatorsEnabled,
    theme: state.theme,
    chartHeight: state.chartHeight,
    setTimeframe: state.setTimeframe,
    toggleIndicator: state.toggleIndicator,
    setShowVolume: state.setShowVolume,
    setShowIndicators: state.setShowIndicators,
    setTheme: state.setTheme,
    setChartHeight: state.setChartHeight,
  }))

// ============================================
// GRANULAR SELECTORS (Maximum Performance)
// ============================================

// Market Data Selectors
export const selectMarketData = (state: TradingStore) => state.marketData
export const selectIsConnected = (state: TradingStore) => state.isConnected
export const selectLatestPrice = (state: TradingStore) => state.marketData.price

// Signal Selectors
export const selectLatestSignal = (state: TradingStore) => state.latestSignal
export const selectSignals = (state: TradingStore) => state.signals
export const selectSignalLoading = (state: TradingStore) => state.isLoading

// Position Selectors
export const selectActivePosition = (state: TradingStore) => state.activePosition
export const selectPositions = (state: TradingStore) => state.positions
export const selectTotalPnl = (state: TradingStore) => state.totalPnl

// Trade Selectors
export const selectTodayTrades = (state: TradingStore) => state.todayTrades
export const selectAllTrades = (state: TradingStore) => state.trades

// UI Selectors
export const selectTimeframe = (state: TradingStore) => state.selectedTimeframe
export const selectTheme = (state: TradingStore) => state.theme
export const selectIndicators = (state: TradingStore) => state.indicatorsEnabled

// ============================================
// BACKWARDS COMPATIBLE HOOKS
// ============================================

/**
 * @deprecated Use useMarketDataSlice() instead for better performance
 */
export const useMarketData = () => useTradingStore(selectMarketData)

/**
 * @deprecated Use useSignalsSlice() instead for better performance
 */
export const useLatestSignal = () => useTradingStore(selectLatestSignal)

/**
 * @deprecated Use usePositionsSlice() instead for better performance
 */
export const useActivePosition = () => useTradingStore(selectActivePosition)

/**
 * @deprecated Use useTradesSlice() instead for better performance
 */
export const useTodayTrades = () => useTradingStore(selectTodayTrades)

/**
 * @deprecated Use useUISlice() instead for better performance
 */
export const useUIState = () => useTradingStore((state) => ({
  selectedTimeframe: state.selectedTimeframe,
  showVolume: state.showVolume,
  showIndicators: state.showIndicators,
  indicatorsEnabled: state.indicatorsEnabled,
  theme: state.theme,
  chartHeight: state.chartHeight,
}))

/**
 * @deprecated Use useMarketDataSlice() instead for better performance
 */
export const useConnectionStatus = () => useTradingStore(selectIsConnected)

// ============================================
// STORE TYPE EXPORTS
// ============================================

export type { TradingStore }
export type {
  MarketDataSlice,
  SignalsSlice,
  PositionsSlice,
  TradesSlice,
  UISlice,
}

// Re-export types from slices for convenience
export type { MarketData } from './slices/marketDataSlice'
export type { Timeframe, Theme, IndicatorsEnabled } from './slices/uiSlice'
