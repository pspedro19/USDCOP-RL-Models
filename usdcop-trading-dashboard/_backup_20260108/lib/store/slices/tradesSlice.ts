/**
 * Trades Slice
 * =============
 *
 * Responsible for:
 * - Trade history
 * - Today's trades filtering
 * - Pending orders tracking
 */

import { StateCreator } from 'zustand'
import { Trade } from '@/types/trading'

// ============================================
// TYPES
// ============================================

export interface TradesSlice {
  // State
  trades: Trade[]
  todayTrades: Trade[]
  pendingOrders: number

  // Actions
  addTrade: (trade: Trade) => void
  setTrades: (trades: Trade[]) => void
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  trades: [],
  todayTrades: [],
  pendingOrders: 0,
}

// ============================================
// SLICE CREATOR
// ============================================

export const createTradesSlice: StateCreator<TradesSlice> = (set) => ({
  // State
  ...initialState,

  // Actions
  addTrade: (trade) =>
    set(
      (state) => {
        const today = new Date().toDateString()
        const isToday = new Date(trade.timestamp).toDateString() === today

        return {
          trades: [...state.trades, trade].slice(-500), // Keep last 500 trades
          todayTrades: isToday
            ? [...state.todayTrades, trade]
            : state.todayTrades,
        }
      },
      false,
      'addTrade'
    ),

  setTrades: (trades) => {
    const today = new Date().toDateString()
    const todayTrades = trades.filter(
      (t) => new Date(t.timestamp).toDateString() === today
    )

    set(
      {
        trades,
        todayTrades,
        pendingOrders: 0,
      },
      false,
      'setTrades'
    )
  },
})

// ============================================
// SELECTORS
// ============================================

export const selectTrades = (state: TradesSlice) => state.trades
export const selectTodayTrades = (state: TradesSlice) => state.todayTrades
export const selectPendingOrders = (state: TradesSlice) => state.pendingOrders
