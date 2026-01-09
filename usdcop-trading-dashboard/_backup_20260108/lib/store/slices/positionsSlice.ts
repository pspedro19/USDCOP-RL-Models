/**
 * Positions Slice
 * ================
 *
 * Responsible for:
 * - Current trading positions
 * - Active position tracking
 * - PnL calculations (realized and unrealized)
 */

import { StateCreator } from 'zustand'
import { Position, OrderSide, OrderStatus } from '@/types/trading'

// ============================================
// TYPES
// ============================================

export interface PositionsSlice {
  // State
  positions: Position[]
  activePosition: Position | null
  totalPnl: number
  totalUnrealizedPnl: number

  // Actions
  setPositions: (positions: Position[]) => void
  updatePosition: (id: string, updates: Partial<Position>) => void
  setActivePosition: (position: Position | null) => void
  closePosition: (id: string) => void

  // Computed
  getPositionSide: () => OrderSide
  getTotalPnl: () => number
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  positions: [],
  activePosition: null,
  totalPnl: 0,
  totalUnrealizedPnl: 0,
}

// ============================================
// SLICE CREATOR
// ============================================

export const createPositionsSlice: StateCreator<PositionsSlice> = (set, get) => ({
  // State
  ...initialState,

  // Actions
  setPositions: (positions) => {
    const totalPnl = positions.reduce((sum, p) => sum + p.realizedPnl, 0)
    const totalUnrealizedPnl = positions.reduce((sum, p) => sum + p.unrealizedPnl, 0)
    const activePosition = positions.find((p) => p.status === OrderStatus.OPEN) || null

    set(
      {
        positions,
        activePosition,
        totalPnl,
        totalUnrealizedPnl,
      },
      false,
      'setPositions'
    )
  },

  updatePosition: (id, updates) =>
    set(
      (state) => ({
        positions: state.positions.map((p) =>
          p.id === id ? { ...p, ...updates } : p
        ),
      }),
      false,
      'updatePosition'
    ),

  setActivePosition: (position) =>
    set(
      { activePosition: position },
      false,
      'setActivePosition'
    ),

  closePosition: (id) =>
    set(
      (state) => ({
        positions: state.positions.map((p) =>
          p.id === id ? { ...p, status: OrderStatus.CLOSED } : p
        ),
        activePosition:
          state.activePosition?.id === id ? null : state.activePosition,
      }),
      false,
      'closePosition'
    ),

  // Computed
  getPositionSide: () => {
    const activePos = get().activePosition
    return activePos?.side || OrderSide.FLAT
  },

  getTotalPnl: () => {
    const { totalPnl, totalUnrealizedPnl } = get()
    return totalPnl + totalUnrealizedPnl
  },
})

// ============================================
// SELECTORS
// ============================================

export const selectPositions = (state: PositionsSlice) => state.positions
export const selectActivePosition = (state: PositionsSlice) => state.activePosition
export const selectTotalPnl = (state: PositionsSlice) => state.totalPnl
export const selectTotalUnrealizedPnl = (state: PositionsSlice) => state.totalUnrealizedPnl
export const selectPositionSide = (state: PositionsSlice) =>
  state.activePosition?.side || OrderSide.FLAT
