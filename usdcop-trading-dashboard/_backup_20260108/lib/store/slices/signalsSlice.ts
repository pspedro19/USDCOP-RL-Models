/**
 * Signals Slice
 * ==============
 *
 * Responsible for:
 * - Trading signals from PPO-LSTM model
 * - Signal history and latest signal
 * - Loading and error states
 */

import { StateCreator } from 'zustand'
import { TradingSignal } from '@/types/trading'

// ============================================
// TYPES
// ============================================

export interface SignalsSlice {
  // State
  signals: TradingSignal[]
  latestSignal: TradingSignal | null
  isLoading: boolean
  error: string | null

  // Actions
  addSignal: (signal: TradingSignal) => void
  setSignals: (signals: TradingSignal[]) => void
  clearSignals: () => void
  setSignalLoading: (loading: boolean) => void
  setSignalError: (error: string | null) => void
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  signals: [],
  latestSignal: null,
  isLoading: false,
  error: null,
}

// ============================================
// SLICE CREATOR
// ============================================

export const createSignalsSlice: StateCreator<SignalsSlice> = (set) => ({
  // State
  ...initialState,

  // Actions
  addSignal: (signal) =>
    set(
      (state) => ({
        signals: [...state.signals, signal].slice(-100), // Keep last 100 signals
        latestSignal: signal,
      }),
      false,
      'addSignal'
    ),

  setSignals: (signals) =>
    set(
      {
        signals,
        latestSignal: signals[signals.length - 1] || null,
      },
      false,
      'setSignals'
    ),

  clearSignals: () =>
    set(
      {
        signals: [],
        latestSignal: null,
      },
      false,
      'clearSignals'
    ),

  setSignalLoading: (loading) =>
    set(
      { isLoading: loading },
      false,
      'setSignalLoading'
    ),

  setSignalError: (error) =>
    set(
      { error },
      false,
      'setSignalError'
    ),
})

// ============================================
// SELECTORS
// ============================================

export const selectSignals = (state: SignalsSlice) => state.signals
export const selectLatestSignal = (state: SignalsSlice) => state.latestSignal
export const selectSignalLoading = (state: SignalsSlice) => state.isLoading
export const selectSignalError = (state: SignalsSlice) => state.error
