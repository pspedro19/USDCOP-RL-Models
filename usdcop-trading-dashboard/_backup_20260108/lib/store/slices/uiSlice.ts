/**
 * UI Slice
 * =========
 *
 * Responsible for:
 * - Chart display preferences (timeframe, height)
 * - Technical indicators toggles
 * - Theme settings
 * - Volume and indicator visibility
 */

import { StateCreator } from 'zustand'

// ============================================
// TYPES
// ============================================

export type Timeframe = '5m' | '15m' | '30m' | '1h' | '4h' | '1d'
export type Theme = 'dark' | 'light'

export interface IndicatorsEnabled {
  ema: boolean
  bb: boolean
  rsi: boolean
  macd: boolean
}

export interface UISlice {
  // State
  selectedTimeframe: Timeframe
  showVolume: boolean
  showIndicators: boolean
  indicatorsEnabled: IndicatorsEnabled
  theme: Theme
  chartHeight: number

  // Actions
  setTimeframe: (timeframe: Timeframe) => void
  toggleIndicator: (indicator: keyof IndicatorsEnabled) => void
  setShowVolume: (show: boolean) => void
  setShowIndicators: (show: boolean) => void
  setTheme: (theme: Theme) => void
  setChartHeight: (height: number) => void
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  selectedTimeframe: '5m' as Timeframe,
  showVolume: true,
  showIndicators: true,
  indicatorsEnabled: {
    ema: true,
    bb: true,
    rsi: false,
    macd: false,
  },
  theme: 'dark' as Theme,
  chartHeight: 500,
}

// ============================================
// SLICE CREATOR
// ============================================

export const createUISlice: StateCreator<UISlice> = (set) => ({
  // State
  ...initialState,

  // Actions
  setTimeframe: (timeframe) =>
    set(
      { selectedTimeframe: timeframe },
      false,
      'setTimeframe'
    ),

  toggleIndicator: (indicator) =>
    set(
      (state) => ({
        indicatorsEnabled: {
          ...state.indicatorsEnabled,
          [indicator]: !state.indicatorsEnabled[indicator],
        },
      }),
      false,
      'toggleIndicator'
    ),

  setShowVolume: (show) =>
    set(
      { showVolume: show },
      false,
      'setShowVolume'
    ),

  setShowIndicators: (show) =>
    set(
      { showIndicators: show },
      false,
      'setShowIndicators'
    ),

  setTheme: (theme) =>
    set(
      { theme },
      false,
      'setTheme'
    ),

  setChartHeight: (height) =>
    set(
      { chartHeight: height },
      false,
      'setChartHeight'
    ),
})

// ============================================
// SELECTORS
// ============================================

export const selectTimeframe = (state: UISlice) => state.selectedTimeframe
export const selectTheme = (state: UISlice) => state.theme
export const selectShowVolume = (state: UISlice) => state.showVolume
export const selectShowIndicators = (state: UISlice) => state.showIndicators
export const selectIndicatorsEnabled = (state: UISlice) => state.indicatorsEnabled
export const selectChartHeight = (state: UISlice) => state.chartHeight
