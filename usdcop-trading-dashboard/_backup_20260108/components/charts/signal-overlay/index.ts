/**
 * Signal Overlay Exports
 * =======================
 *
 * Central export point for all signal overlay components and utilities
 */

// Components
export { default as SignalMarker } from './SignalMarker'
export { default as PositionShading } from './PositionShading'
export { default as StopLossTakeProfit } from './StopLossTakeProfit'

// Types
export type {
  SignalData,
  PositionArea,
  SignalPriceLine,
  SignalMarkerConfig,
  SignalFilterOptions,
  SignalStats,
  SignalTooltipData,
  SignalUpdate,
  SignalOverlayProps,
  SignalPerformance,
} from './types'

// Marker utilities
export {
  getMarkerConfig,
  getExitMarkerConfig,
  useSignalMarkers,
  getConfidenceColor,
  getConfidenceSize,
  filterSignalsByDateRange,
  filterSignalsByConfidence,
  filterSignalsByType,
  filterSignalsByStatus,
} from './SignalMarker'

// Position utilities
export {
  usePositionAreas,
  getShadingColor,
  getBorderColor,
  calculateOpacity,
  createPositionHistogram,
  calculatePositionStats,
  getPositionDuration,
  calculateReturnPercent,
  formatPnL,
  formatReturnPercent,
  getPositionTooltip,
} from './PositionShading'

export type { HistogramData, PositionStats } from './PositionShading'

// Price line utilities
export {
  useSignalPriceLines,
  createPriceLineOptions,
  calculateRiskReward,
  getStopLossDistance,
  getTakeProfitDistance,
  hasHitStopLoss,
  hasHitTakeProfit,
  getPriceLineColor,
  formatPriceLineTitle,
  calculatePriceLineStats,
  usePriceLineManagement,
} from './StopLossTakeProfit'

export type { PriceLineStats } from './StopLossTakeProfit'
