/**
 * Signal Overlay Types
 * ====================
 *
 * Type definitions for the trading signal overlay system
 */

import { Time } from 'lightweight-charts'
import { OrderType } from '@/types/trading'

/**
 * Trading signal with overlay display properties
 */
export interface SignalData {
  id: string
  timestamp: string
  time: Time
  type: OrderType
  confidence: number
  price: number
  stopLoss?: number
  takeProfit?: number
  reasoning: string[]
  riskScore: number
  expectedReturn: number
  timeHorizon: string
  modelSource: string
  latency: number
  exitPrice?: number
  exitTimestamp?: string
  exitTime?: Time
  pnl?: number
  status: 'active' | 'closed' | 'cancelled'
}

/**
 * Position shading area
 */
export interface PositionArea {
  id: string
  entryTime: Time
  exitTime: Time
  entryPrice: number
  exitPrice: number
  type: OrderType
  pnl: number
  confidence: number
}

/**
 * Price line for SL/TP
 */
export interface SignalPriceLine {
  id: string
  signalId: string
  price: number
  type: 'stopLoss' | 'takeProfit'
  color: string
  isActive: boolean
}

/**
 * Signal marker configuration
 */
export interface SignalMarkerConfig {
  time: Time
  position: 'aboveBar' | 'belowBar' | 'inBar'
  color: string
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square'
  text: string
  size?: number
  id: string
}

/**
 * Signal filter options
 */
export interface SignalFilterOptions {
  startDate?: Date
  endDate?: Date
  actionTypes?: OrderType[]
  minConfidence?: number
  maxConfidence?: number
  showHold?: boolean
  showActive?: boolean
  showClosed?: boolean
  modelSources?: string[]
}

/**
 * Signal statistics
 */
export interface SignalStats {
  total: number
  active: number
  closed: number
  winRate: number
  avgConfidence: number
  totalPnl: number
  avgPnl: number
  bestSignal?: SignalData
  worstSignal?: SignalData
}

/**
 * Tooltip data for signal hover
 */
export interface SignalTooltipData {
  signal: SignalData
  mouseX: number
  mouseY: number
  visible: boolean
}

/**
 * WebSocket signal update
 */
export interface SignalUpdate {
  type: 'new_signal' | 'update_signal' | 'close_signal'
  signal: SignalData
  timestamp: string
}

/**
 * Signal overlay props
 */
export interface SignalOverlayProps {
  signals: SignalData[]
  filter?: SignalFilterOptions
  onSignalClick?: (signal: SignalData) => void
  onSignalHover?: (signal: SignalData | null) => void
  showStopLoss?: boolean
  showTakeProfit?: boolean
  showPositionAreas?: boolean
  showTooltips?: boolean
  autoUpdate?: boolean
  websocketUrl?: string
}

/**
 * Signal performance metrics
 */
export interface SignalPerformance {
  signalId: string
  winRate: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  sharpeRatio: number
  totalTrades: number
  activeTrades: number
  returnPercent: number
  maxDrawdown: number
}
