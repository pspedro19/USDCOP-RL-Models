/**
 * Real Data Trading Chart - Type Definitions
 * ==========================================
 *
 * Shared types for the professional trading chart components.
 */

import { CandlestickData, TechnicalIndicators } from '@/lib/services/market-data-service'

export type ChartType = 'candlestick' | 'line'
export type TimeframePeriod = '5m' | '15m' | '30m' | '1h' | '1d'

export interface ChartSettings {
  viewRange: { start: number; end: number }
  zoomLevel: number
  candleWidth: number
  showVolume: boolean
  showCrosshair: boolean
  autoScroll: boolean
}

export interface CrosshairInfo {
  x: number
  y: number
  priceY: number
  timeIndex: number
  price: number
  visible: boolean
}

export interface HoverInfo {
  x: number
  y: number
  candle: CandlestickData & { indicators?: TechnicalIndicators }
  index: number
  visible: boolean
}

export interface ChartMargins {
  top: number
  right: number
  bottom: number
  left: number
}

export interface ChartDimensions {
  width: number
  height: number
  chartWidth: number
  mainChartHeight: number
  volumeHeight: number
  margins: ChartMargins
}

export interface DrawChartParams {
  ctx: CanvasRenderingContext2D
  dimensions: ChartDimensions
  candlestickData: CandlestickData[]
  chartSettings: ChartSettings
  crosshair: CrosshairInfo
  chartType: ChartType
  selectedTimeframe: TimeframePeriod
  showIndicators: boolean
  isConnected: boolean
  lastUpdate: Date | null
  loading: boolean
  symbol: string
}

export interface RealDataTradingChartProps {
  symbol?: string
  timeframe?: string
  height?: number
  className?: string
}

export const DEFAULT_CHART_SETTINGS: ChartSettings = {
  viewRange: { start: 0, end: 100 },
  zoomLevel: 1,
  candleWidth: 8,
  showVolume: true,
  showCrosshair: true,
  autoScroll: true
}

export const DEFAULT_CHART_MARGINS: ChartMargins = {
  top: 40,
  right: 150,
  bottom: 80,
  left: 80
}

export const createEmptyHoverInfo = (): HoverInfo => ({
  x: 0,
  y: 0,
  candle: { time: 0, open: 0, high: 0, low: 0, close: 0, volume: 0 },
  index: 0,
  visible: false
})

export const createEmptyCrosshair = (): CrosshairInfo => ({
  x: 0,
  y: 0,
  priceY: 0,
  timeIndex: 0,
  price: 0,
  visible: false
})
