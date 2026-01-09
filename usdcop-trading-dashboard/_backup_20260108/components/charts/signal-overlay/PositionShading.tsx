'use client'

/**
 * PositionShading Component
 * ==========================
 *
 * Renders shaded areas between entry and exit points showing P&L
 * Green for profitable trades, red for losses
 */

import { useMemo } from 'react'
import { Time, ISeriesApi } from 'lightweight-charts'
import { SignalData, PositionArea } from './types'
import { OrderType } from '@/types/trading'

interface PositionShadingProps {
  signals: SignalData[]
  candleSeries: ISeriesApi<'Candlestick'> | null
}

/**
 * Convert closed signals to position areas
 */
export function usePositionAreas(signals: SignalData[]): PositionArea[] {
  return useMemo(() => {
    return signals
      .filter((signal) => signal.status === 'closed' && signal.exitTime && signal.exitPrice)
      .map((signal) => ({
        id: signal.id,
        entryTime: signal.time,
        exitTime: signal.exitTime!,
        entryPrice: signal.price,
        exitPrice: signal.exitPrice!,
        type: signal.type,
        pnl: signal.pnl || 0,
        confidence: signal.confidence,
      }))
  }, [signals])
}

/**
 * Get shading color based on P&L
 */
export function getShadingColor(pnl: number, opacity: number = 0.15): string {
  const isProfitable = pnl > 0

  if (isProfitable) {
    // Green gradient for profit
    return `rgba(34, 197, 94, ${opacity})`
  } else {
    // Red gradient for loss
    return `rgba(239, 68, 68, ${opacity})`
  }
}

/**
 * Get border color for position area
 */
export function getBorderColor(pnl: number): string {
  return pnl > 0 ? '#22c55e' : '#ef4444'
}

/**
 * Calculate area opacity based on confidence and P&L
 */
export function calculateOpacity(
  confidence: number,
  pnl: number,
  baseOpacity: number = 0.15
): number {
  // Higher confidence = more visible
  const confidenceFactor = confidence / 100

  // Larger P&L = more visible
  const pnlFactor = Math.min(Math.abs(pnl) / 100, 1)

  return baseOpacity + (confidenceFactor * pnlFactor * 0.1)
}

/**
 * Create histogram data for position area shading
 * This is a workaround since lightweight-charts doesn't have native area shading
 * We use the histogram series with custom coloring
 */
export interface HistogramData {
  time: Time
  value: number
  color: string
}

export function createPositionHistogram(
  area: PositionArea,
  candleData: { time: Time; close: number }[]
): HistogramData[] {
  const histogram: HistogramData[] = []
  const color = getShadingColor(area.pnl, 0.3)

  // Find all candles between entry and exit
  const entryIndex = candleData.findIndex((c) => c.time === area.entryTime)
  const exitIndex = candleData.findIndex((c) => c.time === area.exitTime)

  if (entryIndex === -1 || exitIndex === -1) return []

  const start = Math.min(entryIndex, exitIndex)
  const end = Math.max(entryIndex, exitIndex)

  for (let i = start; i <= end; i++) {
    const candle = candleData[i]

    // Calculate position of shading relative to entry/exit prices
    const minPrice = Math.min(area.entryPrice, area.exitPrice)
    const maxPrice = Math.max(area.entryPrice, area.exitPrice)

    // Use the price range as the histogram value
    histogram.push({
      time: candle.time,
      value: maxPrice - minPrice,
      color,
    })
  }

  return histogram
}

/**
 * Calculate position statistics
 */
export interface PositionStats {
  totalPositions: number
  profitablePositions: number
  losingPositions: number
  totalPnl: number
  avgPnl: number
  winRate: number
  largestWin: number
  largestLoss: number
}

export function calculatePositionStats(areas: PositionArea[]): PositionStats {
  if (areas.length === 0) {
    return {
      totalPositions: 0,
      profitablePositions: 0,
      losingPositions: 0,
      totalPnl: 0,
      avgPnl: 0,
      winRate: 0,
      largestWin: 0,
      largestLoss: 0,
    }
  }

  const profitablePositions = areas.filter((a) => a.pnl > 0)
  const losingPositions = areas.filter((a) => a.pnl < 0)
  const totalPnl = areas.reduce((sum, a) => sum + a.pnl, 0)

  const wins = profitablePositions.map((a) => a.pnl)
  const losses = losingPositions.map((a) => a.pnl)

  return {
    totalPositions: areas.length,
    profitablePositions: profitablePositions.length,
    losingPositions: losingPositions.length,
    totalPnl,
    avgPnl: totalPnl / areas.length,
    winRate: (profitablePositions.length / areas.length) * 100,
    largestWin: wins.length > 0 ? Math.max(...wins) : 0,
    largestLoss: losses.length > 0 ? Math.min(...losses) : 0,
  }
}

/**
 * Get position duration in minutes
 */
export function getPositionDuration(area: PositionArea): number {
  const entryTimestamp = typeof area.entryTime === 'number'
    ? area.entryTime
    : new Date(area.entryTime as string).getTime() / 1000

  const exitTimestamp = typeof area.exitTime === 'number'
    ? area.exitTime
    : new Date(area.exitTime as string).getTime() / 1000

  return (exitTimestamp - entryTimestamp) / 60 // Convert to minutes
}

/**
 * Calculate return percentage
 */
export function calculateReturnPercent(area: PositionArea): number {
  return (area.pnl / area.entryPrice) * 100
}

/**
 * Format P&L for display
 */
export function formatPnL(pnl: number): string {
  const sign = pnl >= 0 ? '+' : ''
  return `${sign}${pnl.toFixed(2)}`
}

/**
 * Format return percentage for display
 */
export function formatReturnPercent(returnPercent: number): string {
  const sign = returnPercent >= 0 ? '+' : ''
  return `${sign}${returnPercent.toFixed(2)}%`
}

/**
 * Get position area tooltip text
 */
export function getPositionTooltip(area: PositionArea): string {
  const duration = getPositionDuration(area)
  const returnPercent = calculateReturnPercent(area)

  return [
    `${area.type === OrderType.BUY ? 'LONG' : 'SHORT'} Position`,
    `Entry: ${area.entryPrice.toFixed(2)}`,
    `Exit: ${area.exitPrice.toFixed(2)}`,
    `P&L: ${formatPnL(area.pnl)} (${formatReturnPercent(returnPercent)})`,
    `Duration: ${duration.toFixed(0)} min`,
    `Confidence: ${area.confidence.toFixed(0)}%`,
  ].join('\n')
}

export default function PositionShading({ signals, candleSeries }: PositionShadingProps) {
  const positionAreas = usePositionAreas(signals)

  // This component doesn't render anything directly in React
  // The shading is applied directly to the chart series
  // This is handled by the parent SignalOverlay component

  return null
}
