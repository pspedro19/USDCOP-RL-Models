'use client'

/**
 * SignalMarker Component
 * ======================
 *
 * Renders individual signal markers on the chart using lightweight-charts markers API
 */

import { useMemo } from 'react'
import { SeriesMarker, Time } from 'lightweight-charts'
import { SignalData, SignalMarkerConfig } from './types'
import { OrderType } from '@/types/trading'

interface SignalMarkerProps {
  signals: SignalData[]
  showHold?: boolean
}

/**
 * Get marker configuration based on signal type
 */
export function getMarkerConfig(signal: SignalData): SignalMarkerConfig {
  const baseConfig = {
    time: signal.time,
    id: signal.id,
    size: 1.5,
  }

  switch (signal.type) {
    case OrderType.BUY:
      return {
        ...baseConfig,
        position: 'belowBar' as const,
        color: '#22c55e',
        shape: 'arrowUp' as const,
        text: `BUY ${signal.confidence.toFixed(0)}%`,
      }

    case OrderType.SELL:
      return {
        ...baseConfig,
        position: 'aboveBar' as const,
        color: '#ef4444',
        shape: 'arrowDown' as const,
        text: `SELL ${signal.confidence.toFixed(0)}%`,
      }

    case OrderType.HOLD:
      return {
        ...baseConfig,
        position: 'inBar' as const,
        color: '#f59e0b',
        shape: 'circle' as const,
        text: signal.confidence > 70 ? 'HOLD' : '',
      }

    default:
      return {
        ...baseConfig,
        position: 'inBar' as const,
        color: '#6b7280',
        shape: 'circle' as const,
        text: '',
      }
  }
}

/**
 * Get exit marker configuration
 */
export function getExitMarkerConfig(signal: SignalData): SignalMarkerConfig | null {
  if (!signal.exitTime || signal.status !== 'closed') return null

  const isProfitable = (signal.pnl ?? 0) > 0

  return {
    time: signal.exitTime,
    id: `${signal.id}_exit`,
    position: 'inBar' as const,
    color: isProfitable ? '#86efac' : '#fca5a5',
    shape: 'square' as const,
    text: `${isProfitable ? '+' : ''}${signal.pnl?.toFixed(2) || '0'}`,
    size: 1.2,
  }
}

/**
 * Convert signal markers to lightweight-charts format
 */
export function useSignalMarkers(
  signals: SignalData[],
  showHold: boolean = false
): SeriesMarker<Time>[] {
  return useMemo(() => {
    const markers: SeriesMarker<Time>[] = []

    signals.forEach((signal) => {
      // Skip HOLD signals if not enabled
      if (signal.type === OrderType.HOLD && !showHold) {
        return
      }

      // Entry marker
      const entryConfig = getMarkerConfig(signal)
      markers.push({
        time: entryConfig.time,
        position: entryConfig.position,
        color: entryConfig.color,
        shape: entryConfig.shape,
        text: entryConfig.text,
        size: entryConfig.size,
      })

      // Exit marker (if signal is closed)
      const exitConfig = getExitMarkerConfig(signal)
      if (exitConfig) {
        markers.push({
          time: exitConfig.time,
          position: exitConfig.position,
          color: exitConfig.color,
          shape: exitConfig.shape,
          text: exitConfig.text,
          size: exitConfig.size,
        })
      }
    })

    return markers
  }, [signals, showHold])
}

/**
 * Get marker color based on confidence and type
 */
export function getConfidenceColor(
  type: OrderType,
  confidence: number
): string {
  if (confidence < 60) {
    return '#6b7280' // Gray for low confidence
  }

  switch (type) {
    case OrderType.BUY:
      if (confidence >= 85) return '#16a34a' // Dark green
      if (confidence >= 70) return '#22c55e' // Green
      return '#86efac' // Light green

    case OrderType.SELL:
      if (confidence >= 85) return '#dc2626' // Dark red
      if (confidence >= 70) return '#ef4444' // Red
      return '#fca5a5' // Light red

    case OrderType.HOLD:
      return '#f59e0b' // Amber

    default:
      return '#6b7280' // Gray
  }
}

/**
 * Get marker size based on confidence
 */
export function getConfidenceSize(confidence: number): number {
  if (confidence >= 90) return 2.0
  if (confidence >= 80) return 1.7
  if (confidence >= 70) return 1.5
  if (confidence >= 60) return 1.3
  return 1.0
}

/**
 * Filter signals by date range
 */
export function filterSignalsByDateRange(
  signals: SignalData[],
  startDate?: Date,
  endDate?: Date
): SignalData[] {
  return signals.filter((signal) => {
    const signalDate = new Date(signal.timestamp)

    if (startDate && signalDate < startDate) return false
    if (endDate && signalDate > endDate) return false

    return true
  })
}

/**
 * Filter signals by confidence threshold
 */
export function filterSignalsByConfidence(
  signals: SignalData[],
  minConfidence?: number,
  maxConfidence?: number
): SignalData[] {
  return signals.filter((signal) => {
    if (minConfidence !== undefined && signal.confidence < minConfidence) {
      return false
    }
    if (maxConfidence !== undefined && signal.confidence > maxConfidence) {
      return false
    }
    return true
  })
}

/**
 * Filter signals by action type
 */
export function filterSignalsByType(
  signals: SignalData[],
  types?: OrderType[]
): SignalData[] {
  if (!types || types.length === 0) return signals
  return signals.filter((signal) => types.includes(signal.type))
}

/**
 * Filter signals by status
 */
export function filterSignalsByStatus(
  signals: SignalData[],
  showActive: boolean = true,
  showClosed: boolean = true
): SignalData[] {
  return signals.filter((signal) => {
    if (signal.status === 'active' && !showActive) return false
    if (signal.status === 'closed' && !showClosed) return false
    return true
  })
}

export default function SignalMarker({ signals, showHold = false }: SignalMarkerProps) {
  // This component doesn't render anything directly
  // It's used as a utility component to generate markers
  // The actual rendering is done by the parent component using the hooks
  return null
}
