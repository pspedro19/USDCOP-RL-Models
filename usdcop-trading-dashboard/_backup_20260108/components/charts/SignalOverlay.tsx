'use client'

/**
 * SignalOverlay Component
 * =======================
 *
 * Main component that overlays trading signals on candlestick charts.
 * Integrates markers, position shading, and SL/TP lines.
 *
 * Features:
 * - BUY/SELL/HOLD markers with confidence indicators
 * - Stop Loss and Take Profit horizontal lines
 * - Shaded position areas showing P&L
 * - Interactive tooltips on hover
 * - Real-time signal updates via WebSocket
 * - Comprehensive filtering options
 *
 * @example
 * ```tsx
 * <SignalOverlay
 *   signals={signals}
 *   filter={{ minConfidence: 70, showHold: false }}
 *   showStopLoss={true}
 *   showTakeProfit={true}
 *   showPositionAreas={true}
 *   autoUpdate={true}
 * />
 * ```
 */

import React, { useEffect, useState, useRef, useMemo } from 'react'
import { ISeriesApi, Time } from 'lightweight-charts'
import {
  SignalData,
  SignalFilterOptions,
  SignalTooltipData,
  SignalOverlayProps,
} from './signal-overlay/types'
import {
  useSignalMarkers,
  filterSignalsByDateRange,
  filterSignalsByConfidence,
  filterSignalsByType,
  filterSignalsByStatus,
} from './signal-overlay/SignalMarker'
import { usePositionAreas } from './signal-overlay/PositionShading'
import {
  useSignalPriceLines,
  usePriceLineManagement,
} from './signal-overlay/StopLossTakeProfit'
import useSignalOverlay from '@/hooks/useSignalOverlay'

interface SignalOverlayComponentProps extends SignalOverlayProps {
  candleSeries: ISeriesApi<'Candlestick'> | null
  chartContainer?: HTMLDivElement | null
}

export default function SignalOverlay({
  signals: externalSignals,
  candleSeries,
  chartContainer,
  filter = {},
  onSignalClick,
  onSignalHover,
  showStopLoss = true,
  showTakeProfit = true,
  showPositionAreas = true,
  showTooltips = true,
  autoUpdate = false,
  websocketUrl,
}: SignalOverlayComponentProps) {
  // State
  const [tooltipData, setTooltipData] = useState<SignalTooltipData>({
    signal: externalSignals?.[0],
    mouseX: 0,
    mouseY: 0,
    visible: false,
  })

  // Use the hook for auto-updating signals if enabled
  const {
    signals: hookSignals,
    loading,
    isConnected,
  } = useSignalOverlay({
    autoRefresh: autoUpdate,
    enableWebSocket: !!websocketUrl,
    websocketUrl,
    filter,
  })

  // Use external signals if provided, otherwise use hook signals
  const signals = externalSignals && externalSignals.length > 0 ? externalSignals : hookSignals

  // Apply filters
  const filteredSignals = useMemo(() => {
    let filtered = [...signals]

    if (filter.startDate || filter.endDate) {
      filtered = filterSignalsByDateRange(filtered, filter.startDate, filter.endDate)
    }

    if (filter.minConfidence || filter.maxConfidence) {
      filtered = filterSignalsByConfidence(
        filtered,
        filter.minConfidence,
        filter.maxConfidence
      )
    }

    if (filter.actionTypes) {
      filtered = filterSignalsByType(filtered, filter.actionTypes)
    }

    if (filter.showActive !== undefined || filter.showClosed !== undefined) {
      filtered = filterSignalsByStatus(
        filtered,
        filter.showActive ?? true,
        filter.showClosed ?? true
      )
    }

    return filtered
  }, [signals, filter])

  // Generate markers for the chart
  const markers = useSignalMarkers(filteredSignals, filter.showHold)

  // Generate price lines for SL/TP
  const priceLines = useSignalPriceLines(
    filteredSignals,
    showStopLoss,
    showTakeProfit
  )

  // Generate position areas for shading
  const positionAreas = usePositionAreas(filteredSignals)

  // Apply markers to candlestick series
  useEffect(() => {
    if (!candleSeries || markers.length === 0) return

    candleSeries.setMarkers(markers)

    return () => {
      // Clear markers on unmount
      candleSeries.setMarkers([])
    }
  }, [candleSeries, markers])

  // Manage price lines
  usePriceLineManagement(candleSeries, priceLines)

  // Handle mouse events for tooltips
  useEffect(() => {
    if (!chartContainer || !showTooltips) return

    const handleMouseMove = (event: MouseEvent) => {
      // This is a simplified version
      // In production, you'd need to convert mouse coordinates to chart time/price
      // and find the nearest signal

      const rect = chartContainer.getBoundingClientRect()
      const mouseX = event.clientX - rect.left
      const mouseY = event.clientY - rect.top

      // Find signal near mouse position (simplified)
      // You would need to implement proper coordinate conversion
      const nearbySignal = filteredSignals[0] // Placeholder

      if (nearbySignal) {
        setTooltipData({
          signal: nearbySignal,
          mouseX,
          mouseY,
          visible: true,
        })

        if (onSignalHover) {
          onSignalHover(nearbySignal)
        }
      }
    }

    const handleMouseLeave = () => {
      setTooltipData((prev) => ({ ...prev, visible: false }))
      if (onSignalHover) {
        onSignalHover(null)
      }
    }

    chartContainer.addEventListener('mousemove', handleMouseMove)
    chartContainer.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      chartContainer.removeEventListener('mousemove', handleMouseMove)
      chartContainer.removeEventListener('mouseleave', handleMouseLeave)
    }
  }, [chartContainer, filteredSignals, showTooltips, onSignalHover])

  // Tooltip component
  const SignalTooltip = () => {
    if (!tooltipData.visible || !showTooltips || !tooltipData.signal) return null

    const { signal, mouseX, mouseY } = tooltipData

    return (
      <div
        className="absolute z-50 pointer-events-none"
        style={{
          left: mouseX + 10,
          top: mouseY + 10,
        }}
      >
        <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-3 min-w-[250px]">
          {/* Signal Type Badge */}
          <div className="flex items-center justify-between mb-2">
            <span
              className={`px-2 py-1 rounded text-xs font-bold ${
                signal.type === 'BUY'
                  ? 'bg-green-900/50 text-green-400'
                  : signal.type === 'SELL'
                  ? 'bg-red-900/50 text-red-400'
                  : 'bg-amber-900/50 text-amber-400'
              }`}
            >
              {signal.type}
            </span>
            <span className="text-xs text-gray-400">
              {new Date(signal.timestamp).toLocaleTimeString()}
            </span>
          </div>

          {/* Price and Confidence */}
          <div className="space-y-1 text-sm mb-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Price:</span>
              <span className="text-white font-mono">${signal.price.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Confidence:</span>
              <span className="text-white font-bold">
                {signal.confidence.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Stop Loss / Take Profit */}
          {(signal.stopLoss || signal.takeProfit) && (
            <div className="space-y-1 text-xs border-t border-gray-700 pt-2 mb-2">
              {signal.stopLoss && (
                <div className="flex justify-between">
                  <span className="text-red-400">Stop Loss:</span>
                  <span className="text-white font-mono">
                    ${signal.stopLoss.toFixed(2)}
                  </span>
                </div>
              )}
              {signal.takeProfit && (
                <div className="flex justify-between">
                  <span className="text-green-400">Take Profit:</span>
                  <span className="text-white font-mono">
                    ${signal.takeProfit.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Reasoning */}
          {signal.reasoning && signal.reasoning.length > 0 && (
            <div className="border-t border-gray-700 pt-2">
              <div className="text-xs text-gray-400 mb-1">Analysis:</div>
              <ul className="space-y-0.5">
                {signal.reasoning.slice(0, 3).map((reason, idx) => (
                  <li key={idx} className="text-xs text-gray-300 flex items-start">
                    <span className="text-blue-400 mr-1">•</span>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Risk/Return */}
          <div className="flex justify-between text-xs border-t border-gray-700 pt-2 mt-2">
            <div>
              <span className="text-gray-400">Risk:</span>{' '}
              <span className="text-orange-400">{signal.riskScore.toFixed(1)}/10</span>
            </div>
            <div>
              <span className="text-gray-400">Return:</span>{' '}
              <span className="text-green-400">
                {(signal.expectedReturn * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Model Source */}
          <div className="text-xs text-gray-500 mt-2 flex justify-between items-center">
            <span>{signal.modelSource}</span>
            <span>{signal.latency.toFixed(0)}ms</span>
          </div>

          {/* P&L if closed */}
          {signal.status === 'closed' && signal.pnl !== undefined && (
            <div className="border-t border-gray-700 pt-2 mt-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">P&L:</span>
                <span
                  className={`text-sm font-bold ${
                    signal.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {signal.pnl >= 0 ? '+' : ''}
                  ${signal.pnl.toFixed(2)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <>
      {/* Tooltip overlay */}
      {showTooltips && <SignalTooltip />}

      {/* Loading indicator */}
      {loading && autoUpdate && (
        <div className="absolute top-2 right-2 z-10">
          <div className="bg-gray-900/90 backdrop-blur-sm px-3 py-1 rounded-lg text-xs text-gray-400 flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
            Loading signals...
          </div>
        </div>
      )}

      {/* WebSocket status */}
      {websocketUrl && (
        <div className="absolute top-2 right-2 z-10">
          <div
            className={`bg-gray-900/90 backdrop-blur-sm px-3 py-1 rounded-lg text-xs flex items-center gap-2 ${
              isConnected ? 'text-green-400' : 'text-gray-400'
            }`}
          >
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
              }`}
            />
            {isConnected ? 'Live' : 'Disconnected'}
          </div>
        </div>
      )}

      {/* Signal stats overlay */}
      {filteredSignals.length > 0 && (
        <div className="absolute bottom-2 left-2 z-10">
          <div className="bg-gray-900/90 backdrop-blur-sm px-3 py-2 rounded-lg text-xs">
            <div className="flex gap-4 text-gray-400">
              <div>
                <span className="text-gray-500">Signals:</span>{' '}
                <span className="text-white font-medium">{filteredSignals.length}</span>
              </div>
              <div>
                <span className="text-gray-500">Active:</span>{' '}
                <span className="text-green-400 font-medium">
                  {filteredSignals.filter((s) => s.status === 'active').length}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Closed:</span>{' '}
                <span className="text-blue-400 font-medium">
                  {filteredSignals.filter((s) => s.status === 'closed').length}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

/**
 * Export utility functions
 */
export { useSignalMarkers, usePositionAreas, useSignalPriceLines }

/**
 * Export types
 */
export type { SignalData, SignalFilterOptions, SignalTooltipData }
