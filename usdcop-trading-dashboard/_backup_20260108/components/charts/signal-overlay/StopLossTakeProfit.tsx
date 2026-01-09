'use client'

/**
 * StopLossTakeProfit Component
 * =============================
 *
 * Renders horizontal dashed lines for Stop Loss and Take Profit levels
 */

import { useMemo, useEffect } from 'react'
import { ISeriesApi, LineStyle, IPriceLine } from 'lightweight-charts'
import { SignalData, SignalPriceLine } from './types'
import { OrderType } from '@/types/trading'

interface StopLossTakeProfitProps {
  signals: SignalData[]
  candleSeries: ISeriesApi<'Candlestick'> | null
  showStopLoss?: boolean
  showTakeProfit?: boolean
}

/**
 * Extract price lines from active signals
 */
export function useSignalPriceLines(
  signals: SignalData[],
  showStopLoss: boolean = true,
  showTakeProfit: boolean = true
): SignalPriceLine[] {
  return useMemo(() => {
    const priceLines: SignalPriceLine[] = []

    // Only show lines for active signals
    const activeSignals = signals.filter((s) => s.status === 'active')

    activeSignals.forEach((signal) => {
      // Stop Loss line
      if (showStopLoss && signal.stopLoss) {
        priceLines.push({
          id: `${signal.id}_sl`,
          signalId: signal.id,
          price: signal.stopLoss,
          type: 'stopLoss',
          color: '#ef4444',
          isActive: true,
        })
      }

      // Take Profit line
      if (showTakeProfit && signal.takeProfit) {
        priceLines.push({
          id: `${signal.id}_tp`,
          signalId: signal.id,
          price: signal.takeProfit,
          type: 'takeProfit',
          color: '#22c55e',
          isActive: true,
        })
      }
    })

    return priceLines
  }, [signals, showStopLoss, showTakeProfit])
}

/**
 * Create price line options for lightweight-charts
 */
export function createPriceLineOptions(line: SignalPriceLine) {
  return {
    price: line.price,
    color: line.color,
    lineWidth: 1.5,
    lineStyle: LineStyle.Dashed,
    axisLabelVisible: true,
    title: line.type === 'stopLoss' ? 'SL' : 'TP',
  }
}

/**
 * Calculate risk-reward ratio
 */
export function calculateRiskReward(signal: SignalData): number | null {
  if (!signal.stopLoss || !signal.takeProfit) return null

  const risk = Math.abs(signal.price - signal.stopLoss)
  const reward = Math.abs(signal.takeProfit - signal.price)

  return reward / risk
}

/**
 * Calculate distance to Stop Loss in percentage
 */
export function getStopLossDistance(signal: SignalData): number | null {
  if (!signal.stopLoss) return null
  return ((signal.price - signal.stopLoss) / signal.price) * 100
}

/**
 * Calculate distance to Take Profit in percentage
 */
export function getTakeProfitDistance(signal: SignalData): number | null {
  if (!signal.takeProfit) return null
  return ((signal.takeProfit - signal.price) / signal.price) * 100
}

/**
 * Check if price has hit Stop Loss
 */
export function hasHitStopLoss(
  signal: SignalData,
  currentPrice: number
): boolean {
  if (!signal.stopLoss || signal.status !== 'active') return false

  if (signal.type === OrderType.BUY) {
    return currentPrice <= signal.stopLoss
  } else {
    return currentPrice >= signal.stopLoss
  }
}

/**
 * Check if price has hit Take Profit
 */
export function hasHitTakeProfit(
  signal: SignalData,
  currentPrice: number
): boolean {
  if (!signal.takeProfit || signal.status !== 'active') return false

  if (signal.type === OrderType.BUY) {
    return currentPrice >= signal.takeProfit
  } else {
    return currentPrice <= signal.takeProfit
  }
}

/**
 * Get price line color based on signal type and line type
 */
export function getPriceLineColor(
  signalType: OrderType,
  lineType: 'stopLoss' | 'takeProfit',
  isPriceNear: boolean = false
): string {
  if (isPriceNear) {
    // Highlight if price is near the line
    return lineType === 'stopLoss' ? '#dc2626' : '#16a34a'
  }

  return lineType === 'stopLoss' ? '#ef4444' : '#22c55e'
}

/**
 * Format price line title
 */
export function formatPriceLineTitle(
  line: SignalPriceLine,
  signal: SignalData
): string {
  const prefix = line.type === 'stopLoss' ? 'SL' : 'TP'
  const price = line.price.toFixed(2)

  if (line.type === 'stopLoss') {
    const distance = getStopLossDistance(signal)
    if (distance) {
      return `${prefix} ${price} (${Math.abs(distance).toFixed(1)}%)`
    }
  } else {
    const distance = getTakeProfitDistance(signal)
    if (distance) {
      return `${prefix} ${price} (${Math.abs(distance).toFixed(1)}%)`
    }
  }

  return `${prefix} ${price}`
}

/**
 * Get all active price lines statistics
 */
export interface PriceLineStats {
  totalStopLoss: number
  totalTakeProfit: number
  avgRiskReward: number
  avgStopDistance: number
  avgTakeProfitDistance: number
}

export function calculatePriceLineStats(
  signals: SignalData[]
): PriceLineStats {
  const activeSignals = signals.filter((s) => s.status === 'active')

  if (activeSignals.length === 0) {
    return {
      totalStopLoss: 0,
      totalTakeProfit: 0,
      avgRiskReward: 0,
      avgStopDistance: 0,
      avgTakeProfitDistance: 0,
    }
  }

  const riskRewards = activeSignals
    .map(calculateRiskReward)
    .filter((rr): rr is number => rr !== null)

  const stopDistances = activeSignals
    .map(getStopLossDistance)
    .filter((d): d is number => d !== null)

  const tpDistances = activeSignals
    .map(getTakeProfitDistance)
    .filter((d): d is number => d !== null)

  return {
    totalStopLoss: activeSignals.filter((s) => s.stopLoss).length,
    totalTakeProfit: activeSignals.filter((s) => s.takeProfit).length,
    avgRiskReward:
      riskRewards.length > 0
        ? riskRewards.reduce((a, b) => a + b, 0) / riskRewards.length
        : 0,
    avgStopDistance:
      stopDistances.length > 0
        ? stopDistances.reduce((a, b) => a + b, 0) / stopDistances.length
        : 0,
    avgTakeProfitDistance:
      tpDistances.length > 0
        ? tpDistances.reduce((a, b) => a + b, 0) / tpDistances.length
        : 0,
  }
}

/**
 * Hook to manage price lines on the chart
 */
export function usePriceLineManagement(
  candleSeries: ISeriesApi<'Candlestick'> | null,
  priceLines: SignalPriceLine[]
) {
  const priceLineRefs = useMemo<Map<string, IPriceLine>>(
    () => new Map(),
    []
  )

  useEffect(() => {
    if (!candleSeries) return

    // Remove all existing price lines
    priceLineRefs.forEach((line) => {
      candleSeries.removePriceLine(line)
    })
    priceLineRefs.clear()

    // Add new price lines
    priceLines.forEach((line) => {
      const options = createPriceLineOptions(line)
      const priceLine = candleSeries.createPriceLine(options)
      priceLineRefs.set(line.id, priceLine)
    })

    // Cleanup on unmount
    return () => {
      priceLineRefs.forEach((line) => {
        candleSeries.removePriceLine(line)
      })
      priceLineRefs.clear()
    }
  }, [candleSeries, priceLines])

  return priceLineRefs
}

export default function StopLossTakeProfit({
  signals,
  candleSeries,
  showStopLoss = true,
  showTakeProfit = true,
}: StopLossTakeProfitProps) {
  const priceLines = useSignalPriceLines(signals, showStopLoss, showTakeProfit)
  usePriceLineManagement(candleSeries, priceLines)

  // This component doesn't render anything in React DOM
  // All rendering is done directly on the chart
  return null
}
