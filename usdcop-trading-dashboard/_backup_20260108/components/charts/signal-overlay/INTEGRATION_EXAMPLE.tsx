/**
 * SignalOverlay Integration Example
 * ==================================
 *
 * This example shows how to integrate the SignalOverlay component
 * with an existing lightweight-charts candlestick chart.
 */

'use client'

import React, { useEffect, useRef, useState } from 'react'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  CrosshairMode,
  Time,
} from 'lightweight-charts'
import SignalOverlay from '../SignalOverlay'
import { SignalData } from './types'
import { OrderType } from '@/types/trading'

export default function ChartWithSignalOverlay() {
  // Refs
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // State
  const [signals, setSignals] = useState<SignalData[]>([])
  const [loading, setLoading] = useState(true)

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
    })

    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    // Sample candle data
    const candleData = [
      { time: 1702810800 as Time, open: 4280, high: 4290, low: 4275, close: 4285 },
      { time: 1702811100 as Time, open: 4285, high: 4295, low: 4282, close: 4290 },
      { time: 1702811400 as Time, open: 4290, high: 4298, low: 4288, close: 4295 },
      { time: 1702811700 as Time, open: 4295, high: 4305, low: 4292, close: 4300 },
      { time: 1702812000 as Time, open: 4300, high: 4310, low: 4298, close: 4305 },
    ]

    candleSeries.setData(candleData)
    chart.timeScale().fitContent()

    chartRef.current = chart
    candleSeriesRef.current = candleSeries

    // Cleanup
    return () => {
      chart.remove()
    }
  }, [])

  // Fetch signals
  useEffect(() => {
    async function fetchSignals() {
      try {
        const response = await fetch('/api/trading/signals')
        const data = await response.json()

        if (data.success && data.signals) {
          // Convert API signals to SignalData format
          const convertedSignals: SignalData[] = data.signals.map((s: any) => ({
            id: s.id,
            timestamp: s.timestamp,
            time: Math.floor(new Date(s.timestamp).getTime() / 1000) as Time,
            type: s.type as OrderType,
            confidence: s.confidence,
            price: s.price,
            stopLoss: s.stopLoss,
            takeProfit: s.takeProfit,
            reasoning: s.reasoning || [],
            riskScore: s.riskScore || 5,
            expectedReturn: s.expectedReturn || 0,
            timeHorizon: s.timeHorizon || '15-30 min',
            modelSource: s.modelSource || 'Unknown',
            latency: s.latency || 0,
            status: 'active' as const,
          }))

          setSignals(convertedSignals)
        }
      } catch (error) {
        console.error('Error fetching signals:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchSignals()

    // Poll for updates every 30 seconds
    const interval = setInterval(fetchSignals, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <h2 className="text-white text-xl mb-4">Chart with Signal Overlay</h2>

      <div className="relative">
        {/* Chart Container */}
        <div ref={chartContainerRef} className="w-full" />

        {/* Signal Overlay */}
        {!loading && candleSeriesRef.current && (
          <SignalOverlay
            signals={signals}
            candleSeries={candleSeriesRef.current}
            chartContainer={chartContainerRef.current}
            filter={{
              minConfidence: 70,
              showHold: false,
            }}
            showStopLoss={true}
            showTakeProfit={true}
            showPositionAreas={true}
            showTooltips={true}
            onSignalClick={(signal) => {
              console.log('Signal clicked:', signal)
              // Handle signal click (e.g., show modal with details)
            }}
          />
        )}
      </div>

      {/* Signal Stats */}
      <div className="mt-4 flex gap-4 text-sm">
        <div className="bg-gray-800 px-4 py-2 rounded">
          <span className="text-gray-400">Total Signals: </span>
          <span className="text-white font-bold">{signals.length}</span>
        </div>
        <div className="bg-gray-800 px-4 py-2 rounded">
          <span className="text-gray-400">Active: </span>
          <span className="text-green-400 font-bold">
            {signals.filter((s) => s.status === 'active').length}
          </span>
        </div>
        <div className="bg-gray-800 px-4 py-2 rounded">
          <span className="text-gray-400">Avg Confidence: </span>
          <span className="text-blue-400 font-bold">
            {signals.length > 0
              ? (
                  signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length
                ).toFixed(1)
              : 0}
            %
          </span>
        </div>
      </div>
    </div>
  )
}

/**
 * Example 2: With Filter Controls
 */
export function ChartWithFilterControls() {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  const [signals, setSignals] = useState<SignalData[]>([])
  const [minConfidence, setMinConfidence] = useState(70)
  const [showHold, setShowHold] = useState(false)
  const [showStopLoss, setShowStopLoss] = useState(true)
  const [showTakeProfit, setShowTakeProfit] = useState(true)

  // ... chart initialization (same as above)

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      {/* Controls */}
      <div className="mb-4 flex gap-4 items-center">
        <div>
          <label className="text-gray-400 text-sm">Min Confidence:</label>
          <input
            type="range"
            min="0"
            max="100"
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
            className="ml-2"
          />
          <span className="text-white ml-2">{minConfidence}%</span>
        </div>

        <label className="flex items-center gap-2 text-gray-400">
          <input
            type="checkbox"
            checked={showHold}
            onChange={(e) => setShowHold(e.target.checked)}
          />
          Show HOLD
        </label>

        <label className="flex items-center gap-2 text-gray-400">
          <input
            type="checkbox"
            checked={showStopLoss}
            onChange={(e) => setShowStopLoss(e.target.checked)}
          />
          Stop Loss
        </label>

        <label className="flex items-center gap-2 text-gray-400">
          <input
            type="checkbox"
            checked={showTakeProfit}
            onChange={(e) => setShowTakeProfit(e.target.checked)}
          />
          Take Profit
        </label>
      </div>

      {/* Chart */}
      <div className="relative">
        <div ref={chartContainerRef} className="w-full" />

        <SignalOverlay
          signals={signals}
          candleSeries={candleSeriesRef.current}
          chartContainer={chartContainerRef.current}
          filter={{
            minConfidence,
            showHold,
          }}
          showStopLoss={showStopLoss}
          showTakeProfit={showTakeProfit}
          showPositionAreas={true}
        />
      </div>
    </div>
  )
}

/**
 * Example 3: With Real-time WebSocket
 */
export function ChartWithRealTimeSignals() {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  // ... chart initialization

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="relative">
        <div ref={chartContainerRef} className="w-full" />

        <SignalOverlay
          signals={[]} // Empty - will be populated by hook
          candleSeries={candleSeriesRef.current}
          chartContainer={chartContainerRef.current}
          autoUpdate={true}
          websocketUrl="ws://localhost:3001"
          filter={{
            minConfidence: 75,
            showActive: true,
          }}
        />
      </div>
    </div>
  )
}
