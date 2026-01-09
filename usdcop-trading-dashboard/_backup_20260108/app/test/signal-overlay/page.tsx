'use client'

/**
 * Signal Overlay Test Page
 * =========================
 *
 * Test and demo page for the SignalOverlay component.
 * Shows various configurations and use cases.
 */

import React, { useEffect, useRef, useState } from 'react'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  CrosshairMode,
  Time,
} from 'lightweight-charts'
import SignalOverlay from '@/components/charts/SignalOverlay'
import { SignalData } from '@/components/charts/signal-overlay/types'
import { OrderType } from '@/types/trading'

export default function SignalOverlayTestPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  const [showStopLoss, setShowStopLoss] = useState(true)
  const [showTakeProfit, setShowTakeProfit] = useState(true)
  const [showHold, setShowHold] = useState(false)
  const [minConfidence, setMinConfidence] = useState(0)
  const [selectedSignal, setSelectedSignal] = useState<SignalData | null>(null)

  // Sample signals for testing
  const mockSignals: SignalData[] = [
    {
      id: 'sig_001',
      timestamp: '2025-12-17T10:30:00Z',
      time: 1702810800 as Time,
      type: OrderType.BUY,
      confidence: 87.5,
      price: 4285.50,
      stopLoss: 4270.00,
      takeProfit: 4320.00,
      reasoning: [
        'RSI oversold at 28.5',
        'MACD bullish crossover',
        'Support level at 4280',
        'High ML confidence (87.5%)',
      ],
      riskScore: 3.2,
      expectedReturn: 0.0081,
      timeHorizon: '15-30 min',
      modelSource: 'L5_PPO_LSTM_v2.1',
      latency: 42,
      status: 'active',
    },
    {
      id: 'sig_002',
      timestamp: '2025-12-17T10:35:00Z',
      time: 1702811100 as Time,
      type: OrderType.SELL,
      confidence: 72.3,
      price: 4295.25,
      stopLoss: 4310.00,
      takeProfit: 4260.00,
      reasoning: [
        'RSI overbought at 74.2',
        'Resistance at 4300',
        'Volume divergence',
      ],
      riskScore: 4.8,
      expectedReturn: 0.0082,
      timeHorizon: '20-40 min',
      modelSource: 'L5_PPO_LSTM_v2.1',
      latency: 38,
      exitPrice: 4275.00,
      exitTimestamp: '2025-12-17T10:55:00Z',
      exitTime: 1702812300 as Time,
      pnl: 202.50,
      status: 'closed',
    },
    {
      id: 'sig_003',
      timestamp: '2025-12-17T10:40:00Z',
      time: 1702811400 as Time,
      type: OrderType.HOLD,
      confidence: 65.0,
      price: 4290.00,
      reasoning: [
        'Mixed signals',
        'Low volume',
        'Awaiting confirmation',
      ],
      riskScore: 5.0,
      expectedReturn: 0,
      timeHorizon: '5-15 min',
      modelSource: 'L5_PPO_LSTM_v2.1',
      latency: 45,
      status: 'active',
    },
    {
      id: 'sig_004',
      timestamp: '2025-12-17T10:45:00Z',
      time: 1702811700 as Time,
      type: OrderType.BUY,
      confidence: 92.1,
      price: 4288.75,
      stopLoss: 4275.00,
      takeProfit: 4330.00,
      reasoning: [
        'Strong bullish momentum',
        'Volume spike detected',
        'Breakout above SMA20',
        'High ML confidence (92.1%)',
      ],
      riskScore: 2.5,
      expectedReturn: 0.0095,
      timeHorizon: '30-45 min',
      modelSource: 'L5_PPO_LSTM_v2.1',
      latency: 35,
      exitPrice: 4325.00,
      exitTimestamp: '2025-12-17T11:15:00Z',
      exitTime: 1702813500 as Time,
      pnl: 362.50,
      status: 'closed',
    },
    {
      id: 'sig_005',
      timestamp: '2025-12-17T10:50:00Z',
      time: 1702812000 as Time,
      type: OrderType.BUY,
      confidence: 78.5,
      price: 4305.00,
      stopLoss: 4292.00,
      takeProfit: 4340.00,
      reasoning: [
        'Continuation pattern',
        'Support at EMA12',
        'Positive trend',
      ],
      riskScore: 3.8,
      expectedReturn: 0.0081,
      timeHorizon: '15-30 min',
      modelSource: 'L5_PPO_LSTM_v2.1',
      latency: 41,
      status: 'active',
    },
  ]

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

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
        vertLine: { color: '#475569', labelBackgroundColor: '#1e293b' },
        horzLine: { color: '#475569', labelBackgroundColor: '#1e293b' },
      },
      rightPriceScale: {
        borderColor: '#334155',
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: 600,
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    // Generate sample candle data
    const candleData = []
    let basePrice = 4280
    for (let i = 0; i < 50; i++) {
      const time = (1702810800 + i * 300) as Time
      const change = (Math.random() - 0.5) * 20
      const open = basePrice
      const close = basePrice + change
      const high = Math.max(open, close) + Math.random() * 10
      const low = Math.min(open, close) - Math.random() * 10

      candleData.push({
        time,
        open,
        high,
        low,
        close,
      })

      basePrice = close
    }

    candleSeries.setData(candleData)
    chart.timeScale().fitContent()

    chartRef.current = chart
    candleSeriesRef.current = candleSeries

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-6">
          Signal Overlay Test & Demo
        </h1>

        {/* Controls Panel */}
        <div className="bg-gray-900 rounded-lg p-4 mb-6">
          <h2 className="text-white text-lg mb-4">Controls</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Min Confidence Slider */}
            <div>
              <label className="text-gray-400 text-sm block mb-2">
                Min Confidence: {minConfidence}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                className="w-full"
              />
            </div>

            {/* Checkboxes */}
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showStopLoss}
                  onChange={(e) => setShowStopLoss(e.target.checked)}
                  className="w-4 h-4"
                />
                Show Stop Loss
              </label>
              <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showTakeProfit}
                  onChange={(e) => setShowTakeProfit(e.target.checked)}
                  className="w-4 h-4"
                />
                Show Take Profit
              </label>
            </div>

            <div>
              <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showHold}
                  onChange={(e) => setShowHold(e.target.checked)}
                  className="w-4 h-4"
                />
                Show HOLD Signals
              </label>
            </div>

            {/* Stats */}
            <div className="text-sm">
              <div className="text-gray-400">Filtered Signals:</div>
              <div className="text-white text-2xl font-bold">
                {mockSignals.filter((s) => s.confidence >= minConfidence).length}
              </div>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-gray-900 rounded-lg p-4 mb-6">
          <div className="relative">
            <div ref={chartContainerRef} className="w-full" />

            {candleSeriesRef.current && (
              <SignalOverlay
                signals={mockSignals}
                candleSeries={candleSeriesRef.current}
                chartContainer={chartContainerRef.current}
                filter={{
                  minConfidence,
                  showHold,
                }}
                showStopLoss={showStopLoss}
                showTakeProfit={showTakeProfit}
                showPositionAreas={true}
                showTooltips={true}
                onSignalClick={(signal) => setSelectedSignal(signal)}
              />
            )}
          </div>
        </div>

        {/* Signal Details */}
        {selectedSignal && (
          <div className="bg-gray-900 rounded-lg p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-white text-lg">Selected Signal Details</h2>
              <button
                onClick={() => setSelectedSignal(null)}
                className="text-gray-400 hover:text-white"
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">ID:</span>
                <span className="text-white ml-2">{selectedSignal.id}</span>
              </div>
              <div>
                <span className="text-gray-400">Type:</span>
                <span
                  className={`ml-2 font-bold ${
                    selectedSignal.type === 'BUY'
                      ? 'text-green-400'
                      : selectedSignal.type === 'SELL'
                      ? 'text-red-400'
                      : 'text-amber-400'
                  }`}
                >
                  {selectedSignal.type}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Confidence:</span>
                <span className="text-white ml-2">
                  {selectedSignal.confidence.toFixed(1)}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Price:</span>
                <span className="text-white ml-2 font-mono">
                  ${selectedSignal.price.toFixed(2)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Risk Score:</span>
                <span className="text-orange-400 ml-2">
                  {selectedSignal.riskScore.toFixed(1)}/10
                </span>
              </div>
              <div>
                <span className="text-gray-400">Expected Return:</span>
                <span className="text-green-400 ml-2">
                  {(selectedSignal.expectedReturn * 100).toFixed(2)}%
                </span>
              </div>
              <div>
                <span className="text-gray-400">Status:</span>
                <span className="text-white ml-2">{selectedSignal.status}</span>
              </div>
              {selectedSignal.pnl !== undefined && (
                <div>
                  <span className="text-gray-400">P&L:</span>
                  <span
                    className={`ml-2 font-bold ${
                      selectedSignal.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {selectedSignal.pnl >= 0 ? '+' : ''}$
                    {selectedSignal.pnl.toFixed(2)}
                  </span>
                </div>
              )}
            </div>

            {selectedSignal.reasoning.length > 0 && (
              <div className="mt-4">
                <div className="text-gray-400 text-sm mb-2">Reasoning:</div>
                <ul className="space-y-1">
                  {selectedSignal.reasoning.map((reason, idx) => (
                    <li key={idx} className="text-gray-300 text-sm flex items-start">
                      <span className="text-blue-400 mr-2">•</span>
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Signal List */}
        <div className="bg-gray-900 rounded-lg p-4 mt-6">
          <h2 className="text-white text-lg mb-4">All Signals</h2>
          <div className="space-y-2">
            {mockSignals.map((signal) => (
              <div
                key={signal.id}
                className="bg-gray-800 p-3 rounded flex justify-between items-center cursor-pointer hover:bg-gray-700"
                onClick={() => setSelectedSignal(signal)}
              >
                <div className="flex items-center gap-4">
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
                  <span className="text-white font-mono">
                    ${signal.price.toFixed(2)}
                  </span>
                  <span className="text-gray-400 text-sm">
                    {signal.confidence.toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <span
                    className={`text-sm ${
                      signal.status === 'active' ? 'text-green-400' : 'text-gray-400'
                    }`}
                  >
                    {signal.status}
                  </span>
                  {signal.pnl !== undefined && (
                    <span
                      className={`font-bold ${
                        signal.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {signal.pnl >= 0 ? '+' : ''}${signal.pnl.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
