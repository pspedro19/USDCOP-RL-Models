'use client'

/**
 * ChartWithPositions Component
 *
 * Gráfica de velas japonesas con marcadores de posiciones del agente RL.
 * Utiliza Lightweight Charts para rendering de alto rendimiento.
 *
 * Features:
 * - Velas japonesas en tiempo real
 * - Marcadores de entrada/salida del agente
 * - Línea de posición actual
 * - Panel de equity curve
 * - Indicadores técnicos opcionales
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  CrosshairMode,
  SeriesMarker,
  Time,
} from 'lightweight-charts'

// Types
interface CandleData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface AgentAction {
  action_id: number
  timestamp_cot: string
  bar_number: number
  action_type: string
  side: string | null
  price_at_action: number
  position_after: number
  marker_type: string
  marker_color: string
}

interface EquityPoint {
  time: Time
  value: number
}

// API response types
interface CandlestickAPIResponse {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface EquityCurveAPIResponse {
  timestamp_cot: string
  equity_value: number
}

interface ChartWithPositionsProps {
  symbol?: string
  timeframe?: string
  height?: number
  showEquityCurve?: boolean
  showPositionLine?: boolean
  showIndicators?: boolean
  autoRefresh?: boolean
  refreshInterval?: number
}

// Marker configuration based on action type
const getMarkerConfig = (action: AgentAction): {
  position: 'aboveBar' | 'belowBar' | 'inBar'
  color: string
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square'
  text: string
} => {
  switch (action.action_type) {
    case 'ENTRY_LONG':
    case 'FLIP_LONG':
      return { position: 'belowBar', color: '#22c55e', shape: 'arrowUp', text: 'L' }
    case 'ENTRY_SHORT':
    case 'FLIP_SHORT':
      return { position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: 'S' }
    case 'EXIT_LONG':
    case 'EXIT_SHORT':
      return { position: 'inBar', color: '#f59e0b', shape: 'square', text: 'X' }
    case 'INCREASE_LONG':
      return { position: 'belowBar', color: '#86efac', shape: 'arrowUp', text: '+L' }
    case 'INCREASE_SHORT':
      return { position: 'aboveBar', color: '#fca5a5', shape: 'arrowDown', text: '+S' }
    default:
      return { position: 'inBar', color: '#6b7280', shape: 'circle', text: '' }
  }
}

export default function ChartWithPositions({
  symbol = 'USDCOP',
  timeframe = '5m',
  height = 500,
  showEquityCurve = true,
  showPositionLine = true,
  showIndicators = false,
  autoRefresh = true,
  refreshInterval = 30000,
}: ChartWithPositionsProps) {
  // Refs
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const equityContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const equityChartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const equitySeriesRef = useRef<ISeriesApi<'Area'> | null>(null)
  const positionLineRef = useRef<ReturnType<ISeriesApi<'Candlestick'>['createPriceLine']> | null>(null)

  // State
  const [candleData, setCandleData] = useState<CandleData[]>([])
  const [actions, setActions] = useState<AgentAction[]>([])
  const [equityData, setEquityData] = useState<EquityPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [currentPosition, setCurrentPosition] = useState(0)
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)
  const [isLive, setIsLive] = useState(false)

  // Fetch candle data
  const fetchCandleData = useCallback(async () => {
    try {
      const response = await fetch(
        `/api/market/candlesticks?symbol=${symbol}&timeframe=${timeframe}&limit=200`
      )
      const data = await response.json()

      if (data.data) {
        const candles = data.data.map((d: CandlestickAPIResponse) => ({
          time: Math.floor(new Date(d.time).getTime() / 1000) as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }))
        setCandleData(candles)

        if (candles.length > 0) {
          setCurrentPrice(candles[candles.length - 1].close)
        }
      }
    } catch (err) {
      console.error('Error fetching candle data:', err)
      setError('Error loading market data')
    }
  }, [symbol, timeframe])

  // Fetch agent actions
  const fetchActions = useCallback(async () => {
    try {
      const response = await fetch('/api/agent/actions?action=today')
      const data = await response.json()

      if (data.success) {
        setActions(data.data.actions || [])
        setIsLive(data.data.isLive || false)

        // Extract equity curve
        if (data.data.equityCurve) {
          const equity = data.data.equityCurve.map((e: EquityCurveAPIResponse) => ({
            time: Math.floor(new Date(e.timestamp_cot).getTime() / 1000) as Time,
            value: e.equity_value,
          }))
          setEquityData(equity)
        }

        // Get current position
        if (data.data.actions.length > 0) {
          const lastAction = data.data.actions[data.data.actions.length - 1]
          setCurrentPosition(lastAction.position_after)
        }

        setError(null)
      }
    } catch (err) {
      console.error('Error fetching actions:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  // Initialize charts
  useEffect(() => {
    if (!chartContainerRef.current) return

    // Create main chart
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
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
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

    chartRef.current = chart
    candleSeriesRef.current = candleSeries

    // Create equity chart if enabled
    if (showEquityCurve && equityContainerRef.current) {
      const equityChart = createChart(equityContainerRef.current, {
        layout: {
          background: { type: ColorType.Solid, color: '#0f172a' },
          textColor: '#94a3b8',
        },
        grid: {
          vertLines: { color: '#1e293b' },
          horzLines: { color: '#1e293b' },
        },
        rightPriceScale: {
          borderColor: '#334155',
        },
        timeScale: {
          borderColor: '#334155',
          visible: false,
        },
        width: equityContainerRef.current.clientWidth,
        height: 100,
      })

      const equitySeries = equityChart.addAreaSeries({
        lineColor: '#3b82f6',
        topColor: 'rgba(59, 130, 246, 0.4)',
        bottomColor: 'rgba(59, 130, 246, 0.0)',
        lineWidth: 2,
      })

      equityChartRef.current = equityChart
      equitySeriesRef.current = equitySeries
    }

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
      if (equityContainerRef.current && equityChartRef.current) {
        equityChartRef.current.applyOptions({
          width: equityContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
      equityChartRef.current?.remove()
    }
  }, [height, showEquityCurve])

  // Update chart data
  useEffect(() => {
    if (!candleSeriesRef.current || candleData.length === 0) return

    candleSeriesRef.current.setData(candleData)

    // Add markers for agent actions
    if (actions.length > 0) {
      const markers: SeriesMarker<Time>[] = actions
        .filter(a => a.action_type !== 'HOLD')
        .map(action => {
          const config = getMarkerConfig(action)
          const timestamp = Math.floor(new Date(action.timestamp_cot).getTime() / 1000)

          return {
            time: timestamp as Time,
            position: config.position,
            color: config.color,
            shape: config.shape,
            text: config.text,
            size: 1.5,
          }
        })

      candleSeriesRef.current.setMarkers(markers)
    }

    // Update position line
    if (showPositionLine && currentPrice) {
      // Remove existing position line
      if (positionLineRef.current) {
        candleSeriesRef.current.removePriceLine(positionLineRef.current)
      }

      // Add new position line
      const posColor = currentPosition > 0 ? '#22c55e' :
                       currentPosition < 0 ? '#ef4444' : '#6b7280'

      positionLineRef.current = candleSeriesRef.current.createPriceLine({
        price: currentPrice,
        color: posColor,
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: currentPosition > 0 ? `LONG ${(currentPosition * 100).toFixed(0)}%` :
               currentPosition < 0 ? `SHORT ${(Math.abs(currentPosition) * 100).toFixed(0)}%` :
               'FLAT',
      })
    }

    // Fit content
    chartRef.current?.timeScale().fitContent()
  }, [candleData, actions, currentPosition, currentPrice, showPositionLine])

  // Update equity chart
  useEffect(() => {
    if (!equitySeriesRef.current || equityData.length === 0) return
    equitySeriesRef.current.setData(equityData)
    equityChartRef.current?.timeScale().fitContent()
  }, [equityData])

  // Initial data load and polling
  useEffect(() => {
    const loadData = async () => {
      await Promise.all([fetchCandleData(), fetchActions()])
    }

    loadData()

    if (autoRefresh) {
      const interval = setInterval(loadData, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [fetchCandleData, fetchActions, autoRefresh, refreshInterval])

  // Position indicator component
  const PositionIndicator = () => {
    const isLong = currentPosition > 0.1
    const isShort = currentPosition < -0.1

    return (
      <div className="flex items-center gap-3">
        <div className={`px-3 py-1 rounded-lg font-bold ${
          isLong ? 'bg-green-900/50 text-green-400' :
          isShort ? 'bg-red-900/50 text-red-400' :
          'bg-gray-800 text-gray-400'
        }`}>
          {isLong ? `LONG ${(currentPosition * 100).toFixed(0)}%` :
           isShort ? `SHORT ${(Math.abs(currentPosition) * 100).toFixed(0)}%` :
           'FLAT'}
        </div>
        {currentPrice && (
          <div className="text-white font-mono">
            ${currentPrice.toFixed(2)}
          </div>
        )}
      </div>
    )
  }

  // Legend component
  const Legend = () => (
    <div className="flex flex-wrap gap-4 text-xs text-gray-400">
      <div className="flex items-center gap-1">
        <span className="text-green-400 text-lg">▲</span>
        <span>Entry Long</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-red-400 text-lg">▼</span>
        <span>Entry Short</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-amber-400 text-lg">■</span>
        <span>Exit</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-green-300 text-lg">↑</span>
        <span>Increase Long</span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-red-300 text-lg">↓</span>
        <span>Increase Short</span>
      </div>
    </div>
  )

  if (loading && candleData.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="flex items-center justify-center" style={{ height }}>
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
          <span className="ml-3 text-gray-400">Cargando gráfica...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-bold text-white">
            {symbol} - Posiciones del Agente
          </h3>
          {isLive && (
            <span className="flex items-center gap-1.5 px-2 py-1 bg-green-900/30 rounded text-xs text-green-400">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              LIVE
            </span>
          )}
        </div>
        <PositionIndicator />
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 text-red-400 px-4 py-2 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Main chart */}
      <div ref={chartContainerRef} className="w-full rounded-lg overflow-hidden" />

      {/* Equity curve */}
      {showEquityCurve && (
        <div className="space-y-2">
          <div className="text-sm text-gray-400">Curva de Capital</div>
          <div ref={equityContainerRef} className="w-full rounded-lg overflow-hidden" />
        </div>
      )}

      {/* Legend */}
      <Legend />

      {/* Stats */}
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="bg-gray-800/50 px-3 py-1 rounded">
          <span className="text-gray-400">Trades hoy: </span>
          <span className="text-white font-medium">
            {actions.filter(a => a.action_type !== 'HOLD').length}
          </span>
        </div>
        <div className="bg-gray-800/50 px-3 py-1 rounded">
          <span className="text-gray-400">Última barra: </span>
          <span className="text-white font-medium">
            #{actions.length > 0 ? actions[actions.length - 1].bar_number : 0}
          </span>
        </div>
        {equityData.length > 0 && (
          <div className="bg-gray-800/50 px-3 py-1 rounded">
            <span className="text-gray-400">Equity: </span>
            <span className="text-white font-medium">
              ${equityData[equityData.length - 1]?.value.toFixed(2)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
