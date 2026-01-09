'use client'

/**
 * Ultra-Professional Trading Chart - Superior to Bloomberg & TradingView
 * ==================================================================
 *
 * Refactored modular implementation with extracted components.
 * See components/charts/real-data-chart/ for modular parts.
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { MarketDataService, CandlestickData } from '@/lib/services/market-data-service'
import { createLogger } from '@/lib/utils/logger'
import { useRealTimePrice } from '@/hooks/useRealTimePrice'
import { useDbStats } from '@/hooks/useDbStats'
import { AlertCircle, ChevronLeft, ChevronRight, Play, Pause } from 'lucide-react'
import { Button } from '@/components/ui/button'

// Import modular components
import {
  ChartType,
  TimeframePeriod,
  ChartSettings,
  CrosshairInfo,
  HoverInfo,
  DEFAULT_CHART_SETTINGS,
  createEmptyHoverInfo,
  createEmptyCrosshair
} from './real-data-chart/types'
import {
  calculateDimensions,
  createScaleFunctions,
  drawBackground,
  drawLoadingState,
  drawGrid,
  drawCandlesticks,
  drawLineChart,
  drawVolumeBars,
  drawIndicators,
  drawCrosshair,
  drawHeader
} from './real-data-chart/ChartDrawingEngine'
import { ChartControls, TimeframeSelector, ChartLegend, ChartInfoPanel } from './real-data-chart/ChartControls'
import { ChartTooltip } from './real-data-chart/ChartTooltip'

const logger = createLogger('RealDataTradingChart')

interface RealDataTradingChartProps {
  symbol?: string
  timeframe?: string
  height?: number
  className?: string
  showSignals?: boolean
  showPositions?: boolean
}

export default function RealDataTradingChart({
  symbol = 'USDCOP',
  timeframe = '5m',
  height = 700,
  className = '',
  showSignals = false,
  showPositions = false
}: RealDataTradingChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showIndicators, setShowIndicators] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [chartType, setChartType] = useState<ChartType>('candlestick')
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframePeriod>(timeframe as TimeframePeriod)
  const [hoverInfo, setHoverInfo] = useState<HoverInfo>(createEmptyHoverInfo())
  const [crosshair, setCrosshair] = useState<CrosshairInfo>(createEmptyCrosshair())
  const [chartSettings, setChartSettings] = useState<ChartSettings>(DEFAULT_CHART_SETTINGS)

  // Signal and position state
  const [signals, setSignals] = useState<any[]>([])
  const [positions, setPositions] = useState<any[]>([])
  const [signalsLoading, setSignalsLoading] = useState(false)

  const { stats: dbStats } = useDbStats(60000)
  const { currentPrice, isConnected } = useRealTimePrice(symbol)

  /**
   * Draw signal markers on the chart
   */
  const drawSignalMarkers = useCallback((ctx: CanvasRenderingContext2D, signals: any[], visibleData: any[], scales: any, dimensions: any) => {
    signals.forEach((signal) => {
      const signalTime = new Date(signal.timestamp).getTime()
      const signalIndex = visibleData.findIndex(d => Math.abs(new Date(d.time).getTime() - signalTime) < 5 * 60 * 1000)

      if (signalIndex === -1) return

      const x = dimensions.margins.left + (signalIndex / (visibleData.length - 1)) * dimensions.chartWidth
      const y = scales.yScale(signal.price)

      const isBuy = signal.type === 'BUY'
      const color = isBuy ? '#22c55e' : '#ef4444'

      // Draw arrow marker
      ctx.save()
      ctx.fillStyle = color
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2

      ctx.beginPath()
      if (isBuy) {
        // Up arrow
        ctx.moveTo(x, y + 20)
        ctx.lineTo(x - 8, y + 32)
        ctx.lineTo(x + 8, y + 32)
      } else {
        // Down arrow
        ctx.moveTo(x, y - 20)
        ctx.lineTo(x - 8, y - 32)
        ctx.lineTo(x + 8, y - 32)
      }
      ctx.closePath()
      ctx.fill()
      ctx.stroke()

      // Draw confidence badge
      ctx.fillStyle = '#fff'
      ctx.font = 'bold 10px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(`${signal.confidence.toFixed(0)}%`, x, isBuy ? y + 28 : y - 24)

      ctx.restore()
    })
  }, [])

  /**
   * Draw position overlays
   */
  const drawPositionOverlays = useCallback((ctx: CanvasRenderingContext2D, positions: any[], visibleData: any[], scales: any, dimensions: any) => {
    positions.forEach((pos) => {
      if (!pos.timestamp_cot) return

      const posTime = new Date(pos.timestamp_cot).getTime()
      const posIndex = visibleData.findIndex(d => Math.abs(new Date(d.time).getTime() - posTime) < 5 * 60 * 1000)

      if (posIndex === -1) return

      const x = dimensions.margins.left + (posIndex / (visibleData.length - 1)) * dimensions.chartWidth
      const y = scales.yScale(pos.price_at_action)

      // Draw position marker
      ctx.save()
      ctx.fillStyle = pos.position_after > 0 ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'
      ctx.strokeStyle = pos.position_after > 0 ? '#22c55e' : '#ef4444'
      ctx.lineWidth = 2

      ctx.beginPath()
      ctx.arc(x, y, 6, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()

      ctx.restore()
    })
  }, [])

  // Fetch trading signals for PPO model
  const fetchSignals = useCallback(async () => {
    if (!showSignals) return

    try {
      setSignalsLoading(true)
      // Fetch signals specifically for ppo_v1 model, with higher limit to cover 2025
      const response = await fetch('/api/trading/signals?action=recent&limit=500&model_id=ppo_v1')
      const data = await response.json()

      if (data.success && data.signals) {
        logger.debug(`Loaded ${data.signals.length} PPO signals`)
        setSignals(data.signals)
      }
    } catch (err) {
      logger.error('Error fetching signals:', err)
    } finally {
      setSignalsLoading(false)
    }
  }, [showSignals])

  // Fetch positions
  const fetchPositions = useCallback(async () => {
    if (!showPositions) return

    try {
      const response = await fetch('/api/agent/actions?action=today')
      const data = await response.json()

      if (data.success && data.data?.actions) {
        setPositions(data.data.actions)
      }
    } catch (err) {
      logger.error('Error fetching positions:', err)
    }
  }, [showPositions])

  // Fetch historical data - use filtered endpoint for bars with movement only
  const fetchCandlestickData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      // Use filtered endpoint that returns all bars within market hours
      // Historical data (2020-Dec 2025) shows doji bars from TwelveData (single-tick per 5min)
      const startDateStr = '2020-01-01'  // Full historical data from 2020
      const endDateStr = new Date().toISOString().split('T')[0]

      const response = await fetch(
        `/api/market/candlesticks-filtered?start_date=${startDateStr}&end_date=${endDateStr}&limit=100000`
      )

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`)
      }

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch candlestick data')
      }

      logger.debug(`Fetched ${result.count} candlesticks from ${startDateStr} to ${endDateStr}`)

      setCandlestickData(result.data)
      setLastUpdate(new Date())

      if (chartSettings.autoScroll && result.data.length > 100) {
        setChartSettings((prev) => ({
          ...prev,
          viewRange: { start: result.data.length - 100, end: result.data.length }
        }))
      }
    } catch (err) {
      logger.error('Error fetching candlestick data:', err)
      setError(err instanceof Error ? err.message : 'Error fetching data')
    } finally {
      setLoading(false)
    }
  }, [chartSettings.autoScroll])

  useEffect(() => {
    fetchCandlestickData()
    if (showSignals) fetchSignals()
    if (showPositions) fetchPositions()
  }, [fetchCandlestickData, fetchSignals, fetchPositions, showSignals, showPositions])

  // Mouse handling
  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current
      if (!canvas || candlestickData.length === 0) return

      const rect = canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top

      const dimensions = calculateDimensions(rect.width, rect.height, chartSettings.showVolume)
      const visibleData = candlestickData.slice(chartSettings.viewRange.start, chartSettings.viewRange.end)
      if (visibleData.length === 0) return

      const relativeX = x - dimensions.margins.left
      if (
        relativeX >= 0 &&
        relativeX <= dimensions.chartWidth &&
        y >= dimensions.margins.top &&
        y <= rect.height - dimensions.margins.bottom
      ) {
        const index = Math.round((relativeX / dimensions.chartWidth) * (visibleData.length - 1))
        const actualIndex = chartSettings.viewRange.start + index

        if (index >= 0 && index < visibleData.length && actualIndex < candlestickData.length) {
          const candle = candlestickData[actualIndex]

          const prices = visibleData.flatMap((d) => [d.high, d.low])
          const minPrice = Math.min(...prices)
          const maxPrice = Math.max(...prices)
          const priceRange = maxPrice - minPrice
          const price = maxPrice - ((y - dimensions.margins.top) / dimensions.mainChartHeight) * priceRange

          setCrosshair({
            x,
            y,
            priceY: y,
            timeIndex: actualIndex,
            price,
            visible: chartSettings.showCrosshair
          })

          setHoverInfo({
            x: event.clientX,
            y: event.clientY,
            candle,
            index: actualIndex,
            visible: true
          })
        }
      } else {
        setCrosshair((prev) => ({ ...prev, visible: false }))
        setHoverInfo((prev) => ({ ...prev, visible: false }))
      }
    },
    [candlestickData, chartSettings]
  )

  const handleMouseLeave = useCallback(() => {
    setCrosshair((prev) => ({ ...prev, visible: false }))
    setHoverInfo((prev) => ({ ...prev, visible: false }))
  }, [])

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setChartSettings((prev) => {
      const currentRange = prev.viewRange.end - prev.viewRange.start
      const newRange = Math.max(20, Math.floor(currentRange * 0.7))
      const center = (prev.viewRange.start + prev.viewRange.end) / 2

      return {
        ...prev,
        viewRange: {
          start: Math.max(0, Math.floor(center - newRange / 2)),
          end: Math.min(candlestickData.length, Math.floor(center + newRange / 2))
        },
        zoomLevel: prev.zoomLevel * 1.4,
        candleWidth: Math.min(20, prev.candleWidth * 1.2)
      }
    })
  }, [candlestickData.length])

  const handleZoomOut = useCallback(() => {
    setChartSettings((prev) => {
      const currentRange = prev.viewRange.end - prev.viewRange.start
      const newRange = Math.min(candlestickData.length, Math.floor(currentRange * 1.4))
      const center = (prev.viewRange.start + prev.viewRange.end) / 2

      return {
        ...prev,
        viewRange: {
          start: Math.max(0, Math.floor(center - newRange / 2)),
          end: Math.min(candlestickData.length, Math.floor(center + newRange / 2))
        },
        zoomLevel: prev.zoomLevel * 0.7,
        candleWidth: Math.max(2, prev.candleWidth * 0.8)
      }
    })
  }, [candlestickData.length])

  const handlePanLeft = useCallback(() => {
    setChartSettings((prev) => {
      const range = prev.viewRange.end - prev.viewRange.start
      const step = Math.floor(range * 0.1)

      return {
        ...prev,
        viewRange: {
          start: Math.max(0, prev.viewRange.start - step),
          end: Math.max(range, prev.viewRange.end - step)
        }
      }
    })
  }, [])

  const handlePanRight = useCallback(() => {
    setChartSettings((prev) => {
      const range = prev.viewRange.end - prev.viewRange.start
      const step = Math.floor(range * 0.1)

      return {
        ...prev,
        viewRange: {
          start: Math.min(candlestickData.length - range, prev.viewRange.start + step),
          end: Math.min(candlestickData.length, prev.viewRange.end + step)
        }
      }
    })
  }, [candlestickData.length])

  const handleResetView = useCallback(() => {
    setChartSettings((prev) => ({
      ...prev,
      viewRange: { start: Math.max(0, candlestickData.length - 100), end: candlestickData.length },
      zoomLevel: 1,
      candleWidth: 8
    }))
  }, [candlestickData.length])

  // Chart drawing
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    drawBackground(ctx, rect.width, rect.height)

    if (candlestickData.length === 0) {
      if (loading) {
        drawLoadingState(ctx, rect.width, rect.height, selectedTimeframe, chartType)
      }
      return
    }

    const dimensions = calculateDimensions(rect.width, rect.height, chartSettings.showVolume)
    const visibleData = candlestickData.slice(chartSettings.viewRange.start, chartSettings.viewRange.end)
    if (visibleData.length === 0) return

    const prices = visibleData.flatMap((d) => [d.high, d.low])
    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)
    const priceRange = maxPrice - minPrice
    const padding = priceRange * 0.05

    const scales = createScaleFunctions(dimensions, visibleData, { min: minPrice, max: maxPrice, range: priceRange, padding })

    drawGrid(ctx, dimensions, visibleData, scales, minPrice, priceRange, selectedTimeframe)

    if (chartType === 'candlestick') {
      drawCandlesticks(ctx, visibleData, scales, chartSettings, dimensions.chartWidth)
    } else {
      drawLineChart(ctx, visibleData, scales, dimensions)
    }

    if (chartSettings.showVolume) {
      drawVolumeBars(ctx, visibleData, scales, dimensions, chartSettings)
    }

    if (showIndicators) {
      drawIndicators(ctx, visibleData, scales)
    }

    // Draw signal markers
    if (showSignals && signals.length > 0) {
      drawSignalMarkers(ctx, signals, visibleData, scales, dimensions)
    }

    // Draw position overlays
    if (showPositions && positions.length > 0) {
      drawPositionOverlays(ctx, positions, visibleData, scales, dimensions)
    }

    drawCrosshair(ctx, crosshair, dimensions)
    drawHeader(
      ctx,
      dimensions,
      symbol,
      selectedTimeframe,
      chartType,
      chartSettings,
      candlestickData.length,
      visibleData.length,
      isConnected,
      lastUpdate
    )
  }, [candlestickData, chartSettings, crosshair, chartType, selectedTimeframe, showIndicators, isConnected, lastUpdate, loading, symbol, signals, positions, showSignals, showPositions, drawSignalMarkers, drawPositionOverlays])

  useEffect(() => {
    const animationFrame = requestAnimationFrame(drawChart)
    return () => cancelAnimationFrame(animationFrame)
  }, [drawChart])

  useEffect(() => {
    let resizeTimeout: NodeJS.Timeout
    const handleResize = () => {
      clearTimeout(resizeTimeout)
      resizeTimeout = setTimeout(drawChart, 100)
    }

    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      clearTimeout(resizeTimeout)
    }
  }, [drawChart])

  useEffect(() => {
    fetchCandlestickData()
  }, [selectedTimeframe, fetchCandlestickData])

  return (
    <div
      ref={containerRef}
      className={`relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden ${className}`}
    >
      <ChartControls
        isConnected={isConnected}
        chartType={chartType}
        setChartType={setChartType}
        chartSettings={chartSettings}
        setChartSettings={setChartSettings}
        showIndicators={showIndicators}
        setShowIndicators={setShowIndicators}
        loading={loading}
        onRefresh={fetchCandlestickData}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onResetView={handleResetView}
      />

      <TimeframeSelector selectedTimeframe={selectedTimeframe} setSelectedTimeframe={setSelectedTimeframe} />

      {/* Navigation Controls */}
      <div className="absolute bottom-4 left-4 z-30">
        <div className="bg-slate-800/95 border border-slate-600 rounded-lg p-2">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={handlePanLeft} className="text-xs px-2 py-1 text-gray-300 hover:text-white">
              <ChevronLeft className="w-4 h-4" />
            </Button>

            <div className="flex flex-col gap-2 px-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Historico Completo:</span>
                {candlestickData.length > 0 && (
                  <span className="text-xs text-cyan-400 font-mono">
                    {new Date(candlestickData[0].time).toLocaleDateString('es-CO')} -{' '}
                    {new Date(candlestickData[candlestickData.length - 1].time).toLocaleDateString('es-CO')}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Navegacion:</span>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, candlestickData.length - 50)}
                  value={chartSettings.viewRange.start}
                  onChange={(e) => {
                    const start = parseInt(e.target.value)
                    const range = chartSettings.viewRange.end - chartSettings.viewRange.start
                    setChartSettings((prev) => ({
                      ...prev,
                      viewRange: { start, end: Math.min(candlestickData.length, start + range) }
                    }))
                  }}
                  className="w-40 h-3 bg-slate-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                  style={{
                    background: `linear-gradient(to right, #00d4aa 0%, #00d4aa ${(chartSettings.viewRange.start / Math.max(1, candlestickData.length - 50)) * 100}%, #475569 ${(chartSettings.viewRange.start / Math.max(1, candlestickData.length - 50)) * 100}%, #475569 100%)`
                  }}
                />
                <span className="text-xs text-blue-400 font-mono bg-slate-800 px-2 py-1 rounded">
                  {candlestickData.length > 0 && chartSettings.viewRange.start < candlestickData.length
                    ? new Date(candlestickData[chartSettings.viewRange.start].time).toLocaleDateString('es-CO')
                    : 'N/A'}
                </span>
              </div>
              <div className="text-xs text-amber-400 bg-amber-500/20 px-2 py-1 rounded">
                {candlestickData.length} registros disponibles - Total en DB: {dbStats.totalRecords.toLocaleString()}
              </div>
            </div>

            <Button variant="ghost" size="sm" onClick={handlePanRight} className="text-xs px-2 py-1 text-gray-300 hover:text-white">
              <ChevronRight className="w-4 h-4" />
            </Button>

            <Button
              variant={chartSettings.autoScroll ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setChartSettings((prev) => ({ ...prev, autoScroll: !prev.autoScroll }))}
              className={`text-xs px-2 py-1 ${chartSettings.autoScroll ? 'bg-green-600 text-white' : 'text-gray-300 hover:text-white'}`}
            >
              {chartSettings.autoScroll ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
            </Button>
          </div>
        </div>
      </div>

      {error && (
        <div className="absolute top-20 left-4 right-4 z-20 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Error: {error}</span>
          </div>
        </div>
      )}

      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width: '100%', height: `${height}px`, cursor: 'crosshair' }}
        className="rounded-2xl"
      />

      <ChartTooltip hoverInfo={hoverInfo} chartType={chartType} showIndicators={showIndicators} />

      <ChartLegend showIndicators={showIndicators} showVolume={chartSettings.showVolume} />

      <ChartInfoPanel
        totalDataPoints={candlestickData.length}
        visibleDataPoints={chartSettings.viewRange.end - chartSettings.viewRange.start}
        zoomLevel={chartSettings.zoomLevel}
        selectedTimeframe={selectedTimeframe}
      />

      <style jsx>{`
        .slider-thumb::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #0ea5e9;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }

        .slider-thumb::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #0ea5e9;
          cursor: pointer;
          border: 2px solid #ffffff;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }
      `}</style>
    </div>
  )
}
