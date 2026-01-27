'use client'

/**
 * TradingChartWithSignals Component
 * ==================================
 *
 * Integrated trading chart with:
 * - Candlestick chart (real market data from 2024-02 onwards)
 * - Signal overlay (BUY/SELL markers)
 * - Period separators (Train/Val/Test/OOS)
 * - Day/Week grid lines
 * - Synchronized with TradesTable
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
import {
  createChart,
  createSeriesMarkers,
  IChartApi,
  ISeriesApi,
  ISeriesMarkersPluginApi,
  ColorType,
  CrosshairMode,
  SeriesMarker,
  Time,
  CandlestickSeries
} from 'lightweight-charts'
import { useIntegratedChart } from '@/hooks/useIntegratedChart'
import { TradingSignal, Position } from '@/types/trading'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Loader2,
  AlertCircle,
  Activity,
  Target,
  Shield
} from 'lucide-react'
import {
  TRAINING_PERIODS,
  PERIOD_COLORS,
  getDataType
} from '@/lib/config/training-periods'

interface TradingChartWithSignalsProps {
  symbol?: string
  timeframe?: string
  height?: number
  showSignals?: boolean
  showPositions?: boolean
  showStopLossTakeProfit?: boolean
  enableRealTime?: boolean
  className?: string
  startDate?: string  // Custom start date
  endDate?: Date      // Replay mode: end date for filtering candlesticks
  modelId?: string    // Model ID for filtering signals (ppo_v19_prod, ppo_v20_prod)
  isReplayMode?: boolean  // Whether replay mode is active
  replayVisibleTradeIds?: Set<string>  // Only show signals for these trade IDs during replay
  replayTrades?: Array<{  // Trades to display as signals during replay mode
    trade_id: number
    timestamp?: string
    entry_time?: string
    side: string
    entry_price: number
    pnl?: number
    status?: string
  }>
}

// Signal overlay position type
interface SignalOverlay {
  id: string
  x: number
  y: number
  type: 'BUY' | 'SELL'
  price: number
  confidence: number
  timestamp: string
  visible: boolean
}

// P1-6: Helper para color de marker basado en confidence
const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 90) return '#00C853'  // Alto - verde brillante
  if (confidence >= 75) return '#4CAF50'  // Bueno - verde
  if (confidence >= 60) return '#FFC107'  // Medio - amarillo
  return '#FF5722'                         // Bajo - naranja
}

export default function TradingChartWithSignals({
  symbol = 'USDCOP',
  timeframe = '5m',
  height = 700,
  showSignals = true,
  showPositions = true,
  showStopLossTakeProfit = true,
  enableRealTime = false,
  className = '',
  startDate = '2024-02-01',  // Default: start from just before validation
  endDate,                    // Replay mode: end date for filtering candlesticks
  modelId = 'ppo_v19_prod',  // Default: V19 production model
  isReplayMode = false,       // Whether replay mode is active
  replayVisibleTradeIds,       // Only show signals for these trade IDs during replay
  replayTrades = []           // Trades to display as signals during replay mode
}: TradingChartWithSignalsProps) {
  // Refs
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const markersPluginRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null)

  // Price lines for SL/TP and period separators
  const priceLineRefs = useRef<Map<string, any>>(new Map())

  // Signal overlays state for HTML-based markers
  const [signalOverlays, setSignalOverlays] = useState<SignalOverlay[]>([])

  // Local state
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [selectedSignal, setSelectedSignal] = useState<TradingSignal | null>(null)

  // Tooltip state for OHLCV hover
  const [tooltipData, setTooltipData] = useState<{
    visible: boolean
    x: number
    y: number
    time: number
    open: number
    high: number
    low: number
    close: number
    change: number
    changePct: number
  } | null>(null)

  // Integrated data hook
  const {
    candlestickData: rawCandlestickData,
    signals: rawSignals,
    positions,
    isLoading,
    hasError,
    errorMessage,
    refresh
  } = useIntegratedChart({
    symbol,
    timeframe,
    startDate,  // Pass custom start date to load historical data
    // Don't pass endDate to hook - we fetch all data and filter client-side for display
    enableSignals: showSignals && !isReplayMode,  // In replay mode, signals come from replayTrades
    enablePositions: showPositions,
    enableRealTime: enableRealTime && !isReplayMode,  // Disable real-time in replay mode
    refreshInterval: enableRealTime && !isReplayMode ? 5000 : 0,
    limit: 50000,  // Request more data to cover from 2024-02 to present
    modelId  // Pass model ID for filtering signals
  })

  // Filter candlestick data by endDate for replay mode
  // Note: API returns time in milliseconds, so we use milliseconds for comparison
  const candlestickData = React.useMemo(() => {
    if (!rawCandlestickData || rawCandlestickData.length === 0) {
      return []
    }
    if (!endDate) {
      return rawCandlestickData
    }
    const endTimestamp = endDate.getTime()
    return rawCandlestickData.filter((candle: any) => {
      const candleTime = typeof candle.time === 'number' ? candle.time : new Date(candle.time).getTime()
      return candleTime <= endTimestamp
    })
  }, [rawCandlestickData, endDate])

  // Filter signals by endDate and visible trade IDs for replay mode
  // In replay mode, convert replayTrades to signals format instead of using API signals
  const signals = React.useMemo(() => {
    // In replay mode with trades, use those directly as signals
    if (isReplayMode && replayTrades && replayTrades.length > 0) {
      return replayTrades.map((trade) => ({
        id: trade.trade_id,
        trade_id: trade.trade_id,
        timestamp: trade.timestamp || trade.entry_time || '',
        time: trade.timestamp || trade.entry_time || '',
        type: ['BUY', 'LONG'].includes((trade.side || '').toUpperCase()) ? 'BUY' : 'SELL',
        price: trade.entry_price,
        // P1-6 FIX: Use real confidence from trade data with proper fallback chain
        confidence: trade.entry_confidence
          ?? trade.confidence
          ?? trade.model_metadata?.confidence
          ?? 75,  // Fallback only when no confidence data available
        stopLoss: trade.stop_loss ?? null,
        takeProfit: trade.take_profit ?? null,
        modelVersion: trade.model_version ?? 'unknown',
        entropy: trade.model_metadata?.entropy ?? null,
        featuresSnapshot: trade.features_snapshot ?? null,
      }))
    }

    if (!rawSignals) return rawSignals

    let filtered = rawSignals

    // Filter by endDate if provided
    if (endDate) {
      filtered = filtered.filter((signal: any) => {
        const signalTime = new Date(signal.timestamp || signal.time)
        return signalTime <= endDate
      })
    }

    // In replay mode with visible trade IDs, only show signals for visible trades
    // This provides more precise control than just endDate filtering
    if (isReplayMode && replayVisibleTradeIds && replayVisibleTradeIds.size > 0) {
      filtered = filtered.filter((signal: any) => {
        const tradeId = String(signal.trade_id || signal.tradeId || signal.id)
        return replayVisibleTradeIds.has(tradeId)
      })
    }

    return filtered
  }, [rawSignals, endDate, isReplayMode, replayVisibleTradeIds, replayTrades])

  /**
   * Format timestamp to COT (Colombia Time) - Short format for axis
   *
   * IMPORTANT: Database has two timestamp formats:
   * - OLD FORMAT (before Dec 17, 2025): True UTC (13:00-17:55 UTC = 8:00-12:55 COT)
   * - NEW FORMAT (after Dec 17, 2025): COT stored as UTC offset (08:00-12:55 +00)
   *
   * Detection: If hour is 8-12, data is already COT. If hour is 13-17, data is UTC.
   */
  const formatTimeToCOT = (timestamp: number): string => {
    // timestamp is in seconds (Unix time)
    const rawDate = new Date(timestamp * 1000)
    const rawHour = rawDate.getUTCHours()

    // Detect format: 8-12 = COT format (no conversion), 13-17 = UTC format (subtract 5h)
    let cotDate: Date
    if (rawHour >= 13 && rawHour <= 17) {
      // OLD FORMAT: True UTC, convert to COT by subtracting 5 hours
      cotDate = new Date(rawDate.getTime() - 5 * 60 * 60 * 1000)
    } else {
      // NEW FORMAT: Already COT, no conversion needed
      cotDate = rawDate
    }

    const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    const day = cotDate.getUTCDate().toString().padStart(2, '0')
    const month = months[cotDate.getUTCMonth()]
    const hours = cotDate.getUTCHours().toString().padStart(2, '0')
    const mins = cotDate.getUTCMinutes().toString().padStart(2, '0')

    return `${day} ${month} ${hours}:${mins}`
  }

  /**
   * Format timestamp to COT - Full format for tooltip
   * Same logic as formatTimeToCOT for format detection
   */
  const formatTimeToCOTFull = (timestamp: number): string => {
    const rawDate = new Date(timestamp * 1000)
    const rawHour = rawDate.getUTCHours()

    // Detect format: 8-12 = COT format, 13-17 = UTC format
    let cotDate: Date
    if (rawHour >= 13 && rawHour <= 17) {
      cotDate = new Date(rawDate.getTime() - 5 * 60 * 60 * 1000)
    } else {
      cotDate = rawDate
    }

    const days = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
    const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

    const dayName = days[cotDate.getUTCDay()]
    const day = cotDate.getUTCDate().toString().padStart(2, '0')
    const month = months[cotDate.getUTCMonth()]
    const year = cotDate.getUTCFullYear()
    const hours = cotDate.getUTCHours().toString().padStart(2, '0')
    const mins = cotDate.getUTCMinutes().toString().padStart(2, '0')

    return `${dayName} ${day} ${month} ${year} • ${hours}:${mins} COT`
  }

  /**
   * Initialize chart
   */
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' },
        textColor: '#94a3b8'
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' }
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#475569', labelBackgroundColor: '#1e293b' },
        horzLine: { color: '#475569', labelBackgroundColor: '#1e293b' }
      },
      rightPriceScale: {
        borderColor: '#334155',
        scaleMargins: { top: 0.1, bottom: 0.2 }
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => formatTimeToCOT(time)
      },
      localization: {
        timeFormatter: (time: number) => formatTimeToCOT(time),
        locale: 'es-CO'
      },
      width: chartContainerRef.current.clientWidth,
      height: height
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444'
    })

    chartRef.current = chart
    candleSeriesRef.current = candleSeries

    // Crosshair move handler for tooltip
    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
        setTooltipData(null)
        return
      }

      const data = param.seriesData.get(candleSeries)
      if (data && 'open' in data) {
        const ohlc = data as { open: number; high: number; low: number; close: number }
        const change = ohlc.close - ohlc.open
        const changePct = (change / ohlc.open) * 100

        setTooltipData({
          visible: true,
          x: param.point.x,
          y: param.point.y,
          time: param.time as number,
          open: ohlc.open,
          high: ohlc.high,
          low: ohlc.low,
          close: ohlc.close,
          change,
          changePct
        })
      }
    })

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      markersPluginRef.current = null
      chart.remove()
    }
  }, [height])

  /**
   * Update candlestick data with period-based coloring
   */
  useEffect(() => {
    if (!candleSeriesRef.current || candlestickData.length === 0) return

    // Period boundary timestamps for coloring
    const valStartTs = TRAINING_PERIODS.VAL_START.getTime()
    const testStartTs = TRAINING_PERIODS.TEST_START.getTime()

    const formattedData = candlestickData
      .map((d) => {
        const candleTs = new Date(d.time).getTime()
        const isUp = d.close >= d.open

        // Determine period-based colors (subtle tint)
        let upColor = '#22c55e'      // Default green
        let downColor = '#ef4444'    // Default red
        let borderUp = '#22c55e'
        let borderDown = '#ef4444'
        let wickUp = '#22c55e'
        let wickDown = '#ef4444'

        if (candleTs < valStartTs) {
          // Train period - slight blue tint
          upColor = '#3B82F6'
          downColor = '#60A5FA'
          borderUp = '#3B82F6'
          borderDown = '#60A5FA'
          wickUp = '#3B82F6'
          wickDown = '#60A5FA'
        } else if (candleTs < testStartTs) {
          // Validation period - slight purple tint
          upColor = '#8B5CF6'
          downColor = '#A78BFA'
          borderUp = '#8B5CF6'
          borderDown = '#A78BFA'
          wickUp = '#8B5CF6'
          wickDown = '#A78BFA'
        }
        // Test/OOS period uses default green/red

        return {
          time: Math.floor(candleTs / 1000) as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
          color: isUp ? upColor : downColor,
          borderColor: isUp ? borderUp : borderDown,
          wickColor: isUp ? wickUp : wickDown
        }
      })
      // Sort by time ascending (required by lightweight-charts v5+)
      .sort((a, b) => (a.time as number) - (b.time as number))
      // Remove duplicates (keep last occurrence)
      .filter((item, index, arr) =>
        index === arr.length - 1 || item.time !== arr[index + 1].time
      )

    if (formattedData.length > 0) {
      candleSeriesRef.current.setData(formattedData)
      chartRef.current?.timeScale().fitContent()
    }
  }, [candlestickData])

  /**
   * Update signal markers - snap to nearest candlestick timestamp
   */
  useEffect(() => {
    const series = candleSeriesRef.current
    if (!series || !showSignals || signals.length === 0 || candlestickData.length === 0) return

    // Create a set of valid candlestick timestamps (in seconds)
    const candleTimestamps = new Set(
      candlestickData.map(c => Math.floor(new Date(c.time).getTime() / 1000))
    )

    // Find nearest candlestick timestamp for a signal (with timezone offset handling)
    // OHLCV timestamps: stored as COT (8am-12:55pm) but labeled as UTC
    // Trade timestamps: stored as actual UTC (13:00-17:55 UTC = 8am-12:55pm COT)
    // To match: subtract 5 hours from trade timestamp to align with OHLCV
    const findNearestCandleTime = (rawTs: number): number | null => {
      const candleTimes = Array.from(candleTimestamps).sort((a, b) => a - b)
      // Try raw, -5h (UTC trade to COT OHLCV), and +5h for edge cases
      const tsOptions = [rawTs, rawTs - 18000, rawTs + 18000] // 5 hours offset both directions

      let bestNearest = candleTimes[0]
      let bestDiff = Infinity

      for (const signalTs of tsOptions) {
        for (const ct of candleTimes) {
          const diff = Math.abs(signalTs - ct)
          if (diff < bestDiff) {
            bestDiff = diff
            bestNearest = ct
          }
        }
      }

      // In replay mode with synthetic trades, use wider tolerance
      const maxDiff = isReplayMode ? 86400 : 600
      return bestDiff <= maxDiff ? bestNearest : null
    }

    const markers: SeriesMarker<Time>[] = signals
      .filter((s) => s.type !== 'HOLD')
      .map((signal) => {
        const rawSignalTs = Math.floor(new Date(signal.timestamp).getTime() / 1000)
        const snappedTs = findNearestCandleTime(rawSignalTs)

        if (snappedTs === null) return null

        const isBuy = signal.type === 'BUY'

        return {
          time: snappedTs as Time,
          position: isBuy ? 'belowBar' : 'aboveBar',
          color: isBuy ? '#22c55e' : '#ef4444',
          shape: isBuy ? 'arrowUp' : 'arrowDown',
          text: `${signal.type} ${signal.confidence.toFixed(0)}%`,
          size: 1.5
        }
      })
      .filter((m): m is SeriesMarker<Time> => m !== null)

    // Sort markers by time (required by lightweight-charts)
    markers.sort((a, b) => (a.time as number) - (b.time as number))

    console.log(`[TradingChart] Setting ${markers.length} signal markers from ${signals.length} signals (isReplayMode=${isReplayMode}, replayTrades=${replayTrades?.length || 0})`)

    try {
      // lightweight-charts v5: use createSeriesMarkers plugin API
      if (!markersPluginRef.current) {
        markersPluginRef.current = createSeriesMarkers(series, markers) as ISeriesMarkersPluginApi<Time>
      } else {
        markersPluginRef.current.setMarkers(markers)
      }
    } catch (err) {
      // Fallback: try legacy series.setMarkers (v3/v4)
      try {
        (series as any).setMarkers(markers)
      } catch {
        console.warn('[TradingChart] Error setting markers (using HTML overlays instead):', err)
      }
    }
  }, [signals, showSignals, candlestickData, isReplayMode, replayTrades])

  /**
   * Calculate HTML overlay positions for signals
   * Uses chart coordinate conversion to position overlays on top of chart
   */
  useEffect(() => {
    const chart = chartRef.current
    const series = candleSeriesRef.current
    if (!chart || !series || !showSignals || signals.length === 0 || candlestickData.length === 0) {
      setSignalOverlays([])
      return
    }

    // Function to calculate overlay positions
    const updateOverlayPositions = () => {
      const overlays: SignalOverlay[] = []
      const visibleRange = chart.timeScale().getVisibleRange()

      if (!visibleRange) return

      // Create a map of candlestick data for quick lookup
      const candleMap = new Map<number, { high: number; low: number }>()
      candlestickData.forEach(c => {
        const ts = Math.floor(new Date(c.time).getTime() / 1000)
        candleMap.set(ts, { high: c.high, low: c.low })
      })

      signals
        .filter((s) => s.type !== 'HOLD')
        .forEach((signal, idx) => {
          // Signal timestamps: stored as actual UTC (13:00-17:55 UTC = 8am-12:55pm COT)
          // OHLCV timestamps: stored as COT (8am-12:55pm) but labeled as UTC
          // To match: subtract 5 hours from signal timestamp to align with OHLCV
          const rawSignalTs = Math.floor(new Date(signal.timestamp).getTime() / 1000)
          // Try raw, -5h (UTC signal to COT OHLCV), and +5h for edge cases
          const signalTsOptions = [rawSignalTs, rawSignalTs - 18000, rawSignalTs + 18000]

          let bestMatch = { ts: 0, diff: Infinity, candleTs: 0 }

          // Find nearest candlestick for any timestamp interpretation
          const candleTimes = Array.from(candleMap.keys()).sort((a, b) => a - b)

          for (const signalTs of signalTsOptions) {
            for (const ct of candleTimes) {
              const diff = Math.abs(signalTs - ct)
              if (diff < bestMatch.diff) {
                bestMatch = { ts: signalTs, diff, candleTs: ct }
              }
            }
          }

          // Only show if within tolerance of a candlestick
          // In replay mode with synthetic trades, use wider tolerance (1 day = 86400s)
          const maxDiff = isReplayMode ? 86400 : 600
          if (bestMatch.diff > maxDiff) return

          const nearestTs = bestMatch.candleTs

          const candle = candleMap.get(nearestTs)
          if (!candle) return

          // Get x coordinate from time
          const x = chart.timeScale().timeToCoordinate(nearestTs as Time)
          if (x === null) return

          // Get y coordinate from price (use high for SELL, low for BUY)
          const isBuy = signal.type === 'BUY'
          const priceForPosition = isBuy ? candle.low : candle.high
          const y = series.priceToCoordinate(priceForPosition)
          if (y === null) return

          overlays.push({
            id: `signal-${idx}-${nearestTs}`,
            x: x,
            y: isBuy ? y + 5 : y - 5, // Offset slightly from candle
            type: signal.type as 'BUY' | 'SELL',
            price: signal.price,
            confidence: signal.confidence,
            timestamp: signal.timestamp,
            visible: true
          })
        })

      console.log(`[TradingChart] Calculated ${overlays.length} HTML overlays from ${signals.length} signals (isReplayMode=${isReplayMode})`)
      setSignalOverlays(overlays)
    }

    // Initial calculation
    updateOverlayPositions()

    // Subscribe to visible range changes
    const handleVisibleRangeChange = () => {
      requestAnimationFrame(updateOverlayPositions)
    }

    chart.timeScale().subscribeVisibleTimeRangeChange(handleVisibleRangeChange)

    // Also update on resize
    const resizeObserver = new ResizeObserver(() => {
      requestAnimationFrame(updateOverlayPositions)
    })

    if (chartContainerRef.current) {
      resizeObserver.observe(chartContainerRef.current)
    }

    return () => {
      chart.timeScale().unsubscribeVisibleTimeRangeChange(handleVisibleRangeChange)
      resizeObserver.disconnect()
    }
  }, [signals, showSignals, candlestickData])

  /**
   * Update Stop Loss / Take Profit lines
   */
  useEffect(() => {
    if (!candleSeriesRef.current || !showStopLossTakeProfit) return

    // Clear existing price lines
    priceLineRefs.current.forEach((line) => {
      candleSeriesRef.current?.removePriceLine(line)
    })
    priceLineRefs.current.clear()

    // Add SL/TP lines for visible signals
    signals.forEach((signal, idx) => {
      if (signal.stopLoss) {
        const slLine = candleSeriesRef.current!.createPriceLine({
          price: signal.stopLoss,
          color: '#ef4444',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: `SL ${signal.stopLoss.toFixed(2)}`
        })
        priceLineRefs.current.set(`sl-${idx}`, slLine)
      }

      if (signal.takeProfit) {
        const tpLine = candleSeriesRef.current!.createPriceLine({
          price: signal.takeProfit,
          color: '#22c55e',
          lineWidth: 1,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: `TP ${signal.takeProfit.toFixed(2)}`
        })
        priceLineRefs.current.set(`tp-${idx}`, tpLine)
      }
    })
  }, [signals, showStopLossTakeProfit])

  /**
   * Handle zoom controls
   */
  const handleZoomIn = useCallback(() => {
    chartRef.current?.timeScale().scrollPosition()
    // Implement zoom logic
  }, [])

  const handleZoomOut = useCallback(() => {
    chartRef.current?.timeScale().fitContent()
  }, [])

  /**
   * Stats calculations
   */
  const stats = React.useMemo(() => {
    const activeSignals = signals.filter((s) => s.type !== 'HOLD').length
    const buySignals = signals.filter((s) => s.type === 'BUY').length
    const sellSignals = signals.filter((s) => s.type === 'SELL').length
    const avgConfidence = signals.length > 0
      ? signals.reduce((acc, s) => acc + s.confidence, 0) / signals.length
      : 0

    const openPositions = positions.filter((p) => p.status === 'open').length
    const totalPnl = positions.reduce((acc, p) => acc + (p.realizedPnl || 0) + (p.unrealizedPnl || 0), 0)

    return {
      activeSignals,
      buySignals,
      sellSignals,
      avgConfidence,
      openPositions,
      totalPnl
    }
  }, [signals, positions])

  return (
    <div
      className={`relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden ${className} ${
        isFullscreen ? 'fixed inset-0 z-50' : ''
      }`}
    >
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                {symbol}
                <Badge variant="outline" className="text-xs">
                  {timeframe}
                </Badge>
                <Badge variant="outline" className="text-[10px] bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
                  COT (UTC-5)
                </Badge>
                {isReplayMode && (
                  <Badge variant="outline" className="text-[10px] bg-cyan-500/20 text-cyan-400 border-cyan-500/40 animate-pulse">
                    REPLAY
                  </Badge>
                )}
              </h3>
              <p className="text-sm text-slate-400">Horario Colombia • Lun-Vie 8:00-12:55</p>
            </div>

            {candlestickData.length > 0 && (
              <div className="flex items-center gap-3">
                <div className="text-2xl font-mono text-white">
                  ${candlestickData[candlestickData.length - 1].close.toFixed(2)}
                </div>
                {enableRealTime && (
                  <Badge variant="default" className="bg-green-600 animate-pulse">
                    <Activity className="w-3 h-3 mr-1" />
                    LIVE
                  </Badge>
                )}
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={refresh} disabled={isLoading}>
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button variant="ghost" size="sm" onClick={handleZoomIn}>
              <ZoomIn className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={handleZoomOut}>
              <ZoomOut className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setIsFullscreen(!isFullscreen)}>
              <Maximize2 className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="px-6 py-3 bg-slate-800/30 border-b border-slate-700/30">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-6">
            {showSignals && (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Signals:</span>
                  <span className="font-mono text-white">{stats.activeSignals}</span>
                  <span className="text-green-400">↑{stats.buySignals}</span>
                  <span className="text-red-400">↓{stats.sellSignals}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Avg Confidence:</span>
                  <span className="font-mono text-cyan-400">{stats.avgConfidence.toFixed(1)}%</span>
                </div>
              </>
            )}
            {showPositions && (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Open Positions:</span>
                  <span className="font-mono text-white">{stats.openPositions}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400">Total P&L:</span>
                  <span
                    className={`font-mono font-bold ${
                      stats.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    ${stats.totalPnl.toFixed(2)}
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="text-xs text-slate-500">
            {candlestickData.length.toLocaleString()} data points loaded
          </div>
        </div>
      </div>

      {/* Error Display */}
      {hasError && (
        <div className="mx-6 mt-4 bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span className="text-red-400 text-sm">{errorMessage}</span>
        </div>
      )}

      {/* Loading Overlay */}
      {isLoading && candlestickData.length === 0 && (
        <div className="flex items-center justify-center py-20">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
            <span className="text-slate-400">Loading chart data...</span>
          </div>
        </div>
      )}

      {/* Chart Container with Signal Overlays */}
      <div className="relative w-full">
        <div ref={chartContainerRef} className="w-full" />

        {/* OHLCV Tooltip */}
        {tooltipData && tooltipData.visible && (
          <div
            className="absolute z-20 pointer-events-none bg-slate-900/95 border border-slate-600 rounded-lg shadow-xl backdrop-blur-sm px-4 py-3 min-w-[220px]"
            style={{
              left: Math.min(tooltipData.x + 15, (chartContainerRef.current?.clientWidth || 800) - 240),
              top: Math.max(tooltipData.y - 100, 10),
            }}
          >
            {/* Date Header */}
            <div className="text-cyan-400 font-semibold text-sm mb-2 pb-2 border-b border-slate-700">
              {formatTimeToCOTFull(tooltipData.time)}
            </div>

            {/* OHLCV Values */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">Open:</span>
                <span className="font-mono text-white">${tooltipData.open.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">High:</span>
                <span className="font-mono text-green-400">${tooltipData.high.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Low:</span>
                <span className="font-mono text-red-400">${tooltipData.low.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Close:</span>
                <span className="font-mono text-white">${tooltipData.close.toFixed(2)}</span>
              </div>
            </div>

            {/* Change indicator */}
            <div className="mt-2 pt-2 border-t border-slate-700 flex justify-between items-center text-xs">
              <span className="text-slate-400">Cambio:</span>
              <span className={`font-mono font-bold ${tooltipData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {tooltipData.change >= 0 ? '+' : ''}{tooltipData.change.toFixed(2)} ({tooltipData.changePct >= 0 ? '+' : ''}{tooltipData.changePct.toFixed(3)}%)
              </span>
            </div>
          </div>
        )}

        {/* HTML Signal Overlays - Positioned absolutely over the chart */}
        {signalOverlays.map((overlay) => (
          overlay.visible && (
            <div
              key={overlay.id}
              className={`absolute z-10 transform -translate-x-1/2 pointer-events-auto cursor-pointer transition-all duration-150 hover:scale-125 ${
                overlay.type === 'BUY' ? '-translate-y-full' : 'translate-y-0'
              }`}
              style={{
                left: `${overlay.x}px`,
                top: `${overlay.y}px`,
              }}
              title={`${overlay.type} @ ${overlay.price.toFixed(2)} (${overlay.confidence.toFixed(0)}%)\n${new Date(overlay.timestamp).toLocaleString()}`}
            >
              {overlay.type === 'BUY' ? (
                <div className="flex flex-col items-center">
                  <div className="w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-b-[14px] border-b-green-500 drop-shadow-[0_0_6px_rgba(34,197,94,0.8)]" />
                  <div className="bg-green-500/90 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-sm mt-0.5 shadow-lg">
                    BUY
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <div className="bg-red-500/90 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-sm mb-0.5 shadow-lg">
                    SELL
                  </div>
                  <div className="w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-t-[14px] border-t-red-500 drop-shadow-[0_0_6px_rgba(239,68,68,0.8)]" />
                </div>
              )}
            </div>
          )
        ))}
      </div>


      {/* Legend */}
      <div className="px-6 py-3 bg-slate-800/20 border-t border-slate-700/30">
        <div className="flex flex-wrap items-center gap-6 text-xs">
          {/* Period Colors Legend */}
          <div className="flex items-center gap-4 border-r border-slate-700/50 pr-4">
            <span className="text-slate-500 font-medium">Periods:</span>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-blue-500" />
              <span className="text-slate-400">Train</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-purple-500" />
              <span className="text-slate-400">Val</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-green-500" />
              <span className="text-slate-400">Test/OOS</span>
            </div>
          </div>

          {/* Signal Icons Legend */}
          {showSignals && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <TrendingUp className="w-3 h-3 text-green-500" />
                <span className="text-slate-400">BUY</span>
              </div>
              <div className="flex items-center gap-1.5">
                <TrendingDown className="w-3 h-3 text-red-500" />
                <span className="text-slate-400">SELL</span>
              </div>
            </div>
          )}
          {showStopLossTakeProfit && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                <Shield className="w-3 h-3 text-red-500" />
                <span className="text-slate-400">SL</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Target className="w-3 h-3 text-green-500" />
                <span className="text-slate-400">TP</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
