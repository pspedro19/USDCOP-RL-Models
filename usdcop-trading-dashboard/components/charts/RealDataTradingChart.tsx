'use client'

/**
 * Ultra-Professional Trading Chart - Superior to Bloomberg & TradingView
 * ==================================================================
 *
 * Características avanzadas:
 * - Velas japonesas perfectamente visibles con sombras 3D
 * - Navegación completa del histórico con slider profesional
 * - Zoom y pan superior a TradingView
 * - Controles visuales mejorados
 * - Barras de volumen integradas
 * - Crosshair profesional con valores en tiempo real
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { MarketDataService, CandlestickResponse, CandlestickData } from '@/lib/services/market-data-service'
import { useRealTimePrice } from '@/hooks/useRealTimePrice'
import { useDbStats } from '@/hooks/useDbStats'
import {
  Activity, BarChart3, TrendingUp, Eye, EyeOff, RefreshCw, AlertCircle,
  Clock, ZoomIn, ZoomOut, Move, RotateCcw, Maximize2, Volume2,
  ChevronLeft, ChevronRight, Play, Pause, Settings
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

type ChartType = 'candlestick' | 'line'
type TimeframePeriod = '5m' | '15m' | '30m' | '1h' | '1d'

interface ChartSettings {
  viewRange: { start: number; end: number }
  zoomLevel: number
  candleWidth: number
  showVolume: boolean
  showCrosshair: boolean
  autoScroll: boolean
}

interface CrosshairInfo {
  x: number
  y: number
  priceY: number
  timeIndex: number
  price: number
  visible: boolean
}

interface HoverInfo {
  x: number
  y: number
  candle: CandlestickData & { indicators?: any }
  index: number
  visible: boolean
}

interface RealDataTradingChartProps {
  symbol?: string
  timeframe?: string
  height?: number
  className?: string
}

export default function RealDataTradingChart({
  symbol = 'USDCOP',
  timeframe = '5m',
  height = 700,
  className = ''
}: RealDataTradingChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [candlestickData, setCandlestickData] = useState<CandlestickData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showIndicators, setShowIndicators] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [chartType, setChartType] = useState<ChartType>('candlestick')
  const [selectedTimeframe, setSelectedTimeframe] = useState<TimeframePeriod>(timeframe as TimeframePeriod)
  const [hoverInfo, setHoverInfo] = useState<HoverInfo>({ x: 0, y: 0, candle: {} as any, index: 0, visible: false })
  const [crosshair, setCrosshair] = useState<CrosshairInfo>({ x: 0, y: 0, priceY: 0, timeIndex: 0, price: 0, visible: false })

  // Get DB stats for dynamic record count
  const { stats: dbStats } = useDbStats(60000) // Refresh every 60 seconds

  // Advanced chart settings
  const [chartSettings, setChartSettings] = useState<ChartSettings>({
    viewRange: { start: 0, end: 100 },
    zoomLevel: 1,
    candleWidth: 8,
    showVolume: true,
    showCrosshair: true,
    autoScroll: true
  })

  // Real-time price updates
  const { currentPrice, isConnected, formattedPrice } = useRealTimePrice(symbol)

  // Fetch more historical data for better navigation
  const fetchCandlestickData = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      // Fetch all available historical data including from 2020
      const response = await MarketDataService.getCandlestickData(
        symbol,
        selectedTimeframe,
        '2020-01-01',
        '2025-12-31',
        dbStats.totalRecords || 100000, // Dynamic: Get from API health endpoint
        showIndicators
      )

      console.log(`[Ultra Chart] Fetched ${response.count} candlesticks for complete historical view`)

      setCandlestickData(response.data)
      setLastUpdate(new Date())

      // Auto-scroll to latest data
      if (chartSettings.autoScroll && response.data.length > 100) {
        setChartSettings(prev => ({
          ...prev,
          viewRange: { start: response.data.length - 100, end: response.data.length }
        }))
      }
    } catch (err) {
      console.error('Error fetching candlestick data:', err)
      setError(err instanceof Error ? err.message : 'Error fetching data')
    } finally {
      setLoading(false)
    }
  }, [symbol, selectedTimeframe, showIndicators, chartSettings.autoScroll])

  // Initial data load
  useEffect(() => {
    fetchCandlestickData()
  }, [fetchCandlestickData])

  // Advanced mouse handling for crosshair and navigation
  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas || candlestickData.length === 0) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Chart dimensions
    const chartMargin = { top: 40, right: 150, bottom: 80, left: 80 }
    const chartWidth = rect.width - chartMargin.left - chartMargin.right
    const chartHeight = rect.height - chartMargin.top - chartMargin.bottom

    // Visible data range
    const visibleData = candlestickData.slice(chartSettings.viewRange.start, chartSettings.viewRange.end)
    if (visibleData.length === 0) return

    // Calculate position
    const relativeX = x - chartMargin.left
    if (relativeX >= 0 && relativeX <= chartWidth && y >= chartMargin.top && y <= rect.height - chartMargin.bottom) {
      const index = Math.round((relativeX / chartWidth) * (visibleData.length - 1))
      const actualIndex = chartSettings.viewRange.start + index

      if (index >= 0 && index < visibleData.length && actualIndex < candlestickData.length) {
        const candle = candlestickData[actualIndex]

        // Price calculation for crosshair
        const prices = visibleData.flatMap(d => [d.high, d.low])
        const minPrice = Math.min(...prices)
        const maxPrice = Math.max(...prices)
        const priceRange = maxPrice - minPrice
        const price = maxPrice - ((y - chartMargin.top) / chartHeight) * priceRange

        // Update crosshair
        setCrosshair({
          x,
          y,
          priceY: y,
          timeIndex: actualIndex,
          price,
          visible: chartSettings.showCrosshair
        })

        // Update hover info
        setHoverInfo({
          x: event.clientX,
          y: event.clientY,
          candle,
          index: actualIndex,
          visible: true
        })
      }
    } else {
      setCrosshair(prev => ({ ...prev, visible: false }))
      setHoverInfo(prev => ({ ...prev, visible: false }))
    }
  }, [candlestickData, chartSettings])

  const handleMouseLeave = useCallback(() => {
    setCrosshair(prev => ({ ...prev, visible: false }))
    setHoverInfo(prev => ({ ...prev, visible: false }))
  }, [])

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    setChartSettings(prev => {
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
    setChartSettings(prev => {
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

  // Navigation controls
  const handlePanLeft = useCallback(() => {
    setChartSettings(prev => {
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
    setChartSettings(prev => {
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
    setChartSettings(prev => ({
      ...prev,
      viewRange: { start: Math.max(0, candlestickData.length - 100), end: candlestickData.length },
      zoomLevel: 1,
      candleWidth: 8
    }))
  }, [candlestickData.length])

  // Ultra-professional chart drawing
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // High-DPI setup
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    // Ultra-professional background
    const bgGradient = ctx.createLinearGradient(0, 0, 0, rect.height)
    bgGradient.addColorStop(0, '#0a0e1a')
    bgGradient.addColorStop(0.3, '#1a1f2e')
    bgGradient.addColorStop(0.7, '#2a2f3e')
    bgGradient.addColorStop(1, '#1a1f2e')
    ctx.fillStyle = bgGradient
    ctx.fillRect(0, 0, rect.width, rect.height)

    if (candlestickData.length === 0) {
      // Ultra-professional loading
      const centerX = rect.width / 2
      const centerY = rect.height / 2

      if (loading) {
        // Bloomberg-style loading animation
        ctx.strokeStyle = '#00d4aa'
        ctx.lineWidth = 4
        const time = Date.now() * 0.002

        for (let i = 0; i < 8; i++) {
          const angle = (i / 8) * Math.PI * 2 + time
          const radius = 30 + Math.sin(time + i) * 10
          ctx.globalAlpha = 0.3 + 0.7 * ((Math.sin(time + i) + 1) / 2)

          ctx.beginPath()
          ctx.arc(
            centerX + Math.cos(angle) * radius,
            centerY + Math.sin(angle) * radius,
            8, 0, Math.PI * 2
          )
          ctx.stroke()
        }
        ctx.globalAlpha = 1

        // Professional loading text
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 24px Inter'
        ctx.textAlign = 'center'
        ctx.fillText('🚀 Cargando Datos Históricos Completos', centerX, centerY + 80)

        ctx.fillStyle = '#00d4aa'
        ctx.font = '16px Inter'
        ctx.fillText(`${selectedTimeframe.toUpperCase()} • ${chartType === 'candlestick' ? 'Velas Japonesas' : 'Línea Suavizada'}`, centerX, centerY + 110)

        // Progress indication
        ctx.fillStyle = 'rgba(0, 212, 170, 0.2)'
        ctx.fillRect(centerX - 150, centerY + 130, 300, 6)
        ctx.fillStyle = '#00d4aa'
        ctx.fillRect(centerX - 150, centerY + 130, 300 * ((Date.now() % 3000) / 3000), 6)
      }
      return
    }

    // Chart dimensions
    const chartMargin = { top: 40, right: 150, bottom: 80, left: 80 }
    const chartWidth = rect.width - chartMargin.left - chartMargin.right
    const mainChartHeight = chartSettings.showVolume ? (rect.height - chartMargin.top - chartMargin.bottom) * 0.75 : rect.height - chartMargin.top - chartMargin.bottom
    const volumeHeight = chartSettings.showVolume ? (rect.height - chartMargin.top - chartMargin.bottom) * 0.25 : 0

    // Visible data
    const visibleData = candlestickData.slice(chartSettings.viewRange.start, chartSettings.viewRange.end)
    if (visibleData.length === 0) return

    // Price range calculation
    const prices = visibleData.flatMap(d => [d.high, d.low])
    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)
    const priceRange = maxPrice - minPrice
    const padding = priceRange * 0.05

    // Scale functions
    const xScale = (index: number) => chartMargin.left + (index / Math.max(1, visibleData.length - 1)) * chartWidth
    const yScale = (price: number) => chartMargin.top + (1 - (price - minPrice + padding) / (priceRange + 2 * padding)) * mainChartHeight

    // Ultra-professional grid
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.08)'
    ctx.lineWidth = 0.5

    // Horizontal grid (price levels)
    const priceSteps = 12
    for (let i = 0; i <= priceSteps; i++) {
      const price = minPrice + (priceRange * i) / priceSteps
      const y = yScale(price)

      // Grid line with gradient
      const gridGradient = ctx.createLinearGradient(chartMargin.left, y, rect.width - chartMargin.right, y)
      gridGradient.addColorStop(0, 'rgba(100, 116, 139, 0.02)')
      gridGradient.addColorStop(0.5, 'rgba(100, 116, 139, 0.15)')
      gridGradient.addColorStop(1, 'rgba(100, 116, 139, 0.02)')

      ctx.strokeStyle = gridGradient
      ctx.beginPath()
      ctx.moveTo(chartMargin.left, y)
      ctx.lineTo(rect.width - chartMargin.right, y)
      ctx.stroke()

      // Professional price labels
      ctx.fillStyle = 'rgba(15, 23, 42, 0.95)'
      ctx.fillRect(rect.width - chartMargin.right + 5, y - 12, 100, 24)

      // Price label with gradient
      const priceGradient = ctx.createLinearGradient(0, y - 10, 0, y + 10)
      priceGradient.addColorStop(0, '#00d4aa')
      priceGradient.addColorStop(1, '#00b894')
      ctx.fillStyle = priceGradient
      ctx.font = 'bold 12px Inter'
      ctx.textAlign = 'left'
      ctx.fillText(`$${price.toFixed(2)}`, rect.width - chartMargin.right + 10, y + 4)
    }

    // Vertical grid (time)
    const timeSteps = Math.min(12, visibleData.length)
    for (let i = 0; i <= timeSteps; i++) {
      const x = chartMargin.left + (i / timeSteps) * chartWidth

      ctx.strokeStyle = 'rgba(100, 116, 139, 0.08)'
      ctx.beginPath()
      ctx.moveTo(x, chartMargin.top)
      ctx.lineTo(x, chartMargin.top + mainChartHeight)
      ctx.stroke()

      // Time labels
      if (i < visibleData.length) {
        const dataIndex = Math.floor((i / timeSteps) * (visibleData.length - 1))
        const timestamp = new Date(visibleData[dataIndex].time)

        ctx.fillStyle = 'rgba(15, 23, 42, 0.95)'
        ctx.fillRect(x - 40, rect.height - chartMargin.bottom + 10, 80, 20)

        ctx.fillStyle = '#64748b'
        ctx.font = 'bold 11px Inter'
        ctx.textAlign = 'center'

        const timeLabel = selectedTimeframe === '1d'
          ? timestamp.toLocaleDateString('es-CO', { month: 'short', day: 'numeric' })
          : timestamp.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' })

        ctx.fillText(timeLabel, x, rect.height - chartMargin.bottom + 23)
      }
    }

    // Ultra-professional candlesticks
    if (chartType === 'candlestick') {
      visibleData.forEach((candle, index) => {
        const x = xScale(index)
        // Candlestick width adaptativo y más ancho
        const candleWidth = Math.max(4, Math.min(chartSettings.candleWidth, chartWidth / visibleData.length * 0.8))

        const openY = yScale(candle.open)
        const closeY = yScale(candle.close)
        const highY = yScale(candle.high)
        const lowY = yScale(candle.low)

        const isGreen = candle.close >= candle.open
        const baseColor = isGreen ? '#00d4aa' : '#ff4757'
        const bodyColor = isGreen ? '#00b894' : '#e84393'

        // Professional wick with multiple gradients
        const wickGradient = ctx.createLinearGradient(0, highY, 0, lowY)
        wickGradient.addColorStop(0, baseColor)
        wickGradient.addColorStop(0.5, `${baseColor}CC`)
        wickGradient.addColorStop(1, `${baseColor}88`)

        ctx.strokeStyle = wickGradient
        ctx.lineWidth = Math.max(1.5, candleWidth * 0.2)
        ctx.beginPath()
        ctx.moveTo(x, highY)
        ctx.lineTo(x, lowY)
        ctx.stroke()

        // Professional body with 3D effect
        const bodyHeight = Math.abs(closeY - openY)
        const bodyY = Math.min(openY, closeY)

        // 3D shadow effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
        ctx.fillRect(x - candleWidth / 2 + 2, bodyY + 2, candleWidth, Math.max(2, bodyHeight))

        // Main body with gradient
        const bodyGradient = ctx.createLinearGradient(x - candleWidth / 2, bodyY, x + candleWidth / 2, bodyY)
        if (isGreen) {
          bodyGradient.addColorStop(0, '#00b894')
          bodyGradient.addColorStop(0.5, '#00d4aa')
          bodyGradient.addColorStop(1, '#00b894')
        } else {
          bodyGradient.addColorStop(0, '#e84393')
          bodyGradient.addColorStop(0.5, '#ff4757')
          bodyGradient.addColorStop(1, '#e84393')
        }

        ctx.fillStyle = bodyGradient
        ctx.fillRect(x - candleWidth / 2, bodyY, candleWidth, Math.max(2, bodyHeight))

        // Border with glow effect
        ctx.shadowBlur = 3
        ctx.shadowColor = baseColor
        ctx.strokeStyle = baseColor
        ctx.lineWidth = 1
        ctx.strokeRect(x - candleWidth / 2, bodyY, candleWidth, Math.max(2, bodyHeight))
        ctx.shadowBlur = 0

        // High/Low markers for better visibility
        if (candleWidth > 6) {
          ctx.fillStyle = baseColor
          ctx.beginPath()
          ctx.arc(x, highY, 1.5, 0, 2 * Math.PI)
          ctx.fill()
          ctx.beginPath()
          ctx.arc(x, lowY, 1.5, 0, 2 * Math.PI)
          ctx.fill()
        }
      })
    } else {
      // Ultra-professional line chart
      // Gradient fill
      const fillGradient = ctx.createLinearGradient(0, chartMargin.top, 0, chartMargin.top + mainChartHeight)
      fillGradient.addColorStop(0, 'rgba(0, 212, 170, 0.4)')
      fillGradient.addColorStop(0.3, 'rgba(0, 212, 170, 0.2)')
      fillGradient.addColorStop(0.7, 'rgba(0, 212, 170, 0.1)')
      fillGradient.addColorStop(1, 'rgba(0, 212, 170, 0.02)')

      ctx.beginPath()
      visibleData.forEach((candle, index) => {
        const x = xScale(index)
        const y = yScale(candle.close)

        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      // Fill area
      ctx.lineTo(xScale(visibleData.length - 1), chartMargin.top + mainChartHeight)
      ctx.lineTo(chartMargin.left, chartMargin.top + mainChartHeight)
      ctx.closePath()
      ctx.fillStyle = fillGradient
      ctx.fill()

      // Main line with glow effect
      ctx.beginPath()
      visibleData.forEach((candle, index) => {
        const x = xScale(index)
        const y = yScale(candle.close)

        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      // Glow effect
      ctx.shadowBlur = 8
      ctx.shadowColor = '#00d4aa'
      ctx.strokeStyle = '#00d4aa'
      ctx.lineWidth = 3
      ctx.stroke()
      ctx.shadowBlur = 0

      // Data points
      visibleData.forEach((candle, index) => {
        const x = xScale(index)
        const y = yScale(candle.close)

        ctx.beginPath()
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI)
        ctx.fillStyle = '#ffffff'
        ctx.fill()
        ctx.strokeStyle = '#00d4aa'
        ctx.lineWidth = 2
        ctx.stroke()
      })
    }

    // Volume bars (if enabled)
    if (chartSettings.showVolume) {
      const volumeTop = chartMargin.top + mainChartHeight + 10
      const maxVolume = Math.max(...visibleData.map(d => d.volume))

      visibleData.forEach((candle, index) => {
        const x = xScale(index)
        const candleWidth = Math.max(2, Math.min(chartSettings.candleWidth * 0.8, chartWidth / visibleData.length * 0.6))
        const volumeBarHeight = (candle.volume / maxVolume) * volumeHeight * 0.8

        const isGreen = candle.close >= candle.open
        const volumeColor = isGreen ? 'rgba(0, 184, 148, 0.7)' : 'rgba(255, 71, 87, 0.7)'

        ctx.fillStyle = volumeColor
        ctx.fillRect(
          x - candleWidth / 2,
          volumeTop + volumeHeight - volumeBarHeight,
          candleWidth,
          volumeBarHeight
        )
      })
    }

    // Professional technical indicators
    if (showIndicators) {
      // EMA lines with improved visibility
      const emaConfigs = [
        { key: 'ema_20', color: '#74b9ff', width: 2 },
        { key: 'ema_50', color: '#fdcb6e', width: 2 }
      ]

      emaConfigs.forEach(config => {
        ctx.strokeStyle = config.color
        ctx.lineWidth = config.width
        ctx.setLineDash([])

        ctx.beginPath()
        let hasData = false

        visibleData.forEach((candle, index) => {
          if (candle.indicators?.[config.key]) {
            const x = xScale(index)
            const y = yScale(candle.indicators[config.key])

            if (!hasData) {
              ctx.moveTo(x, y)
              hasData = true
            } else {
              ctx.lineTo(x, y)
            }
          }
        })

        if (hasData) {
          ctx.shadowBlur = 3
          ctx.shadowColor = config.color
          ctx.stroke()
          ctx.shadowBlur = 0
        }
      })
    }

    // Professional crosshair
    if (crosshair.visible) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])

      // Vertical line
      ctx.beginPath()
      ctx.moveTo(crosshair.x, chartMargin.top)
      ctx.lineTo(crosshair.x, chartMargin.top + mainChartHeight)
      ctx.stroke()

      // Horizontal line
      ctx.beginPath()
      ctx.moveTo(chartMargin.left, crosshair.priceY)
      ctx.lineTo(rect.width - chartMargin.right, crosshair.priceY)
      ctx.stroke()

      ctx.setLineDash([])

      // Price label on crosshair
      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'
      ctx.fillRect(rect.width - chartMargin.right + 5, crosshair.priceY - 12, 100, 24)
      ctx.fillStyle = '#000000'
      ctx.font = 'bold 12px Inter'
      ctx.textAlign = 'left'
      ctx.fillText(`$${crosshair.price.toFixed(2)}`, rect.width - chartMargin.right + 10, crosshair.priceY + 4)
    }

    // Ultra-professional header
    const headerGradient = ctx.createLinearGradient(0, 0, rect.width, 0)
    headerGradient.addColorStop(0, 'rgba(0, 212, 170, 0.15)')
    headerGradient.addColorStop(0.5, 'rgba(116, 185, 255, 0.1)')
    headerGradient.addColorStop(1, 'rgba(255, 99, 72, 0.15)')
    ctx.fillStyle = headerGradient
    ctx.fillRect(0, 0, rect.width, 35)

    // Title with professional styling
    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 16px Inter'
    ctx.textAlign = 'left'
    ctx.fillText(`${symbol} Ultra-Professional Chart`, chartMargin.left, 22)

    // Range indicator
    ctx.fillStyle = '#00d4aa'
    ctx.font = 'bold 12px Inter'
    ctx.fillText(
      `${selectedTimeframe.toUpperCase()} • ${chartType === 'candlestick' ? 'Velas 3D' : 'Línea Pro'} • Viendo ${chartSettings.viewRange.start + 1}-${chartSettings.viewRange.end} de ${candlestickData.length}`,
      chartMargin.left + 300, 22
    )

    // Professional data info
    ctx.fillStyle = '#64748b'
    ctx.font = '11px Inter'
    ctx.textAlign = 'right'
    ctx.fillText(
      `${visibleData.length} velas visibles • Zoom: ${chartSettings.zoomLevel.toFixed(1)}x • ${lastUpdate?.toLocaleTimeString() || 'N/A'}`,
      rect.width - chartMargin.right,
      22
    )

    // Live indicator with pulse effect
    if (isConnected) {
      const pulseRadius = 4 + Math.sin(Date.now() * 0.008) * 2
      ctx.fillStyle = `rgba(0, 184, 148, ${0.7 + Math.sin(Date.now() * 0.008) * 0.3})`
      ctx.beginPath()
      ctx.arc(rect.width - 20, 20, pulseRadius, 0, 2 * Math.PI)
      ctx.fill()
    }

  }, [candlestickData, chartSettings, crosshair, chartType, selectedTimeframe, showIndicators, currentPrice, isConnected, lastUpdate])

  // Redraw chart
  useEffect(() => {
    const animationFrame = requestAnimationFrame(drawChart)
    return () => cancelAnimationFrame(animationFrame)
  }, [drawChart])

  // Handle resize with debouncing
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

  // Refetch data when timeframe changes
  useEffect(() => {
    fetchCandlestickData()
  }, [selectedTimeframe, fetchCandlestickData])

  return (
    <div ref={containerRef} className={`relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 rounded-2xl border border-slate-700 shadow-2xl overflow-hidden ${className}`}>
      {/* Ultra-Professional Controls */}
      <div className="absolute top-4 right-4 z-30 flex flex-wrap items-center gap-2">
        {/* Connection Status */}
        <Badge variant={isConnected ? "default" : "secondary"} className={`${isConnected ? "bg-green-600 text-white" : "bg-gray-600 text-gray-200"} font-bold`}>
          <Activity className="w-3 h-3 mr-1" />
          {isConnected ? 'EN VIVO' : 'OFFLINE'}
        </Badge>

        {/* Chart Type Toggle */}
        <div className="flex bg-slate-800/90 rounded-lg p-1 border border-slate-600">
          <Button
            variant={chartType === 'candlestick' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setChartType('candlestick')}
            className={`text-xs px-3 py-1 ${chartType === 'candlestick' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'}`}
          >
            <BarChart3 className="w-3 h-3 mr-1" />
            Velas 3D
          </Button>
          <Button
            variant={chartType === 'line' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setChartType('line')}
            className={`text-xs px-3 py-1 ${chartType === 'line' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:text-white'}`}
          >
            <TrendingUp className="w-3 h-3 mr-1" />
            Línea Pro
          </Button>
        </div>

        {/* Advanced Controls */}
        <div className="flex bg-slate-800/90 rounded-lg p-1 border border-slate-600">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleZoomIn}
            className="text-xs px-2 py-1 text-gray-300 hover:text-white"
          >
            <ZoomIn className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleZoomOut}
            className="text-xs px-2 py-1 text-gray-300 hover:text-white"
          >
            <ZoomOut className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleResetView}
            className="text-xs px-2 py-1 text-gray-300 hover:text-white"
          >
            <RotateCcw className="w-3 h-3" />
          </Button>
        </div>

        {/* Volume & Indicators */}
        <div className="flex bg-slate-800/90 rounded-lg p-1 border border-slate-600">
          <Button
            variant={chartSettings.showVolume ? "default" : "ghost"}
            size="sm"
            onClick={() => setChartSettings(prev => ({ ...prev, showVolume: !prev.showVolume }))}
            className={`text-xs px-3 py-1 ${chartSettings.showVolume ? 'bg-purple-600 text-white' : 'text-gray-300 hover:text-white'}`}
          >
            <Volume2 className="w-3 h-3 mr-1" />
            Vol
          </Button>
          <Button
            variant={showIndicators ? "default" : "ghost"}
            size="sm"
            onClick={() => setShowIndicators(!showIndicators)}
            className={`text-xs px-3 py-1 ${showIndicators ? 'bg-orange-600 text-white' : 'text-gray-300 hover:text-white'}`}
          >
            {showIndicators ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
            Tech
          </Button>
        </div>

        {/* Refresh */}
        <Button
          variant="outline"
          size="sm"
          onClick={fetchCandlestickData}
          disabled={loading}
          className="text-xs border-slate-600 text-gray-300 hover:text-white"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {/* Professional Timeframe Selector */}
      <div className="absolute top-4 left-4 z-30">
        <div className="bg-slate-800/95 border border-slate-600 rounded-lg p-2">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-400" />
            <div className="flex gap-1">
              {(['5m', '15m', '30m', '1h', '1d'] as TimeframePeriod[]).map(tf => (
                <Button
                  key={tf}
                  variant={selectedTimeframe === tf ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setSelectedTimeframe(tf)}
                  className={`text-xs px-3 py-1 ${
                    selectedTimeframe === tf
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:text-white hover:bg-slate-700'
                  }`}
                >
                  {tf.toUpperCase()}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Professional Navigation Controls */}
      <div className="absolute bottom-4 left-4 z-30">
        <div className="bg-slate-800/95 border border-slate-600 rounded-lg p-2">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handlePanLeft}
              className="text-xs px-2 py-1 text-gray-300 hover:text-white"
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>

            {/* Professional Historical Range Slider */}
            <div className="flex flex-col gap-2 px-2">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Histórico Completo:</span>
                {candlestickData.length > 0 && (
                  <span className="text-xs text-cyan-400 font-mono">
                    {new Date(candlestickData[0].time).toLocaleDateString('es-CO')} - {new Date(candlestickData[candlestickData.length - 1].time).toLocaleDateString('es-CO')}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-400">Navegación:</span>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, candlestickData.length - 50)}
                  value={chartSettings.viewRange.start}
                  onChange={(e) => {
                    const start = parseInt(e.target.value)
                    const range = chartSettings.viewRange.end - chartSettings.viewRange.start
                    setChartSettings(prev => ({
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
                  {candlestickData.length > 0 && chartSettings.viewRange.start < candlestickData.length ?
                    new Date(candlestickData[chartSettings.viewRange.start].time).toLocaleDateString('es-CO') : 'N/A'}
                </span>
              </div>
              <div className="text-xs text-amber-400 bg-amber-500/20 px-2 py-1 rounded">
                📊 {candlestickData.length} registros disponibles • Total en DB: {dbStats.totalRecords.toLocaleString()}
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={handlePanRight}
              className="text-xs px-2 py-1 text-gray-300 hover:text-white"
            >
              <ChevronRight className="w-4 h-4" />
            </Button>

            <Button
              variant={chartSettings.autoScroll ? "default" : "ghost"}
              size="sm"
              onClick={() => setChartSettings(prev => ({ ...prev, autoScroll: !prev.autoScroll }))}
              className={`text-xs px-2 py-1 ${chartSettings.autoScroll ? 'bg-green-600 text-white' : 'text-gray-300 hover:text-white'}`}
            >
              {chartSettings.autoScroll ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="absolute top-20 left-4 right-4 z-20 bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <div className="flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Error: {error}</span>
          </div>
        </div>
      )}

      {/* Ultra-Professional Chart Canvas */}
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width: '100%', height: `${height}px`, cursor: 'crosshair' }}
        className="rounded-2xl"
      />

      {/* Professional Hover Tooltip */}
      {hoverInfo.visible && (
        <div
          className="fixed bg-slate-900/95 border border-slate-600 rounded-xl p-4 pointer-events-none z-40 shadow-2xl backdrop-blur-sm"
          style={{
            left: hoverInfo.x + 15,
            top: hoverInfo.y - 15,
            transform: hoverInfo.x > window.innerWidth - 300 ? 'translateX(-100%)' : 'none'
          }}
        >
          <div className="text-white font-bold text-sm mb-3 border-b border-slate-600 pb-2">
            📊 {new Date(hoverInfo.candle.time).toLocaleString('es-CO')}
          </div>

          {chartType === 'candlestick' ? (
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex justify-between">
                  <span className="text-gray-400">🟢 Apertura:</span>
                  <span className="text-cyan-400 font-mono font-bold">${hoverInfo.candle.open?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">🔺 Máximo:</span>
                  <span className="text-green-400 font-mono font-bold">${hoverInfo.candle.high?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">🔻 Mínimo:</span>
                  <span className="text-red-400 font-mono font-bold">${hoverInfo.candle.low?.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">🎯 Cierre:</span>
                  <span className="text-yellow-400 font-mono font-bold">${hoverInfo.candle.close?.toFixed(2)}</span>
                </div>
              </div>

              <div className="flex justify-between pt-2 border-t border-slate-700">
                <span className="text-gray-400">📊 Volumen:</span>
                <span className="text-purple-400 font-mono font-bold">{hoverInfo.candle.volume?.toLocaleString()}</span>
              </div>

              {showIndicators && hoverInfo.candle.indicators && (
                <>
                  <div className="border-t border-slate-600 pt-2 mt-2">
                    <div className="text-blue-300 font-bold text-xs mb-1">📈 Indicadores Técnicos</div>
                    {hoverInfo.candle.indicators.ema_20 && (
                      <div className="flex justify-between">
                        <span className="text-blue-400">EMA 20:</span>
                        <span className="text-blue-400 font-mono">${hoverInfo.candle.indicators.ema_20.toFixed(2)}</span>
                      </div>
                    )}
                    {hoverInfo.candle.indicators.ema_50 && (
                      <div className="flex justify-between">
                        <span className="text-orange-400">EMA 50:</span>
                        <span className="text-orange-400 font-mono">${hoverInfo.candle.indicators.ema_50.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">💰 Precio:</span>
                <span className="text-cyan-400 font-mono font-bold">${hoverInfo.candle.close?.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">📊 Volumen:</span>
                <span className="text-purple-400 font-mono font-bold">{hoverInfo.candle.volume?.toLocaleString()}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Professional Legend */}
      {showIndicators && (
        <div className="absolute bottom-4 right-4 bg-slate-900/95 border border-slate-600 rounded-xl p-4 backdrop-blur-sm">
          <div className="text-white font-bold text-xs mb-2">📊 Indicadores</div>
          <div className="grid grid-cols-1 gap-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-1 bg-blue-400 rounded shadow-lg"></div>
              <span className="text-blue-400 font-bold">EMA 20</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-1 bg-orange-400 rounded shadow-lg"></div>
              <span className="text-orange-400 font-bold">EMA 50</span>
            </div>
            {chartSettings.showVolume && (
              <div className="flex items-center gap-2">
                <div className="w-4 h-1 bg-purple-400 rounded shadow-lg"></div>
                <span className="text-purple-400 font-bold">Volumen</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Professional Info Panel */}
      <div className="absolute top-20 right-4 bg-slate-900/95 border border-slate-600 rounded-xl p-3 backdrop-blur-sm">
        <div className="text-xs space-y-1">
          <div className="text-gray-300 font-bold">📈 Estado del Gráfico</div>
          <div className="text-green-400 font-mono">{candlestickData.length} puntos totales</div>
          <div className="text-blue-400 font-mono">{chartSettings.viewRange.end - chartSettings.viewRange.start} visibles</div>
          <div className="text-purple-400 font-mono">Zoom: {chartSettings.zoomLevel.toFixed(1)}x</div>
          <div className="text-orange-400 font-mono">{selectedTimeframe.toUpperCase()}</div>
        </div>
      </div>

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