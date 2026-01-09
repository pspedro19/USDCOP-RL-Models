/**
 * Chart Drawing Engine
 * ====================
 *
 * Professional canvas-based chart rendering engine.
 * Handles all drawing operations for candlestick and line charts.
 */

import { CandlestickData } from '@/lib/services/market-data-service'
import {
  ChartDimensions,
  ChartSettings,
  CrosshairInfo,
  ChartType,
  TimeframePeriod,
  DEFAULT_CHART_MARGINS
} from './types'

interface ScaleFunctions {
  xScale: (index: number) => number
  yScale: (price: number) => number
}

/**
 * Creates scale functions for mapping data to canvas coordinates
 */
export function createScaleFunctions(
  dimensions: ChartDimensions,
  visibleData: CandlestickData[],
  priceRange: { min: number; max: number; range: number; padding: number }
): ScaleFunctions {
  const { margins, chartWidth, mainChartHeight } = dimensions

  return {
    xScale: (index: number) =>
      margins.left + (index / Math.max(1, visibleData.length - 1)) * chartWidth,
    yScale: (price: number) =>
      margins.top +
      (1 - (price - priceRange.min + priceRange.padding) / (priceRange.range + 2 * priceRange.padding)) *
        mainChartHeight
  }
}

/**
 * Calculates chart dimensions based on canvas size and settings
 */
export function calculateDimensions(
  width: number,
  height: number,
  showVolume: boolean
): ChartDimensions {
  const margins = DEFAULT_CHART_MARGINS
  const chartWidth = width - margins.left - margins.right
  const totalChartHeight = height - margins.top - margins.bottom
  const mainChartHeight = showVolume ? totalChartHeight * 0.75 : totalChartHeight
  const volumeHeight = showVolume ? totalChartHeight * 0.25 : 0

  return {
    width,
    height,
    chartWidth,
    mainChartHeight,
    volumeHeight,
    margins
  }
}

/**
 * Draws the professional background gradient
 */
export function drawBackground(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  const bgGradient = ctx.createLinearGradient(0, 0, 0, height)
  bgGradient.addColorStop(0, '#0a0e1a')
  bgGradient.addColorStop(0.3, '#1a1f2e')
  bgGradient.addColorStop(0.7, '#2a2f3e')
  bgGradient.addColorStop(1, '#1a1f2e')
  ctx.fillStyle = bgGradient
  ctx.fillRect(0, 0, width, height)
}

/**
 * Draws loading animation
 */
export function drawLoadingState(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  selectedTimeframe: TimeframePeriod,
  chartType: ChartType
): void {
  const centerX = width / 2
  const centerY = height / 2
  const time = Date.now() * 0.002

  ctx.strokeStyle = '#00d4aa'
  ctx.lineWidth = 4

  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2 + time
    const radius = 30 + Math.sin(time + i) * 10
    ctx.globalAlpha = 0.3 + 0.7 * ((Math.sin(time + i) + 1) / 2)

    ctx.beginPath()
    ctx.arc(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius, 8, 0, Math.PI * 2)
    ctx.stroke()
  }
  ctx.globalAlpha = 1

  ctx.fillStyle = '#ffffff'
  ctx.font = 'bold 24px Inter'
  ctx.textAlign = 'center'
  ctx.fillText('Cargando Datos Historicos Completos', centerX, centerY + 80)

  ctx.fillStyle = '#00d4aa'
  ctx.font = '16px Inter'
  ctx.fillText(
    `${selectedTimeframe.toUpperCase()} - ${chartType === 'candlestick' ? 'Velas Japonesas' : 'Linea Suavizada'}`,
    centerX,
    centerY + 110
  )

  ctx.fillStyle = 'rgba(0, 212, 170, 0.2)'
  ctx.fillRect(centerX - 150, centerY + 130, 300, 6)
  ctx.fillStyle = '#00d4aa'
  ctx.fillRect(centerX - 150, centerY + 130, 300 * ((Date.now() % 3000) / 3000), 6)
}

/**
 * Draws the grid lines
 */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  dimensions: ChartDimensions,
  visibleData: CandlestickData[],
  scales: ScaleFunctions,
  minPrice: number,
  priceRange: number,
  selectedTimeframe: TimeframePeriod
): void {
  const { width, margins, mainChartHeight } = dimensions

  ctx.strokeStyle = 'rgba(100, 116, 139, 0.08)'
  ctx.lineWidth = 0.5

  // Horizontal grid (price levels)
  const priceSteps = 12
  for (let i = 0; i <= priceSteps; i++) {
    const price = minPrice + (priceRange * i) / priceSteps
    const y = scales.yScale(price)

    const gridGradient = ctx.createLinearGradient(margins.left, y, width - margins.right, y)
    gridGradient.addColorStop(0, 'rgba(100, 116, 139, 0.02)')
    gridGradient.addColorStop(0.5, 'rgba(100, 116, 139, 0.15)')
    gridGradient.addColorStop(1, 'rgba(100, 116, 139, 0.02)')

    ctx.strokeStyle = gridGradient
    ctx.beginPath()
    ctx.moveTo(margins.left, y)
    ctx.lineTo(width - margins.right, y)
    ctx.stroke()

    // Price labels
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)'
    ctx.fillRect(width - margins.right + 5, y - 12, 100, 24)

    const priceGradient = ctx.createLinearGradient(0, y - 10, 0, y + 10)
    priceGradient.addColorStop(0, '#00d4aa')
    priceGradient.addColorStop(1, '#00b894')
    ctx.fillStyle = priceGradient
    ctx.font = 'bold 12px Inter'
    ctx.textAlign = 'left'
    ctx.fillText(`$${price.toFixed(2)}`, width - margins.right + 10, y + 4)
  }

  // Vertical grid (time)
  const timeSteps = Math.min(12, visibleData.length)
  for (let i = 0; i <= timeSteps; i++) {
    const x = margins.left + (i / timeSteps) * dimensions.chartWidth

    ctx.strokeStyle = 'rgba(100, 116, 139, 0.08)'
    ctx.beginPath()
    ctx.moveTo(x, margins.top)
    ctx.lineTo(x, margins.top + mainChartHeight)
    ctx.stroke()

    if (i < visibleData.length) {
      const dataIndex = Math.floor((i / timeSteps) * (visibleData.length - 1))
      const timestamp = new Date(visibleData[dataIndex].time)

      ctx.fillStyle = 'rgba(15, 23, 42, 0.95)'
      ctx.fillRect(x - 40, dimensions.height - margins.bottom + 10, 80, 20)

      ctx.fillStyle = '#64748b'
      ctx.font = 'bold 11px Inter'
      ctx.textAlign = 'center'

      const timeLabel =
        selectedTimeframe === '1d'
          ? timestamp.toLocaleDateString('es-CO', { month: 'short', day: 'numeric' })
          : timestamp.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' })

      ctx.fillText(timeLabel, x, dimensions.height - margins.bottom + 23)
    }
  }
}

/**
 * Draws candlestick chart
 */
export function drawCandlesticks(
  ctx: CanvasRenderingContext2D,
  visibleData: CandlestickData[],
  scales: ScaleFunctions,
  chartSettings: ChartSettings,
  chartWidth: number
): void {
  visibleData.forEach((candle, index) => {
    const x = scales.xScale(index)
    const candleWidth = Math.max(4, Math.min(chartSettings.candleWidth, (chartWidth / visibleData.length) * 0.8))

    const openY = scales.yScale(candle.open)
    const closeY = scales.yScale(candle.close)
    const highY = scales.yScale(candle.high)
    const lowY = scales.yScale(candle.low)

    const isGreen = candle.close >= candle.open
    const baseColor = isGreen ? '#00d4aa' : '#ff4757'

    // Wick
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

    // Body
    const bodyHeight = Math.abs(closeY - openY)
    const bodyY = Math.min(openY, closeY)

    // 3D shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
    ctx.fillRect(x - candleWidth / 2 + 2, bodyY + 2, candleWidth, Math.max(2, bodyHeight))

    // Main body gradient
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

    // Border with glow
    ctx.shadowBlur = 3
    ctx.shadowColor = baseColor
    ctx.strokeStyle = baseColor
    ctx.lineWidth = 1
    ctx.strokeRect(x - candleWidth / 2, bodyY, candleWidth, Math.max(2, bodyHeight))
    ctx.shadowBlur = 0

    // High/Low markers
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
}

/**
 * Draws line chart with gradient fill
 */
export function drawLineChart(
  ctx: CanvasRenderingContext2D,
  visibleData: CandlestickData[],
  scales: ScaleFunctions,
  dimensions: ChartDimensions
): void {
  const { margins, mainChartHeight } = dimensions

  // Gradient fill
  const fillGradient = ctx.createLinearGradient(0, margins.top, 0, margins.top + mainChartHeight)
  fillGradient.addColorStop(0, 'rgba(0, 212, 170, 0.4)')
  fillGradient.addColorStop(0.3, 'rgba(0, 212, 170, 0.2)')
  fillGradient.addColorStop(0.7, 'rgba(0, 212, 170, 0.1)')
  fillGradient.addColorStop(1, 'rgba(0, 212, 170, 0.02)')

  ctx.beginPath()
  visibleData.forEach((candle, index) => {
    const x = scales.xScale(index)
    const y = scales.yScale(candle.close)

    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })

  ctx.lineTo(scales.xScale(visibleData.length - 1), margins.top + mainChartHeight)
  ctx.lineTo(margins.left, margins.top + mainChartHeight)
  ctx.closePath()
  ctx.fillStyle = fillGradient
  ctx.fill()

  // Main line with glow
  ctx.beginPath()
  visibleData.forEach((candle, index) => {
    const x = scales.xScale(index)
    const y = scales.yScale(candle.close)

    if (index === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })

  ctx.shadowBlur = 8
  ctx.shadowColor = '#00d4aa'
  ctx.strokeStyle = '#00d4aa'
  ctx.lineWidth = 3
  ctx.stroke()
  ctx.shadowBlur = 0

  // Data points
  visibleData.forEach((candle, index) => {
    const x = scales.xScale(index)
    const y = scales.yScale(candle.close)

    ctx.beginPath()
    ctx.arc(x, y, 2.5, 0, 2 * Math.PI)
    ctx.fillStyle = '#ffffff'
    ctx.fill()
    ctx.strokeStyle = '#00d4aa'
    ctx.lineWidth = 2
    ctx.stroke()
  })
}

/**
 * Draws volume bars
 */
export function drawVolumeBars(
  ctx: CanvasRenderingContext2D,
  visibleData: CandlestickData[],
  scales: ScaleFunctions,
  dimensions: ChartDimensions,
  chartSettings: ChartSettings
): void {
  const { margins, mainChartHeight, volumeHeight, chartWidth } = dimensions
  const volumeTop = margins.top + mainChartHeight + 10
  const maxVolume = Math.max(...visibleData.map((d) => d.volume))

  visibleData.forEach((candle, index) => {
    const x = scales.xScale(index)
    const candleWidth = Math.max(2, Math.min(chartSettings.candleWidth * 0.8, (chartWidth / visibleData.length) * 0.6))
    const volumeBarHeight = (candle.volume / maxVolume) * volumeHeight * 0.8

    const isGreen = candle.close >= candle.open
    const volumeColor = isGreen ? 'rgba(0, 184, 148, 0.7)' : 'rgba(255, 71, 87, 0.7)'

    ctx.fillStyle = volumeColor
    ctx.fillRect(x - candleWidth / 2, volumeTop + volumeHeight - volumeBarHeight, candleWidth, volumeBarHeight)
  })
}

/**
 * Draws technical indicators (EMA lines)
 */
export function drawIndicators(
  ctx: CanvasRenderingContext2D,
  visibleData: (CandlestickData & { indicators?: Record<string, number> })[],
  scales: ScaleFunctions
): void {
  const emaConfigs = [
    { key: 'ema_20', color: '#74b9ff', width: 2 },
    { key: 'ema_50', color: '#fdcb6e', width: 2 }
  ]

  emaConfigs.forEach((config) => {
    ctx.strokeStyle = config.color
    ctx.lineWidth = config.width
    ctx.setLineDash([])

    ctx.beginPath()
    let hasData = false

    visibleData.forEach((candle, index) => {
      if (candle.indicators?.[config.key]) {
        const x = scales.xScale(index)
        const y = scales.yScale(candle.indicators[config.key])

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

/**
 * Draws crosshair lines
 */
export function drawCrosshair(
  ctx: CanvasRenderingContext2D,
  crosshair: CrosshairInfo,
  dimensions: ChartDimensions
): void {
  if (!crosshair.visible) return

  const { width, margins, mainChartHeight } = dimensions

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)'
  ctx.lineWidth = 1
  ctx.setLineDash([5, 5])

  // Vertical line
  ctx.beginPath()
  ctx.moveTo(crosshair.x, margins.top)
  ctx.lineTo(crosshair.x, margins.top + mainChartHeight)
  ctx.stroke()

  // Horizontal line
  ctx.beginPath()
  ctx.moveTo(margins.left, crosshair.priceY)
  ctx.lineTo(width - margins.right, crosshair.priceY)
  ctx.stroke()

  ctx.setLineDash([])

  // Price label
  ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'
  ctx.fillRect(width - margins.right + 5, crosshair.priceY - 12, 100, 24)
  ctx.fillStyle = '#000000'
  ctx.font = 'bold 12px Inter'
  ctx.textAlign = 'left'
  ctx.fillText(`$${crosshair.price.toFixed(2)}`, width - margins.right + 10, crosshair.priceY + 4)
}

/**
 * Draws chart header
 */
export function drawHeader(
  ctx: CanvasRenderingContext2D,
  dimensions: ChartDimensions,
  symbol: string,
  selectedTimeframe: TimeframePeriod,
  chartType: ChartType,
  chartSettings: ChartSettings,
  totalDataLength: number,
  visibleDataLength: number,
  isConnected: boolean,
  lastUpdate: Date | null
): void {
  const { width, margins } = dimensions

  const headerGradient = ctx.createLinearGradient(0, 0, width, 0)
  headerGradient.addColorStop(0, 'rgba(0, 212, 170, 0.15)')
  headerGradient.addColorStop(0.5, 'rgba(116, 185, 255, 0.1)')
  headerGradient.addColorStop(1, 'rgba(255, 99, 72, 0.15)')
  ctx.fillStyle = headerGradient
  ctx.fillRect(0, 0, width, 35)

  // Title
  ctx.fillStyle = '#ffffff'
  ctx.font = 'bold 16px Inter'
  ctx.textAlign = 'left'
  ctx.fillText(`${symbol} Ultra-Professional Chart`, margins.left, 22)

  // Range indicator
  ctx.fillStyle = '#00d4aa'
  ctx.font = 'bold 12px Inter'
  ctx.fillText(
    `${selectedTimeframe.toUpperCase()} - ${chartType === 'candlestick' ? 'Velas 3D' : 'Linea Pro'} - Viendo ${chartSettings.viewRange.start + 1}-${chartSettings.viewRange.end} de ${totalDataLength}`,
    margins.left + 300,
    22
  )

  // Info
  ctx.fillStyle = '#64748b'
  ctx.font = '11px Inter'
  ctx.textAlign = 'right'
  ctx.fillText(
    `${visibleDataLength} velas visibles - Zoom: ${chartSettings.zoomLevel.toFixed(1)}x - ${lastUpdate?.toLocaleTimeString() || 'N/A'}`,
    width - margins.right,
    22
  )

  // Live indicator
  if (isConnected) {
    const pulseRadius = 4 + Math.sin(Date.now() * 0.008) * 2
    ctx.fillStyle = `rgba(0, 184, 148, ${0.7 + Math.sin(Date.now() * 0.008) * 0.3})`
    ctx.beginPath()
    ctx.arc(width - 20, 20, pulseRadius, 0, 2 * Math.PI)
    ctx.fill()
  }
}
