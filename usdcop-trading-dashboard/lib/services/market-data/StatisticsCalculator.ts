/**
 * Statistics Calculator - Market Statistics Calculations
 * ======================================================
 *
 * Single Responsibility: Calculate market statistics and metrics
 * - Price changes and percentages
 * - Moving averages
 * - Volatility metrics
 * - Statistical aggregations
 */

import { createLogger } from '@/lib/utils/logger'
import type { CandlestickData } from './types'

const logger = createLogger('StatisticsCalculator')

export interface PriceChange {
  change: number
  changePercent: number
  isPositive: boolean
}

export interface PriceStatistics {
  mean: number
  median: number
  stdDev: number
  min: number
  max: number
  range: number
}

export interface VolumeStatistics {
  total: number
  average: number
  min: number
  max: number
}

export class StatisticsCalculator {
  /**
   * Calculate price change and percentage
   */
  static calculatePriceChange(current: number, previous: number): PriceChange {
    const change = current - previous
    const changePercent = previous !== 0 ? (change / previous) * 100 : 0

    return {
      change,
      changePercent,
      isPositive: change >= 0,
    }
  }

  /**
   * Calculate simple moving average
   */
  static calculateSMA(prices: number[], period: number): number | null {
    if (prices.length < period) {
      logger.warn(`Not enough data points for SMA calculation. Need ${period}, got ${prices.length}`)
      return null
    }

    const relevantPrices = prices.slice(-period)
    const sum = relevantPrices.reduce((acc, price) => acc + price, 0)
    return sum / period
  }

  /**
   * Calculate exponential moving average
   */
  static calculateEMA(prices: number[], period: number): number | null {
    if (prices.length < period) {
      logger.warn(`Not enough data points for EMA calculation. Need ${period}, got ${prices.length}`)
      return null
    }

    const multiplier = 2 / (period + 1)
    let ema = this.calculateSMA(prices.slice(0, period), period)

    if (ema === null) return null

    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema
    }

    return ema
  }

  /**
   * Calculate standard deviation
   */
  static calculateStdDev(values: number[]): number {
    if (values.length === 0) return 0

    const mean = values.reduce((acc, val) => acc + val, 0) / values.length
    const squaredDiffs = values.map((val) => Math.pow(val - mean, 2))
    const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length

    return Math.sqrt(variance)
  }

  /**
   * Calculate price statistics from candlestick data
   */
  static calculatePriceStats(candles: CandlestickData[]): PriceStatistics {
    if (candles.length === 0) {
      return {
        mean: 0,
        median: 0,
        stdDev: 0,
        min: 0,
        max: 0,
        range: 0,
      }
    }

    const prices = candles.map((c) => c.close)
    const sorted = [...prices].sort((a, b) => a - b)

    const mean = prices.reduce((acc, price) => acc + price, 0) / prices.length
    const median =
      sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)]

    const min = Math.min(...prices)
    const max = Math.max(...prices)

    return {
      mean,
      median,
      stdDev: this.calculateStdDev(prices),
      min,
      max,
      range: max - min,
    }
  }

  /**
   * Calculate volume statistics from candlestick data
   */
  static calculateVolumeStats(candles: CandlestickData[]): VolumeStatistics {
    if (candles.length === 0) {
      return {
        total: 0,
        average: 0,
        min: 0,
        max: 0,
      }
    }

    const volumes = candles.map((c) => c.volume)
    const total = volumes.reduce((acc, vol) => acc + vol, 0)

    return {
      total,
      average: total / volumes.length,
      min: Math.min(...volumes),
      max: Math.max(...volumes),
    }
  }

  /**
   * Calculate volatility (standard deviation of returns)
   */
  static calculateVolatility(candles: CandlestickData[]): number {
    if (candles.length < 2) return 0

    const returns: number[] = []
    for (let i = 1; i < candles.length; i++) {
      const ret = (candles[i].close - candles[i - 1].close) / candles[i - 1].close
      returns.push(ret)
    }

    return this.calculateStdDev(returns)
  }

  /**
   * Calculate RSI (Relative Strength Index)
   */
  static calculateRSI(prices: number[], period: number = 14): number | null {
    if (prices.length < period + 1) {
      logger.warn(`Not enough data points for RSI calculation. Need ${period + 1}, got ${prices.length}`)
      return null
    }

    const changes: number[] = []
    for (let i = 1; i < prices.length; i++) {
      changes.push(prices[i] - prices[i - 1])
    }

    const gains = changes.map((c) => (c > 0 ? c : 0))
    const losses = changes.map((c) => (c < 0 ? Math.abs(c) : 0))

    const avgGain = gains.slice(-period).reduce((acc, g) => acc + g, 0) / period
    const avgLoss = losses.slice(-period).reduce((acc, l) => acc + l, 0) / period

    if (avgLoss === 0) return 100

    const rs = avgGain / avgLoss
    const rsi = 100 - 100 / (1 + rs)

    return rsi
  }

  /**
   * Calculate Bollinger Bands
   */
  static calculateBollingerBands(
    prices: number[],
    period: number = 20,
    stdDevMultiplier: number = 2
  ): { upper: number; middle: number; lower: number } | null {
    const sma = this.calculateSMA(prices, period)

    if (sma === null) {
      logger.warn('Cannot calculate Bollinger Bands without SMA')
      return null
    }

    const relevantPrices = prices.slice(-period)
    const stdDev = this.calculateStdDev(relevantPrices)

    return {
      upper: sma + stdDevMultiplier * stdDev,
      middle: sma,
      lower: sma - stdDevMultiplier * stdDev,
    }
  }

  /**
   * Calculate percentage change over period
   */
  static calculatePercentChange(startValue: number, endValue: number): number {
    if (startValue === 0) {
      logger.warn('Cannot calculate percentage change with zero start value')
      return 0
    }
    return ((endValue - startValue) / startValue) * 100
  }

  /**
   * Calculate VWAP (Volume Weighted Average Price)
   */
  static calculateVWAP(candles: CandlestickData[]): number {
    if (candles.length === 0) return 0

    let totalPriceVolume = 0
    let totalVolume = 0

    for (const candle of candles) {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3
      totalPriceVolume += typicalPrice * candle.volume
      totalVolume += candle.volume
    }

    if (totalVolume === 0) {
      logger.warn('Cannot calculate VWAP with zero total volume')
      return 0
    }

    return totalPriceVolume / totalVolume
  }

  /**
   * Calculate True Range (TR) for ATR calculation
   */
  private static calculateTrueRange(candles: CandlestickData[]): number[] {
    const trueRanges: number[] = []

    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].high
      const low = candles[i].low
      const prevClose = candles[i - 1].close

      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      )

      trueRanges.push(tr)
    }

    return trueRanges
  }

  /**
   * Calculate ATR (Average True Range)
   */
  static calculateATR(candles: CandlestickData[], period: number = 14): number | null {
    if (candles.length < period + 1) {
      logger.warn(`Not enough data points for ATR calculation. Need ${period + 1}, got ${candles.length}`)
      return null
    }

    const trueRanges = this.calculateTrueRange(candles)
    return this.calculateSMA(trueRanges, period)
  }

  /**
   * Calculate 24h statistics
   */
  static calculate24hStats(candles: CandlestickData[]) {
    if (candles.length === 0) {
      return {
        open: 0,
        high: 0,
        low: 0,
        close: 0,
        volume: 0,
        change: 0,
        changePercent: 0,
      }
    }

    const open = candles[0].open
    const close = candles[candles.length - 1].close
    const high = Math.max(...candles.map((c) => c.high))
    const low = Math.min(...candles.map((c) => c.low))
    const volume = candles.reduce((sum, c) => sum + c.volume, 0)

    const priceChange = this.calculatePriceChange(open, close)

    return {
      open,
      high,
      low,
      close,
      volume,
      change: priceChange.change,
      changePercent: priceChange.changePercent,
    }
  }
}
