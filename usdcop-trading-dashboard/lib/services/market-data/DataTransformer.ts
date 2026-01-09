/**
 * Data Transformer - Data Formatting and Transformation
 * =====================================================
 *
 * Single Responsibility: Transform and format data
 * - Format prices for display
 * - Transform API responses
 * - Data normalization
 */

import { createLogger } from '@/lib/utils/logger'

const logger = createLogger('DataTransformer')

export interface FormatOptions {
  locale?: string
  currency?: string
  minimumFractionDigits?: number
  maximumFractionDigits?: number
}

export class DataTransformer {
  /**
   * Format price for display with currency
   */
  static formatPrice(
    price: number,
    decimals: number = 4,
    options: FormatOptions = {}
  ): string {
    const {
      locale = 'es-CO',
      currency = 'COP',
      minimumFractionDigits = decimals,
      maximumFractionDigits = decimals,
    } = options

    try {
      return new Intl.NumberFormat(locale, {
        style: 'currency',
        currency,
        minimumFractionDigits,
        maximumFractionDigits,
      }).format(price)
    } catch (error) {
      logger.error('Error formatting price:', error)
      return price.toFixed(decimals)
    }
  }

  /**
   * Format number without currency symbol
   */
  static formatNumber(
    value: number,
    decimals: number = 2,
    locale: string = 'es-CO'
  ): string {
    try {
      return new Intl.NumberFormat(locale, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      }).format(value)
    } catch (error) {
      logger.error('Error formatting number:', error)
      return value.toFixed(decimals)
    }
  }

  /**
   * Format percentage
   */
  static formatPercent(
    value: number,
    decimals: number = 2,
    locale: string = 'es-CO'
  ): string {
    try {
      return new Intl.NumberFormat(locale, {
        style: 'percent',
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      }).format(value / 100)
    } catch (error) {
      logger.error('Error formatting percentage:', error)
      return `${value.toFixed(decimals)}%`
    }
  }

  /**
   * Format volume (abbreviate large numbers)
   */
  static formatVolume(volume: number, locale: string = 'es-CO'): string {
    try {
      if (volume >= 1_000_000_000) {
        return `${(volume / 1_000_000_000).toFixed(2)}B`
      } else if (volume >= 1_000_000) {
        return `${(volume / 1_000_000).toFixed(2)}M`
      } else if (volume >= 1_000) {
        return `${(volume / 1_000).toFixed(2)}K`
      }
      return volume.toFixed(0)
    } catch (error) {
      logger.error('Error formatting volume:', error)
      return volume.toString()
    }
  }

  /**
   * Format timestamp to readable date
   */
  static formatTimestamp(
    timestamp: number,
    locale: string = 'es-CO',
    options: Intl.DateTimeFormatOptions = {}
  ): string {
    try {
      const defaultOptions: Intl.DateTimeFormatOptions = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        ...options,
      }

      return new Intl.DateTimeFormat(locale, defaultOptions).format(
        new Date(timestamp)
      )
    } catch (error) {
      logger.error('Error formatting timestamp:', error)
      return new Date(timestamp).toISOString()
    }
  }

  /**
   * Format ISO date string to readable format
   */
  static formatDateString(
    dateString: string,
    locale: string = 'es-CO',
    options: Intl.DateTimeFormatOptions = {}
  ): string {
    try {
      return this.formatTimestamp(new Date(dateString).getTime(), locale, options)
    } catch (error) {
      logger.error('Error formatting date string:', error)
      return dateString
    }
  }

  /**
   * Normalize symbol (uppercase, trim)
   */
  static normalizeSymbol(symbol: string): string {
    return symbol.trim().toUpperCase()
  }

  /**
   * Parse timeframe string to minutes
   */
  static timeframeToMinutes(timeframe: string): number {
    const match = timeframe.match(/^(\d+)([mhd])$/)
    if (!match) {
      logger.warn(`Invalid timeframe format: ${timeframe}`)
      return 5 // Default to 5 minutes
    }

    const [, value, unit] = match
    const num = parseInt(value, 10)

    switch (unit) {
      case 'm':
        return num
      case 'h':
        return num * 60
      case 'd':
        return num * 60 * 24
      default:
        return 5
    }
  }

  /**
   * Convert minutes to timeframe string
   */
  static minutesToTimeframe(minutes: number): string {
    if (minutes < 60) {
      return `${minutes}m`
    } else if (minutes < 1440) {
      return `${minutes / 60}h`
    } else {
      return `${minutes / 1440}d`
    }
  }

  /**
   * Truncate string to max length
   */
  static truncate(str: string, maxLength: number, suffix: string = '...'): string {
    if (str.length <= maxLength) {
      return str
    }
    return str.substring(0, maxLength - suffix.length) + suffix
  }

  /**
   * Safe parse JSON with fallback
   */
  static safeParseJSON<T>(json: string, fallback: T): T {
    try {
      return JSON.parse(json) as T
    } catch (error) {
      logger.error('Error parsing JSON:', error)
      return fallback
    }
  }

  /**
   * Deep clone object
   */
  static deepClone<T>(obj: T): T {
    try {
      return JSON.parse(JSON.stringify(obj))
    } catch (error) {
      logger.error('Error cloning object:', error)
      return obj
    }
  }

  /**
   * Convert timestamp to ISO string
   */
  static timestampToISO(timestamp: number): string {
    try {
      return new Date(timestamp).toISOString()
    } catch (error) {
      logger.error('Error converting timestamp to ISO:', error)
      return new Date().toISOString()
    }
  }

  /**
   * Convert ISO string to timestamp
   */
  static isoToTimestamp(iso: string): number {
    try {
      return new Date(iso).getTime()
    } catch (error) {
      logger.error('Error converting ISO to timestamp:', error)
      return Date.now()
    }
  }
}
