/**
 * Integration Tests for Market Data Services
 * ==========================================
 *
 * Tests to verify the refactored modules work correctly together
 * and maintain backwards compatibility.
 */

import {
  MarketDataService,
  WebSocketConnector,
  MarketDataFetcher,
  DataTransformer,
  StatisticsCalculator,
} from '../index'

describe('Market Data Services - Integration', () => {
  describe('Backwards Compatibility', () => {
    it('should export MarketDataService class', () => {
      expect(MarketDataService).toBeDefined()
      expect(typeof MarketDataService).toBe('function')
    })

    it('should have all static methods from original API', () => {
      expect(typeof MarketDataService.getRealTimeData).toBe('function')
      expect(typeof MarketDataService.getCandlestickData).toBe('function')
      expect(typeof MarketDataService.getSymbolStats).toBe('function')
      expect(typeof MarketDataService.checkAPIHealth).toBe('function')
      expect(typeof MarketDataService.isMarketOpen).toBe('function')
      expect(typeof MarketDataService.formatPrice).toBe('function')
      expect(typeof MarketDataService.calculatePriceChange).toBe('function')
    })
  })

  describe('Module Exports', () => {
    it('should export WebSocketConnector', () => {
      expect(WebSocketConnector).toBeDefined()
      expect(typeof WebSocketConnector).toBe('function')
    })

    it('should export MarketDataFetcher', () => {
      expect(MarketDataFetcher).toBeDefined()
      expect(typeof MarketDataFetcher).toBe('function')
    })

    it('should export DataTransformer', () => {
      expect(DataTransformer).toBeDefined()
      expect(typeof DataTransformer).toBe('function')
    })

    it('should export StatisticsCalculator', () => {
      expect(StatisticsCalculator).toBeDefined()
      expect(typeof StatisticsCalculator).toBe('function')
    })
  })

  describe('DataTransformer', () => {
    it('should format prices correctly', () => {
      const formatted = DataTransformer.formatPrice(4123.45, 2)
      expect(formatted).toContain('4')
      expect(formatted).toContain('123')
    })

    it('should format volumes with abbreviations', () => {
      expect(DataTransformer.formatVolume(1500000)).toBe('1.50M')
      expect(DataTransformer.formatVolume(2500000000)).toBe('2.50B')
      expect(DataTransformer.formatVolume(5500)).toBe('5.50K')
      expect(DataTransformer.formatVolume(500)).toBe('500')
    })

    it('should convert timeframes correctly', () => {
      expect(DataTransformer.timeframeToMinutes('5m')).toBe(5)
      expect(DataTransformer.timeframeToMinutes('1h')).toBe(60)
      expect(DataTransformer.timeframeToMinutes('1d')).toBe(1440)
    })

    it('should normalize symbols', () => {
      expect(DataTransformer.normalizeSymbol('usdcop')).toBe('USDCOP')
      expect(DataTransformer.normalizeSymbol('  btcusd  ')).toBe('BTCUSD')
    })
  })

  describe('StatisticsCalculator', () => {
    it('should calculate price changes correctly', () => {
      const change = StatisticsCalculator.calculatePriceChange(4100, 4050)
      expect(change.change).toBe(50)
      expect(change.changePercent).toBeCloseTo(1.23, 1)
      expect(change.isPositive).toBe(true)

      const negChange = StatisticsCalculator.calculatePriceChange(4000, 4100)
      expect(negChange.change).toBe(-100)
      expect(negChange.isPositive).toBe(false)
    })

    it('should calculate SMA correctly', () => {
      const prices = [100, 102, 101, 103, 105]
      const sma = StatisticsCalculator.calculateSMA(prices, 5)
      expect(sma).toBeCloseTo(102.2, 1)
    })

    it('should return null for SMA with insufficient data', () => {
      const prices = [100, 102]
      const sma = StatisticsCalculator.calculateSMA(prices, 5)
      expect(sma).toBeNull()
    })

    it('should calculate standard deviation', () => {
      const values = [2, 4, 4, 4, 5, 5, 7, 9]
      const stdDev = StatisticsCalculator.calculateStdDev(values)
      expect(stdDev).toBeCloseTo(2, 0)
    })

    it('should calculate percentage change', () => {
      const pct = StatisticsCalculator.calculatePercentChange(100, 110)
      expect(pct).toBe(10)

      const pctNeg = StatisticsCalculator.calculatePercentChange(110, 100)
      expect(pctNeg).toBeCloseTo(-9.09, 1)
    })
  })

  describe('MarketDataService Facade', () => {
    it('should delegate formatPrice to DataTransformer', () => {
      const result1 = MarketDataService.formatPrice(4123.45, 2)
      const result2 = DataTransformer.formatPrice(4123.45, 2)
      expect(result1).toBe(result2)
    })

    it('should delegate calculatePriceChange to StatisticsCalculator', () => {
      const result1 = MarketDataService.calculatePriceChange(4100, 4050)
      const result2 = StatisticsCalculator.calculatePriceChange(4100, 4050)
      expect(result1).toEqual(result2)
    })
  })

  describe('Type Definitions', () => {
    it('should allow creating MarketDataPoint', () => {
      const point = {
        symbol: 'USDCOP',
        price: 4123.45,
        timestamp: Date.now(),
        volume: 1000,
        source: 'test',
      }
      expect(point.symbol).toBe('USDCOP')
      expect(point.price).toBe(4123.45)
    })

    it('should allow creating CandlestickData', () => {
      const candle = {
        time: Date.now(),
        open: 4100,
        high: 4150,
        low: 4090,
        close: 4123,
        volume: 1000,
      }
      expect(candle.high).toBeGreaterThanOrEqual(candle.low)
    })
  })
})
