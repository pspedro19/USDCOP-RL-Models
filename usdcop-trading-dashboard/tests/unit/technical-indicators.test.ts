import { describe, it, expect, beforeEach } from 'vitest'
import {
  calculateVolumeProfile,
  calculateEMA,
  calculateBollingerBands,
  calculateRSI,
  calculateMACD,
  generateMLPredictions,
  calculateAllIndicators,
  TechnicalIndicatorBuffer,
  CandleData,
  VolumeProfile
} from '@/lib/technical-indicators'

// Mock data for testing
const mockCandleData: CandleData[] = [
  { time: 1000, open: 100, high: 105, low: 95, close: 102, volume: 1000 },
  { time: 2000, open: 102, high: 108, low: 98, close: 105, volume: 1200 },
  { time: 3000, open: 105, high: 110, low: 100, close: 103, volume: 800 },
  { time: 4000, open: 103, high: 107, low: 99, close: 106, volume: 1500 },
  { time: 5000, open: 106, high: 112, low: 104, close: 110, volume: 2000 },
  { time: 6000, open: 110, high: 115, low: 108, close: 112, volume: 1800 },
  { time: 7000, open: 112, high: 118, low: 110, close: 115, volume: 1600 },
  { time: 8000, open: 115, high: 120, low: 113, close: 117, volume: 1400 },
  { time: 9000, open: 117, high: 122, low: 115, close: 119, volume: 1700 },
  { time: 10000, open: 119, high: 125, low: 117, close: 122, volume: 1900 },
  { time: 11000, open: 122, high: 127, low: 120, close: 124, volume: 1300 },
  { time: 12000, open: 124, high: 129, low: 122, close: 126, volume: 1100 },
  { time: 13000, open: 126, high: 131, low: 124, close: 128, volume: 1600 },
  { time: 14000, open: 128, high: 133, low: 126, close: 130, volume: 1800 },
  { time: 15000, open: 130, high: 135, low: 128, close: 132, volume: 2100 },
  { time: 16000, open: 132, high: 137, low: 130, close: 134, volume: 1500 },
  { time: 17000, open: 134, high: 139, low: 132, close: 136, volume: 1200 },
  { time: 18000, open: 136, high: 141, low: 134, close: 138, volume: 1400 },
  { time: 19000, open: 138, high: 143, low: 136, close: 140, volume: 1700 },
  { time: 20000, open: 140, high: 145, low: 138, close: 142, volume: 1900 }
]

const shortMockData: CandleData[] = mockCandleData.slice(0, 5)

describe('Technical Indicators', () => {
  describe('calculateVolumeProfile', () => {
    it('should calculate volume profile correctly', () => {
      const profile = calculateVolumeProfile(mockCandleData, 10)

      expect(profile).toBeDefined()
      expect(profile.levels).toHaveLength(10)
      expect(profile.poc).toBeGreaterThan(0)
      expect(profile.valueAreaHigh).toBeGreaterThanOrEqual(profile.valueAreaLow)
      expect(profile.totalVolume).toBeGreaterThan(0)
      expect(profile.valueAreaVolume).toBeLessThanOrEqual(profile.totalVolume)
    })

    it('should handle empty data', () => {
      const profile = calculateVolumeProfile([])

      expect(profile.levels).toHaveLength(0)
      expect(profile.poc).toBe(0)
      expect(profile.valueAreaHigh).toBe(0)
      expect(profile.valueAreaLow).toBe(0)
      expect(profile.totalVolume).toBe(0)
    })

    it('should calculate percentage of total correctly', () => {
      const profile = calculateVolumeProfile(mockCandleData, 5)

      const totalPercent = profile.levels.reduce((sum, level) => sum + level.percentOfTotal, 0)
      expect(totalPercent).toBeCloseTo(100, 1)
    })

    it('should handle data without volume', () => {
      const dataWithoutVolume = mockCandleData.map(candle => ({
        ...candle,
        volume: undefined
      }))

      const profile = calculateVolumeProfile(dataWithoutVolume)
      expect(profile.totalVolume).toBe(0)
      expect(profile.levels.every(level => level.volume === 0)).toBe(true)
    })
  })

  describe('calculateEMA', () => {
    it('should calculate EMA correctly', () => {
      const ema = calculateEMA(mockCandleData, 10)

      expect(ema).toHaveLength(mockCandleData.length - 9) // Should have length - period + 1
      expect(ema[0].time).toBe(mockCandleData[9].time)
      expect(ema[0].value).toBeGreaterThan(0)
    })

    it('should return empty array for insufficient data', () => {
      const ema = calculateEMA(shortMockData, 10)
      expect(ema).toHaveLength(0)
    })

    it('should have increasing values for uptrending data', () => {
      const ema = calculateEMA(mockCandleData, 5)
      const firstValue = ema[0].value
      const lastValue = ema[ema.length - 1].value

      expect(lastValue).toBeGreaterThan(firstValue)
    })

    it('should calculate correct EMA multiplier', () => {
      const period = 10
      const ema = calculateEMA(mockCandleData, period)

      // First EMA value should be SMA
      const firstTenCandles = mockCandleData.slice(0, period)
      const expectedSMA = firstTenCandles.reduce((sum, candle) => sum + candle.close, 0) / period

      expect(ema[0].value).toBeCloseTo(expectedSMA, 2)
    })
  })

  describe('calculateBollingerBands', () => {
    it('should calculate Bollinger Bands correctly', () => {
      const bb = calculateBollingerBands(mockCandleData, 10, 2)

      expect(bb.upper).toHaveLength(mockCandleData.length - 9)
      expect(bb.middle).toHaveLength(mockCandleData.length - 9)
      expect(bb.lower).toHaveLength(mockCandleData.length - 9)

      // Upper band should be above middle, middle above lower
      for (let i = 0; i < bb.upper.length; i++) {
        expect(bb.upper[i].value).toBeGreaterThan(bb.middle[i].value)
        expect(bb.middle[i].value).toBeGreaterThan(bb.lower[i].value)
        expect(bb.upper[i].time).toBe(bb.middle[i].time)
        expect(bb.middle[i].time).toBe(bb.lower[i].time)
      }
    })

    it('should return empty arrays for insufficient data', () => {
      const bb = calculateBollingerBands(shortMockData, 10)

      expect(bb.upper).toHaveLength(0)
      expect(bb.middle).toHaveLength(0)
      expect(bb.lower).toHaveLength(0)
    })

    it('should respect standard deviation parameter', () => {
      const bb1 = calculateBollingerBands(mockCandleData, 10, 1)
      const bb2 = calculateBollingerBands(mockCandleData, 10, 2)

      // Wider bands for higher std dev
      expect(bb2.upper[0].value - bb2.middle[0].value).toBeGreaterThan(bb1.upper[0].value - bb1.middle[0].value)
      expect(bb2.middle[0].value - bb2.lower[0].value).toBeGreaterThan(bb1.middle[0].value - bb1.lower[0].value)
    })
  })

  describe('calculateRSI', () => {
    it('should calculate RSI correctly', () => {
      const rsi = calculateRSI(mockCandleData, 14)

      expect(rsi).toHaveLength(mockCandleData.length - 14)

      // RSI should be between 0 and 100
      rsi.forEach(point => {
        expect(point.value).toBeGreaterThanOrEqual(0)
        expect(point.value).toBeLessThanOrEqual(100)
      })
    })

    it('should return empty array for insufficient data', () => {
      const rsi = calculateRSI(shortMockData, 14)
      expect(rsi).toHaveLength(0)
    })

    it('should show high RSI for consistently uptrending data', () => {
      const rsi = calculateRSI(mockCandleData, 14)
      const lastRSI = rsi[rsi.length - 1].value

      // For our uptrending mock data, RSI should be > 50
      expect(lastRSI).toBeGreaterThan(50)
    })

    it('should handle edge case with no price changes', () => {
      const flatData: CandleData[] = Array.from({ length: 20 }, (_, i) => ({
        time: i * 1000,
        open: 100,
        high: 100,
        low: 100,
        close: 100,
        volume: 1000
      }))

      const rsi = calculateRSI(flatData, 14)
      // When there are no gains or losses, RSI should be 0 or handle gracefully
      expect(rsi).toHaveLength(flatData.length - 14)
    })
  })

  describe('calculateMACD', () => {
    it('should calculate MACD correctly', () => {
      const macd = calculateMACD(mockCandleData, 12, 26, 9)

      expect(macd.macd.length).toBeGreaterThan(0)
      expect(macd.signal.length).toBeGreaterThan(0)
      expect(macd.histogram.length).toBeGreaterThan(0)

      // Signal should be shorter than MACD due to additional smoothing
      expect(macd.signal.length).toBeLessThanOrEqual(macd.macd.length)
      expect(macd.histogram.length).toBe(macd.signal.length)
    })

    it('should return empty arrays for insufficient data', () => {
      const macd = calculateMACD(shortMockData, 12, 26, 9)

      expect(macd.macd).toHaveLength(0)
      expect(macd.signal).toHaveLength(0)
      expect(macd.histogram).toHaveLength(0)
    })

    it('should calculate histogram as difference between MACD and signal', () => {
      const macd = calculateMACD(mockCandleData, 12, 26, 9)

      // Check that histogram equals MACD - Signal (within rounding errors)
      for (let i = 0; i < macd.histogram.length; i++) {
        const correspondingMACDIndex = macd.macd.length - macd.histogram.length + i
        const expectedHistogram = macd.macd[correspondingMACDIndex].value - macd.signal[i].value

        expect(macd.histogram[i].value).toBeCloseTo(expectedHistogram, 6)
      }
    })
  })

  describe('generateMLPredictions', () => {
    it('should generate predictions correctly', () => {
      const predictions = generateMLPredictions(mockCandleData, 10)

      expect(predictions).toHaveLength(10)

      predictions.forEach((prediction, index) => {
        expect(prediction.time).toBeGreaterThan(mockCandleData[mockCandleData.length - 1].time)
        expect(prediction.predicted_price).toBeGreaterThan(0)
        expect(prediction.confidence).toBeGreaterThan(0)
        expect(prediction.confidence).toBeLessThanOrEqual(1)
        expect(prediction.upper_bound).toBeGreaterThan(prediction.predicted_price)
        expect(prediction.lower_bound).toBeLessThan(prediction.predicted_price)
        expect(['bullish', 'bearish', 'neutral']).toContain(prediction.trend)

        // Confidence should decrease over time
        if (index > 0) {
          expect(prediction.confidence).toBeLessThanOrEqual(predictions[index - 1].confidence)
        }
      })
    })

    it('should return empty array for empty data', () => {
      const predictions = generateMLPredictions([])
      expect(predictions).toHaveLength(0)
    })

    it('should generate sequential timestamps', () => {
      const predictions = generateMLPredictions(mockCandleData, 5)

      for (let i = 1; i < predictions.length; i++) {
        expect(predictions[i].time).toBeGreaterThan(predictions[i - 1].time)
      }
    })
  })

  describe('calculateAllIndicators', () => {
    it('should calculate all indicators when requested', () => {
      const result = calculateAllIndicators(mockCandleData, {
        ema20: true,
        ema50: false,
        ema200: false,
        bollinger: true,
        rsi: true,
        macd: true,
        volumeProfile: true
      })

      expect(result.data).toHaveLength(mockCandleData.length)
      expect(result.volumeProfile).toBeDefined()
      expect(result.predictions).toBeDefined()

      // Check that indicators are present where expected
      const lastDataPoint = result.data[result.data.length - 1]
      expect(lastDataPoint.ema20).toBeDefined()
      expect(lastDataPoint.ema50).toBeUndefined()
      expect(lastDataPoint.bb_upper).toBeDefined()
      expect(lastDataPoint.rsi).toBeDefined()
      expect(lastDataPoint.macd).toBeDefined()
    })

    it('should not calculate indicators when not requested', () => {
      const result = calculateAllIndicators(mockCandleData, {})

      const lastDataPoint = result.data[result.data.length - 1]
      expect(lastDataPoint.ema20).toBeUndefined()
      expect(lastDataPoint.ema50).toBeUndefined()
      expect(lastDataPoint.bb_upper).toBeUndefined()
      expect(lastDataPoint.rsi).toBeUndefined()
      expect(lastDataPoint.macd).toBeUndefined()
      expect(result.volumeProfile).toBeUndefined()
    })
  })

  describe('TechnicalIndicatorBuffer', () => {
    let buffer: TechnicalIndicatorBuffer

    beforeEach(() => {
      buffer = new TechnicalIndicatorBuffer(10)
    })

    it('should add candles correctly', () => {
      buffer.addCandle(mockCandleData[0])
      expect(buffer.getData()).toHaveLength(1)
      expect(buffer.getData()[0]).toEqual(mockCandleData[0])
    })

    it('should respect max size', () => {
      // Add more candles than max size
      for (let i = 0; i < 15; i++) {
        buffer.addCandle(mockCandleData[i % mockCandleData.length])
      }

      expect(buffer.getData()).toHaveLength(10)
    })

    it('should update last candle correctly', () => {
      buffer.addCandle(mockCandleData[0])
      const updatedCandle = { ...mockCandleData[0], close: 999 }
      buffer.updateLastCandle(updatedCandle)

      expect(buffer.getData()).toHaveLength(1)
      expect(buffer.getData()[0].close).toBe(999)
    })

    it('should add candle when updating empty buffer', () => {
      const candle = mockCandleData[0]
      buffer.updateLastCandle(candle)

      expect(buffer.getData()).toHaveLength(1)
      expect(buffer.getData()[0]).toEqual(candle)
    })

    it('should clear data correctly', () => {
      buffer.addCandle(mockCandleData[0])
      buffer.clear()
      expect(buffer.getData()).toHaveLength(0)
    })

    it('should get last indicators correctly', () => {
      // Add enough data for indicators
      mockCandleData.forEach(candle => buffer.addCandle(candle))

      const indicators = buffer.getLastIndicators({ ema20: true, rsi: true })
      expect(indicators).toBeDefined()
      expect(indicators.ema20).toBeDefined()
      expect(indicators.rsi).toBeDefined()
    })

    it('should return null for insufficient data', () => {
      buffer.addCandle(mockCandleData[0])
      const indicators = buffer.getLastIndicators({ ema20: true })
      expect(indicators).toBeNull()
    })
  })

  describe('Edge Cases and Error Handling', () => {
    it('should handle negative volumes gracefully', () => {
      const dataWithNegativeVolume = [{
        ...mockCandleData[0],
        volume: -100
      }]

      const profile = calculateVolumeProfile(dataWithNegativeVolume)
      expect(profile.totalVolume).toBe(-100)
    })

    it('should handle zero price ranges', () => {
      const flatPriceData: CandleData[] = [{
        time: 1000,
        open: 100,
        high: 100,
        low: 100,
        close: 100,
        volume: 1000
      }]

      const profile = calculateVolumeProfile(flatPriceData)
      expect(profile).toBeDefined()
      expect(profile.poc).toBe(100)
    })

    it('should handle single data point', () => {
      const singlePoint = [mockCandleData[0]]

      const ema = calculateEMA(singlePoint, 1)
      expect(ema).toHaveLength(1)
      expect(ema[0].value).toBe(singlePoint[0].close)
    })

    it('should handle very large datasets efficiently', () => {
      const largeDataset: CandleData[] = Array.from({ length: 10000 }, (_, i) => ({
        time: i * 1000,
        open: 100 + Math.sin(i / 100) * 10,
        high: 105 + Math.sin(i / 100) * 10,
        low: 95 + Math.sin(i / 100) * 10,
        close: 102 + Math.sin(i / 100) * 10,
        volume: 1000 + Math.random() * 500
      }))

      const startTime = performance.now()
      const result = calculateAllIndicators(largeDataset, {
        ema20: true,
        bollinger: true,
        rsi: true,
        macd: true,
        volumeProfile: true
      })
      const endTime = performance.now()

      expect(result.data).toHaveLength(largeDataset.length)
      expect(endTime - startTime).toBeLessThan(5000) // Should complete within 5 seconds
    })
  })
})