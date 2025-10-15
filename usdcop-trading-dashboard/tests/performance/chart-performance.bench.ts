import { bench, describe } from 'vitest'
import { calculateEMA, calculateBollingerBands, calculateRSI, calculateMACD, calculateVolumeProfile, TechnicalIndicatorBuffer } from '@/lib/technical-indicators'

// Generate test data of different sizes
const generateTestData = (size: number) => {
  return Array.from({ length: size }, (_, i) => ({
    time: Date.now() + i * 60000,
    open: 4000 + Math.sin(i / 10) * 50 + (Math.random() - 0.5) * 20,
    high: 4000 + Math.sin(i / 10) * 50 + Math.random() * 30,
    low: 4000 + Math.sin(i / 10) * 50 - Math.random() * 30,
    close: 4000 + Math.sin(i / 10) * 50 + (Math.random() - 0.5) * 20,
    volume: 1000000 + Math.random() * 500000
  }))
}

// Test datasets
const smallDataset = generateTestData(100)
const mediumDataset = generateTestData(1000)
const largeDataset = generateTestData(10000)
const extraLargeDataset = generateTestData(50000)

describe('Technical Indicators Performance', () => {
  describe('EMA Calculation', () => {
    bench('EMA-20 with 100 data points', () => {
      calculateEMA(smallDataset, 20)
    })

    bench('EMA-20 with 1k data points', () => {
      calculateEMA(mediumDataset, 20)
    })

    bench('EMA-20 with 10k data points', () => {
      calculateEMA(largeDataset, 20)
    })

    bench('EMA-20 with 50k data points', () => {
      calculateEMA(extraLargeDataset, 20)
    })

    bench('EMA-200 with 10k data points', () => {
      calculateEMA(largeDataset, 200)
    })
  })

  describe('Bollinger Bands Calculation', () => {
    bench('Bollinger Bands with 100 data points', () => {
      calculateBollingerBands(smallDataset, 20, 2)
    })

    bench('Bollinger Bands with 1k data points', () => {
      calculateBollingerBands(mediumDataset, 20, 2)
    })

    bench('Bollinger Bands with 10k data points', () => {
      calculateBollingerBands(largeDataset, 20, 2)
    })

    bench('Bollinger Bands with 50k data points', () => {
      calculateBollingerBands(extraLargeDataset, 20, 2)
    })
  })

  describe('RSI Calculation', () => {
    bench('RSI-14 with 100 data points', () => {
      calculateRSI(smallDataset, 14)
    })

    bench('RSI-14 with 1k data points', () => {
      calculateRSI(mediumDataset, 14)
    })

    bench('RSI-14 with 10k data points', () => {
      calculateRSI(largeDataset, 14)
    })

    bench('RSI-14 with 50k data points', () => {
      calculateRSI(extraLargeDataset, 14)
    })
  })

  describe('MACD Calculation', () => {
    bench('MACD with 100 data points', () => {
      calculateMACD(smallDataset, 12, 26, 9)
    })

    bench('MACD with 1k data points', () => {
      calculateMACD(mediumDataset, 12, 26, 9)
    })

    bench('MACD with 10k data points', () => {
      calculateMACD(largeDataset, 12, 26, 9)
    })

    bench('MACD with 50k data points', () => {
      calculateMACD(extraLargeDataset, 12, 26, 9)
    })
  })

  describe('Volume Profile Calculation', () => {
    bench('Volume Profile with 100 data points, 50 levels', () => {
      calculateVolumeProfile(smallDataset, 50)
    })

    bench('Volume Profile with 1k data points, 50 levels', () => {
      calculateVolumeProfile(mediumDataset, 50)
    })

    bench('Volume Profile with 10k data points, 50 levels', () => {
      calculateVolumeProfile(largeDataset, 50)
    })

    bench('Volume Profile with 10k data points, 100 levels', () => {
      calculateVolumeProfile(largeDataset, 100)
    })

    bench('Volume Profile with 50k data points, 50 levels', () => {
      calculateVolumeProfile(extraLargeDataset, 50)
    })
  })

  describe('Buffer Operations', () => {
    bench('Buffer add single candle', () => {
      const buffer = new TechnicalIndicatorBuffer(1000)
      buffer.addCandle(smallDataset[0])
    })

    bench('Buffer add 100 candles', () => {
      const buffer = new TechnicalIndicatorBuffer(1000)
      smallDataset.forEach(candle => buffer.addCandle(candle))
    })

    bench('Buffer update last candle', () => {
      const buffer = new TechnicalIndicatorBuffer(1000)
      mediumDataset.forEach(candle => buffer.addCandle(candle))
      buffer.updateLastCandle(mediumDataset[mediumDataset.length - 1])
    })

    bench('Buffer get indicators', () => {
      const buffer = new TechnicalIndicatorBuffer(1000)
      mediumDataset.forEach(candle => buffer.addCandle(candle))
      buffer.getLastIndicators({
        ema20: true,
        ema50: true,
        bollinger: true,
        rsi: true,
        macd: true
      })
    })
  })

  describe('Combined Calculations', () => {
    bench('All indicators with 1k data points', () => {
      calculateEMA(mediumDataset, 20)
      calculateEMA(mediumDataset, 50)
      calculateBollingerBands(mediumDataset, 20, 2)
      calculateRSI(mediumDataset, 14)
      calculateMACD(mediumDataset, 12, 26, 9)
    })

    bench('All indicators with 10k data points', () => {
      calculateEMA(largeDataset, 20)
      calculateEMA(largeDataset, 50)
      calculateBollingerBands(largeDataset, 20, 2)
      calculateRSI(largeDataset, 14)
      calculateMACD(largeDataset, 12, 26, 9)
    })

    bench('Memory-intensive operations with 50k data points', () => {
      const results = {
        ema20: calculateEMA(extraLargeDataset, 20),
        ema50: calculateEMA(extraLargeDataset, 50),
        ema200: calculateEMA(extraLargeDataset, 200),
        bb: calculateBollingerBands(extraLargeDataset, 20, 2),
        rsi: calculateRSI(extraLargeDataset, 14),
        macd: calculateMACD(extraLargeDataset, 12, 26, 9),
        volumeProfile: calculateVolumeProfile(extraLargeDataset, 50)
      }
      // Force garbage collection by nullifying large objects
      Object.keys(results).forEach(key => {
        ;(results as any)[key] = null
      })
    })
  })
})

describe('Data Processing Performance', () => {
  describe('Array Operations', () => {
    bench('Array creation (10k elements)', () => {
      Array.from({ length: 10000 }, (_, i) => i)
    })

    bench('Array mapping (10k elements)', () => {
      largeDataset.map(item => ({ ...item, processed: true }))
    })

    bench('Array filtering (10k elements)', () => {
      largeDataset.filter(item => item.close > 4000)
    })

    bench('Array reduction (10k elements)', () => {
      largeDataset.reduce((sum, item) => sum + item.volume, 0)
    })

    bench('Array sorting (10k elements)', () => {
      [...largeDataset].sort((a, b) => b.volume - a.volume)
    })

    bench('Array slicing (10k elements)', () => {
      largeDataset.slice(-1000)
    })
  })

  describe('Mathematical Operations', () => {
    const numbers = Array.from({ length: 10000 }, () => Math.random() * 1000)

    bench('Math.sin operations (10k)', () => {
      numbers.map(n => Math.sin(n))
    })

    bench('Math.sqrt operations (10k)', () => {
      numbers.map(n => Math.sqrt(n))
    })

    bench('Math.pow operations (10k)', () => {
      numbers.map(n => Math.pow(n, 2))
    })

    bench('Variance calculation (10k)', () => {
      const mean = numbers.reduce((sum, n) => sum + n, 0) / numbers.length
      const variance = numbers.reduce((sum, n) => sum + Math.pow(n - mean, 2), 0) / numbers.length
      return variance
    })

    bench('Standard deviation calculation (10k)', () => {
      const mean = numbers.reduce((sum, n) => sum + n, 0) / numbers.length
      const variance = numbers.reduce((sum, n) => sum + Math.pow(n - mean, 2), 0) / numbers.length
      return Math.sqrt(variance)
    })
  })

  describe('JSON Operations', () => {
    const jsonData = JSON.stringify(largeDataset)

    bench('JSON.stringify (10k objects)', () => {
      JSON.stringify(largeDataset)
    })

    bench('JSON.parse (10k objects)', () => {
      JSON.parse(jsonData)
    })

    bench('Deep clone via JSON (10k objects)', () => {
      JSON.parse(JSON.stringify(largeDataset))
    })
  })

  describe('Memory Allocation', () => {
    bench('Object creation (10k objects)', () => {
      Array.from({ length: 10000 }, (_, i) => ({
        id: i,
        timestamp: Date.now(),
        value: Math.random() * 1000,
        metadata: { index: i, type: 'test' }
      }))
    })

    bench('Array destructuring (10k operations)', () => {
      largeDataset.forEach(({ time, open, high, low, close, volume }) => {
        return { time, open, high, low, close, volume }
      })
    })

    bench('Object spread (10k operations)', () => {
      largeDataset.map(item => ({ ...item, updated: Date.now() }))
    })
  })
})

describe('Real-time Simulation Performance', () => {
  describe('High-frequency Updates', () => {
    bench('Process 1000 price updates', () => {
      const buffer = new TechnicalIndicatorBuffer(5000)

      // Initialize with historical data
      mediumDataset.forEach(candle => buffer.addCandle(candle))

      // Simulate 1000 rapid updates
      for (let i = 0; i < 1000; i++) {
        const update = {
          time: Date.now() + i * 1000,
          open: 4000 + Math.random() * 20,
          high: 4000 + Math.random() * 25,
          low: 4000 - Math.random() * 25,
          close: 4000 + Math.random() * 20,
          volume: 100000 + Math.random() * 50000
        }
        buffer.updateLastCandle(update)
      }
    })

    bench('Calculate indicators on each update (100 iterations)', () => {
      const buffer = new TechnicalIndicatorBuffer(1000)
      mediumDataset.forEach(candle => buffer.addCandle(candle))

      for (let i = 0; i < 100; i++) {
        const update = {
          time: Date.now() + i * 1000,
          open: 4000 + Math.random() * 20,
          high: 4000 + Math.random() * 25,
          low: 4000 - Math.random() * 25,
          close: 4000 + Math.random() * 20,
          volume: 100000 + Math.random() * 50000
        }
        buffer.updateLastCandle(update)
        buffer.getLastIndicators({
          ema20: true,
          ema50: true,
          bollinger: true,
          rsi: true,
          macd: true
        })
      }
    })
  })

  describe('Concurrent Operations', () => {
    bench('Parallel indicator calculations', async () => {
      const promises = [
        () => calculateEMA(largeDataset, 20),
        () => calculateEMA(largeDataset, 50),
        () => calculateBollingerBands(largeDataset, 20, 2),
        () => calculateRSI(largeDataset, 14),
        () => calculateMACD(largeDataset, 12, 26, 9)
      ]

      await Promise.all(promises.map(fn => Promise.resolve(fn())))
    })

    bench('Sequential indicator calculations', () => {
      calculateEMA(largeDataset, 20)
      calculateEMA(largeDataset, 50)
      calculateBollingerBands(largeDataset, 20, 2)
      calculateRSI(largeDataset, 14)
      calculateMACD(largeDataset, 12, 26, 9)
    })
  })
})

describe('Memory Usage Benchmarks', () => {
  describe('Memory Efficiency', () => {
    bench('Large dataset processing with cleanup', () => {
      // Create large dataset
      let data = generateTestData(100000)

      // Process data
      const results = {
        ema: calculateEMA(data, 20),
        bb: calculateBollingerBands(data, 20, 2),
        rsi: calculateRSI(data, 14)
      }

      // Cleanup
      data = null as any
      results.ema.length = 0
      results.bb.upper.length = 0
      results.bb.middle.length = 0
      results.bb.lower.length = 0
      results.rsi.length = 0
    })

    bench('Buffer memory management', () => {
      const buffer = new TechnicalIndicatorBuffer(1000)

      // Fill buffer beyond capacity
      for (let i = 0; i < 2000; i++) {
        buffer.addCandle({
          time: Date.now() + i * 60000,
          open: 4000 + Math.random() * 20,
          high: 4000 + Math.random() * 25,
          low: 4000 - Math.random() * 25,
          close: 4000 + Math.random() * 20,
          volume: 100000 + Math.random() * 50000
        })
      }

      // Should maintain max size
      expect(buffer.getData().length).toBeLessThanOrEqual(1000)

      buffer.clear()
    })
  })
})

// Performance targets for CI/CD
export const performanceTargets = {
  ema: {
    '1k_data_points': 10, // milliseconds
    '10k_data_points': 50,
    '50k_data_points': 200
  },
  bollingerBands: {
    '1k_data_points': 15,
    '10k_data_points': 75,
    '50k_data_points': 300
  },
  rsi: {
    '1k_data_points': 20,
    '10k_data_points': 100,
    '50k_data_points': 400
  },
  macd: {
    '1k_data_points': 25,
    '10k_data_points': 125,
    '50k_data_points': 500
  },
  volumeProfile: {
    '1k_data_points': 30,
    '10k_data_points': 150,
    '50k_data_points': 600
  }
}