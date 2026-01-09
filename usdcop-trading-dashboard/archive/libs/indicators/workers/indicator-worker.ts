/**
 * Indicator Calculation Web Worker
 * ===============================
 *
 * High-performance Web Worker for parallel technical indicator calculations.
 * Uses the technicalindicators library for proven algorithms.
 */

import * as Comlink from 'comlink';
import {
  SMA, EMA, WMA, DEMA, TEMA, KAMA, HullMA,
  RSI, Stochastic, WilliamsR, CCI, ROC, MOM,
  MACD, PPO, TRIX, ADX, Aroon, PSAR,
  BollingerBands, KeltnerChannels, DonchianChannel,
  ATR, TrueRange, ADL, OBV, VWAP,
  IchimokuCloud, StochasticRSI, AwesomeOscillator,
  UltimateOscillator, MoneyFlowIndex, ChaikinMoneyFlow,
  VolumeWeightedAveragePrice, AverageDirectionalIndex,
  CommodityChannelIndex, RateOfChange, Momentum
} from 'technicalindicators';

import {
  CandleData,
  IndicatorConfig,
  IndicatorValue,
  MultiLineIndicator,
  BollingerBands as BBResult,
  MACD as MACDResult,
  Stochastic as StochasticResult,
  WorkerMessage,
  WorkerResponse,
  VolumeProfile,
  OrderFlow,
  MarketMicrostructure,
  CorrelationMatrix
} from '../types';

export class IndicatorWorker {
  private performanceMonitor = {
    startTime: 0,
    calculations: 0,
    totalTime: 0
  };

  /**
   * Calculate Single Moving Averages
   */
  private calculateMovingAverages(data: CandleData[], config: IndicatorConfig): IndicatorValue[] {
    const { type, period = 20, source = 'close' } = config;
    const values = this.extractValues(data, source);

    let result: number[] = [];

    switch (type) {
      case 'sma':
        result = SMA.calculate({ period, values });
        break;
      case 'ema':
        result = EMA.calculate({ period, values });
        break;
      case 'wma':
        result = WMA.calculate({ period, values });
        break;
      case 'dema':
        result = DEMA.calculate({ period, values });
        break;
      case 'tema':
        result = TEMA.calculate({ period, values });
        break;
      case 'hull':
        result = HullMA.calculate({ period, values });
        break;
      case 'kama':
        result = KAMA.calculate({ period, values });
        break;
      default:
        throw new Error(`Unsupported moving average type: ${type}`);
    }

    return this.createIndicatorValues(data, result, period - 1);
  }

  /**
   * Calculate Momentum Oscillators
   */
  private calculateMomentumIndicators(data: CandleData[], config: IndicatorConfig): IndicatorValue[] {
    const { type, period = 14 } = config;
    const values = this.extractValues(data, config.source || 'close');

    let result: number[] = [];

    switch (type) {
      case 'rsi':
        result = RSI.calculate({ period, values });
        break;
      case 'williams':
        const williamsData = data.map(d => ({ high: d.high, low: d.low, close: d.close }));
        result = WilliamsR.calculate({ period, high: data.map(d => d.high), low: data.map(d => d.low), close: data.map(d => d.close) });
        break;
      case 'cci':
        result = CCI.calculate({
          period,
          high: data.map(d => d.high),
          low: data.map(d => d.low),
          close: data.map(d => d.close)
        });
        break;
      case 'roc':
        result = ROC.calculate({ period, values });
        break;
      case 'momentum':
        result = MOM.calculate({ period, values });
        break;
      default:
        throw new Error(`Unsupported momentum indicator: ${type}`);
    }

    return this.createIndicatorValues(data, result, period - 1);
  }

  /**
   * Calculate MACD with signal and histogram
   */
  private calculateMACD(data: CandleData[], config: IndicatorConfig): MACDResult[] {
    const {
      fastPeriod = 12,
      slowPeriod = 26,
      signalPeriod = 9
    } = config.parameters || {};

    const values = this.extractValues(data, config.source || 'close');
    const macdResult = MACD.calculate({
      values,
      fastPeriod: Number(fastPeriod),
      slowPeriod: Number(slowPeriod),
      signalPeriod: Number(signalPeriod),
      SimpleMAOscillator: false,
      SimpleMASignal: false
    });

    const startIndex = data.length - macdResult.length;

    return macdResult.map((macd, index) => ({
      timestamp: data[startIndex + index].timestamp,
      value: macd.MACD || 0,
      macd: macd.MACD || 0,
      signal: macd.signal || 0,
      histogram: macd.histogram || 0,
      divergence: this.detectMACDDivergence(macdResult, index)
    }));
  }

  /**
   * Calculate Stochastic Oscillator
   */
  private calculateStochastic(data: CandleData[], config: IndicatorConfig): StochasticResult[] {
    const {
      kPeriod = 14,
      dPeriod = 3,
      slowing = 3
    } = config.parameters || {};

    const stochResult = Stochastic.calculate({
      high: data.map(d => d.high),
      low: data.map(d => d.low),
      close: data.map(d => d.close),
      period: Number(kPeriod),
      signalPeriod: Number(dPeriod)
    });

    const startIndex = data.length - stochResult.length;

    return stochResult.map((stoch, index) => ({
      timestamp: data[startIndex + index].timestamp,
      value: stoch.k || 0,
      k: stoch.k || 0,
      d: stoch.d || 0,
      crossover: this.detectStochasticCrossover(stochResult, index)
    }));
  }

  /**
   * Calculate Bollinger Bands
   */
  private calculateBollingerBands(data: CandleData[], config: IndicatorConfig): BBResult[] {
    const {
      period = 20,
      stdDev = 2
    } = config.parameters || {};

    const values = this.extractValues(data, config.source || 'close');
    const bbResult = BollingerBands.calculate({
      period: Number(period),
      values,
      stdDev: Number(stdDev)
    });

    const startIndex = data.length - bbResult.length;

    return bbResult.map((bb, index) => ({
      timestamp: data[startIndex + index].timestamp,
      value: bb.middle || 0,
      upper: bb.upper || 0,
      middle: bb.middle || 0,
      lower: bb.lower || 0,
      width: ((bb.upper || 0) - (bb.lower || 0)) / (bb.middle || 1),
      percentB: ((data[startIndex + index].close - (bb.lower || 0)) / ((bb.upper || 0) - (bb.lower || 0))) * 100
    }));
  }

  /**
   * Calculate Volume Profile
   */
  private calculateVolumeProfile(data: CandleData[], options: any = {}): VolumeProfile {
    const { levels = 50, valueAreaPercent = 0.7 } = options;

    if (!data.length) {
      return {
        levels: [],
        poc: 0,
        valueAreaHigh: 0,
        valueAreaLow: 0,
        valueAreaVolume: 0,
        totalVolume: 0,
        profiles: { session: [], composite: [] }
      };
    }

    // Find price range
    const prices = data.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceStep = (maxPrice - minPrice) / levels;

    // Initialize volume levels
    const volumeLevels = new Map<number, {
      volume: number;
      buyVolume: number;
      sellVolume: number;
    }>();

    let totalVolume = 0;

    // Calculate volume distribution
    data.forEach(candle => {
      const volume = candle.volume || 0;
      const range = candle.high - candle.low;
      totalVolume += volume;

      // Distribute volume across price levels in the candle
      for (let price = candle.low; price <= candle.high; price += priceStep / 10) {
        const levelPrice = Math.round(price / priceStep) * priceStep;

        if (!volumeLevels.has(levelPrice)) {
          volumeLevels.set(levelPrice, { volume: 0, buyVolume: 0, sellVolume: 0 });
        }

        const level = volumeLevels.get(levelPrice)!;
        const volumeWeight = range > 0 ? (priceStep / 10) / range : 1;
        const distributedVolume = volume * volumeWeight;

        level.volume += distributedVolume;

        // Estimate buy/sell volume based on price position within candle
        const pricePosition = (price - candle.low) / range;
        if (candle.close > candle.open) {
          // Bullish candle - more buying pressure at higher prices
          level.buyVolume += distributedVolume * (0.3 + 0.7 * pricePosition);
          level.sellVolume += distributedVolume * (0.7 - 0.7 * pricePosition);
        } else {
          // Bearish candle - more selling pressure at higher prices
          level.buyVolume += distributedVolume * (0.7 - 0.7 * pricePosition);
          level.sellVolume += distributedVolume * (0.3 + 0.7 * pricePosition);
        }
      }
    });

    // Convert to array and sort by volume
    const levels = Array.from(volumeLevels.entries())
      .map(([price, data]) => ({
        price,
        volume: data.volume,
        buyVolume: data.buyVolume,
        sellVolume: data.sellVolume,
        percentOfTotal: totalVolume > 0 ? (data.volume / totalVolume) * 100 : 0,
        delta: data.buyVolume - data.sellVolume
      }))
      .sort((a, b) => b.volume - a.volume);

    // Find Point of Control (highest volume level)
    const poc = levels.length > 0 ? levels[0].price : minPrice;

    // Calculate Value Area
    const targetVolume = totalVolume * valueAreaPercent;
    let valueAreaVolume = 0;
    const valueAreaPrices: number[] = [];

    // Start from POC and expand outward
    const sortedByPrice = [...levels].sort((a, b) => a.price - b.price);
    const pocIndex = sortedByPrice.findIndex(level => level.price === poc);

    let upIndex = pocIndex;
    let downIndex = pocIndex;

    while (valueAreaVolume < targetVolume && (upIndex >= 0 || downIndex < sortedByPrice.length)) {
      const upLevel = upIndex >= 0 ? sortedByPrice[upIndex] : null;
      const downLevel = downIndex < sortedByPrice.length ? sortedByPrice[downIndex] : null;

      if (upLevel && downLevel) {
        if (upLevel.volume >= downLevel.volume) {
          valueAreaVolume += upLevel.volume;
          valueAreaPrices.push(upLevel.price);
          upIndex--;
        } else {
          valueAreaVolume += downLevel.volume;
          valueAreaPrices.push(downLevel.price);
          downIndex++;
        }
      } else if (upLevel) {
        valueAreaVolume += upLevel.volume;
        valueAreaPrices.push(upLevel.price);
        upIndex--;
      } else if (downLevel) {
        valueAreaVolume += downLevel.volume;
        valueAreaPrices.push(downLevel.price);
        downIndex++;
      } else {
        break;
      }
    }

    const valueAreaHigh = valueAreaPrices.length > 0 ? Math.max(...valueAreaPrices) : maxPrice;
    const valueAreaLow = valueAreaPrices.length > 0 ? Math.min(...valueAreaPrices) : minPrice;

    return {
      levels,
      poc,
      valueAreaHigh,
      valueAreaLow,
      valueAreaVolume,
      totalVolume,
      profiles: {
        session: levels,
        composite: levels
      }
    };
  }

  /**
   * Calculate correlation matrix between multiple indicators
   */
  private calculateCorrelationMatrix(data: CandleData[], configs: IndicatorConfig[]): CorrelationMatrix {
    const indicators: { [key: string]: number[] } = {};

    // Calculate all indicators
    configs.forEach(config => {
      try {
        const result = this.calculateSingleIndicator(data, config);
        indicators[config.name] = result.map(r => r.value);
      } catch (error) {
        console.warn(`Failed to calculate ${config.name}:`, error);
      }
    });

    const assets = Object.keys(indicators);
    const n = assets.length;
    const matrix: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));

    // Calculate correlation coefficients
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1;
        } else {
          matrix[i][j] = this.pearsonCorrelation(indicators[assets[i]], indicators[assets[j]]);
        }
      }
    }

    // Calculate eigenvalues and principal components
    const { eigenvalues, eigenvectors } = this.eigenDecomposition(matrix);

    // Perform hierarchical clustering
    const clustered = this.hierarchicalClustering(assets, matrix);

    return {
      assets,
      matrix,
      eigenvalues,
      principalComponents: eigenvectors,
      clustered
    };
  }

  /**
   * Main calculation dispatcher
   */
  public calculateIndicator(data: CandleData[], config: IndicatorConfig): any {
    this.performanceMonitor.startTime = performance.now();
    this.performanceMonitor.calculations++;

    try {
      let result: any;

      switch (config.type) {
        case 'sma':
        case 'ema':
        case 'wma':
        case 'dema':
        case 'tema':
        case 'hull':
        case 'kama':
          result = this.calculateMovingAverages(data, config);
          break;

        case 'rsi':
        case 'williams':
        case 'cci':
        case 'roc':
        case 'momentum':
          result = this.calculateMomentumIndicators(data, config);
          break;

        case 'macd':
          result = this.calculateMACD(data, config);
          break;

        case 'stochastic':
          result = this.calculateStochastic(data, config);
          break;

        case 'bollinger':
          result = this.calculateBollingerBands(data, config);
          break;

        case 'volume_profile':
          result = this.calculateVolumeProfile(data, config.parameters);
          break;

        default:
          result = this.calculateSingleIndicator(data, config);
      }

      const executionTime = performance.now() - this.performanceMonitor.startTime;
      this.performanceMonitor.totalTime += executionTime;

      return result;
    } catch (error) {
      console.error(`Error calculating ${config.type}:`, error);
      throw error;
    }
  }

  /**
   * Calculate multiple indicators in batch
   */
  public calculateBatch(data: CandleData[], configs: IndicatorConfig[]): { [key: string]: any } {
    const results: { [key: string]: any } = {};

    configs.forEach(config => {
      try {
        results[config.name] = this.calculateIndicator(data, config);
      } catch (error) {
        console.error(`Failed to calculate ${config.name}:`, error);
        results[config.name] = null;
      }
    });

    return results;
  }

  // Helper methods
  private extractValues(data: CandleData[], source: string): number[] {
    switch (source) {
      case 'open': return data.map(d => d.open);
      case 'high': return data.map(d => d.high);
      case 'low': return data.map(d => d.low);
      case 'close': return data.map(d => d.close);
      case 'volume': return data.map(d => d.volume);
      case 'hlc3': return data.map(d => (d.high + d.low + d.close) / 3);
      case 'ohlc4': return data.map(d => (d.open + d.high + d.low + d.close) / 4);
      default: return data.map(d => d.close);
    }
  }

  private createIndicatorValues(data: CandleData[], values: number[], offset: number): IndicatorValue[] {
    const startIndex = Math.max(0, data.length - values.length);
    return values.map((value, index) => ({
      timestamp: data[startIndex + index].timestamp,
      value: value || 0
    }));
  }

  private calculateSingleIndicator(data: CandleData[], config: IndicatorConfig): IndicatorValue[] {
    // Generic fallback for other indicators
    const values = this.extractValues(data, config.source || 'close');
    const period = config.period || 14;

    // Simple SMA as fallback
    const result = SMA.calculate({ period, values });
    return this.createIndicatorValues(data, result, period - 1);
  }

  private detectMACDDivergence(macdData: any[], index: number): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const current = macdData[index];
    const previous = macdData[index - 5];

    if (!current || !previous) return null;

    // Simple divergence detection logic
    if (current.MACD > previous.MACD && current.histogram < previous.histogram) {
      return 'BEARISH';
    } else if (current.MACD < previous.MACD && current.histogram > previous.histogram) {
      return 'BULLISH';
    }

    return null;
  }

  private detectStochasticCrossover(stochData: any[], index: number): 'BULLISH' | 'BEARISH' | null {
    if (index < 1) return null;

    const current = stochData[index];
    const previous = stochData[index - 1];

    if (!current || !previous) return null;

    if (previous.k <= previous.d && current.k > current.d) {
      return 'BULLISH';
    } else if (previous.k >= previous.d && current.k < current.d) {
      return 'BEARISH';
    }

    return null;
  }

  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;

    const sumX = x.slice(0, n).reduce((a, b) => a + b, 0);
    const sumY = y.slice(0, n).reduce((a, b) => a + b, 0);
    const sumXY = x.slice(0, n).reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.slice(0, n).reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.slice(0, n).reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  private eigenDecomposition(matrix: number[][]): { eigenvalues: number[], eigenvectors: number[][] } {
    // Simplified eigenvalue decomposition (would use a proper library in production)
    const n = matrix.length;
    return {
      eigenvalues: Array(n).fill(1),
      eigenvectors: matrix
    };
  }

  private hierarchicalClustering(assets: string[], correlationMatrix: number[][]): {
    groups: string[][];
    distances: number[][];
  } {
    // Simplified clustering algorithm
    const n = assets.length;
    const distances = correlationMatrix.map(row => row.map(val => 1 - Math.abs(val)));

    // Simple threshold-based clustering
    const groups: string[][] = [];
    const used = new Set<number>();

    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue;

      const group = [assets[i]];
      used.add(i);

      for (let j = i + 1; j < n; j++) {
        if (used.has(j)) continue;
        if (distances[i][j] < 0.3) { // 70% correlation threshold
          group.push(assets[j]);
          used.add(j);
        }
      }

      groups.push(group);
    }

    return { groups, distances };
  }

  public getPerformanceStats() {
    return {
      ...this.performanceMonitor,
      averageTime: this.performanceMonitor.totalTime / this.performanceMonitor.calculations
    };
  }
}

// Expose the worker API
const worker = new IndicatorWorker();
Comlink.expose(worker);