'use client'

/**
 * Technical Indicators Library
 * ===========================
 *
 * Professional-grade technical indicator calculations for trading charts.
 * Includes Volume Profile, enhanced EMAs, Bollinger Bands, and ML prediction overlays.
 */

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface VolumeProfileLevel {
  price: number;
  volume: number;
  percentOfTotal: number;
}

export interface VolumeProfile {
  levels: VolumeProfileLevel[];
  poc: number; // Point of Control (highest volume price)
  valueAreaHigh: number;
  valueAreaLow: number;
  valueAreaVolume: number; // 70% of total volume
  totalVolume: number;
}

export interface TechnicalIndicators {
  ema20?: number;
  ema50?: number;
  ema200?: number;
  bb_upper?: number;
  bb_middle?: number;
  bb_lower?: number;
  rsi?: number;
  macd?: number;
  macd_signal?: number;
  macd_histogram?: number;
  volume_sma?: number;
}

export interface MLPrediction {
  time: number;
  predicted_price: number;
  confidence: number;
  upper_bound: number;
  lower_bound: number;
  trend: 'bullish' | 'bearish' | 'neutral';
}

/**
 * Calculate Volume Profile for given price data
 */
export function calculateVolumeProfile(
  data: CandleData[],
  numberOfLevels: number = 50
): VolumeProfile {
  if (!data.length) {
    return {
      levels: [],
      poc: 0,
      valueAreaHigh: 0,
      valueAreaLow: 0,
      valueAreaVolume: 0,
      totalVolume: 0
    };
  }

  // Find price range
  const prices = data.flatMap(d => [d.high, d.low]);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceStep = (maxPrice - minPrice) / numberOfLevels;

  // Initialize volume levels
  const volumeLevels: { [price: number]: number } = {};
  let totalVolume = 0;

  // Calculate volume at each price level
  data.forEach(candle => {
    const volume = candle.volume || 0;
    const avgPrice = (candle.high + candle.low + candle.close) / 3;

    // Distribute volume across the candle's price range
    const startLevel = Math.floor((candle.low - minPrice) / priceStep);
    const endLevel = Math.floor((candle.high - minPrice) / priceStep);

    for (let level = startLevel; level <= Math.min(endLevel, numberOfLevels - 1); level++) {
      const levelPrice = minPrice + (level * priceStep);
      if (!volumeLevels[levelPrice]) {
        volumeLevels[levelPrice] = 0;
      }
      // Weight volume by how much of the candle's range this level represents
      const levelWeight = Math.min(priceStep, candle.high - Math.max(candle.low, levelPrice));
      const candleRange = candle.high - candle.low;
      volumeLevels[levelPrice] += volume * (candleRange > 0 ? levelWeight / candleRange : 1);
    }

    totalVolume += volume;
  });

  // Convert to sorted array
  const levels: VolumeProfileLevel[] = Object.entries(volumeLevels)
    .map(([price, volume]) => ({
      price: parseFloat(price),
      volume,
      percentOfTotal: totalVolume > 0 ? (volume / totalVolume) * 100 : 0
    }))
    .sort((a, b) => b.volume - a.volume);

  // Find Point of Control (highest volume)
  const poc = levels.length > 0 ? levels[0].price : 0;

  // Calculate Value Area (70% of volume)
  const targetVolume = totalVolume * 0.7;
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

    // Choose the level with higher volume
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
    totalVolume
  };
}

/**
 * Calculate Enhanced EMA with multiple periods
 */
export function calculateEMA(data: CandleData[], period: number): Array<{time: number, value: number}> {
  if (data.length < period) return [];

  const multiplier = 2 / (period + 1);
  const result: Array<{time: number, value: number}> = [];

  // Start with SMA for the first value
  let ema = data.slice(0, period).reduce((sum, candle) => sum + candle.close, 0) / period;
  result.push({ time: data[period - 1].time, value: ema });

  // Calculate EMA for remaining values
  for (let i = period; i < data.length; i++) {
    ema = (data[i].close - ema) * multiplier + ema;
    result.push({ time: data[i].time, value: ema });
  }

  return result;
}

/**
 * Calculate Bollinger Bands with configurable parameters
 */
export function calculateBollingerBands(
  data: CandleData[],
  period: number = 20,
  stdDev: number = 2
): {
  upper: Array<{time: number, value: number}>;
  middle: Array<{time: number, value: number}>;
  lower: Array<{time: number, value: number}>;
} {
  if (data.length < period) {
    return { upper: [], middle: [], lower: [] };
  }

  const upper: Array<{time: number, value: number}> = [];
  const middle: Array<{time: number, value: number}> = [];
  const lower: Array<{time: number, value: number}> = [];

  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const sma = slice.reduce((sum, candle) => sum + candle.close, 0) / period;
    const variance = slice.reduce((sum, candle) => sum + Math.pow(candle.close - sma, 2), 0) / period;
    const std = Math.sqrt(variance);

    const time = data[i].time;
    upper.push({ time, value: sma + std * stdDev });
    middle.push({ time, value: sma });
    lower.push({ time, value: sma - std * stdDev });
  }

  return { upper, middle, lower };
}

/**
 * Calculate RSI (Relative Strength Index)
 */
export function calculateRSI(data: CandleData[], period: number = 14): Array<{time: number, value: number}> {
  if (data.length < period + 1) return [];

  const result: Array<{time: number, value: number}> = [];
  const gains: number[] = [];
  const losses: number[] = [];

  // Calculate initial gains and losses
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close;
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }

  // Calculate initial averages
  let avgGain = gains.slice(0, period).reduce((sum, gain) => sum + gain, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((sum, loss) => sum + loss, 0) / period;

  // Calculate RSI for initial period
  let rs = avgGain / avgLoss;
  let rsi = 100 - (100 / (1 + rs));
  result.push({ time: data[period].time, value: rsi });

  // Calculate RSI for remaining periods using Wilder's smoothing
  for (let i = period + 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = ((avgGain * (period - 1)) + gain) / period;
    avgLoss = ((avgLoss * (period - 1)) + loss) / period;

    rs = avgGain / avgLoss;
    rsi = 100 - (100 / (1 + rs));
    result.push({ time: data[i].time, value: rsi });
  }

  return result;
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 */
export function calculateMACD(
  data: CandleData[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): {
  macd: Array<{time: number, value: number}>;
  signal: Array<{time: number, value: number}>;
  histogram: Array<{time: number, value: number}>;
} {
  if (data.length < slowPeriod) {
    return { macd: [], signal: [], histogram: [] };
  }

  const fastEMA = calculateEMA(data, fastPeriod);
  const slowEMA = calculateEMA(data, slowPeriod);

  // Calculate MACD line
  const macdLine: Array<{time: number, value: number}> = [];
  const startIndex = Math.max(fastEMA.length, slowEMA.length) - Math.min(fastEMA.length, slowEMA.length);

  for (let i = startIndex; i < Math.min(fastEMA.length, slowEMA.length); i++) {
    const fastValue = fastEMA[i]?.value || 0;
    const slowValue = slowEMA[i]?.value || 0;
    macdLine.push({
      time: fastEMA[i].time,
      value: fastValue - slowValue
    });
  }

  // Calculate signal line (EMA of MACD)
  const signalEMA = calculateEMAFromValues(macdLine.map(m => m.value), signalPeriod);
  const signal: Array<{time: number, value: number}> = [];

  for (let i = signalPeriod - 1; i < macdLine.length; i++) {
    signal.push({
      time: macdLine[i].time,
      value: signalEMA[i - signalPeriod + 1]
    });
  }

  // Calculate histogram
  const histogram: Array<{time: number, value: number}> = [];
  const startHistogramIndex = macdLine.length - signal.length;

  for (let i = 0; i < signal.length; i++) {
    const macdValue = macdLine[startHistogramIndex + i]?.value || 0;
    histogram.push({
      time: signal[i].time,
      value: macdValue - signal[i].value
    });
  }

  return { macd: macdLine, signal, histogram };
}

/**
 * Helper function to calculate EMA from array of values
 */
function calculateEMAFromValues(values: number[], period: number): number[] {
  if (values.length < period) return [];

  const multiplier = 2 / (period + 1);
  const result: number[] = [];

  // Start with SMA
  let ema = values.slice(0, period).reduce((sum, val) => sum + val, 0) / period;
  result.push(ema);

  // Calculate EMA for remaining values
  for (let i = period; i < values.length; i++) {
    ema = (values[i] - ema) * multiplier + ema;
    result.push(ema);
  }

  return result;
}

/**
 * Generate ML prediction placeholder data
 * This would connect to your actual ML model in production
 */
export function generateMLPredictions(
  data: CandleData[],
  predictionSteps: number = 24
): MLPrediction[] {
  if (data.length === 0) return [];

  const lastCandle = data[data.length - 1];
  const recentPrices = data.slice(-20).map(d => d.close);
  const avgPrice = recentPrices.reduce((sum, price) => sum + price, 0) / recentPrices.length;
  const volatility = Math.sqrt(recentPrices.reduce((sum, price) => sum + Math.pow(price - avgPrice, 2), 0) / recentPrices.length);

  const predictions: MLPrediction[] = [];
  let currentPrice = lastCandle.close;
  const timeStep = 5 * 60 * 1000; // 5 minutes in milliseconds

  for (let i = 1; i <= predictionSteps; i++) {
    // Simple trend analysis based on recent price movement
    const recentTrend = data.slice(-5).reduce((sum, candle, idx) => {
      if (idx === 0) return 0;
      return sum + (candle.close - data[data.length - 5 + idx - 1].close);
    }, 0) / 4;

    // Generate predicted price using trend analysis only (no random noise in production)
    const trendFactor = recentTrend * 0.1;
    // REMOVED: const randomFactor = (Math.random() - 0.5) * volatility * 0.5;
    // ML predictions should be deterministic, not random
    const predictedPrice = currentPrice + trendFactor;

    // Calculate confidence (decreases with time)
    const confidence = Math.max(0.5, 0.95 - (i * 0.02));

    // Calculate bounds
    const boundRange = volatility * (1 - confidence) * 2;
    const upperBound = predictedPrice + boundRange;
    const lowerBound = predictedPrice - boundRange;

    // Determine trend
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (predictedPrice > currentPrice * 1.001) trend = 'bullish';
    else if (predictedPrice < currentPrice * 0.999) trend = 'bearish';

    predictions.push({
      time: lastCandle.time + (i * timeStep / 1000), // Convert to seconds
      predicted_price: predictedPrice,
      confidence,
      upper_bound: upperBound,
      lower_bound: lowerBound,
      trend
    });

    currentPrice = predictedPrice;
  }

  return predictions;
}

/**
 * Calculate comprehensive technical indicators for a dataset
 */
export function calculateAllIndicators(
  data: CandleData[],
  config: {
    ema20?: boolean;
    ema50?: boolean;
    ema200?: boolean;
    bollinger?: boolean;
    rsi?: boolean;
    macd?: boolean;
    volumeProfile?: boolean;
  } = {}
): {
  data: (CandleData & TechnicalIndicators)[];
  volumeProfile?: VolumeProfile;
  predictions?: MLPrediction[];
} {
  const result: (CandleData & TechnicalIndicators)[] = data.map(candle => ({ ...candle }));

  // Calculate EMAs
  if (config.ema20) {
    const ema20 = calculateEMA(data, 20);
    ema20.forEach((ema, index) => {
      const dataIndex = data.findIndex(d => d.time === ema.time);
      if (dataIndex >= 0) {
        result[dataIndex].ema20 = ema.value;
      }
    });
  }

  if (config.ema50) {
    const ema50 = calculateEMA(data, 50);
    ema50.forEach((ema, index) => {
      const dataIndex = data.findIndex(d => d.time === ema.time);
      if (dataIndex >= 0) {
        result[dataIndex].ema50 = ema.value;
      }
    });
  }

  if (config.ema200) {
    const ema200 = calculateEMA(data, 200);
    ema200.forEach((ema, index) => {
      const dataIndex = data.findIndex(d => d.time === ema.time);
      if (dataIndex >= 0) {
        result[dataIndex].ema200 = ema.value;
      }
    });
  }

  // Calculate Bollinger Bands
  if (config.bollinger) {
    const bb = calculateBollingerBands(data);
    bb.upper.forEach((band, index) => {
      const dataIndex = data.findIndex(d => d.time === band.time);
      if (dataIndex >= 0) {
        result[dataIndex].bb_upper = bb.upper[index].value;
        result[dataIndex].bb_middle = bb.middle[index].value;
        result[dataIndex].bb_lower = bb.lower[index].value;
      }
    });
  }

  // Calculate RSI
  if (config.rsi) {
    const rsi = calculateRSI(data);
    rsi.forEach((rsiPoint, index) => {
      const dataIndex = data.findIndex(d => d.time === rsiPoint.time);
      if (dataIndex >= 0) {
        result[dataIndex].rsi = rsiPoint.value;
      }
    });
  }

  // Calculate MACD
  if (config.macd) {
    const macd = calculateMACD(data);
    macd.macd.forEach((macdPoint, index) => {
      const dataIndex = data.findIndex(d => d.time === macdPoint.time);
      if (dataIndex >= 0) {
        result[dataIndex].macd = macdPoint.value;
      }
    });

    macd.signal.forEach((signalPoint, index) => {
      const dataIndex = data.findIndex(d => d.time === signalPoint.time);
      if (dataIndex >= 0) {
        result[dataIndex].macd_signal = signalPoint.value;
      }
    });

    macd.histogram.forEach((histPoint, index) => {
      const dataIndex = data.findIndex(d => d.time === histPoint.time);
      if (dataIndex >= 0) {
        result[dataIndex].macd_histogram = histPoint.value;
      }
    });
  }

  const response: {
    data: (CandleData & TechnicalIndicators)[];
    volumeProfile?: VolumeProfile;
    predictions?: MLPrediction[];
  } = { data: result };

  // Calculate Volume Profile
  if (config.volumeProfile) {
    response.volumeProfile = calculateVolumeProfile(data);
  }

  // Generate ML predictions
  response.predictions = generateMLPredictions(data);

  return response;
}

/**
 * Real-time data buffer for efficient updates
 */
export class TechnicalIndicatorBuffer {
  private data: CandleData[] = [];
  private maxSize: number;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  addCandle(candle: CandleData): void {
    this.data.push(candle);
    if (this.data.length > this.maxSize) {
      this.data.shift();
    }
  }

  updateLastCandle(candle: CandleData): void {
    if (this.data.length > 0) {
      this.data[this.data.length - 1] = candle;
    } else {
      this.addCandle(candle);
    }
  }

  getData(): CandleData[] {
    return [...this.data];
  }

  getLastIndicators(config: Record<string, unknown>): Record<string, unknown> | null {
    if (this.data.length < 20) return null;

    const indicators = calculateAllIndicators(this.data, config);
    return indicators.data[indicators.data.length - 1];
  }

  clear(): void {
    this.data = [];
  }
}