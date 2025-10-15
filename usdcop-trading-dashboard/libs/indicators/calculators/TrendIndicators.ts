/**
 * Trend Indicators
 * ===============
 *
 * Professional implementations of trend-following indicators
 * with enhanced signal detection and divergence analysis.
 */

import {
  SMA, EMA, WMA, DEMA, TEMA, KAMA, HullMA,
  MACD, PSAR, Supertrend, IchimokuCloud, TrueRange, ATR
} from 'technicalindicators';

import { CandleData, IndicatorValue, MultiLineIndicator } from '../types';

export class TrendIndicators {
  /**
   * Enhanced Moving Averages with adaptive smoothing
   */
  static calculateAdaptiveEMA(data: CandleData[], period: number = 21, sensitivity: number = 2): IndicatorValue[] {
    if (data.length < period) return [];

    const closes = data.map(d => d.close);
    const volatility = this.calculateVolatility(data, period);

    const result: IndicatorValue[] = [];
    let ema = closes.slice(0, period).reduce((sum, price) => sum + price, 0) / period;

    for (let i = period; i < data.length; i++) {
      const vol = volatility[i - period] || 1;
      const adaptiveAlpha = (2 / (period + 1)) * (1 + sensitivity * vol);
      const clampedAlpha = Math.min(Math.max(adaptiveAlpha, 0.1), 0.9);

      ema = closes[i] * clampedAlpha + ema * (1 - clampedAlpha);

      result.push({
        timestamp: data[i].timestamp,
        value: ema,
        signal: this.generateTrendSignal(closes[i], ema, closes[i - 1], result[result.length - 1]?.value)
      });
    }

    return result;
  }

  /**
   * Multi-timeframe Moving Average Convergence
   */
  static calculateMACD(
    data: CandleData[],
    fastPeriod: number = 12,
    slowPeriod: number = 26,
    signalPeriod: number = 9
  ): Array<{
    timestamp: number;
    macd: number;
    signal: number;
    histogram: number;
    divergence?: 'BULLISH' | 'BEARISH' | null;
    crossover?: 'BULLISH' | 'BEARISH' | null;
  }> {
    if (data.length < slowPeriod + signalPeriod) return [];

    const closes = data.map(d => d.close);
    const macdData = MACD.calculate({
      values: closes,
      fastPeriod,
      slowPeriod,
      signalPeriod,
      SimpleMAOscillator: false,
      SimpleMASignal: false
    });

    const startIndex = data.length - macdData.length;

    return macdData.map((macd, index) => {
      const dataIndex = startIndex + index;

      return {
        timestamp: data[dataIndex].timestamp,
        macd: macd.MACD || 0,
        signal: macd.signal || 0,
        histogram: macd.histogram || 0,
        divergence: this.detectMACDDivergence(data, macdData, index, dataIndex),
        crossover: this.detectMACDCrossover(macdData, index)
      };
    });
  }

  /**
   * Enhanced Parabolic SAR with trend strength
   */
  static calculateParabolicSAR(
    data: CandleData[],
    step: number = 0.02,
    max: number = 0.2
  ): Array<{
    timestamp: number;
    value: number;
    trend: 'UP' | 'DOWN';
    strength: number;
    signal?: 'BUY' | 'SELL';
  }> {
    if (data.length < 10) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    const psarData = PSAR.calculate({ high: highs, low: lows, step, max });
    const startIndex = data.length - psarData.length;

    return psarData.map((psar, index) => {
      const dataIndex = startIndex + index;
      const currentPrice = closes[dataIndex];
      const trend = currentPrice > psar ? 'UP' : 'DOWN';

      // Calculate trend strength based on price distance from SAR
      const strength = Math.abs(currentPrice - psar) / currentPrice;

      // Generate signals on trend change
      let signal: 'BUY' | 'SELL' | undefined;
      if (index > 0) {
        const prevPsar = psarData[index - 1];
        const prevPrice = closes[dataIndex - 1];
        const prevTrend = prevPrice > prevPsar ? 'UP' : 'DOWN';

        if (trend !== prevTrend) {
          signal = trend === 'UP' ? 'BUY' : 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: psar,
        trend,
        strength: Math.min(strength * 100, 100),
        signal
      };
    });
  }

  /**
   * Supertrend with volatility adjustment
   */
  static calculateSupertrend(
    data: CandleData[],
    period: number = 10,
    multiplier: number = 3
  ): Array<{
    timestamp: number;
    value: number;
    trend: 'UP' | 'DOWN';
    support: number;
    resistance: number;
    signal?: 'BUY' | 'SELL';
  }> {
    if (data.length < period + 1) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    // Calculate ATR for volatility
    const atrData = ATR.calculate({
      high: highs,
      low: lows,
      close: closes,
      period
    });

    const result: Array<{
      timestamp: number;
      value: number;
      trend: 'UP' | 'DOWN';
      support: number;
      resistance: number;
      signal?: 'BUY' | 'SELL';
    }> = [];

    const startIndex = data.length - atrData.length;
    let trend = 'UP' as 'UP' | 'DOWN';
    let supertrend = 0;

    for (let i = 0; i < atrData.length; i++) {
      const dataIndex = startIndex + i;
      const hl2 = (highs[dataIndex] + lows[dataIndex]) / 2;
      const atr = atrData[i];

      const upperBand = hl2 + (multiplier * atr);
      const lowerBand = hl2 - (multiplier * atr);

      // Calculate supertrend
      const prevClose = i > 0 ? closes[dataIndex - 1] : closes[dataIndex];
      const prevSupertrend = i > 0 ? result[i - 1].value : lowerBand;

      if (closes[dataIndex] > prevSupertrend) {
        trend = 'UP';
        supertrend = Math.max(lowerBand, prevSupertrend);
      } else {
        trend = 'DOWN';
        supertrend = Math.min(upperBand, prevSupertrend);
      }

      // Generate signals
      let signal: 'BUY' | 'SELL' | undefined;
      if (i > 0) {
        const prevTrend = result[i - 1].trend;
        if (trend !== prevTrend) {
          signal = trend === 'UP' ? 'BUY' : 'SELL';
        }
      }

      result.push({
        timestamp: data[dataIndex].timestamp,
        value: supertrend,
        trend,
        support: trend === 'UP' ? supertrend : 0,
        resistance: trend === 'DOWN' ? supertrend : 0,
        signal
      });
    }

    return result;
  }

  /**
   * Ichimoku Cloud with full analysis
   */
  static calculateIchimoku(
    data: CandleData[],
    tenkanPeriod: number = 9,
    kijunPeriod: number = 26,
    senkouPeriod: number = 52,
    displacement: number = 26
  ): Array<{
    timestamp: number;
    tenkanSen: number;
    kijunSen: number;
    senkouSpanA: number;
    senkouSpanB: number;
    chikouSpan: number;
    cloudColor: 'GREEN' | 'RED';
    pricePosition: 'ABOVE_CLOUD' | 'IN_CLOUD' | 'BELOW_CLOUD';
    signal?: 'BUY' | 'SELL' | 'NEUTRAL';
  }> {
    if (data.length < senkouPeriod + displacement) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    const ichimokuData = IchimokuCloud.calculate({
      high: highs,
      low: lows,
      close: closes,
      tenkanPeriod,
      kijunPeriod,
      senkouPeriod,
      displacement
    });

    const startIndex = data.length - ichimokuData.length;

    return ichimokuData.map((ichimoku, index) => {
      const dataIndex = startIndex + index;
      const currentPrice = closes[dataIndex];

      // Determine cloud color
      const cloudColor = (ichimoku.senkouSpanA || 0) > (ichimoku.senkouSpanB || 0) ? 'GREEN' : 'RED';

      // Determine price position relative to cloud
      const cloudTop = Math.max(ichimoku.senkouSpanA || 0, ichimoku.senkouSpanB || 0);
      const cloudBottom = Math.min(ichimoku.senkouSpanA || 0, ichimoku.senkouSpanB || 0);

      let pricePosition: 'ABOVE_CLOUD' | 'IN_CLOUD' | 'BELOW_CLOUD';
      if (currentPrice > cloudTop) {
        pricePosition = 'ABOVE_CLOUD';
      } else if (currentPrice < cloudBottom) {
        pricePosition = 'BELOW_CLOUD';
      } else {
        pricePosition = 'IN_CLOUD';
      }

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevIchimoku = ichimokuData[index - 1];
        const tenkanCrossover = this.detectLineCrossover(
          ichimoku.tenkanSen || 0,
          ichimoku.kijunSen || 0,
          prevIchimoku.tenkanSen || 0,
          prevIchimoku.kijunSen || 0
        );

        if (tenkanCrossover === 'BULLISH' && pricePosition === 'ABOVE_CLOUD') {
          signal = 'BUY';
        } else if (tenkanCrossover === 'BEARISH' && pricePosition === 'BELOW_CLOUD') {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        tenkanSen: ichimoku.tenkanSen || 0,
        kijunSen: ichimoku.kijunSen || 0,
        senkouSpanA: ichimoku.senkouSpanA || 0,
        senkouSpanB: ichimoku.senkouSpanB || 0,
        chikouSpan: ichimoku.chikouSpan || 0,
        cloudColor,
        pricePosition,
        signal
      };
    });
  }

  /**
   * Triple Exponential Moving Average (TEMA) with trend analysis
   */
  static calculateTEMA(data: CandleData[], period: number = 21): IndicatorValue[] {
    if (data.length < period * 3) return [];

    const closes = data.map(d => d.close);
    const temaData = TEMA.calculate({ period, values: closes });
    const startIndex = data.length - temaData.length;

    return temaData.map((tema, index) => {
      const dataIndex = startIndex + index;

      // Calculate trend strength
      const trendStrength = index > 5 ?
        this.calculateTrendStrength(temaData.slice(Math.max(0, index - 5), index + 1)) : 0;

      return {
        timestamp: data[dataIndex].timestamp,
        value: tema,
        confidence: Math.min(trendStrength * 100, 100),
        signal: this.generateTrendSignal(
          closes[dataIndex],
          tema,
          closes[dataIndex - 1] || closes[dataIndex],
          temaData[index - 1] || tema
        )
      };
    });
  }

  /**
   * Kaufman's Adaptive Moving Average (KAMA)
   */
  static calculateKAMA(data: CandleData[], period: number = 14): IndicatorValue[] {
    if (data.length < period + 10) return [];

    const closes = data.map(d => d.close);
    const kamaData = KAMA.calculate({ period, values: closes });
    const startIndex = data.length - kamaData.length;

    return kamaData.map((kama, index) => {
      const dataIndex = startIndex + index;

      // Calculate efficiency ratio for confidence
      const efficiency = index >= period ?
        this.calculateEfficiencyRatio(closes.slice(dataIndex - period, dataIndex + 1)) : 0;

      return {
        timestamp: data[dataIndex].timestamp,
        value: kama,
        confidence: efficiency * 100,
        signal: this.generateTrendSignal(
          closes[dataIndex],
          kama,
          closes[dataIndex - 1] || closes[dataIndex],
          kamaData[index - 1] || kama
        )
      };
    });
  }

  // Helper methods

  private static calculateVolatility(data: CandleData[], period: number): number[] {
    const returns: number[] = [];

    for (let i = 1; i < data.length; i++) {
      const ret = Math.log(data[i].close / data[i - 1].close);
      returns.push(ret);
    }

    const volatility: number[] = [];

    for (let i = period - 1; i < returns.length; i++) {
      const slice = returns.slice(i - period + 1, i + 1);
      const mean = slice.reduce((sum, ret) => sum + ret, 0) / slice.length;
      const variance = slice.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / slice.length;
      volatility.push(Math.sqrt(variance));
    }

    return volatility;
  }

  private static generateTrendSignal(
    currentPrice: number,
    currentMA: number,
    previousPrice: number,
    previousMA: number
  ): 'BUY' | 'SELL' | 'HOLD' {
    const wasAbove = previousPrice > previousMA;
    const isAbove = currentPrice > currentMA;

    if (!wasAbove && isAbove) return 'BUY';
    if (wasAbove && !isAbove) return 'SELL';
    return 'HOLD';
  }

  private static detectMACDDivergence(
    data: CandleData[],
    macdData: any[],
    index: number,
    dataIndex: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const lookback = 5;
    const currentPrice = data[dataIndex].close;
    const currentMACD = macdData[index].MACD;
    const pastPrice = data[dataIndex - lookback].close;
    const pastMACD = macdData[index - lookback].MACD;

    // Bullish divergence: price makes lower low, MACD makes higher low
    if (currentPrice < pastPrice && currentMACD > pastMACD) {
      return 'BULLISH';
    }

    // Bearish divergence: price makes higher high, MACD makes lower high
    if (currentPrice > pastPrice && currentMACD < pastMACD) {
      return 'BEARISH';
    }

    return null;
  }

  private static detectMACDCrossover(macdData: any[], index: number): 'BULLISH' | 'BEARISH' | null {
    if (index < 1) return null;

    const current = macdData[index];
    const previous = macdData[index - 1];

    if (previous.MACD <= previous.signal && current.MACD > current.signal) {
      return 'BULLISH';
    }

    if (previous.MACD >= previous.signal && current.MACD < current.signal) {
      return 'BEARISH';
    }

    return null;
  }

  private static detectLineCrossover(
    currentLine1: number,
    currentLine2: number,
    previousLine1: number,
    previousLine2: number
  ): 'BULLISH' | 'BEARISH' | null {
    const wasAbove = previousLine1 > previousLine2;
    const isAbove = currentLine1 > currentLine2;

    if (!wasAbove && isAbove) return 'BULLISH';
    if (wasAbove && !isAbove) return 'BEARISH';
    return null;
  }

  private static calculateTrendStrength(values: number[]): number {
    if (values.length < 2) return 0;

    let upMoves = 0;
    let downMoves = 0;

    for (let i = 1; i < values.length; i++) {
      if (values[i] > values[i - 1]) upMoves++;
      else if (values[i] < values[i - 1]) downMoves++;
    }

    const totalMoves = upMoves + downMoves;
    if (totalMoves === 0) return 0;

    return Math.abs(upMoves - downMoves) / totalMoves;
  }

  private static calculateEfficiencyRatio(prices: number[]): number {
    if (prices.length < 2) return 0;

    const change = Math.abs(prices[prices.length - 1] - prices[0]);
    let volatility = 0;

    for (let i = 1; i < prices.length; i++) {
      volatility += Math.abs(prices[i] - prices[i - 1]);
    }

    return volatility === 0 ? 0 : change / volatility;
  }
}