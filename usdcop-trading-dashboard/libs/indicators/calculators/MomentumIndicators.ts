/**
 * Momentum Indicators
 * ==================
 *
 * Professional implementations of momentum oscillators
 * with advanced divergence detection and signal generation.
 */

import {
  RSI, Stochastic, StochasticRSI, WilliamsR, CCI, ROC, MOM,
  UltimateOscillator, AwesomeOscillator, MoneyFlowIndex
} from 'technicalindicators';

import { CandleData, IndicatorValue, Stochastic as StochasticResult } from '../types';

export class MomentumIndicators {
  /**
   * Enhanced RSI with divergence detection and multi-timeframe analysis
   */
  static calculateEnhancedRSI(
    data: CandleData[],
    period: number = 14,
    overbought: number = 70,
    oversold: number = 30
  ): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    divergence?: 'BULLISH' | 'BEARISH' | null;
    zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
    momentum: number;
  }> {
    if (data.length < period + 10) return [];

    const closes = data.map(d => d.close);
    const rsiData = RSI.calculate({ period, values: closes });
    const startIndex = data.length - rsiData.length;

    return rsiData.map((rsi, index) => {
      const dataIndex = startIndex + index;

      // Determine zone
      let zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL' = 'NEUTRAL';
      if (rsi >= overbought) zone = 'OVERBOUGHT';
      else if (rsi <= oversold) zone = 'OVERSOLD';

      // Calculate momentum (rate of change of RSI)
      const momentum = index > 0 ? rsi - rsiData[index - 1] : 0;

      // Detect divergences
      const divergence = this.detectRSIDivergence(data, rsiData, index, dataIndex);

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevRSI = rsiData[index - 1];

        // Oversold bounce
        if (prevRSI <= oversold && rsi > oversold && momentum > 0) {
          signal = 'BUY';
        }
        // Overbought rejection
        else if (prevRSI >= overbought && rsi < overbought && momentum < 0) {
          signal = 'SELL';
        }
        // Divergence signals
        else if (divergence === 'BULLISH' && zone !== 'OVERBOUGHT') {
          signal = 'BUY';
        } else if (divergence === 'BEARISH' && zone !== 'OVERSOLD') {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: rsi,
        signal,
        divergence,
        zone,
        momentum
      };
    });
  }

  /**
   * Advanced Stochastic Oscillator with %K and %D analysis
   */
  static calculateAdvancedStochastic(
    data: CandleData[],
    kPeriod: number = 14,
    dPeriod: number = 3,
    slowing: number = 3
  ): Array<{
    timestamp: number;
    k: number;
    d: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    crossover?: 'BULLISH' | 'BEARISH' | null;
    zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
    divergence?: 'BULLISH' | 'BEARISH' | null;
  }> {
    if (data.length < kPeriod + dPeriod + slowing) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    const stochData = Stochastic.calculate({
      high: highs,
      low: lows,
      close: closes,
      period: kPeriod,
      signalPeriod: dPeriod
    });

    const startIndex = data.length - stochData.length;

    return stochData.map((stoch, index) => {
      const dataIndex = startIndex + index;

      // Determine zone
      let zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL' = 'NEUTRAL';
      if (stoch.k >= 80) zone = 'OVERBOUGHT';
      else if (stoch.k <= 20) zone = 'OVERSOLD';

      // Detect crossovers
      const crossover = this.detectStochasticCrossover(stochData, index);

      // Detect divergences
      const divergence = this.detectStochasticDivergence(data, stochData, index, dataIndex);

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (crossover === 'BULLISH' && zone === 'OVERSOLD') {
        signal = 'BUY';
      } else if (crossover === 'BEARISH' && zone === 'OVERBOUGHT') {
        signal = 'SELL';
      } else if (divergence === 'BULLISH' && zone !== 'OVERBOUGHT') {
        signal = 'BUY';
      } else if (divergence === 'BEARISH' && zone !== 'OVERSOLD') {
        signal = 'SELL';
      }

      return {
        timestamp: data[dataIndex].timestamp,
        k: stoch.k || 0,
        d: stoch.d || 0,
        signal,
        crossover,
        zone,
        divergence
      };
    });
  }

  /**
   * Stochastic RSI with enhanced signal detection
   */
  static calculateStochasticRSI(
    data: CandleData[],
    rsiPeriod: number = 14,
    stochPeriod: number = 14,
    kPeriod: number = 3,
    dPeriod: number = 3
  ): Array<{
    timestamp: number;
    k: number;
    d: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    momentum: 'INCREASING' | 'DECREASING' | 'NEUTRAL';
  }> {
    if (data.length < rsiPeriod + stochPeriod + Math.max(kPeriod, dPeriod)) return [];

    const closes = data.map(d => d.close);
    const stochRSIData = StochasticRSI.calculate({
      values: closes,
      rsiPeriod,
      stochasticPeriod: stochPeriod,
      kPeriod,
      dPeriod
    });

    const startIndex = data.length - stochRSIData.length;

    return stochRSIData.map((stochRSI, index) => {
      const dataIndex = startIndex + index;

      // Calculate momentum
      let momentum: 'INCREASING' | 'DECREASING' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevK = stochRSIData[index - 1].k || 0;
        const currentK = stochRSI.k || 0;

        if (currentK > prevK) momentum = 'INCREASING';
        else if (currentK < prevK) momentum = 'DECREASING';
      }

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      const k = stochRSI.k || 0;
      const d = stochRSI.d || 0;

      if (index > 0) {
        const prevStochRSI = stochRSIData[index - 1];
        const prevK = prevStochRSI.k || 0;
        const prevD = prevStochRSI.d || 0;

        // Bullish crossover in oversold territory
        if (prevK <= prevD && k > d && k < 20) {
          signal = 'BUY';
        }
        // Bearish crossover in overbought territory
        else if (prevK >= prevD && k < d && k > 80) {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        k,
        d,
        signal,
        momentum
      };
    });
  }

  /**
   * Williams %R with volatility adjustment
   */
  static calculateWilliamsR(
    data: CandleData[],
    period: number = 14
  ): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
    volatilityAdjusted: number;
  }> {
    if (data.length < period + 5) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    const williamsData = WilliamsR.calculate({
      high: highs,
      low: lows,
      close: closes,
      period
    });

    const startIndex = data.length - williamsData.length;

    return williamsData.map((williams, index) => {
      const dataIndex = startIndex + index;

      // Determine zone
      let zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL' = 'NEUTRAL';
      if (williams >= -20) zone = 'OVERBOUGHT';
      else if (williams <= -80) zone = 'OVERSOLD';

      // Calculate volatility adjustment
      const volatility = this.calculateLocalVolatility(data, dataIndex, 5);
      const volatilityAdjusted = williams * (1 + volatility);

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevWilliams = williamsData[index - 1];

        // Oversold bounce
        if (prevWilliams <= -80 && williams > -80) {
          signal = 'BUY';
        }
        // Overbought rejection
        else if (prevWilliams >= -20 && williams < -20) {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: williams,
        signal,
        zone,
        volatilityAdjusted
      };
    });
  }

  /**
   * Commodity Channel Index (CCI) with trend analysis
   */
  static calculateCCI(
    data: CandleData[],
    period: number = 20
  ): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    extremeLevel: boolean;
  }> {
    if (data.length < period + 5) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    const cciData = CCI.calculate({
      high: highs,
      low: lows,
      close: closes,
      period
    });

    const startIndex = data.length - cciData.length;

    return cciData.map((cci, index) => {
      const dataIndex = startIndex + index;

      // Determine trend
      let trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
      if (index >= 5) {
        const recentCCI = cciData.slice(Math.max(0, index - 4), index + 1);
        const avgChange = recentCCI.reduce((sum, val, i) => {
          return i === 0 ? sum : sum + (val - recentCCI[i - 1]);
        }, 0) / (recentCCI.length - 1);

        if (avgChange > 10) trend = 'BULLISH';
        else if (avgChange < -10) trend = 'BEARISH';
      }

      // Check for extreme levels
      const extremeLevel = Math.abs(cci) > 200;

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevCCI = cciData[index - 1];

        // Extreme oversold bounce
        if (prevCCI <= -200 && cci > -200 && trend !== 'BEARISH') {
          signal = 'BUY';
        }
        // Extreme overbought rejection
        else if (prevCCI >= 200 && cci < 200 && trend !== 'BULLISH') {
          signal = 'SELL';
        }
        // Zero line crosses with trend confirmation
        else if (prevCCI < 0 && cci > 0 && trend === 'BULLISH') {
          signal = 'BUY';
        } else if (prevCCI > 0 && cci < 0 && trend === 'BEARISH') {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: cci,
        signal,
        trend,
        extremeLevel
      };
    });
  }

  /**
   * Rate of Change (ROC) with momentum analysis
   */
  static calculateROC(
    data: CandleData[],
    period: number = 12
  ): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    momentum: 'ACCELERATING' | 'DECELERATING' | 'NEUTRAL';
    magnitude: 'HIGH' | 'MEDIUM' | 'LOW';
  }> {
    if (data.length < period + 5) return [];

    const closes = data.map(d => d.close);
    const rocData = ROC.calculate({ period, values: closes });
    const startIndex = data.length - rocData.length;

    return rocData.map((roc, index) => {
      const dataIndex = startIndex + index;

      // Calculate momentum trend
      let momentum: 'ACCELERATING' | 'DECELERATING' | 'NEUTRAL' = 'NEUTRAL';
      if (index >= 2) {
        const recent = rocData.slice(index - 2, index + 1);
        const isAccelerating = recent.every((val, i) => i === 0 || val > recent[i - 1]);
        const isDecelerating = recent.every((val, i) => i === 0 || val < recent[i - 1]);

        if (isAccelerating) momentum = 'ACCELERATING';
        else if (isDecelerating) momentum = 'DECELERATING';
      }

      // Calculate magnitude
      let magnitude: 'HIGH' | 'MEDIUM' | 'LOW' = 'LOW';
      const absROC = Math.abs(roc);
      if (absROC > 5) magnitude = 'HIGH';
      else if (absROC > 2) magnitude = 'MEDIUM';

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevROC = rocData[index - 1];

        // Zero line crosses with momentum confirmation
        if (prevROC < 0 && roc > 0 && momentum === 'ACCELERATING') {
          signal = 'BUY';
        } else if (prevROC > 0 && roc < 0 && momentum === 'ACCELERATING') {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: roc,
        signal,
        momentum,
        magnitude
      };
    });
  }

  /**
   * Ultimate Oscillator with multi-timeframe analysis
   */
  static calculateUltimateOscillator(
    data: CandleData[],
    period1: number = 7,
    period2: number = 14,
    period3: number = 28
  ): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
    divergence?: 'BULLISH' | 'BEARISH' | null;
  }> {
    if (data.length < period3 + 10) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);

    const uoData = UltimateOscillator.calculate({
      high: highs,
      low: lows,
      close: closes,
      period1,
      period2,
      period3
    });

    const startIndex = data.length - uoData.length;

    return uoData.map((uo, index) => {
      const dataIndex = startIndex + index;

      // Determine zone
      let zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL' = 'NEUTRAL';
      if (uo >= 70) zone = 'OVERBOUGHT';
      else if (uo <= 30) zone = 'OVERSOLD';

      // Detect divergences
      const divergence = this.detectUltimateDivergence(data, uoData, index, dataIndex);

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevUO = uoData[index - 1];

        // Oversold bounce
        if (prevUO <= 30 && uo > 30) {
          signal = 'BUY';
        }
        // Overbought rejection
        else if (prevUO >= 70 && uo < 70) {
          signal = 'SELL';
        }
        // Divergence signals
        else if (divergence === 'BULLISH' && zone !== 'OVERBOUGHT') {
          signal = 'BUY';
        } else if (divergence === 'BEARISH' && zone !== 'OVERSOLD') {
          signal = 'SELL';
        }
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: uo,
        signal,
        zone,
        divergence
      };
    });
  }

  // Helper methods

  private static detectRSIDivergence(
    data: CandleData[],
    rsiData: number[],
    index: number,
    dataIndex: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const lookback = 5;
    const currentPrice = data[dataIndex].close;
    const currentRSI = rsiData[index];
    const pastPrice = data[dataIndex - lookback].close;
    const pastRSI = rsiData[index - lookback];

    // Bullish divergence: price makes lower low, RSI makes higher low
    if (currentPrice < pastPrice && currentRSI > pastRSI) {
      return 'BULLISH';
    }

    // Bearish divergence: price makes higher high, RSI makes lower high
    if (currentPrice > pastPrice && currentRSI < pastRSI) {
      return 'BEARISH';
    }

    return null;
  }

  private static detectStochasticCrossover(
    stochData: any[],
    index: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 1) return null;

    const current = stochData[index];
    const previous = stochData[index - 1];

    if (!current || !previous) return null;

    // %K crosses above %D
    if (previous.k <= previous.d && current.k > current.d) {
      return 'BULLISH';
    }

    // %K crosses below %D
    if (previous.k >= previous.d && current.k < current.d) {
      return 'BEARISH';
    }

    return null;
  }

  private static detectStochasticDivergence(
    data: CandleData[],
    stochData: any[],
    index: number,
    dataIndex: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const lookback = 5;
    const currentPrice = data[dataIndex].close;
    const currentStoch = stochData[index].k;
    const pastPrice = data[dataIndex - lookback].close;
    const pastStoch = stochData[index - lookback].k;

    if (!currentStoch || !pastStoch) return null;

    // Bullish divergence
    if (currentPrice < pastPrice && currentStoch > pastStoch) {
      return 'BULLISH';
    }

    // Bearish divergence
    if (currentPrice > pastPrice && currentStoch < pastStoch) {
      return 'BEARISH';
    }

    return null;
  }

  private static detectUltimateDivergence(
    data: CandleData[],
    uoData: number[],
    index: number,
    dataIndex: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const lookback = 7;
    const currentPrice = data[dataIndex].close;
    const currentUO = uoData[index];
    const pastPrice = data[dataIndex - lookback].close;
    const pastUO = uoData[index - lookback];

    // Bullish divergence
    if (currentPrice < pastPrice && currentUO > pastUO) {
      return 'BULLISH';
    }

    // Bearish divergence
    if (currentPrice > pastPrice && currentUO < pastUO) {
      return 'BEARISH';
    }

    return null;
  }

  private static calculateLocalVolatility(
    data: CandleData[],
    index: number,
    period: number
  ): number {
    if (index < period) return 0;

    const returns: number[] = [];
    for (let i = index - period + 1; i <= index; i++) {
      if (i > 0) {
        const ret = Math.log(data[i].close / data[i - 1].close);
        returns.push(ret);
      }
    }

    if (returns.length === 0) return 0;

    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;

    return Math.sqrt(variance);
  }
}