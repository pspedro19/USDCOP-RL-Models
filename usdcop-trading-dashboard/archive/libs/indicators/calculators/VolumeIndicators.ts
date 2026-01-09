/**
 * Volume Indicators
 * ================
 *
 * Professional implementations of volume-based indicators
 * with advanced analytics for institutional trading.
 */

import {
  OBV, ADL, VWAP, MoneyFlowIndex, ChaikinMoneyFlow, VolumePriceTrend,
  NegativeVolumeIndex, PositiveVolumeIndex, VolumeWeightedAveragePrice
} from 'technicalindicators';

import { CandleData, IndicatorValue, VolumeProfile, VolumeProfileLevel, OrderFlow, MarketMicrostructure } from '../types';

export class VolumeIndicators {
  /**
   * Enhanced On-Balance Volume with trend confirmation
   */
  static calculateEnhancedOBV(data: CandleData[]): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    divergence?: 'BULLISH' | 'BEARISH' | null;
    momentum: number;
  }> {
    if (data.length < 10) return [];

    const closes = data.map(d => d.close);
    const volumes = data.map(d => d.volume);
    const obvData = OBV.calculate({ close: closes, volume: volumes });
    const startIndex = data.length - obvData.length;

    return obvData.map((obv, index) => {
      const dataIndex = startIndex + index;

      // Calculate trend
      let trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
      if (index >= 5) {
        const recentOBV = obvData.slice(Math.max(0, index - 4), index + 1);
        const avgChange = recentOBV.reduce((sum, val, i) => {
          return i === 0 ? sum : sum + (val - recentOBV[i - 1]);
        }, 0) / (recentOBV.length - 1);

        if (avgChange > 0) trend = 'BULLISH';
        else if (avgChange < 0) trend = 'BEARISH';
      }

      // Calculate momentum
      const momentum = index > 0 ? obv - obvData[index - 1] : 0;

      // Detect divergences
      const divergence = this.detectOBVDivergence(data, obvData, index, dataIndex);

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (divergence === 'BULLISH' && trend !== 'BEARISH') {
        signal = 'BUY';
      } else if (divergence === 'BEARISH' && trend !== 'BULLISH') {
        signal = 'SELL';
      } else if (trend === 'BULLISH' && momentum > 0) {
        signal = 'BUY';
      } else if (trend === 'BEARISH' && momentum < 0) {
        signal = 'SELL';
      }

      return {
        timestamp: data[dataIndex].timestamp,
        value: obv,
        signal,
        trend,
        divergence,
        momentum
      };
    });
  }

  /**
   * Professional Volume Profile with POC/VAH/VAL analysis
   */
  static calculateVolumeProfile(
    data: CandleData[],
    levels: number = 50,
    valueAreaPercent: number = 0.7
  ): VolumeProfile {
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

    // Initialize volume levels with buy/sell breakdown
    const volumeLevels = new Map<number, {
      volume: number;
      buyVolume: number;
      sellVolume: number;
      trades: number;
    }>();

    let totalVolume = 0;

    // Calculate volume distribution with order flow estimation
    data.forEach((candle, candleIndex) => {
      const volume = candle.volume || 0;
      const range = candle.high - candle.low;
      totalVolume += volume;

      // Estimate buy/sell pressure based on candle characteristics
      const bodySize = Math.abs(candle.close - candle.open);
      const upperWick = candle.high - Math.max(candle.open, candle.close);
      const lowerWick = Math.min(candle.open, candle.close) - candle.low;

      const bullishFactor = candle.close > candle.open ? 1 : -1;
      const wickBalance = (lowerWick - upperWick) / (range || 1);

      // Distribute volume across price levels
      for (let price = candle.low; price <= candle.high; price += priceStep / 10) {
        const levelPrice = Math.round(price / priceStep) * priceStep;

        if (!volumeLevels.has(levelPrice)) {
          volumeLevels.set(levelPrice, { volume: 0, buyVolume: 0, sellVolume: 0, trades: 0 });
        }

        const level = volumeLevels.get(levelPrice)!;
        const volumeWeight = range > 0 ? (priceStep / 10) / range : 1;
        const distributedVolume = volume * volumeWeight;

        level.volume += distributedVolume;
        level.trades += 1;

        // Advanced buy/sell volume estimation
        const pricePosition = range > 0 ? (price - candle.low) / range : 0.5;

        let buyRatio = 0.5; // Default neutral
        if (candle.close > candle.open) {
          // Bullish candle - more buying at higher prices
          buyRatio = 0.3 + 0.4 * pricePosition + 0.3 * (bodySize / range);
        } else {
          // Bearish candle - more selling at higher prices
          buyRatio = 0.7 - 0.4 * pricePosition - 0.3 * (bodySize / range);
        }

        // Adjust for wick analysis
        buyRatio += wickBalance * 0.1;
        buyRatio = Math.max(0.1, Math.min(0.9, buyRatio));

        level.buyVolume += distributedVolume * buyRatio;
        level.sellVolume += distributedVolume * (1 - buyRatio);
      }
    });

    // Convert to sorted array
    const levelArray: VolumeProfileLevel[] = Array.from(volumeLevels.entries())
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
    const poc = levelArray.length > 0 ? levelArray[0].price : minPrice;

    // Calculate Value Area
    const targetVolume = totalVolume * valueAreaPercent;
    let valueAreaVolume = 0;
    const valueAreaPrices: number[] = [];

    // Start from POC and expand outward
    const sortedByPrice = [...levelArray].sort((a, b) => a.price - b.price);
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
      levels: levelArray,
      poc,
      valueAreaHigh,
      valueAreaLow,
      valueAreaVolume,
      totalVolume,
      profiles: {
        session: levelArray,
        composite: levelArray // Could be extended for multi-session analysis
      }
    };
  }

  /**
   * Enhanced VWAP with standard deviation bands
   */
  static calculateEnhancedVWAP(data: CandleData[]): Array<{
    timestamp: number;
    vwap: number;
    upperBand1: number;
    lowerBand1: number;
    upperBand2: number;
    lowerBand2: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    position: 'ABOVE' | 'BELOW' | 'AT_VWAP';
  }> {
    if (data.length < 5) return [];

    let cumulativePV = 0;
    let cumulativeVolume = 0;
    const vwapValues: number[] = [];
    const prices: number[] = [];

    const result: Array<{
      timestamp: number;
      vwap: number;
      upperBand1: number;
      lowerBand1: number;
      upperBand2: number;
      lowerBand2: number;
      signal: 'BUY' | 'SELL' | 'NEUTRAL';
      position: 'ABOVE' | 'BELOW' | 'AT_VWAP';
    }> = [];

    data.forEach((candle, index) => {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3;
      const volume = candle.volume || 0;

      cumulativePV += typicalPrice * volume;
      cumulativeVolume += volume;

      const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : typicalPrice;
      vwapValues.push(vwap);
      prices.push(typicalPrice);

      // Calculate standard deviation
      let variance = 0;
      let weightedSum = 0;

      for (let i = 0; i <= index; i++) {
        const weight = data[i].volume || 0;
        const priceDiff = prices[i] - vwap;
        variance += weight * priceDiff * priceDiff;
        weightedSum += weight;
      }

      const stdDev = weightedSum > 0 ? Math.sqrt(variance / weightedSum) : 0;

      // Calculate bands
      const upperBand1 = vwap + stdDev;
      const lowerBand1 = vwap - stdDev;
      const upperBand2 = vwap + 2 * stdDev;
      const lowerBand2 = vwap - 2 * stdDev;

      // Determine position
      let position: 'ABOVE' | 'BELOW' | 'AT_VWAP' = 'AT_VWAP';
      if (candle.close > vwap * 1.001) position = 'ABOVE';
      else if (candle.close < vwap * 0.999) position = 'BELOW';

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevResult = result[index - 1];
        const prevPosition = prevResult.position;

        // VWAP reclaim/rejection signals
        if (prevPosition === 'BELOW' && position === 'ABOVE') {
          signal = 'BUY';
        } else if (prevPosition === 'ABOVE' && position === 'BELOW') {
          signal = 'SELL';
        }
        // Band reversal signals
        else if (candle.close <= lowerBand2 && candle.close > data[index - 1].close) {
          signal = 'BUY';
        } else if (candle.close >= upperBand2 && candle.close < data[index - 1].close) {
          signal = 'SELL';
        }
      }

      result.push({
        timestamp: candle.timestamp,
        vwap,
        upperBand1,
        lowerBand1,
        upperBand2,
        lowerBand2,
        signal,
        position
      });
    });

    return result;
  }

  /**
   * Money Flow Index with divergence detection
   */
  static calculateMoneyFlowIndex(
    data: CandleData[],
    period: number = 14
  ): Array<{
    timestamp: number;
    value: number;
    signal: 'BUY' | 'SELL' | 'NEUTRAL';
    zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
    divergence?: 'BULLISH' | 'BEARISH' | null;
    moneyFlow: number;
  }> {
    if (data.length < period + 5) return [];

    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);
    const closes = data.map(d => d.close);
    const volumes = data.map(d => d.volume);

    const mfiData = MoneyFlowIndex.calculate({
      high: highs,
      low: lows,
      close: closes,
      volume: volumes,
      period
    });

    const startIndex = data.length - mfiData.length;

    return mfiData.map((mfi, index) => {
      const dataIndex = startIndex + index;

      // Calculate raw money flow
      const typicalPrice = (data[dataIndex].high + data[dataIndex].low + data[dataIndex].close) / 3;
      const moneyFlow = typicalPrice * data[dataIndex].volume;

      // Determine zone
      let zone: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL' = 'NEUTRAL';
      if (mfi >= 80) zone = 'OVERBOUGHT';
      else if (mfi <= 20) zone = 'OVERSOLD';

      // Detect divergences
      const divergence = this.detectMFIDivergence(data, mfiData, index, dataIndex);

      // Generate signals
      let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      if (index > 0) {
        const prevMFI = mfiData[index - 1];

        // Zone breakout signals
        if (prevMFI <= 20 && mfi > 20) {
          signal = 'BUY';
        } else if (prevMFI >= 80 && mfi < 80) {
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
        value: mfi,
        signal,
        zone,
        divergence,
        moneyFlow
      };
    });
  }

  /**
   * Order Flow Analysis
   */
  static calculateOrderFlow(data: CandleData[]): OrderFlow[] {
    if (data.length < 5) return [];

    return data.map((candle, index) => {
      // Estimate order flow based on price action and volume
      const range = candle.high - candle.low;
      const body = Math.abs(candle.close - candle.open);
      const upperWick = candle.high - Math.max(candle.open, candle.close);
      const lowerWick = Math.min(candle.open, candle.close) - candle.low;

      // Calculate buy/sell pressure
      const bodyRatio = range > 0 ? body / range : 0;
      const wickImbalance = range > 0 ? (lowerWick - upperWick) / range : 0;

      let buyPressure = 0.5; // Default neutral
      if (candle.close > candle.open) {
        buyPressure = 0.5 + 0.3 * bodyRatio + 0.2 * wickImbalance;
      } else {
        buyPressure = 0.5 - 0.3 * bodyRatio - 0.2 * wickImbalance;
      }

      buyPressure = Math.max(0, Math.min(1, buyPressure));
      const sellPressure = 1 - buyPressure;

      // Calculate imbalance
      const imbalance = (buyPressure - sellPressure) * candle.volume;

      // Calculate net volume flow
      const netVolume = imbalance;

      // Calculate VWAP for this period
      let vwap = (candle.high + candle.low + candle.close) / 3;
      if (index > 0) {
        // Simple rolling VWAP approximation
        const prevData = data.slice(Math.max(0, index - 20), index + 1);
        let totalPV = 0;
        let totalV = 0;

        prevData.forEach(d => {
          const tp = (d.high + d.low + d.close) / 3;
          totalPV += tp * d.volume;
          totalV += d.volume;
        });

        vwap = totalV > 0 ? totalPV / totalV : vwap;
      }

      // Estimate market impact
      const avgVolume = index >= 20 ?
        data.slice(index - 19, index + 1).reduce((sum, d) => sum + d.volume, 0) / 20 :
        candle.volume;

      const marketImpact = avgVolume > 0 ? (candle.volume / avgVolume) * Math.abs(imbalance) / candle.volume : 0;

      return {
        timestamp: candle.timestamp,
        imbalance,
        buyPressure,
        sellPressure,
        netVolume,
        vwap,
        marketImpact
      };
    });
  }

  /**
   * Market Microstructure Analysis
   */
  static calculateMarketMicrostructure(data: CandleData[]): MarketMicrostructure[] {
    if (data.length < 20) return [];

    return data.map((candle, index) => {
      // Calculate spread (approximation using high-low)
      const spread = candle.high - candle.low;
      const midPrice = (candle.high + candle.low) / 2;
      const spreadBps = midPrice > 0 ? (spread / midPrice) * 10000 : 0;

      // Calculate depth approximation
      const volume = candle.volume || 0;
      const avgVolume = index >= 10 ?
        data.slice(index - 9, index + 1).reduce((sum, d) => sum + d.volume, 0) / 10 :
        volume;
      const depth = avgVolume > 0 ? volume / avgVolume : 1;

      // Calculate liquidity metric
      const priceChange = index > 0 ? Math.abs(candle.close - data[index - 1].close) : 0;
      const liquidity = volume > 0 && priceChange > 0 ? volume / priceChange : 0;

      // Calculate volatility
      let volatility = 0;
      if (index >= 20) {
        const returns = [];
        for (let i = index - 19; i <= index; i++) {
          if (i > 0) {
            const ret = Math.log(data[i].close / data[i - 1].close);
            returns.push(ret);
          }
        }

        if (returns.length > 0) {
          const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
          const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
          volatility = Math.sqrt(variance) * 100; // Convert to percentage
        }
      }

      // Calculate efficiency (inverse of price impact)
      const efficiency = liquidity > 0 ? Math.min(1, 1000 / liquidity) : 0;

      // Calculate toxicity (how much volume leads to adverse price movement)
      let toxicity = 0;
      if (index >= 5) {
        const recentCandles = data.slice(index - 4, index + 1);
        const totalVolume = recentCandles.reduce((sum, c) => sum + c.volume, 0);
        const priceMovement = Math.abs(candle.close - data[index - 4].close);
        toxicity = totalVolume > 0 ? (priceMovement / candle.close) * (totalVolume / avgVolume) : 0;
      }

      return {
        timestamp: candle.timestamp,
        spread: spreadBps,
        depth,
        liquidity,
        volatility,
        efficiency,
        toxicity: Math.min(1, toxicity)
      };
    });
  }

  // Helper methods

  private static detectOBVDivergence(
    data: CandleData[],
    obvData: number[],
    index: number,
    dataIndex: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const lookback = 7;
    const currentPrice = data[dataIndex].close;
    const currentOBV = obvData[index];
    const pastPrice = data[dataIndex - lookback].close;
    const pastOBV = obvData[index - lookback];

    // Bullish divergence: price makes lower low, OBV makes higher low
    if (currentPrice < pastPrice && currentOBV > pastOBV) {
      return 'BULLISH';
    }

    // Bearish divergence: price makes higher high, OBV makes lower high
    if (currentPrice > pastPrice && currentOBV < pastOBV) {
      return 'BEARISH';
    }

    return null;
  }

  private static detectMFIDivergence(
    data: CandleData[],
    mfiData: number[],
    index: number,
    dataIndex: number
  ): 'BULLISH' | 'BEARISH' | null {
    if (index < 10) return null;

    const lookback = 7;
    const currentPrice = data[dataIndex].close;
    const currentMFI = mfiData[index];
    const pastPrice = data[dataIndex - lookback].close;
    const pastMFI = mfiData[index - lookback];

    // Bullish divergence
    if (currentPrice < pastPrice && currentMFI > pastMFI) {
      return 'BULLISH';
    }

    // Bearish divergence
    if (currentPrice > pastPrice && currentMFI < pastMFI) {
      return 'BEARISH';
    }

    return null;
  }
}