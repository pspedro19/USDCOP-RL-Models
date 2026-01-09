/**
 * Volume Analytics Engine
 * ======================
 *
 * Advanced volume analysis for institutional trading with:
 * - Volume Profile with POC/VAH/VAL
 * - Time and Sales Analysis
 * - Market Profile
 * - Volume Weighted Price Analysis
 * - Order Flow Analytics
 */

import { CandleData, VolumeProfile, VolumeProfileLevel, OrderFlow, MarketMicrostructure } from '../types';

export interface VolumeAnalyticsConfig {
  profileLevels: number;
  valueAreaPercent: number;
  sessionSplit: boolean;
  timeframes: string[];
  includeOrderFlow: boolean;
  microstructureAnalysis: boolean;
}

export interface MarketProfile {
  timeStamp: number;
  priceRanges: {
    price: number;
    timeAtPrice: number;
    volume: number;
    tpo: string[]; // Time Price Opportunity
  }[];
  initialBalance: {
    high: number;
    low: number;
    range: number;
  };
  dayType: 'NORMAL' | 'NORMAL_VARIATION' | 'TREND' | 'NON_TREND';
  valueArea: {
    high: number;
    low: number;
    poc: number;
  };
}

export interface VolumeWeightedMetrics {
  vwap: number;
  standardDeviations: {
    sd1Upper: number;
    sd1Lower: number;
    sd2Upper: number;
    sd2Lower: number;
    sd3Upper: number;
    sd3Lower: number;
  };
  volumeWeightedMovingAverage: number;
  volumeRate: number;
  relativeVolume: number;
}

export interface TimeAndSalesAnalysis {
  timestamp: number;
  aggression: 'PASSIVE' | 'NEUTRAL' | 'AGGRESSIVE';
  tradeSize: 'SMALL' | 'MEDIUM' | 'LARGE' | 'BLOCK';
  direction: 'UP_TICK' | 'DOWN_TICK' | 'ZERO_TICK';
  cumulativeDelta: number;
  volumeAtBid: number;
  volumeAtAsk: number;
  printCount: number;
}

export interface SessionAnalytics {
  session: 'ASIAN' | 'LONDON' | 'NEW_YORK' | 'OVERLAP';
  startTime: number;
  endTime: number;
  volumeProfile: VolumeProfile;
  averageVolume: number;
  volatility: number;
  efficiency: number;
  participationRate: number;
}

export class VolumeAnalytics {
  private config: VolumeAnalyticsConfig;

  constructor(config: Partial<VolumeAnalyticsConfig> = {}) {
    this.config = {
      profileLevels: 50,
      valueAreaPercent: 0.7,
      sessionSplit: true,
      timeframes: ['1D', '4H', '1H'],
      includeOrderFlow: true,
      microstructureAnalysis: true,
      ...config
    };
  }

  /**
   * Enhanced Volume Profile with session analysis
   */
  calculateVolumeProfile(
    data: CandleData[],
    options: {
      levels?: number;
      valueAreaPercent?: number;
      splitSessions?: boolean;
      timeframe?: string;
    } = {}
  ): VolumeProfile {
    const { levels = this.config.profileLevels, valueAreaPercent = this.config.valueAreaPercent } = options;

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

    // Calculate price range and levels
    const priceData = data.flatMap(d => [d.high, d.low, d.open, d.close]);
    const minPrice = Math.min(...priceData);
    const maxPrice = Math.max(...priceData);
    const priceStep = (maxPrice - minPrice) / levels;

    // Initialize volume accumulation
    const volumeLevels = new Map<number, {
      volume: number;
      buyVolume: number;
      sellVolume: number;
      trades: number;
      timeAtLevel: number;
      firstTouch: number;
      lastTouch: number;
    }>();

    let totalVolume = 0;

    // Process each candle
    data.forEach((candle, candleIndex) => {
      const volume = candle.volume || 0;
      const timeWeight = 1; // Could be adjusted based on timeframe
      totalVolume += volume;

      // Calculate intrabar price distribution
      const priceRange = candle.high - candle.low;
      const bodySize = Math.abs(candle.close - candle.open);
      const upperWick = candle.high - Math.max(candle.open, candle.close);
      const lowerWick = Math.min(candle.open, candle.close) - candle.low;

      // Determine candle characteristics
      const isBullish = candle.close > candle.open;
      const bodyRatio = priceRange > 0 ? bodySize / priceRange : 0;
      const wickImbalance = priceRange > 0 ? (lowerWick - upperWick) / priceRange : 0;

      // Distribute volume across price levels
      this.distributeVolumeAcrossLevels(
        candle,
        minPrice,
        priceStep,
        levels,
        volumeLevels,
        { isBullish, bodyRatio, wickImbalance, timeWeight }
      );
    });

    // Convert to array and calculate metrics
    const levelArray: VolumeProfileLevel[] = Array.from(volumeLevels.entries())
      .map(([price, data]) => ({
        price,
        volume: data.volume,
        buyVolume: data.buyVolume,
        sellVolume: data.sellVolume,
        percentOfTotal: totalVolume > 0 ? (data.volume / totalVolume) * 100 : 0,
        delta: data.buyVolume - data.sellVolume
      }))
      .filter(level => level.volume > 0)
      .sort((a, b) => b.volume - a.volume);

    // Find Point of Control
    const poc = levelArray.length > 0 ? levelArray[0].price : minPrice;

    // Calculate Value Area using advanced algorithm
    const { valueAreaHigh, valueAreaLow, valueAreaVolume } = this.calculateValueArea(
      levelArray,
      poc,
      totalVolume,
      valueAreaPercent
    );

    // Calculate session profiles if requested
    const sessionProfiles = options.splitSessions ?
      this.calculateSessionProfiles(data, levels) : levelArray;

    return {
      levels: levelArray,
      poc,
      valueAreaHigh,
      valueAreaLow,
      valueAreaVolume,
      totalVolume,
      profiles: {
        session: sessionProfiles,
        composite: levelArray
      }
    };
  }

  /**
   * Calculate Market Profile (TPO)
   */
  calculateMarketProfile(data: CandleData[], timeframe: '30min' | '1h' = '30min'): MarketProfile {
    if (!data.length) {
      throw new Error('No data provided for market profile calculation');
    }

    const timeframeMins = timeframe === '30min' ? 30 : 60;
    const sessionStart = data[0].timestamp;
    const sessionEnd = data[data.length - 1].timestamp;

    // Calculate price levels (similar to volume profile)
    const priceData = data.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...priceData);
    const maxPrice = Math.max(...priceData);
    const tick = this.calculateOptimalTick(minPrice, maxPrice);

    // Build TPO (Time Price Opportunity) chart
    const tpoData = new Map<number, {
      timeAtPrice: number;
      volume: number;
      tpo: string[];
    }>();

    // Process data in time buckets
    let currentTime = sessionStart;
    let letterIndex = 0;
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

    while (currentTime < sessionEnd) {
      const bucketEnd = currentTime + (timeframeMins * 60);
      const bucketData = data.filter(d =>
        d.timestamp >= currentTime && d.timestamp < bucketEnd
      );

      const bucketLetter = letters[letterIndex % letters.length];

      // Distribute time across price levels for this bucket
      bucketData.forEach(candle => {
        const priceRange = candle.high - candle.low;

        // Create price levels within this candle
        for (let price = candle.low; price <= candle.high; price += tick) {
          const roundedPrice = Math.round(price / tick) * tick;

          if (!tpoData.has(roundedPrice)) {
            tpoData.set(roundedPrice, { timeAtPrice: 0, volume: 0, tpo: [] });
          }

          const levelData = tpoData.get(roundedPrice)!;
          levelData.timeAtPrice += 1;
          levelData.volume += (candle.volume || 0) / Math.max(1, priceRange / tick);

          if (!levelData.tpo.includes(bucketLetter)) {
            levelData.tpo.push(bucketLetter);
          }
        }
      });

      currentTime = bucketEnd;
      letterIndex++;
    }

    // Convert to array and sort
    const priceRanges = Array.from(tpoData.entries())
      .map(([price, data]) => ({
        price,
        timeAtPrice: data.timeAtPrice,
        volume: data.volume,
        tpo: data.tpo
      }))
      .sort((a, b) => b.price - a.price);

    // Calculate Initial Balance (first hour range)
    const firstHourEnd = sessionStart + (60 * 60);
    const firstHourData = data.filter(d => d.timestamp <= firstHourEnd);
    const initialBalance = {
      high: Math.max(...firstHourData.map(d => d.high)),
      low: Math.min(...firstHourData.map(d => d.low)),
      range: 0
    };
    initialBalance.range = initialBalance.high - initialBalance.low;

    // Find POC and Value Area
    const maxTimeAtPrice = Math.max(...priceRanges.map(p => p.timeAtPrice));
    const pocLevel = priceRanges.find(p => p.timeAtPrice === maxTimeAtPrice);
    const poc = pocLevel ? pocLevel.price : (minPrice + maxPrice) / 2;

    // Calculate Value Area (70% of time)
    const totalTime = priceRanges.reduce((sum, p) => sum + p.timeAtPrice, 0);
    const targetTime = totalTime * 0.7;

    const { valueAreaHigh, valueAreaLow } = this.calculateTPOValueArea(
      priceRanges,
      poc,
      targetTime
    );

    // Determine day type
    const dayType = this.classifyDayType(data, initialBalance, { high: valueAreaHigh, low: valueAreaLow, poc });

    return {
      timeStamp: sessionStart,
      priceRanges,
      initialBalance,
      dayType,
      valueArea: {
        high: valueAreaHigh,
        low: valueAreaLow,
        poc
      }
    };
  }

  /**
   * Calculate VWAP with standard deviation bands
   */
  calculateVWAPMetrics(data: CandleData[]): VolumeWeightedMetrics[] {
    if (!data.length) return [];

    const results: VolumeWeightedMetrics[] = [];
    let cumulativePV = 0;
    let cumulativeVolume = 0;
    let cumulativeVariance = 0;

    data.forEach((candle, index) => {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3;
      const volume = candle.volume || 0;

      cumulativePV += typicalPrice * volume;
      cumulativeVolume += volume;

      const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : typicalPrice;

      // Calculate variance for standard deviation bands
      cumulativeVariance += volume * Math.pow(typicalPrice - vwap, 2);
      const variance = cumulativeVolume > 0 ? cumulativeVariance / cumulativeVolume : 0;
      const stdDev = Math.sqrt(variance);

      // Calculate volume metrics
      const avgVolume = index >= 20 ?
        data.slice(Math.max(0, index - 19), index + 1).reduce((sum, d) => sum + d.volume, 0) / 20 :
        volume;

      const volumeRate = avgVolume > 0 ? volume / avgVolume : 1;
      const relativeVolume = this.calculateRelativeVolume(data, index);

      results.push({
        vwap,
        standardDeviations: {
          sd1Upper: vwap + stdDev,
          sd1Lower: vwap - stdDev,
          sd2Upper: vwap + 2 * stdDev,
          sd2Lower: vwap - 2 * stdDev,
          sd3Upper: vwap + 3 * stdDev,
          sd3Lower: vwap - 3 * stdDev
        },
        volumeWeightedMovingAverage: vwap,
        volumeRate,
        relativeVolume
      });
    });

    return results;
  }

  /**
   * Analyze Time and Sales data
   */
  analyzeTimeAndSales(data: CandleData[]): TimeAndSalesAnalysis[] {
    if (data.length < 2) return [];

    const results: TimeAndSalesAnalysis[] = [];
    let cumulativeDelta = 0;

    data.forEach((candle, index) => {
      if (index === 0) return;

      const prevCandle = data[index - 1];
      const priceChange = candle.close - prevCandle.close;
      const volume = candle.volume || 0;

      // Determine tick direction
      let direction: 'UP_TICK' | 'DOWN_TICK' | 'ZERO_TICK' = 'ZERO_TICK';
      if (priceChange > 0) direction = 'UP_TICK';
      else if (priceChange < 0) direction = 'DOWN_TICK';

      // Estimate aggression based on price movement and volume
      let aggression: 'PASSIVE' | 'NEUTRAL' | 'AGGRESSIVE' = 'NEUTRAL';
      const priceImpact = Math.abs(priceChange) / prevCandle.close;
      const volumeRatio = this.calculateVolumeRatio(data, index);

      if (priceImpact > 0.001 && volumeRatio > 1.5) {
        aggression = 'AGGRESSIVE';
      } else if (priceImpact < 0.0005 && volumeRatio < 0.8) {
        aggression = 'PASSIVE';
      }

      // Classify trade size
      const avgVolume = this.calculateAverageVolume(data, index, 20);
      let tradeSize: 'SMALL' | 'MEDIUM' | 'LARGE' | 'BLOCK' = 'MEDIUM';

      if (volume < avgVolume * 0.5) tradeSize = 'SMALL';
      else if (volume > avgVolume * 2) tradeSize = 'LARGE';
      else if (volume > avgVolume * 5) tradeSize = 'BLOCK';

      // Estimate delta (buy volume - sell volume)
      const delta = this.estimateDelta(candle, prevCandle);
      cumulativeDelta += delta;

      // Estimate volume at bid/ask
      const { volumeAtBid, volumeAtAsk } = this.estimateBidAskVolume(candle, delta);

      results.push({
        timestamp: candle.timestamp,
        aggression,
        tradeSize,
        direction,
        cumulativeDelta,
        volumeAtBid,
        volumeAtAsk,
        printCount: 1 // Simplified for candle data
      });
    });

    return results;
  }

  /**
   * Analyze trading sessions
   */
  analyzeSessionVolume(data: CandleData[]): SessionAnalytics[] {
    const sessions = this.splitIntoSessions(data);

    return sessions.map(sessionData => {
      const volumeProfile = this.calculateVolumeProfile(sessionData.data);
      const avgVolume = sessionData.data.reduce((sum, d) => sum + d.volume, 0) / sessionData.data.length;

      // Calculate session volatility
      const returns = sessionData.data.slice(1).map((candle, i) =>
        Math.log(candle.close / sessionData.data[i].close)
      );
      const volatility = this.calculateVolatility(returns);

      // Calculate efficiency (price change / total price movement)
      const priceChange = Math.abs(sessionData.data[sessionData.data.length - 1].close - sessionData.data[0].open);
      const totalMovement = sessionData.data.reduce((sum, candle) => sum + (candle.high - candle.low), 0);
      const efficiency = totalMovement > 0 ? priceChange / totalMovement : 0;

      // Calculate participation rate
      const totalSessionVolume = sessionData.data.reduce((sum, d) => sum + d.volume, 0);
      const avgDailyVolume = this.calculateAverageDailyVolume(data);
      const participationRate = avgDailyVolume > 0 ? totalSessionVolume / avgDailyVolume : 0;

      return {
        session: sessionData.session,
        startTime: sessionData.startTime,
        endTime: sessionData.endTime,
        volumeProfile,
        averageVolume: avgVolume,
        volatility,
        efficiency,
        participationRate
      };
    });
  }

  // Private helper methods

  private distributeVolumeAcrossLevels(
    candle: CandleData,
    minPrice: number,
    priceStep: number,
    levels: number,
    volumeLevels: Map<number, any>,
    context: { isBullish: boolean; bodyRatio: number; wickImbalance: number; timeWeight: number }
  ): void {
    const volume = candle.volume || 0;
    const range = candle.high - candle.low;

    // More sophisticated volume distribution
    const pricePoints = this.generatePriceDistribution(candle, 20);

    pricePoints.forEach(({ price, weight }) => {
      const levelPrice = Math.round((price - minPrice) / priceStep) * priceStep + minPrice;

      if (!volumeLevels.has(levelPrice)) {
        volumeLevels.set(levelPrice, {
          volume: 0,
          buyVolume: 0,
          sellVolume: 0,
          trades: 0,
          timeAtLevel: 0,
          firstTouch: candle.timestamp,
          lastTouch: candle.timestamp
        });
      }

      const level = volumeLevels.get(levelPrice)!;
      const distributedVolume = volume * weight;

      // Enhanced buy/sell estimation
      const buyRatio = this.calculateBuyRatio(candle, price, context);

      level.volume += distributedVolume;
      level.buyVolume += distributedVolume * buyRatio;
      level.sellVolume += distributedVolume * (1 - buyRatio);
      level.trades += weight;
      level.timeAtLevel += context.timeWeight * weight;
      level.lastTouch = candle.timestamp;
    });
  }

  private generatePriceDistribution(candle: CandleData, points: number): { price: number; weight: number }[] {
    const distribution: { price: number; weight: number }[] = [];
    const range = candle.high - candle.low;

    if (range === 0) {
      return [{ price: candle.close, weight: 1 }];
    }

    // Use normal distribution centered around typical price
    const center = (candle.high + candle.low + candle.close) / 3;
    const sigma = range / 6; // 99.7% of volume within the range

    for (let i = 0; i < points; i++) {
      const price = candle.low + (range * i) / (points - 1);
      const weight = this.normalDistribution(price, center, sigma);
      distribution.push({ price, weight });
    }

    // Normalize weights
    const totalWeight = distribution.reduce((sum, d) => sum + d.weight, 0);
    return distribution.map(d => ({ ...d, weight: d.weight / totalWeight }));
  }

  private normalDistribution(x: number, mean: number, sigma: number): number {
    return Math.exp(-0.5 * Math.pow((x - mean) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI));
  }

  private calculateBuyRatio(
    candle: CandleData,
    price: number,
    context: { isBullish: boolean; bodyRatio: number; wickImbalance: number }
  ): number {
    const range = candle.high - candle.low;
    const pricePosition = range > 0 ? (price - candle.low) / range : 0.5;

    let buyRatio = 0.5; // Default neutral

    if (context.isBullish) {
      // Bullish candle: more buying at higher prices
      buyRatio = 0.3 + 0.4 * pricePosition + 0.3 * context.bodyRatio;
    } else {
      // Bearish candle: more selling at higher prices
      buyRatio = 0.7 - 0.4 * pricePosition - 0.3 * context.bodyRatio;
    }

    // Adjust for wick imbalance
    buyRatio += context.wickImbalance * 0.1;

    return Math.max(0.1, Math.min(0.9, buyRatio));
  }

  private calculateValueArea(
    levels: VolumeProfileLevel[],
    poc: number,
    totalVolume: number,
    valueAreaPercent: number
  ): { valueAreaHigh: number; valueAreaLow: number; valueAreaVolume: number } {
    const targetVolume = totalVolume * valueAreaPercent;
    let valueAreaVolume = 0;
    const valueAreaPrices: number[] = [];

    // Sort levels by price
    const sortedByPrice = [...levels].sort((a, b) => a.price - b.price);
    const pocIndex = sortedByPrice.findIndex(level => level.price === poc);

    let upIndex = pocIndex;
    let downIndex = pocIndex;

    // Expand outward from POC
    while (valueAreaVolume < targetVolume && (upIndex >= 0 || downIndex < sortedByPrice.length)) {
      const upLevel = upIndex >= 0 ? sortedByPrice[upIndex] : null;
      const downLevel = downIndex < sortedByPrice.length ? sortedByPrice[downIndex] : null;

      if (upLevel && downLevel) {
        // Choose the level with higher volume
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

    const valueAreaHigh = valueAreaPrices.length > 0 ? Math.max(...valueAreaPrices) : levels[0]?.price || 0;
    const valueAreaLow = valueAreaPrices.length > 0 ? Math.min(...valueAreaPrices) : levels[0]?.price || 0;

    return { valueAreaHigh, valueAreaLow, valueAreaVolume };
  }

  private calculateOptimalTick(minPrice: number, maxPrice: number): number {
    const range = maxPrice - minPrice;
    const targetLevels = 50;
    const baseTick = range / targetLevels;

    // Round to nice numbers
    const magnitude = Math.pow(10, Math.floor(Math.log10(baseTick)));
    const normalized = baseTick / magnitude;

    if (normalized <= 1) return magnitude;
    else if (normalized <= 2) return 2 * magnitude;
    else if (normalized <= 5) return 5 * magnitude;
    else return 10 * magnitude;
  }

  private calculateTPOValueArea(
    priceRanges: any[],
    poc: number,
    targetTime: number
  ): { valueAreaHigh: number; valueAreaLow: number } {
    let accumulatedTime = 0;
    const valueAreaPrices: number[] = [];

    // Sort by distance from POC, then by time at price
    const sortedByPOCDistance = priceRanges
      .sort((a, b) => {
        const distA = Math.abs(a.price - poc);
        const distB = Math.abs(b.price - poc);
        return distA === distB ? b.timeAtPrice - a.timeAtPrice : distA - distB;
      });

    for (const priceRange of sortedByPOCDistance) {
      if (accumulatedTime >= targetTime) break;

      accumulatedTime += priceRange.timeAtPrice;
      valueAreaPrices.push(priceRange.price);
    }

    return {
      valueAreaHigh: Math.max(...valueAreaPrices),
      valueAreaLow: Math.min(...valueAreaPrices)
    };
  }

  private classifyDayType(
    data: CandleData[],
    initialBalance: { high: number; low: number; range: number },
    valueArea: { high: number; low: number; poc: number }
  ): 'NORMAL' | 'NORMAL_VARIATION' | 'TREND' | 'NON_TREND' {
    const dayHigh = Math.max(...data.map(d => d.high));
    const dayLow = Math.min(...data.map(d => d.low));
    const dayRange = dayHigh - dayLow;

    // Check if price stayed within initial balance
    if (dayHigh <= initialBalance.high && dayLow >= initialBalance.low) {
      return 'NON_TREND';
    }

    // Check for trending behavior
    const ibExtension = Math.max(
      dayHigh - initialBalance.high,
      initialBalance.low - dayLow
    );

    if (ibExtension > initialBalance.range * 1.5) {
      return 'TREND';
    }

    // Check value area position
    const valueAreaMid = (valueArea.high + valueArea.low) / 2;
    const ibMid = (initialBalance.high + initialBalance.low) / 2;

    if (Math.abs(valueAreaMid - ibMid) < initialBalance.range * 0.3) {
      return 'NORMAL';
    }

    return 'NORMAL_VARIATION';
  }

  private splitIntoSessions(data: CandleData[]): Array<{
    session: 'ASIAN' | 'LONDON' | 'NEW_YORK' | 'OVERLAP';
    startTime: number;
    endTime: number;
    data: CandleData[];
  }> {
    // Simplified session splitting - would need timezone handling in production
    const sessions: any[] = [];

    // Group data by day and classify by time
    const dataByDay = this.groupByDay(data);

    dataByDay.forEach(dayData => {
      // Asian session: 21:00 - 06:00 UTC
      // London session: 07:00 - 16:00 UTC
      // New York session: 12:00 - 21:00 UTC

      const asianData = dayData.filter(d => {
        const hour = new Date(d.timestamp * 1000).getUTCHours();
        return hour >= 21 || hour < 6;
      });

      const londonData = dayData.filter(d => {
        const hour = new Date(d.timestamp * 1000).getUTCHours();
        return hour >= 7 && hour < 16;
      });

      const nyData = dayData.filter(d => {
        const hour = new Date(d.timestamp * 1000).getUTCHours();
        return hour >= 12 && hour < 21;
      });

      if (asianData.length) {
        sessions.push({
          session: 'ASIAN' as const,
          startTime: asianData[0].timestamp,
          endTime: asianData[asianData.length - 1].timestamp,
          data: asianData
        });
      }

      if (londonData.length) {
        sessions.push({
          session: 'LONDON' as const,
          startTime: londonData[0].timestamp,
          endTime: londonData[londonData.length - 1].timestamp,
          data: londonData
        });
      }

      if (nyData.length) {
        sessions.push({
          session: 'NEW_YORK' as const,
          startTime: nyData[0].timestamp,
          endTime: nyData[nyData.length - 1].timestamp,
          data: nyData
        });
      }
    });

    return sessions;
  }

  private groupByDay(data: CandleData[]): CandleData[][] {
    const groups: { [key: string]: CandleData[] } = {};

    data.forEach(candle => {
      const date = new Date(candle.timestamp * 1000).toISOString().split('T')[0];
      if (!groups[date]) groups[date] = [];
      groups[date].push(candle);
    });

    return Object.values(groups);
  }

  private calculateVolatility(returns: number[]): number {
    if (returns.length === 0) return 0;

    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;

    return Math.sqrt(variance) * Math.sqrt(252); // Annualized
  }

  private calculateRelativeVolume(data: CandleData[], index: number): number {
    if (index < 20) return 1;

    const current = data[index].volume;
    const historical = data.slice(Math.max(0, index - 20), index);
    const avgHistorical = historical.reduce((sum, d) => sum + d.volume, 0) / historical.length;

    return avgHistorical > 0 ? current / avgHistorical : 1;
  }

  private calculateVolumeRatio(data: CandleData[], index: number): number {
    if (index < 5) return 1;

    const current = data[index].volume;
    const recent = data.slice(Math.max(0, index - 4), index);
    const avgRecent = recent.reduce((sum, d) => sum + d.volume, 0) / recent.length;

    return avgRecent > 0 ? current / avgRecent : 1;
  }

  private calculateAverageVolume(data: CandleData[], index: number, period: number): number {
    const start = Math.max(0, index - period + 1);
    const slice = data.slice(start, index + 1);
    return slice.reduce((sum, d) => sum + d.volume, 0) / slice.length;
  }

  private estimateDelta(candle: CandleData, prevCandle: CandleData): number {
    const volume = candle.volume || 0;
    const priceChange = candle.close - prevCandle.close;
    const range = candle.high - candle.low;

    // Simple delta estimation based on price action
    let deltaRatio = 0.5; // Neutral

    if (priceChange > 0) {
      // Price went up - likely more buying
      const priceMovement = priceChange / (range || 0.0001);
      deltaRatio = 0.5 + Math.min(0.4, priceMovement * 2);
    } else if (priceChange < 0) {
      // Price went down - likely more selling
      const priceMovement = Math.abs(priceChange) / (range || 0.0001);
      deltaRatio = 0.5 - Math.min(0.4, priceMovement * 2);
    }

    return volume * (deltaRatio - 0.5) * 2; // Convert to delta
  }

  private estimateBidAskVolume(candle: CandleData, delta: number): { volumeAtBid: number; volumeAtAsk: number } {
    const volume = candle.volume || 0;
    const buyVolume = volume / 2 + delta / 2;
    const sellVolume = volume / 2 - delta / 2;

    return {
      volumeAtBid: Math.max(0, sellVolume),
      volumeAtAsk: Math.max(0, buyVolume)
    };
  }

  private calculateAverageDailyVolume(data: CandleData[]): number {
    const dailyVolumes = this.groupByDay(data).map(dayData =>
      dayData.reduce((sum, d) => sum + d.volume, 0)
    );

    return dailyVolumes.reduce((sum, vol) => sum + vol, 0) / dailyVolumes.length;
  }

  private calculateSessionProfiles(data: CandleData[], levels: number): VolumeProfileLevel[] {
    // Simplified session profile - would split by actual trading sessions
    return this.calculateVolumeProfile(data, { levels }).levels;
  }
}