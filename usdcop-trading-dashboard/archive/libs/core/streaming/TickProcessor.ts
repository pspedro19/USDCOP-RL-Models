/**
 * TickProcessor - High-Performance Tick-by-Tick Data Processing
 * Real-time aggregation, pattern detection, and technical analysis
 */

import { EventEmitter } from 'eventemitter3';
import { Subject, BehaviorSubject, Observable, interval } from 'rxjs';
import {
  buffer,
  bufferTime,
  bufferCount,
  map,
  filter,
  scan,
  distinctUntilChanged,
  takeUntil,
  share
} from 'rxjs/operators';

import type {
  StreamMessage,
  StreamDataType,
  TimeInterval
} from '../types/streaming-types';

import type { MarketTick, OHLCV, Trade } from '../types/market-data';

export interface TickAggregationConfig {
  readonly intervals: readonly TimeInterval[];
  readonly enableVWAP: boolean;
  readonly enableVolumeDelta: boolean;
  readonly enableMicrostructure: boolean;
  readonly maxTicksPerSecond: number;
  readonly enablePatternDetection: boolean;
  readonly enableRealTimeIndicators: boolean;
}

export interface AggregatedTick {
  readonly symbol: string;
  readonly interval: TimeInterval;
  readonly timestamp: number;
  readonly open: number;
  readonly high: number;
  readonly low: number;
  readonly close: number;
  readonly volume: number;
  readonly tickCount: number;
  readonly vwap?: number;
  readonly bid: number;
  readonly ask: number;
  readonly spread: number;
  readonly buyVolume: number;
  readonly sellVolume: number;
  readonly volumeDelta: number;
  readonly microstructure?: MicrostructureMetrics;
}

export interface MicrostructureMetrics {
  readonly effectiveSpread: number;
  readonly realizedSpread: number;
  readonly priceImpact: number;
  readonly orderImbalance: number;
  readonly volatility: number;
  readonly momentum: number;
  readonly flowToxicity: number;
}

export interface TickPattern {
  readonly type: TickPatternType;
  readonly confidence: number;
  readonly timestamp: number;
  readonly symbol: string;
  readonly duration: number;
  readonly description: string;
  readonly metadata: Record<string, any>;
}

export interface VolumeProfile {
  readonly symbol: string;
  readonly timestamp: number;
  readonly interval: TimeInterval;
  readonly levels: VolumePriceLevel[];
  readonly valueAreaHigh: number;
  readonly valueAreaLow: number;
  readonly pointOfControl: number;
  readonly totalVolume: number;
}

export interface VolumePriceLevel {
  readonly price: number;
  readonly volume: number;
  readonly percentage: number;
  readonly buyVolume: number;
  readonly sellVolume: number;
}

export type TickPatternType =
  | 'momentum_surge'
  | 'volume_spike'
  | 'spread_widening'
  | 'price_gap'
  | 'unusual_activity'
  | 'liquidity_drain'
  | 'order_flow_imbalance'
  | 'volatility_cluster';

export class TickProcessor extends EventEmitter {
  private readonly config: TickAggregationConfig;

  // Input streams
  private readonly tickStream$ = new Subject<MarketTick>();
  private readonly tradeStream$ = new Subject<Trade>();
  private readonly destroy$ = new Subject<void>();

  // Output streams
  private readonly aggregatedData$ = new Subject<AggregatedTick>();
  private readonly patterns$ = new Subject<TickPattern>();
  private readonly volumeProfile$ = new Subject<VolumeProfile>();
  private readonly microstructure$ = new Subject<MicrostructureMetrics>();

  // Internal state
  private readonly tickBuffers = new Map<string, TickBuffer>();
  private readonly priceHistory = new Map<string, number[]>();
  private readonly volumeHistory = new Map<string, number[]>();
  private readonly tradeHistory = new Map<string, Trade[]>();

  // Real-time indicators
  private readonly realtimeIndicators = new Map<string, RealtimeIndicators>();

  constructor(config: TickAggregationConfig) {
    super();
    this.config = config;
    this.initialize();
  }

  // ==========================================
  // PUBLIC API
  // ==========================================

  public processTick(tick: MarketTick): void {
    this.tickStream$.next(tick);
  }

  public processTrade(trade: Trade): void {
    this.tradeStream$.next(trade);
  }

  public getAggregatedStream(
    symbol?: string,
    interval?: TimeInterval
  ): Observable<AggregatedTick> {
    return this.aggregatedData$.pipe(
      filter(data =>
        (!symbol || data.symbol === symbol) &&
        (!interval || data.interval === interval)
      ),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getPatternStream(symbol?: string): Observable<TickPattern> {
    return this.patterns$.pipe(
      filter(pattern => !symbol || pattern.symbol === symbol),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getVolumeProfileStream(symbol?: string): Observable<VolumeProfile> {
    return this.volumeProfile$.pipe(
      filter(profile => !symbol || profile.symbol === symbol),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getMicrostructureStream(symbol?: string): Observable<MicrostructureMetrics> {
    return this.microstructure$.pipe(
      takeUntil(this.destroy$),
      share()
    );
  }

  public getRealtimeIndicators(symbol: string): RealtimeIndicators | undefined {
    return this.realtimeIndicators.get(symbol);
  }

  public getTickBuffer(symbol: string, interval: TimeInterval): TickBuffer | undefined {
    const key = `${symbol}_${interval}`;
    return this.tickBuffers.get(key);
  }

  // ==========================================
  // INITIALIZATION & SETUP
  // ==========================================

  private initialize(): void {
    this.setupTickProcessing();
    this.setupTradeProcessing();
    this.setupAggregation();
    this.setupPatternDetection();
    this.setupVolumeProfileAnalysis();
    this.setupMicrostructureAnalysis();
    this.setupRealtimeIndicators();
  }

  private setupTickProcessing(): void {
    // Process incoming ticks
    this.tickStream$.pipe(
      takeUntil(this.destroy$)
    ).subscribe(tick => {
      this.processSingleTick(tick);
    });

    // Throttle tick processing if configured
    if (this.config.maxTicksPerSecond > 0) {
      const throttleInterval = 1000 / this.config.maxTicksPerSecond;

      this.tickStream$.pipe(
        bufferTime(throttleInterval),
        filter(ticks => ticks.length > 0),
        takeUntil(this.destroy$)
      ).subscribe(ticks => {
        // Process latest tick from each symbol
        const latestTicks = this.selectLatestTicks(ticks);
        latestTicks.forEach(tick => this.processSingleTick(tick));
      });
    }
  }

  private setupTradeProcessing(): void {
    this.tradeStream$.pipe(
      takeUntil(this.destroy$)
    ).subscribe(trade => {
      this.processSingleTrade(trade);
    });
  }

  private setupAggregation(): void {
    // Setup aggregation for each configured interval
    this.config.intervals.forEach(interval => {
      this.setupIntervalAggregation(interval);
    });
  }

  private setupIntervalAggregation(interval: TimeInterval): void {
    const intervalMs = this.getIntervalMilliseconds(interval);

    // Time-based aggregation
    this.tickStream$.pipe(
      bufferTime(intervalMs),
      filter(ticks => ticks.length > 0),
      map(ticks => this.aggregateTicks(ticks, interval)),
      takeUntil(this.destroy$)
    ).subscribe(aggregated => {
      aggregated.forEach(agg => {
        this.aggregatedData$.next(agg);
        this.emit('aggregated_data', agg);
      });
    });
  }

  private setupPatternDetection(): void {
    if (!this.config.enablePatternDetection) return;

    // Pattern detection on aggregated data
    this.aggregatedData$.pipe(
      bufferCount(20, 1), // Sliding window of 20 aggregations
      map(window => this.detectPatterns(window)),
      filter(patterns => patterns.length > 0),
      takeUntil(this.destroy$)
    ).subscribe(patterns => {
      patterns.forEach(pattern => {
        this.patterns$.next(pattern);
        this.emit('pattern_detected', pattern);
      });
    });
  }

  private setupVolumeProfileAnalysis(): void {
    // Generate volume profiles every minute
    interval(60000).pipe(
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.generateVolumeProfiles();
    });
  }

  private setupMicrostructureAnalysis(): void {
    if (!this.config.enableMicrostructure) return;

    // Analyze microstructure every 5 seconds
    interval(5000).pipe(
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.analyzeMicrostructure();
    });
  }

  private setupRealtimeIndicators(): void {
    if (!this.config.enableRealTimeIndicators) return;

    // Update indicators on every tick
    this.tickStream$.pipe(
      takeUntil(this.destroy$)
    ).subscribe(tick => {
      this.updateRealtimeIndicators(tick);
    });
  }

  // ==========================================
  // TICK PROCESSING
  // ==========================================

  private processSingleTick(tick: MarketTick): void {
    const symbol = tick.symbol;

    // Update price history
    this.updatePriceHistory(symbol, tick.last);

    // Update volume history
    this.updateVolumeHistory(symbol, tick.volume);

    // Add to tick buffers for each interval
    this.config.intervals.forEach(interval => {
      this.addToTickBuffer(symbol, interval, tick);
    });

    // Emit processed tick
    this.emit('tick_processed', tick);
  }

  private processSingleTrade(trade: Trade): void {
    const symbol = trade.symbol;

    // Update trade history
    this.updateTradeHistory(symbol, trade);

    // Analyze trade flow
    this.analyzeTradeFlow(trade);

    this.emit('trade_processed', trade);
  }

  private selectLatestTicks(ticks: MarketTick[]): MarketTick[] {
    const symbolMap = new Map<string, MarketTick>();

    ticks.forEach(tick => {
      const existing = symbolMap.get(tick.symbol);
      if (!existing || tick.timestamp > existing.timestamp) {
        symbolMap.set(tick.symbol, tick);
      }
    });

    return Array.from(symbolMap.values());
  }

  // ==========================================
  // AGGREGATION LOGIC
  // ==========================================

  private aggregateTicks(ticks: MarketTick[], interval: TimeInterval): AggregatedTick[] {
    const symbolGroups = this.groupTicksBySymbol(ticks);
    const aggregated: AggregatedTick[] = [];

    symbolGroups.forEach((symbolTicks, symbol) => {
      if (symbolTicks.length === 0) return;

      const agg = this.createAggregation(symbol, symbolTicks, interval);
      aggregated.push(agg);
    });

    return aggregated;
  }

  private createAggregation(
    symbol: string,
    ticks: MarketTick[],
    interval: TimeInterval
  ): AggregatedTick {
    // Sort ticks by timestamp
    const sortedTicks = ticks.sort((a, b) => a.timestamp - b.timestamp);

    const first = sortedTicks[0];
    const last = sortedTicks[sortedTicks.length - 1];

    // OHLC calculation
    const prices = sortedTicks.map(t => t.last);
    const open = first.last;
    const close = last.last;
    const high = Math.max(...prices);
    const low = Math.min(...prices);

    // Volume calculations
    const totalVolume = sortedTicks.reduce((sum, t) => sum + t.volume, 0);
    const buyVolume = this.calculateBuyVolume(sortedTicks);
    const sellVolume = totalVolume - buyVolume;
    const volumeDelta = buyVolume - sellVolume;

    // VWAP calculation
    let vwap: number | undefined;
    if (this.config.enableVWAP && totalVolume > 0) {
      const vwapSum = sortedTicks.reduce((sum, t) => sum + (t.last * t.volume), 0);
      vwap = vwapSum / totalVolume;
    }

    // Bid/Ask calculations
    const bid = this.calculateWeightedAverage(sortedTicks, 'bid');
    const ask = this.calculateWeightedAverage(sortedTicks, 'ask');
    const spread = ask - bid;

    // Microstructure metrics
    let microstructure: MicrostructureMetrics | undefined;
    if (this.config.enableMicrostructure) {
      microstructure = this.calculateMicrostructureMetrics(symbol, sortedTicks);
    }

    return {
      symbol,
      interval,
      timestamp: this.alignToInterval(last.timestamp, interval),
      open,
      high,
      low,
      close,
      volume: totalVolume,
      tickCount: sortedTicks.length,
      vwap,
      bid,
      ask,
      spread,
      buyVolume,
      sellVolume,
      volumeDelta,
      microstructure
    };
  }

  private calculateBuyVolume(ticks: MarketTick[]): number {
    // Estimate buy volume based on price movement
    let buyVolume = 0;

    for (let i = 1; i < ticks.length; i++) {
      const current = ticks[i];
      const previous = ticks[i - 1];

      if (current.last > previous.last) {
        // Price went up, assume buying pressure
        buyVolume += current.volume;
      } else if (current.last === previous.last) {
        // No price change, split volume
        buyVolume += current.volume * 0.5;
      }
      // Price went down, assume selling pressure (counted as sell volume)
    }

    return buyVolume;
  }

  private calculateWeightedAverage(ticks: MarketTick[], field: keyof MarketTick): number {
    let weightedSum = 0;
    let totalWeight = 0;

    ticks.forEach(tick => {
      const value = tick[field] as number;
      const weight = tick.volume || 1;

      if (typeof value === 'number' && value > 0) {
        weightedSum += value * weight;
        totalWeight += weight;
      }
    });

    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }

  // ==========================================
  // PATTERN DETECTION
  // ==========================================

  private detectPatterns(window: AggregatedTick[]): TickPattern[] {
    const patterns: TickPattern[] = [];

    if (window.length < 5) return patterns;

    // Group by symbol
    const symbolWindows = this.groupAggregatedBySymbol(window);

    symbolWindows.forEach((symbolData, symbol) => {
      // Detect momentum surge
      const momentumPattern = this.detectMomentumSurge(symbol, symbolData);
      if (momentumPattern) patterns.push(momentumPattern);

      // Detect volume spike
      const volumePattern = this.detectVolumeSpike(symbol, symbolData);
      if (volumePattern) patterns.push(volumePattern);

      // Detect spread widening
      const spreadPattern = this.detectSpreadWidening(symbol, symbolData);
      if (spreadPattern) patterns.push(spreadPattern);

      // Detect unusual activity
      const unusualPattern = this.detectUnusualActivity(symbol, symbolData);
      if (unusualPattern) patterns.push(unusualPattern);
    });

    return patterns;
  }

  private detectMomentumSurge(symbol: string, data: AggregatedTick[]): TickPattern | null {
    if (data.length < 3) return null;

    const recent = data.slice(-3);
    const prices = recent.map(d => d.close);

    // Check for consecutive price increases
    let consecutiveIncrease = 0;
    for (let i = 1; i < prices.length; i++) {
      if (prices[i] > prices[i - 1]) {
        consecutiveIncrease++;
      } else {
        break;
      }
    }

    if (consecutiveIncrease >= 2) {
      const priceChange = prices[prices.length - 1] - prices[0];
      const percentChange = (priceChange / prices[0]) * 100;

      if (Math.abs(percentChange) > 0.1) { // 0.1% threshold
        return {
          type: 'momentum_surge',
          confidence: Math.min(Math.abs(percentChange) * 10, 100),
          timestamp: recent[recent.length - 1].timestamp,
          symbol,
          duration: recent[recent.length - 1].timestamp - recent[0].timestamp,
          description: `Price momentum surge: ${percentChange.toFixed(2)}% in ${consecutiveIncrease} periods`,
          metadata: {
            priceChange,
            percentChange,
            consecutiveMoves: consecutiveIncrease
          }
        };
      }
    }

    return null;
  }

  private detectVolumeSpike(symbol: string, data: AggregatedTick[]): TickPattern | null {
    if (data.length < 5) return null;

    const recent = data[data.length - 1];
    const historical = data.slice(-5, -1);

    const avgHistoricalVolume = historical.reduce((sum, d) => sum + d.volume, 0) / historical.length;
    const volumeRatio = recent.volume / avgHistoricalVolume;

    if (volumeRatio > 2) { // 2x normal volume
      return {
        type: 'volume_spike',
        confidence: Math.min(volumeRatio * 20, 100),
        timestamp: recent.timestamp,
        symbol,
        duration: 0,
        description: `Volume spike: ${volumeRatio.toFixed(2)}x normal volume`,
        metadata: {
          currentVolume: recent.volume,
          avgVolume: avgHistoricalVolume,
          volumeRatio
        }
      };
    }

    return null;
  }

  private detectSpreadWidening(symbol: string, data: AggregatedTick[]): TickPattern | null {
    if (data.length < 3) return null;

    const recent = data.slice(-3);
    const spreads = recent.map(d => d.spread);

    // Check if spread is widening
    let isWidening = true;
    for (let i = 1; i < spreads.length; i++) {
      if (spreads[i] <= spreads[i - 1]) {
        isWidening = false;
        break;
      }
    }

    if (isWidening) {
      const spreadIncrease = spreads[spreads.length - 1] - spreads[0];
      const percentIncrease = (spreadIncrease / spreads[0]) * 100;

      if (percentIncrease > 50) { // 50% spread increase
        return {
          type: 'spread_widening',
          confidence: Math.min(percentIncrease, 100),
          timestamp: recent[recent.length - 1].timestamp,
          symbol,
          duration: recent[recent.length - 1].timestamp - recent[0].timestamp,
          description: `Spread widening: ${percentIncrease.toFixed(1)}% increase`,
          metadata: {
            initialSpread: spreads[0],
            finalSpread: spreads[spreads.length - 1],
            spreadIncrease,
            percentIncrease
          }
        };
      }
    }

    return null;
  }

  private detectUnusualActivity(symbol: string, data: AggregatedTick[]): TickPattern | null {
    if (data.length < 5) return null;

    const recent = data[data.length - 1];
    const historical = data.slice(-5, -1);

    // Check for unusual tick count
    const avgTickCount = historical.reduce((sum, d) => sum + d.tickCount, 0) / historical.length;
    const tickRatio = recent.tickCount / avgTickCount;

    // Check for unusual volume delta
    const avgVolumeDelta = Math.abs(historical.reduce((sum, d) => sum + Math.abs(d.volumeDelta), 0) / historical.length);
    const currentVolumeDelta = Math.abs(recent.volumeDelta);
    const deltaRatio = avgVolumeDelta > 0 ? currentVolumeDelta / avgVolumeDelta : 1;

    if (tickRatio > 3 || deltaRatio > 3) {
      return {
        type: 'unusual_activity',
        confidence: Math.min(Math.max(tickRatio, deltaRatio) * 20, 100),
        timestamp: recent.timestamp,
        symbol,
        duration: 0,
        description: `Unusual activity: ${tickRatio.toFixed(1)}x tick rate, ${deltaRatio.toFixed(1)}x volume delta`,
        metadata: {
          tickRatio,
          deltaRatio,
          currentTickCount: recent.tickCount,
          avgTickCount,
          currentVolumeDelta,
          avgVolumeDelta
        }
      };
    }

    return null;
  }

  // ==========================================
  // MICROSTRUCTURE ANALYSIS
  // ==========================================

  private calculateMicrostructureMetrics(symbol: string, ticks: MarketTick[]): MicrostructureMetrics {
    const trades = this.tradeHistory.get(symbol) || [];
    const recentTrades = trades.filter(t => t.timestamp > Date.now() - 60000); // Last minute

    return {
      effectiveSpread: this.calculateEffectiveSpread(ticks),
      realizedSpread: this.calculateRealizedSpread(recentTrades),
      priceImpact: this.calculatePriceImpact(recentTrades),
      orderImbalance: this.calculateOrderImbalance(recentTrades),
      volatility: this.calculateShortTermVolatility(symbol),
      momentum: this.calculateMomentum(symbol),
      flowToxicity: this.calculateFlowToxicity(recentTrades)
    };
  }

  private calculateEffectiveSpread(ticks: MarketTick[]): number {
    if (ticks.length === 0) return 0;

    const spreads = ticks.map(t => t.ask - t.bid).filter(s => s > 0);
    return spreads.length > 0 ? spreads.reduce((sum, s) => sum + s, 0) / spreads.length : 0;
  }

  private calculateRealizedSpread(trades: Trade[]): number {
    // Simplified realized spread calculation
    if (trades.length < 2) return 0;

    let totalSpread = 0;
    let count = 0;

    for (let i = 1; i < trades.length; i++) {
      const current = trades[i];
      const previous = trades[i - 1];

      if (current.symbol === previous.symbol) {
        const spread = Math.abs(current.price - previous.price);
        totalSpread += spread;
        count++;
      }
    }

    return count > 0 ? totalSpread / count : 0;
  }

  private calculatePriceImpact(trades: Trade[]): number {
    // Simplified price impact measure
    if (trades.length < 5) return 0;

    const buyTrades = trades.filter(t => t.side === 'buy');
    const sellTrades = trades.filter(t => t.side === 'sell');

    if (buyTrades.length === 0 || sellTrades.length === 0) return 0;

    const avgBuyPrice = buyTrades.reduce((sum, t) => sum + t.price, 0) / buyTrades.length;
    const avgSellPrice = sellTrades.reduce((sum, t) => sum + t.price, 0) / sellTrades.length;

    return Math.abs(avgBuyPrice - avgSellPrice);
  }

  private calculateOrderImbalance(trades: Trade[]): number {
    if (trades.length === 0) return 0;

    const buyVolume = trades.filter(t => t.side === 'buy').reduce((sum, t) => sum + t.size, 0);
    const sellVolume = trades.filter(t => t.side === 'sell').reduce((sum, t) => sum + t.size, 0);

    const totalVolume = buyVolume + sellVolume;
    return totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0;
  }

  private calculateShortTermVolatility(symbol: string): number {
    const prices = this.priceHistory.get(symbol) || [];
    if (prices.length < 10) return 0;

    const recentPrices = prices.slice(-10);
    const returns = [];

    for (let i = 1; i < recentPrices.length; i++) {
      const returnValue = Math.log(recentPrices[i] / recentPrices[i - 1]);
      returns.push(returnValue);
    }

    // Calculate standard deviation
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;

    return Math.sqrt(variance);
  }

  private calculateMomentum(symbol: string): number {
    const prices = this.priceHistory.get(symbol) || [];
    if (prices.length < 5) return 0;

    const recent = prices.slice(-5);
    const first = recent[0];
    const last = recent[recent.length - 1];

    return (last - first) / first;
  }

  private calculateFlowToxicity(trades: Trade[]): number {
    // Simplified flow toxicity measure
    if (trades.length < 10) return 0;

    let consecutiveSameSide = 0;
    let maxConsecutive = 0;
    let lastSide = trades[0].side;

    for (let i = 1; i < trades.length; i++) {
      if (trades[i].side === lastSide) {
        consecutiveSameSide++;
      } else {
        maxConsecutive = Math.max(maxConsecutive, consecutiveSameSide);
        consecutiveSameSide = 1;
        lastSide = trades[i].side;
      }
    }

    maxConsecutive = Math.max(maxConsecutive, consecutiveSameSide);
    return maxConsecutive / trades.length;
  }

  private analyzeMicrostructure(): void {
    this.realtimeIndicators.forEach((indicators, symbol) => {
      const metrics = this.calculateMicrostructureMetrics(symbol, []);
      this.microstructure$.next(metrics);
    });
  }

  // ==========================================
  // VOLUME PROFILE ANALYSIS
  // ==========================================

  private generateVolumeProfiles(): void {
    this.tickBuffers.forEach((buffer, key) => {
      const [symbol, interval] = key.split('_');
      const profile = this.createVolumeProfile(symbol, interval as TimeInterval, buffer);
      if (profile) {
        this.volumeProfile$.next(profile);
      }
    });
  }

  private createVolumeProfile(
    symbol: string,
    interval: TimeInterval,
    buffer: TickBuffer
  ): VolumeProfile | null {
    const ticks = buffer.getTicks();
    if (ticks.length === 0) return null;

    // Create price levels
    const priceLevels = this.createPriceLevels(ticks);

    // Calculate value area (70% of volume)
    const totalVolume = priceLevels.reduce((sum, level) => sum + level.volume, 0);
    const targetVolume = totalVolume * 0.7;

    // Sort by volume to find POC (Point of Control)
    const sortedByVolume = [...priceLevels].sort((a, b) => b.volume - a.volume);
    const pointOfControl = sortedByVolume[0].price;

    // Find value area around POC
    const { valueAreaHigh, valueAreaLow } = this.calculateValueArea(priceLevels, targetVolume, pointOfControl);

    return {
      symbol,
      timestamp: Date.now(),
      interval,
      levels: priceLevels,
      valueAreaHigh,
      valueAreaLow,
      pointOfControl,
      totalVolume
    };
  }

  private createPriceLevels(ticks: MarketTick[]): VolumePriceLevel[] {
    const priceMap = new Map<number, { volume: number; buyVolume: number; sellVolume: number }>();

    ticks.forEach(tick => {
      const price = Math.round(tick.last * 100) / 100; // Round to 2 decimals
      const existing = priceMap.get(price) || { volume: 0, buyVolume: 0, sellVolume: 0 };

      existing.volume += tick.volume;
      // Estimate buy/sell volume based on price movement
      if (tick.changePercent > 0) {
        existing.buyVolume += tick.volume * 0.7;
        existing.sellVolume += tick.volume * 0.3;
      } else if (tick.changePercent < 0) {
        existing.buyVolume += tick.volume * 0.3;
        existing.sellVolume += tick.volume * 0.7;
      } else {
        existing.buyVolume += tick.volume * 0.5;
        existing.sellVolume += tick.volume * 0.5;
      }

      priceMap.set(price, existing);
    });

    const totalVolume = Array.from(priceMap.values()).reduce((sum, level) => sum + level.volume, 0);

    return Array.from(priceMap.entries()).map(([price, data]) => ({
      price,
      volume: data.volume,
      percentage: (data.volume / totalVolume) * 100,
      buyVolume: data.buyVolume,
      sellVolume: data.sellVolume
    })).sort((a, b) => b.price - a.price);
  }

  private calculateValueArea(
    levels: VolumePriceLevel[],
    targetVolume: number,
    poc: number
  ): { valueAreaHigh: number; valueAreaLow: number } {
    // Start from POC and expand up and down
    const sortedByPrice = [...levels].sort((a, b) => a.price - b.price);
    const pocIndex = sortedByPrice.findIndex(level => level.price === poc);

    let accumulatedVolume = sortedByPrice[pocIndex].volume;
    let highIndex = pocIndex;
    let lowIndex = pocIndex;

    while (accumulatedVolume < targetVolume && (highIndex < sortedByPrice.length - 1 || lowIndex > 0)) {
      const canExpandUp = highIndex < sortedByPrice.length - 1;
      const canExpandDown = lowIndex > 0;

      if (!canExpandUp && !canExpandDown) break;

      const upVolume = canExpandUp ? sortedByPrice[highIndex + 1].volume : 0;
      const downVolume = canExpandDown ? sortedByPrice[lowIndex - 1].volume : 0;

      if (!canExpandDown || (canExpandUp && upVolume >= downVolume)) {
        highIndex++;
        accumulatedVolume += sortedByPrice[highIndex].volume;
      } else {
        lowIndex--;
        accumulatedVolume += sortedByPrice[lowIndex].volume;
      }
    }

    return {
      valueAreaHigh: sortedByPrice[highIndex].price,
      valueAreaLow: sortedByPrice[lowIndex].price
    };
  }

  // ==========================================
  // REALTIME INDICATORS
  // ==========================================

  private updateRealtimeIndicators(tick: MarketTick): void {
    const symbol = tick.symbol;

    if (!this.realtimeIndicators.has(symbol)) {
      this.realtimeIndicators.set(symbol, new RealtimeIndicators(symbol));
    }

    const indicators = this.realtimeIndicators.get(symbol)!;
    indicators.update(tick);
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  private addToTickBuffer(symbol: string, interval: TimeInterval, tick: MarketTick): void {
    const key = `${symbol}_${interval}`;

    if (!this.tickBuffers.has(key)) {
      this.tickBuffers.set(key, new TickBuffer(symbol, interval));
    }

    const buffer = this.tickBuffers.get(key)!;
    buffer.addTick(tick);
  }

  private updatePriceHistory(symbol: string, price: number): void {
    if (!this.priceHistory.has(symbol)) {
      this.priceHistory.set(symbol, []);
    }

    const history = this.priceHistory.get(symbol)!;
    history.push(price);

    // Keep only last 100 prices
    if (history.length > 100) {
      history.splice(0, history.length - 100);
    }
  }

  private updateVolumeHistory(symbol: string, volume: number): void {
    if (!this.volumeHistory.has(symbol)) {
      this.volumeHistory.set(symbol, []);
    }

    const history = this.volumeHistory.get(symbol)!;
    history.push(volume);

    // Keep only last 100 volumes
    if (history.length > 100) {
      history.splice(0, history.length - 100);
    }
  }

  private updateTradeHistory(symbol: string, trade: Trade): void {
    if (!this.tradeHistory.has(symbol)) {
      this.tradeHistory.set(symbol, []);
    }

    const history = this.tradeHistory.get(symbol)!;
    history.push(trade);

    // Keep only last 1000 trades
    if (history.length > 1000) {
      history.splice(0, history.length - 1000);
    }
  }

  private analyzeTradeFlow(trade: Trade): void {
    // Implement trade flow analysis
    this.emit('trade_flow_analyzed', { trade });
  }

  private groupTicksBySymbol(ticks: MarketTick[]): Map<string, MarketTick[]> {
    const groups = new Map<string, MarketTick[]>();

    ticks.forEach(tick => {
      const symbol = tick.symbol;
      if (!groups.has(symbol)) {
        groups.set(symbol, []);
      }
      groups.get(symbol)!.push(tick);
    });

    return groups;
  }

  private groupAggregatedBySymbol(data: AggregatedTick[]): Map<string, AggregatedTick[]> {
    const groups = new Map<string, AggregatedTick[]>();

    data.forEach(item => {
      const symbol = item.symbol;
      if (!groups.has(symbol)) {
        groups.set(symbol, []);
      }
      groups.get(symbol)!.push(item);
    });

    return groups;
  }

  private getIntervalMilliseconds(interval: TimeInterval): number {
    const map: Record<TimeInterval, number> = {
      '1s': 1000,
      '5s': 5000,
      '15s': 15000,
      '30s': 30000,
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      '30m': 1800000,
      '1h': 3600000,
      '4h': 14400000,
      '1d': 86400000,
      '1w': 604800000,
      '1M': 2592000000
    };

    return map[interval] || 60000;
  }

  private alignToInterval(timestamp: number, interval: TimeInterval): number {
    const intervalMs = this.getIntervalMilliseconds(interval);
    return Math.floor(timestamp / intervalMs) * intervalMs;
  }

  // ==========================================
  // CLEANUP
  // ==========================================

  public destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear all data structures
    this.tickBuffers.clear();
    this.priceHistory.clear();
    this.volumeHistory.clear();
    this.tradeHistory.clear();
    this.realtimeIndicators.clear();

    // Complete subjects
    this.tickStream$.complete();
    this.tradeStream$.complete();
    this.aggregatedData$.complete();
    this.patterns$.complete();
    this.volumeProfile$.complete();
    this.microstructure$.complete();

    this.removeAllListeners();
  }
}

// ==========================================
// HELPER CLASSES
// ==========================================

class TickBuffer {
  private readonly symbol: string;
  private readonly interval: TimeInterval;
  private readonly ticks: MarketTick[] = [];
  private readonly maxSize = 1000;

  constructor(symbol: string, interval: TimeInterval) {
    this.symbol = symbol;
    this.interval = interval;
  }

  public addTick(tick: MarketTick): void {
    this.ticks.push(tick);

    // Maintain buffer size
    if (this.ticks.length > this.maxSize) {
      this.ticks.splice(0, this.ticks.length - this.maxSize);
    }
  }

  public getTicks(): MarketTick[] {
    return [...this.ticks];
  }

  public getRecentTicks(count: number): MarketTick[] {
    return this.ticks.slice(-count);
  }

  public clear(): void {
    this.ticks.length = 0;
  }
}

class RealtimeIndicators {
  private readonly symbol: string;
  private ema9: number = 0;
  private ema21: number = 0;
  private ema50: number = 0;
  private rsi: number = 50;
  private macd: number = 0;
  private initialized = false;

  constructor(symbol: string) {
    this.symbol = symbol;
  }

  public update(tick: MarketTick): void {
    const price = tick.last;

    if (!this.initialized) {
      this.ema9 = price;
      this.ema21 = price;
      this.ema50 = price;
      this.initialized = true;
      return;
    }

    // Update EMAs
    this.ema9 = this.calculateEMA(price, this.ema9, 9);
    this.ema21 = this.calculateEMA(price, this.ema21, 21);
    this.ema50 = this.calculateEMA(price, this.ema50, 50);

    // Update MACD (simplified)
    this.macd = this.ema9 - this.ema21;
  }

  private calculateEMA(price: number, previousEMA: number, period: number): number {
    const multiplier = 2 / (period + 1);
    return (price - previousEMA) * multiplier + previousEMA;
  }

  public getEMA9(): number { return this.ema9; }
  public getEMA21(): number { return this.ema21; }
  public getEMA50(): number { return this.ema50; }
  public getRSI(): number { return this.rsi; }
  public getMACD(): number { return this.macd; }
}