/**
 * OrderBookManager - Advanced Order Book Reconstruction & Analysis
 * Real-time order book maintenance with gap detection, validation, and analytics
 */

import { EventEmitter } from 'eventemitter3';
import { Subject, BehaviorSubject, Observable, interval } from 'rxjs';
import { takeUntil, filter, map, share } from 'rxjs/operators';

import type {
  StreamMessage,
  StreamError
} from '../types/streaming-types';

import type { OrderBook, OrderBookLevel, Trade } from '../types/market-data';

export interface OrderBookConfig {
  readonly maxLevels: number;
  readonly enableValidation: boolean;
  readonly enableGapDetection: boolean;
  readonly enableAnalytics: boolean;
  readonly checksumValidation: boolean;
  readonly autoReconstruct: boolean;
  readonly reconstructionTimeout: number;
  readonly priceToleranceBps: number;
  readonly volumeToleranceBps: number;
}

export interface OrderBookSnapshot {
  readonly symbol: string;
  readonly timestamp: number;
  readonly sequence: number;
  readonly bids: readonly OrderBookLevel[];
  readonly asks: readonly OrderBookLevel[];
  readonly spread: number;
  readonly midPrice: number;
  readonly totalBidVolume: number;
  readonly totalAskVolume: number;
  readonly imbalance: number;
  readonly checksum?: string;
  readonly quality: OrderBookQuality;
}

export interface OrderBookUpdate {
  readonly type: 'snapshot' | 'delta' | 'trade_impact';
  readonly symbol: string;
  readonly timestamp: number;
  readonly sequence: number;
  readonly changes: OrderBookChange[];
  readonly metadata?: Record<string, any>;
}

export interface OrderBookChange {
  readonly side: 'bid' | 'ask';
  readonly price: number;
  readonly size: number;
  readonly action: 'add' | 'update' | 'delete';
  readonly timestamp: number;
}

export interface OrderBookAnalytics {
  readonly symbol: string;
  readonly timestamp: number;
  readonly depth: {
    readonly levels: number;
    readonly bidDepth: number;
    readonly askDepth: number;
    readonly totalDepth: number;
  };
  readonly spread: {
    readonly absolute: number;
    readonly relative: number;
    readonly weightedAverage: number;
  };
  readonly imbalance: {
    readonly ratio: number;
    readonly pressure: 'buy' | 'sell' | 'neutral';
    readonly strength: number;
  };
  readonly liquidity: {
    readonly tightness: number;
    readonly depth: number;
    readonly resilience: number;
    readonly immediacy: number;
  };
  readonly volatility: {
    readonly midPriceVol: number;
    readonly spreadVol: number;
    readonly orderFlowVol: number;
  };
  readonly microstructure: {
    readonly tickSize: number;
    readonly effectiveSpread: number;
    readonly impactCost: number;
    readonly resilience: number;
  };
}

export interface OrderBookGap {
  readonly symbol: string;
  readonly detectedAt: number;
  readonly gapType: 'sequence' | 'price' | 'time';
  readonly severity: 'low' | 'medium' | 'high' | 'critical';
  readonly expectedSequence?: number;
  readonly actualSequence?: number;
  readonly gapSize?: number;
  readonly description: string;
}

export interface OrderBookValidation {
  readonly isValid: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
  readonly qualityScore: number;
  readonly checksumValid: boolean;
  readonly priceConsistency: boolean;
  readonly volumeConsistency: boolean;
  readonly sequenceConsistency: boolean;
}

export type OrderBookQuality = 'excellent' | 'good' | 'fair' | 'poor' | 'invalid';

export class OrderBookManager extends EventEmitter {
  private readonly config: OrderBookConfig;

  // Internal state
  private readonly orderBooks = new Map<string, ManagedOrderBook>();
  private readonly analytics = new Map<string, OrderBookAnalytics>();
  private readonly recentTrades = new Map<string, Trade[]>();

  // Observables
  private readonly snapshots$ = new Subject<OrderBookSnapshot>();
  private readonly updates$ = new Subject<OrderBookUpdate>();
  private readonly analytics$ = new Subject<OrderBookAnalytics>();
  private readonly gaps$ = new Subject<OrderBookGap>();
  private readonly errors$ = new Subject<StreamError>();
  private readonly destroy$ = new Subject<void>();

  // Monitoring
  private analyticsTimer?: NodeJS.Timeout;

  constructor(config: OrderBookConfig) {
    super();
    this.config = config;
    this.initialize();
  }

  // ==========================================
  // PUBLIC API
  // ==========================================

  public processUpdate(symbol: string, update: any): void {
    try {
      const managedBook = this.getOrCreateOrderBook(symbol);
      const processedUpdate = this.normalizeUpdate(symbol, update);

      // Validate update
      if (this.config.enableValidation) {
        const validation = this.validateUpdate(managedBook, processedUpdate);
        if (!validation.isValid) {
          this.handleValidationError(symbol, validation, processedUpdate);
          return;
        }
      }

      // Check for gaps
      if (this.config.enableGapDetection) {
        this.detectGaps(managedBook, processedUpdate);
      }

      // Apply update
      this.applyUpdate(managedBook, processedUpdate);

      // Generate snapshot
      const snapshot = this.generateSnapshot(managedBook);
      this.snapshots$.next(snapshot);

      // Emit update event
      this.updates$.next(processedUpdate);

      this.emit('order_book_updated', { symbol, snapshot, update: processedUpdate });

    } catch (error) {
      this.handleError(symbol, error, 'process_update');
    }
  }

  public processSnapshot(symbol: string, snapshot: any): void {
    try {
      const managedBook = this.getOrCreateOrderBook(symbol);
      const normalizedSnapshot = this.normalizeSnapshot(symbol, snapshot);

      // Reset order book with new snapshot
      managedBook.reset(normalizedSnapshot);

      // Generate and emit snapshot
      const processedSnapshot = this.generateSnapshot(managedBook);
      this.snapshots$.next(processedSnapshot);

      this.emit('order_book_snapshot', { symbol, snapshot: processedSnapshot });

    } catch (error) {
      this.handleError(symbol, error, 'process_snapshot');
    }
  }

  public processTrade(trade: Trade): void {
    try {
      // Update recent trades
      this.updateRecentTrades(trade);

      // Apply trade impact to order book
      const managedBook = this.orderBooks.get(trade.symbol);
      if (managedBook) {
        this.applyTradeImpact(managedBook, trade);

        const snapshot = this.generateSnapshot(managedBook);
        this.snapshots$.next(snapshot);
      }

      this.emit('trade_processed', trade);

    } catch (error) {
      this.handleError(trade.symbol, error, 'process_trade');
    }
  }

  public getOrderBook(symbol: string): OrderBookSnapshot | null {
    const managedBook = this.orderBooks.get(symbol);
    return managedBook ? this.generateSnapshot(managedBook) : null;
  }

  public getSnapshotStream(symbol?: string): Observable<OrderBookSnapshot> {
    return this.snapshots$.pipe(
      filter(snapshot => !symbol || snapshot.symbol === symbol),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getUpdatesStream(symbol?: string): Observable<OrderBookUpdate> {
    return this.updates$.pipe(
      filter(update => !symbol || update.symbol === symbol),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getAnalyticsStream(symbol?: string): Observable<OrderBookAnalytics> {
    return this.analytics$.pipe(
      filter(analytics => !symbol || analytics.symbol === symbol),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getGapsStream(symbol?: string): Observable<OrderBookGap> {
    return this.gaps$.pipe(
      filter(gap => !symbol || gap.symbol === symbol),
      takeUntil(this.destroy$),
      share()
    );
  }

  public getAnalytics(symbol: string): OrderBookAnalytics | null {
    return this.analytics.get(symbol) || null;
  }

  public requestReconstruction(symbol: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      try {
        const managedBook = this.orderBooks.get(symbol);
        if (!managedBook) {
          reject(new Error(`Order book not found: ${symbol}`));
          return;
        }

        // Trigger reconstruction
        this.reconstructOrderBook(managedBook);
        resolve(true);

      } catch (error) {
        reject(error);
      }
    });
  }

  // ==========================================
  // ORDER BOOK MANAGEMENT
  // ==========================================

  private getOrCreateOrderBook(symbol: string): ManagedOrderBook {
    if (!this.orderBooks.has(symbol)) {
      const managedBook = new ManagedOrderBook(symbol, this.config);
      this.orderBooks.set(symbol, managedBook);

      this.emit('order_book_created', { symbol });
    }

    return this.orderBooks.get(symbol)!;
  }

  private applyUpdate(managedBook: ManagedOrderBook, update: OrderBookUpdate): void {
    // Apply changes to the order book
    update.changes.forEach(change => {
      managedBook.applyChange(change);
    });

    // Update sequence
    managedBook.updateSequence(update.sequence, update.timestamp);

    // Validate integrity
    if (this.config.enableValidation) {
      this.validateIntegrity(managedBook);
    }
  }

  private applyTradeImpact(managedBook: ManagedOrderBook, trade: Trade): void {
    // Simulate trade impact on order book
    const side = trade.side === 'buy' ? 'ask' : 'bid';
    const levels = side === 'ask' ? managedBook.asks : managedBook.bids;

    // Find and update affected levels
    let remainingSize = trade.size;
    const changes: OrderBookChange[] = [];

    for (let i = 0; i < levels.length && remainingSize > 0; i++) {
      const level = levels[i];

      if ((side === 'ask' && trade.price >= level.price) ||
          (side === 'bid' && trade.price <= level.price)) {

        const consumedSize = Math.min(remainingSize, level.size);
        const newSize = level.size - consumedSize;
        remainingSize -= consumedSize;

        changes.push({
          side,
          price: level.price,
          size: newSize,
          action: newSize > 0 ? 'update' : 'delete',
          timestamp: trade.timestamp
        });
      }
    }

    // Apply changes
    changes.forEach(change => {
      managedBook.applyChange(change);
    });

    // Create trade impact update
    const tradeUpdate: OrderBookUpdate = {
      type: 'trade_impact',
      symbol: trade.symbol,
      timestamp: trade.timestamp,
      sequence: managedBook.sequence + 1,
      changes,
      metadata: { trade }
    };

    this.updates$.next(tradeUpdate);
  }

  private generateSnapshot(managedBook: ManagedOrderBook): OrderBookSnapshot {
    const bids = [...managedBook.bids].sort((a, b) => b.price - a.price);
    const asks = [...managedBook.asks].sort((a, b) => a.price - b.price);

    const bestBid = bids[0]?.price || 0;
    const bestAsk = asks[0]?.price || Number.MAX_VALUE;
    const spread = bestAsk - bestBid;
    const midPrice = (bestBid + bestAsk) / 2;

    const totalBidVolume = bids.reduce((sum, level) => sum + level.size, 0);
    const totalAskVolume = asks.reduce((sum, level) => sum + level.size, 0);
    const imbalance = (totalBidVolume - totalAskVolume) / (totalBidVolume + totalAskVolume);

    // Calculate quality
    const quality = this.assessOrderBookQuality(managedBook, bids, asks);

    return {
      symbol: managedBook.symbol,
      timestamp: managedBook.lastUpdateTime,
      sequence: managedBook.sequence,
      bids: bids.slice(0, this.config.maxLevels),
      asks: asks.slice(0, this.config.maxLevels),
      spread,
      midPrice,
      totalBidVolume,
      totalAskVolume,
      imbalance,
      checksum: managedBook.checksum,
      quality
    };
  }

  // ==========================================
  // DATA NORMALIZATION
  // ==========================================

  private normalizeUpdate(symbol: string, rawUpdate: any): OrderBookUpdate {
    const changes: OrderBookChange[] = [];

    // Handle different update formats
    if (rawUpdate.bids || rawUpdate.b) {
      const bids = rawUpdate.bids || rawUpdate.b;
      bids.forEach((level: any) => {
        changes.push(this.normalizeLevelChange('bid', level, rawUpdate.timestamp));
      });
    }

    if (rawUpdate.asks || rawUpdate.a) {
      const asks = rawUpdate.asks || rawUpdate.a;
      asks.forEach((level: any) => {
        changes.push(this.normalizeLevelChange('ask', level, rawUpdate.timestamp));
      });
    }

    return {
      type: rawUpdate.type || 'delta',
      symbol,
      timestamp: rawUpdate.timestamp || rawUpdate.T || Date.now(),
      sequence: rawUpdate.sequence || rawUpdate.u || rawUpdate.lastUpdateId || 0,
      changes,
      metadata: rawUpdate.metadata
    };
  }

  private normalizeSnapshot(symbol: string, rawSnapshot: any): OrderBookSnapshot {
    const bids = this.normalizeLevels(rawSnapshot.bids || rawSnapshot.b || []);
    const asks = this.normalizeLevels(rawSnapshot.asks || rawSnapshot.a || []);

    const bestBid = Math.max(...bids.map(l => l.price));
    const bestAsk = Math.min(...asks.map(l => l.price));
    const spread = bestAsk - bestBid;
    const midPrice = (bestBid + bestAsk) / 2;

    const totalBidVolume = bids.reduce((sum, level) => sum + level.size, 0);
    const totalAskVolume = asks.reduce((sum, level) => sum + level.size, 0);
    const imbalance = (totalBidVolume - totalAskVolume) / (totalBidVolume + totalAskVolume);

    return {
      symbol,
      timestamp: rawSnapshot.timestamp || rawSnapshot.T || Date.now(),
      sequence: rawSnapshot.sequence || rawSnapshot.lastUpdateId || 0,
      bids: bids.sort((a, b) => b.price - a.price),
      asks: asks.sort((a, b) => a.price - b.price),
      spread,
      midPrice,
      totalBidVolume,
      totalAskVolume,
      imbalance,
      checksum: rawSnapshot.checksum,
      quality: 'good'
    };
  }

  private normalizeLevelChange(side: 'bid' | 'ask', rawLevel: any, timestamp: number): OrderBookChange {
    const price = parseFloat(Array.isArray(rawLevel) ? rawLevel[0] : rawLevel.price || rawLevel.p);
    const size = parseFloat(Array.isArray(rawLevel) ? rawLevel[1] : rawLevel.size || rawLevel.s);

    let action: 'add' | 'update' | 'delete';
    if (size === 0) {
      action = 'delete';
    } else if (rawLevel.action) {
      action = rawLevel.action;
    } else {
      action = 'update'; // Default assumption
    }

    return {
      side,
      price,
      size,
      action,
      timestamp: timestamp || Date.now()
    };
  }

  private normalizeLevels(rawLevels: any[]): OrderBookLevel[] {
    return rawLevels.map(level => ({
      price: parseFloat(Array.isArray(level) ? level[0] : level.price || level.p),
      size: parseFloat(Array.isArray(level) ? level[1] : level.size || level.s),
      count: level.count || level.c
    }));
  }

  // ==========================================
  // VALIDATION & GAP DETECTION
  // ==========================================

  private validateUpdate(managedBook: ManagedOrderBook, update: OrderBookUpdate): OrderBookValidation {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Sequence validation
    if (update.sequence <= managedBook.sequence) {
      if (update.sequence < managedBook.sequence) {
        errors.push(`Out of order sequence: expected > ${managedBook.sequence}, got ${update.sequence}`);
      } else {
        warnings.push(`Duplicate sequence: ${update.sequence}`);
      }
    }

    // Price validation
    update.changes.forEach(change => {
      if (change.price <= 0) {
        errors.push(`Invalid price: ${change.price}`);
      }

      if (change.size < 0) {
        errors.push(`Invalid size: ${change.size}`);
      }
    });

    // Cross validation (bids should not exceed asks)
    this.validateCrossConsistency(managedBook, update, errors, warnings);

    const qualityScore = Math.max(0, 100 - (errors.length * 25) - (warnings.length * 5));

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      qualityScore,
      checksumValid: this.validateChecksum(managedBook, update),
      priceConsistency: this.validatePriceConsistency(update),
      volumeConsistency: this.validateVolumeConsistency(update),
      sequenceConsistency: update.sequence > managedBook.sequence
    };
  }

  private validateCrossConsistency(
    managedBook: ManagedOrderBook,
    update: OrderBookUpdate,
    errors: string[],
    warnings: string[]
  ): void {
    // Create temporary copy to test update
    const tempBids = [...managedBook.bids];
    const tempAsks = [...managedBook.asks];

    // Apply changes to temp structures
    update.changes.forEach(change => {
      const levels = change.side === 'bid' ? tempBids : tempAsks;
      const index = levels.findIndex(l => l.price === change.price);

      if (change.action === 'delete') {
        if (index >= 0) {
          levels.splice(index, 1);
        }
      } else if (change.action === 'update' || change.action === 'add') {
        if (index >= 0) {
          levels[index] = { ...levels[index], size: change.size };
        } else {
          levels.push({ price: change.price, size: change.size });
        }
      }
    });

    // Check for crossing
    const bestBid = Math.max(...tempBids.map(l => l.price), 0);
    const bestAsk = Math.min(...tempAsks.map(l => l.price), Number.MAX_VALUE);

    if (bestBid >= bestAsk && bestBid > 0 && bestAsk < Number.MAX_VALUE) {
      errors.push(`Crossed book detected: best bid (${bestBid}) >= best ask (${bestAsk})`);
    }
  }

  private validateChecksum(managedBook: ManagedOrderBook, update: OrderBookUpdate): boolean {
    if (!this.config.checksumValidation || !update.metadata?.checksum) {
      return true;
    }

    // Calculate expected checksum
    const expectedChecksum = this.calculateChecksum(managedBook);
    return expectedChecksum === update.metadata.checksum;
  }

  private validatePriceConsistency(update: OrderBookUpdate): boolean {
    // Check for reasonable price movements
    const tolerance = this.config.priceToleranceBps / 10000;

    return update.changes.every(change => {
      // Add price consistency checks
      return change.price > 0 && change.price < Number.MAX_VALUE;
    });
  }

  private validateVolumeConsistency(update: OrderBookUpdate): boolean {
    // Check for reasonable volume changes
    const tolerance = this.config.volumeToleranceBps / 10000;

    return update.changes.every(change => {
      return change.size >= 0; // Volume can be zero (delete)
    });
  }

  private detectGaps(managedBook: ManagedOrderBook, update: OrderBookUpdate): void {
    // Sequence gap detection
    const expectedSequence = managedBook.sequence + 1;
    if (update.sequence > expectedSequence) {
      const gap: OrderBookGap = {
        symbol: managedBook.symbol,
        detectedAt: Date.now(),
        gapType: 'sequence',
        severity: this.getGapSeverity(update.sequence - expectedSequence),
        expectedSequence,
        actualSequence: update.sequence,
        gapSize: update.sequence - expectedSequence,
        description: `Sequence gap: expected ${expectedSequence}, got ${update.sequence}`
      };

      this.gaps$.next(gap);
      this.emit('gap_detected', gap);

      // Auto-reconstruct if configured
      if (this.config.autoReconstruct && gap.severity === 'critical') {
        this.scheduleReconstruction(managedBook);
      }
    }

    // Time gap detection
    const timeDiff = update.timestamp - managedBook.lastUpdateTime;
    if (timeDiff > 60000) { // More than 1 minute
      const gap: OrderBookGap = {
        symbol: managedBook.symbol,
        detectedAt: Date.now(),
        gapType: 'time',
        severity: timeDiff > 300000 ? 'critical' : 'medium',
        gapSize: timeDiff,
        description: `Time gap: ${timeDiff}ms since last update`
      };

      this.gaps$.next(gap);
      this.emit('gap_detected', gap);
    }
  }

  private getGapSeverity(gapSize: number): 'low' | 'medium' | 'high' | 'critical' {
    if (gapSize <= 5) return 'low';
    if (gapSize <= 20) return 'medium';
    if (gapSize <= 100) return 'high';
    return 'critical';
  }

  // ==========================================
  // ANALYTICS & QUALITY ASSESSMENT
  // ==========================================

  private generateAnalytics(): void {
    this.orderBooks.forEach((managedBook, symbol) => {
      const analytics = this.calculateAnalytics(managedBook);
      this.analytics.set(symbol, analytics);
      this.analytics$.next(analytics);
    });
  }

  private calculateAnalytics(managedBook: ManagedOrderBook): OrderBookAnalytics {
    const snapshot = this.generateSnapshot(managedBook);

    return {
      symbol: managedBook.symbol,
      timestamp: Date.now(),
      depth: this.calculateDepthMetrics(snapshot),
      spread: this.calculateSpreadMetrics(snapshot),
      imbalance: this.calculateImbalanceMetrics(snapshot),
      liquidity: this.calculateLiquidityMetrics(snapshot),
      volatility: this.calculateVolatilityMetrics(managedBook),
      microstructure: this.calculateMicrostructureMetrics(managedBook)
    };
  }

  private calculateDepthMetrics(snapshot: OrderBookSnapshot) {
    return {
      levels: Math.max(snapshot.bids.length, snapshot.asks.length),
      bidDepth: snapshot.totalBidVolume,
      askDepth: snapshot.totalAskVolume,
      totalDepth: snapshot.totalBidVolume + snapshot.totalAskVolume
    };
  }

  private calculateSpreadMetrics(snapshot: OrderBookSnapshot) {
    const weightedBid = this.calculateWeightedPrice(snapshot.bids);
    const weightedAsk = this.calculateWeightedPrice(snapshot.asks);
    const weightedSpread = weightedAsk - weightedBid;

    return {
      absolute: snapshot.spread,
      relative: snapshot.midPrice > 0 ? (snapshot.spread / snapshot.midPrice) * 100 : 0,
      weightedAverage: weightedSpread
    };
  }

  private calculateImbalanceMetrics(snapshot: OrderBookSnapshot) {
    const ratio = snapshot.imbalance;
    const absRatio = Math.abs(ratio);

    let pressure: 'buy' | 'sell' | 'neutral';
    if (ratio > 0.1) pressure = 'buy';
    else if (ratio < -0.1) pressure = 'sell';
    else pressure = 'neutral';

    return {
      ratio,
      pressure,
      strength: absRatio
    };
  }

  private calculateLiquidityMetrics(snapshot: OrderBookSnapshot) {
    // Kyle's lambda (price impact)
    const tightness = snapshot.midPrice > 0 ? snapshot.spread / snapshot.midPrice : 0;

    // Market depth at best prices
    const depth = snapshot.bids.length > 0 && snapshot.asks.length > 0 ?
      Math.min(snapshot.bids[0].size, snapshot.asks[0].size) : 0;

    return {
      tightness,
      depth,
      resilience: this.calculateResilience(snapshot),
      immediacy: this.calculateImmediacy(snapshot)
    };
  }

  private calculateVolatilityMetrics(managedBook: ManagedOrderBook) {
    const priceHistory = managedBook.priceHistory.slice(-20);

    if (priceHistory.length < 2) {
      return {
        midPriceVol: 0,
        spreadVol: 0,
        orderFlowVol: 0
      };
    }

    const midPriceVol = this.calculateVolatility(priceHistory.map(h => h.midPrice));
    const spreadVol = this.calculateVolatility(priceHistory.map(h => h.spread));

    return {
      midPriceVol,
      spreadVol,
      orderFlowVol: this.calculateOrderFlowVolatility(managedBook)
    };
  }

  private calculateMicrostructureMetrics(managedBook: ManagedOrderBook) {
    const recentTrades = this.recentTrades.get(managedBook.symbol) || [];

    return {
      tickSize: this.estimateTickSize(managedBook),
      effectiveSpread: this.calculateEffectiveSpread(recentTrades),
      impactCost: this.calculateImpactCost(recentTrades),
      resilience: this.calculateBookResilience(managedBook)
    };
  }

  private assessOrderBookQuality(
    managedBook: ManagedOrderBook,
    bids: OrderBookLevel[],
    asks: OrderBookLevel[]
  ): OrderBookQuality {
    let score = 100;

    // Check basic structure
    if (bids.length === 0 || asks.length === 0) score -= 50;

    // Check for crossing
    const bestBid = bids[0]?.price || 0;
    const bestAsk = asks[0]?.price || Number.MAX_VALUE;
    if (bestBid >= bestAsk) score -= 40;

    // Check spread reasonableness
    if (bestAsk < Number.MAX_VALUE && bestBid > 0) {
      const spread = bestAsk - bestBid;
      const relativeSpread = spread / ((bestBid + bestAsk) / 2);
      if (relativeSpread > 0.01) score -= 20; // 1% spread
    }

    // Check sequence consistency
    if (managedBook.hasSequenceGaps) score -= 15;

    // Check age
    const age = Date.now() - managedBook.lastUpdateTime;
    if (age > 10000) score -= 10; // 10 seconds

    if (score >= 90) return 'excellent';
    if (score >= 70) return 'good';
    if (score >= 50) return 'fair';
    if (score >= 30) return 'poor';
    return 'invalid';
  }

  // ==========================================
  // HELPER CALCULATIONS
  // ==========================================

  private calculateWeightedPrice(levels: readonly OrderBookLevel[]): number {
    if (levels.length === 0) return 0;

    let weightedSum = 0;
    let totalWeight = 0;

    levels.slice(0, 5).forEach(level => { // Top 5 levels
      weightedSum += level.price * level.size;
      totalWeight += level.size;
    });

    return totalWeight > 0 ? weightedSum / totalWeight : levels[0].price;
  }

  private calculateResilience(snapshot: OrderBookSnapshot): number {
    // Measure how quickly the book recovers after trades
    const depthNearMid = this.calculateDepthNearMidpoint(snapshot, 0.001); // 0.1%
    return depthNearMid / Math.max(snapshot.totalBidVolume + snapshot.totalAskVolume, 1);
  }

  private calculateImmediacy(snapshot: OrderBookSnapshot): number {
    // Measure how much volume can be traded immediately
    const immediateVolume = snapshot.bids.slice(0, 3).reduce((sum, level) => sum + level.size, 0) +
                           snapshot.asks.slice(0, 3).reduce((sum, level) => sum + level.size, 0);
    return immediateVolume;
  }

  private calculateDepthNearMidpoint(snapshot: OrderBookSnapshot, percentage: number): number {
    const range = snapshot.midPrice * percentage;
    const lowerBound = snapshot.midPrice - range;
    const upperBound = snapshot.midPrice + range;

    const bidDepth = snapshot.bids
      .filter(level => level.price >= lowerBound)
      .reduce((sum, level) => sum + level.size, 0);

    const askDepth = snapshot.asks
      .filter(level => level.price <= upperBound)
      .reduce((sum, level) => sum + level.size, 0);

    return bidDepth + askDepth;
  }

  private calculateVolatility(values: number[]): number {
    if (values.length < 2) return 0;

    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  private calculateOrderFlowVolatility(managedBook: ManagedOrderBook): number {
    // Simplified order flow volatility
    const changes = managedBook.recentChanges.slice(-10);
    if (changes.length < 2) return 0;

    const volumes = changes.map(change => change.size);
    return this.calculateVolatility(volumes);
  }

  private estimateTickSize(managedBook: ManagedOrderBook): number {
    // Estimate tick size from recent price levels
    const allPrices = [...managedBook.bids, ...managedBook.asks].map(l => l.price);
    if (allPrices.length < 2) return 0.01;

    const differences = [];
    allPrices.sort((a, b) => a - b);

    for (let i = 1; i < allPrices.length; i++) {
      const diff = allPrices[i] - allPrices[i - 1];
      if (diff > 0) differences.push(diff);
    }

    if (differences.length === 0) return 0.01;

    // Find the most common small difference
    differences.sort((a, b) => a - b);
    return differences[0] || 0.01;
  }

  private calculateEffectiveSpread(trades: Trade[]): number {
    if (trades.length < 2) return 0;

    const spreads = [];
    for (let i = 1; i < trades.length; i++) {
      if (trades[i].side !== trades[i - 1].side) {
        spreads.push(Math.abs(trades[i].price - trades[i - 1].price));
      }
    }

    return spreads.length > 0 ? spreads.reduce((sum, s) => sum + s, 0) / spreads.length : 0;
  }

  private calculateImpactCost(trades: Trade[]): number {
    // Simplified impact cost calculation
    if (trades.length === 0) return 0;

    const volumes = trades.map(t => t.size);
    const avgVolume = volumes.reduce((sum, v) => sum + v, 0) / volumes.length;

    const largeTrades = trades.filter(t => t.size > avgVolume * 2);
    return largeTrades.length / trades.length;
  }

  private calculateBookResilience(managedBook: ManagedOrderBook): number {
    // Measure how quickly levels are replenished
    const changes = managedBook.recentChanges.slice(-20);
    const additions = changes.filter(c => c.action === 'add').length;
    const deletions = changes.filter(c => c.action === 'delete').length;

    return deletions > 0 ? additions / deletions : 1;
  }

  private calculateChecksum(managedBook: ManagedOrderBook): string {
    // Simple checksum calculation
    const data = managedBook.bids.concat(managedBook.asks)
      .map(level => `${level.price}:${level.size}`)
      .join('|');

    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    return hash.toString();
  }

  // ==========================================
  // RECONSTRUCTION & ERROR HANDLING
  // ==========================================

  private scheduleReconstruction(managedBook: ManagedOrderBook): void {
    setTimeout(() => {
      this.reconstructOrderBook(managedBook);
    }, this.config.reconstructionTimeout);

    this.emit('reconstruction_scheduled', { symbol: managedBook.symbol });
  }

  private reconstructOrderBook(managedBook: ManagedOrderBook): void {
    // Mark for reconstruction
    managedBook.needsReconstruction = true;

    this.emit('reconstruction_requested', {
      symbol: managedBook.symbol,
      reason: 'gap_detected'
    });
  }

  private validateIntegrity(managedBook: ManagedOrderBook): void {
    // Check for basic integrity issues
    const errors: string[] = [];

    // Check sorting
    if (!this.isSorted(managedBook.bids, 'desc')) {
      errors.push('Bids not properly sorted');
    }

    if (!this.isSorted(managedBook.asks, 'asc')) {
      errors.push('Asks not properly sorted');
    }

    // Check for duplicates
    if (this.hasDuplicatePrices(managedBook.bids) || this.hasDuplicatePrices(managedBook.asks)) {
      errors.push('Duplicate price levels detected');
    }

    if (errors.length > 0) {
      this.emit('integrity_violation', {
        symbol: managedBook.symbol,
        errors
      });
    }
  }

  private isSorted(levels: OrderBookLevel[], order: 'asc' | 'desc'): boolean {
    for (let i = 1; i < levels.length; i++) {
      const current = levels[i].price;
      const previous = levels[i - 1].price;

      if (order === 'desc' && current > previous) return false;
      if (order === 'asc' && current < previous) return false;
    }
    return true;
  }

  private hasDuplicatePrices(levels: OrderBookLevel[]): boolean {
    const prices = new Set(levels.map(l => l.price));
    return prices.size !== levels.length;
  }

  private updateRecentTrades(trade: Trade): void {
    const symbol = trade.symbol;

    if (!this.recentTrades.has(symbol)) {
      this.recentTrades.set(symbol, []);
    }

    const trades = this.recentTrades.get(symbol)!;
    trades.push(trade);

    // Keep only last 100 trades
    if (trades.length > 100) {
      trades.splice(0, trades.length - 100);
    }
  }

  private handleValidationError(
    symbol: string,
    validation: OrderBookValidation,
    update: OrderBookUpdate
  ): void {
    const error: StreamError = {
      id: this.generateId(),
      type: 'data_corruption',
      message: `Order book validation failed: ${validation.errors.join(', ')}`,
      timestamp: Date.now(),
      source: 'orderbook',
      symbol,
      retryable: false,
      context: { validation, update }
    };

    this.errors$.next(error);
    this.emit('validation_error', { symbol, validation, update });
  }

  private handleError(symbol: string, error: any, operation: string): void {
    const streamError: StreamError = {
      id: this.generateId(),
      type: 'protocol_error',
      message: error.message || 'Order book processing error',
      timestamp: Date.now(),
      source: 'orderbook',
      symbol,
      retryable: true,
      context: { operation, error }
    };

    this.errors$.next(streamError);
    this.emit('error', streamError);
  }

  // ==========================================
  // INITIALIZATION & CLEANUP
  // ==========================================

  private initialize(): void {
    // Setup analytics generation
    if (this.config.enableAnalytics) {
      this.analyticsTimer = setInterval(() => {
        this.generateAnalytics();
      }, 5000);
    }
  }

  private generateId(): string {
    return `ob-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }

  public destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear timers
    if (this.analyticsTimer) {
      clearInterval(this.analyticsTimer);
    }

    // Clear data
    this.orderBooks.clear();
    this.analytics.clear();
    this.recentTrades.clear();

    // Complete observables
    this.snapshots$.complete();
    this.updates$.complete();
    this.analytics$.complete();
    this.gaps$.complete();
    this.errors$.complete();

    this.removeAllListeners();
  }
}

// ==========================================
// MANAGED ORDER BOOK CLASS
// ==========================================

class ManagedOrderBook {
  public readonly symbol: string;
  public readonly config: OrderBookConfig;

  public bids: OrderBookLevel[] = [];
  public asks: OrderBookLevel[] = [];
  public sequence = 0;
  public lastUpdateTime = 0;
  public checksum?: string;
  public needsReconstruction = false;
  public hasSequenceGaps = false;

  // History tracking
  public readonly priceHistory: Array<{ timestamp: number; midPrice: number; spread: number }> = [];
  public readonly recentChanges: OrderBookChange[] = [];

  constructor(symbol: string, config: OrderBookConfig) {
    this.symbol = symbol;
    this.config = config;
  }

  public reset(snapshot: OrderBookSnapshot): void {
    this.bids = [...snapshot.bids];
    this.asks = [...snapshot.asks];
    this.sequence = snapshot.sequence;
    this.lastUpdateTime = snapshot.timestamp;
    this.checksum = snapshot.checksum;
    this.needsReconstruction = false;
    this.hasSequenceGaps = false;

    this.updatePriceHistory(snapshot.midPrice, snapshot.spread);
  }

  public applyChange(change: OrderBookChange): void {
    const levels = change.side === 'bid' ? this.bids : this.asks;
    const index = levels.findIndex(level => level.price === change.price);

    switch (change.action) {
      case 'add':
        if (index === -1) {
          levels.push({ price: change.price, size: change.size });
          this.sortLevels(change.side);
        }
        break;

      case 'update':
        if (index >= 0) {
          levels[index] = { ...levels[index], size: change.size };
        } else {
          levels.push({ price: change.price, size: change.size });
          this.sortLevels(change.side);
        }
        break;

      case 'delete':
        if (index >= 0) {
          levels.splice(index, 1);
        }
        break;
    }

    // Limit levels
    if (levels.length > this.config.maxLevels) {
      if (change.side === 'bid') {
        this.bids = this.bids.slice(0, this.config.maxLevels);
      } else {
        this.asks = this.asks.slice(0, this.config.maxLevels);
      }
    }

    // Track change
    this.recentChanges.push(change);
    if (this.recentChanges.length > 100) {
      this.recentChanges.splice(0, this.recentChanges.length - 100);
    }
  }

  public updateSequence(sequence: number, timestamp: number): void {
    this.sequence = sequence;
    this.lastUpdateTime = timestamp;

    // Update price history
    const bestBid = this.bids[0]?.price || 0;
    const bestAsk = this.asks[0]?.price || Number.MAX_VALUE;
    if (bestBid > 0 && bestAsk < Number.MAX_VALUE) {
      const midPrice = (bestBid + bestAsk) / 2;
      const spread = bestAsk - bestBid;
      this.updatePriceHistory(midPrice, spread);
    }
  }

  private sortLevels(side: 'bid' | 'ask'): void {
    if (side === 'bid') {
      this.bids.sort((a, b) => b.price - a.price); // Descending
    } else {
      this.asks.sort((a, b) => a.price - b.price); // Ascending
    }
  }

  private updatePriceHistory(midPrice: number, spread: number): void {
    this.priceHistory.push({
      timestamp: this.lastUpdateTime,
      midPrice,
      spread
    });

    // Keep only last 100 entries
    if (this.priceHistory.length > 100) {
      this.priceHistory.splice(0, this.priceHistory.length - 100);
    }
  }
}