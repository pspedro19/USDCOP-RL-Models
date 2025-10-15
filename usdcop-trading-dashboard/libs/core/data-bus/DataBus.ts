/**
 * DataBus - Unified Data Flow Management for Elite Trading Platform
 * High-performance, type-safe data distribution system
 */

import { Observable, Subject, BehaviorSubject, combineLatest, merge } from 'rxjs';
import { filter, map, debounceTime, distinctUntilChanged, shareReplay, catchError } from 'rxjs/operators';
import { EventEmitter } from 'eventemitter3';
import type {
  MarketTick,
  OrderBook,
  Trade,
  TradingPlatformEvent,
  BaseEvent,
  DataRequest,
  DataResponse,
  CacheEntry
} from '../types';

export interface DataBusConfig {
  readonly maxCacheSize: number;
  readonly defaultTTL: number;
  readonly enablePersistence: boolean;
  readonly compressionThreshold: number;
  readonly batchSize: number;
  readonly flushInterval: number;
}

export interface DataChannel<T = any> {
  readonly name: string;
  readonly type: string;
  readonly subject: Subject<T>;
  readonly cache: Map<string, CacheEntry<T>>;
  readonly subscribers: Set<string>;
  readonly lastUpdate: number;
  readonly messageCount: number;
}

export interface DataSubscription {
  readonly id: string;
  readonly channel: string;
  readonly filter?: (data: any) => boolean;
  readonly transform?: (data: any) => any;
  readonly active: boolean;
  readonly createdAt: number;
}

export class DataBus extends EventEmitter {
  private readonly config: DataBusConfig;
  private readonly channels = new Map<string, DataChannel>();
  private readonly subscriptions = new Map<string, DataSubscription>();
  private readonly requestQueue = new Subject<DataRequest>();
  private readonly responseQueue = new Subject<DataResponse>();
  private readonly errorQueue = new Subject<Error>();

  // Performance tracking
  private readonly metrics = {
    totalMessages: 0,
    messagesPerSecond: 0,
    averageLatency: 0,
    cacheHitRate: 0,
    errorCount: 0,
    lastFlush: Date.now(),
    channelStats: new Map<string, { messageCount: number; lastUpdate: number }>()
  };

  private flushTimer?: NodeJS.Timeout;
  private metricsTimer?: NodeJS.Timeout;

  constructor(config: Partial<DataBusConfig> = {}) {
    super();

    this.config = {
      maxCacheSize: 10000,
      defaultTTL: 300000, // 5 minutes
      enablePersistence: false,
      compressionThreshold: 1024,
      batchSize: 100,
      flushInterval: 1000,
      ...config
    };

    this.initialize();
  }

  private initialize(): void {
    // Setup core channels
    this.createChannel('market.ticks', 'MarketTick');
    this.createChannel('market.orderbooks', 'OrderBook');
    this.createChannel('market.trades', 'Trade');
    this.createChannel('trading.orders', 'Order');
    this.createChannel('trading.positions', 'Position');
    this.createChannel('system.events', 'Event');

    // Setup periodic tasks
    this.startPeriodicTasks();

    // Handle errors globally
    this.errorQueue.subscribe(this.handleError.bind(this));
  }

  /**
   * Create a new data channel
   */
  public createChannel<T>(name: string, type: string): DataChannel<T> {
    if (this.channels.has(name)) {
      throw new Error(`Channel ${name} already exists`);
    }

    const channel: DataChannel<T> = {
      name,
      type,
      subject: new Subject<T>(),
      cache: new Map(),
      subscribers: new Set(),
      lastUpdate: 0,
      messageCount: 0
    };

    this.channels.set(name, channel);
    this.metrics.channelStats.set(name, { messageCount: 0, lastUpdate: 0 });

    this.emit('channel.created', { name, type });
    return channel;
  }

  /**
   * Publish data to a channel
   */
  public publish<T>(channelName: string, data: T, options?: {
    cache?: boolean;
    cacheKey?: string;
    ttl?: number;
    compress?: boolean;
  }): void {
    const channel = this.channels.get(channelName);
    if (!channel) {
      throw new Error(`Channel ${channelName} does not exist`);
    }

    const timestamp = Date.now();
    const message = this.wrapMessage(data, timestamp, channelName);

    try {
      // Update metrics
      this.updateMetrics(channelName, timestamp);

      // Cache if requested
      if (options?.cache) {
        this.cacheMessage(channel, message, options);
      }

      // Publish to all subscribers
      channel.subject.next(message);

      // Update channel state
      (channel as any).lastUpdate = timestamp;
      (channel as any).messageCount++;

      this.emit('data.published', { channel: channelName, timestamp, dataSize: this.getDataSize(data) });

    } catch (error) {
      this.errorQueue.next(error as Error);
    }
  }

  /**
   * Subscribe to a channel
   */
  public subscribe<T>(
    channelName: string,
    callback: (data: T) => void,
    options?: {
      filter?: (data: T) => boolean;
      transform?: (data: T) => any;
      includeCache?: boolean;
    }
  ): string {
    const channel = this.channels.get(channelName);
    if (!channel) {
      throw new Error(`Channel ${channelName} does not exist`);
    }

    const subscriptionId = this.generateId();
    const subscription: DataSubscription = {
      id: subscriptionId,
      channel: channelName,
      filter: options?.filter,
      transform: options?.transform,
      active: true,
      createdAt: Date.now()
    };

    this.subscriptions.set(subscriptionId, subscription);
    channel.subscribers.add(subscriptionId);

    // Create observable with optional filtering and transformation
    let observable = channel.subject.asObservable();

    if (options?.filter) {
      observable = observable.pipe(filter(options.filter));
    }

    if (options?.transform) {
      observable = observable.pipe(map(options.transform));
    }

    // Subscribe and handle errors
    const rxSubscription = observable.pipe(
      catchError((error) => {
        this.errorQueue.next(error);
        return [];
      })
    ).subscribe(callback);

    // Send cached data if requested
    if (options?.includeCache) {
      this.sendCachedData(channel, callback, options);
    }

    this.emit('subscription.created', { id: subscriptionId, channel: channelName });

    // Return unsubscribe function
    return subscriptionId;
  }

  /**
   * Unsubscribe from a channel
   */
  public unsubscribe(subscriptionId: string): boolean {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      return false;
    }

    const channel = this.channels.get(subscription.channel);
    if (channel) {
      channel.subscribers.delete(subscriptionId);
    }

    this.subscriptions.delete(subscriptionId);
    this.emit('subscription.removed', { id: subscriptionId });

    return true;
  }

  /**
   * Get cached data from a channel
   */
  public getCachedData<T>(channelName: string, key?: string): T[] | T | null {
    const channel = this.channels.get(channelName);
    if (!channel) {
      return null;
    }

    if (key) {
      const entry = channel.cache.get(key);
      return entry && !this.isExpired(entry) ? entry.value : null;
    }

    // Return all non-expired cache entries
    const validEntries: T[] = [];
    Array.from(channel.cache.values()).forEach(entry => {
      if (!this.isExpired(entry)) {
        validEntries.push(entry.value);
      }
    });

    return validEntries;
  }

  /**
   * Create a combined observable from multiple channels
   */
  public combineChannels<T1, T2, R>(
    channel1: string,
    channel2: string,
    combiner: (data1: T1, data2: T2) => R
  ): Observable<R> {
    const ch1 = this.channels.get(channel1);
    const ch2 = this.channels.get(channel2);

    if (!ch1 || !ch2) {
      throw new Error('One or more channels do not exist');
    }

    return combineLatest([
      ch1.subject.asObservable(),
      ch2.subject.asObservable()
    ]).pipe(
      map(([data1, data2]) => combiner(data1, data2)),
      shareReplay(1)
    );
  }

  /**
   * Get real-time metrics
   */
  public getMetrics() {
    return {
      ...this.metrics,
      channels: Array.from(this.channels.keys()),
      subscriptionCount: this.subscriptions.size,
      uptime: Date.now() - this.metrics.lastFlush
    };
  }

  /**
   * Clear channel cache
   */
  public clearCache(channelName?: string): void {
    if (channelName) {
      const channel = this.channels.get(channelName);
      if (channel) {
        channel.cache.clear();
        this.emit('cache.cleared', { channel: channelName });
      }
    } else {
      // Clear all caches
      Array.from(this.channels.values()).forEach(channel => {
        channel.cache.clear();
      });
      this.emit('cache.cleared', { channel: 'all' });
    }
  }

  /**
   * Destroy the DataBus and cleanup resources
   */
  public destroy(): void {
    // Clear timers
    if (this.flushTimer) clearInterval(this.flushTimer);
    if (this.metricsTimer) clearInterval(this.metricsTimer);

    // Complete all subjects
    Array.from(this.channels.values()).forEach(channel => {
      channel.subject.complete();
    });

    this.requestQueue.complete();
    this.responseQueue.complete();
    this.errorQueue.complete();

    // Clear all maps
    this.channels.clear();
    this.subscriptions.clear();

    this.emit('databus.destroyed');
    this.removeAllListeners();
  }

  // Private helper methods
  private wrapMessage<T>(data: T, timestamp: number, channel: string): T {
    // For now, return data as-is. Could add metadata wrapper in the future
    return data;
  }

  private cacheMessage<T>(
    channel: DataChannel<T>,
    data: T,
    options: { cacheKey?: string; ttl?: number; compress?: boolean }
  ): void {
    const key = options.cacheKey || this.generateCacheKey();
    const ttl = options.ttl || this.config.defaultTTL;

    // Check cache size limit
    if (channel.cache.size >= this.config.maxCacheSize) {
      this.evictOldestCache(channel);
    }

    const entry: CacheEntry<T> = {
      key,
      value: data,
      timestamp: Date.now(),
      ttl,
      hits: 0,
      size: this.getDataSize(data)
    };

    channel.cache.set(key, entry);
  }

  private sendCachedData<T>(
    channel: DataChannel<T>,
    callback: (data: T) => void,
    options: { filter?: (data: T) => boolean; transform?: (data: T) => any }
  ): void {
    Array.from(channel.cache.values()).forEach(entry => {
      if (!this.isExpired(entry)) {
        let data = entry.value;

        if (options.filter && !options.filter(data)) return;
        if (options.transform) data = options.transform(data);

        callback(data);
        (entry as any).hits++;
      }
    });
  }

  private updateMetrics(channelName: string, timestamp: number): void {
    this.metrics.totalMessages++;

    const channelStats = this.metrics.channelStats.get(channelName);
    if (channelStats) {
      channelStats.messageCount++;
      channelStats.lastUpdate = timestamp;
    }

    // Calculate messages per second
    const timeDiff = timestamp - this.metrics.lastFlush;
    if (timeDiff >= 1000) {
      this.metrics.messagesPerSecond = Math.round(this.metrics.totalMessages / (timeDiff / 1000));
    }
  }

  private evictOldestCache<T>(channel: DataChannel<T>): void {
    let oldestKey: string | null = null;
    let oldestTime = Date.now();

    Array.from(channel.cache.entries()).forEach(([key, entry]) => {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestKey = key;
      }
    });

    if (oldestKey) {
      channel.cache.delete(oldestKey);
    }
  }

  private isExpired(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp > entry.ttl;
  }

  private getDataSize(data: any): number {
    try {
      return JSON.stringify(data).length;
    } catch {
      return 0;
    }
  }

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }

  private generateCacheKey(): string {
    return `cache-${this.generateId()}`;
  }

  private startPeriodicTasks(): void {
    // Cache cleanup
    this.flushTimer = setInterval(() => {
      this.cleanupExpiredCache();
    }, this.config.flushInterval);

    // Metrics update
    this.metricsTimer = setInterval(() => {
      this.updatePerformanceMetrics();
    }, 5000);
  }

  private cleanupExpiredCache(): void {
    Array.from(this.channels.values()).forEach(channel => {
      Array.from(channel.cache.entries()).forEach(([key, entry]) => {
        if (this.isExpired(entry)) {
          channel.cache.delete(key);
        }
      });
    });
  }

  private updatePerformanceMetrics(): void {
    // Calculate cache hit rate
    let totalHits = 0;
    let totalRequests = 0;

    Array.from(this.channels.values()).forEach(channel => {
      Array.from(channel.cache.values()).forEach(entry => {
        totalHits += entry.hits;
        totalRequests += entry.hits > 0 ? 1 : 0;
      });
    });

    this.metrics.cacheHitRate = totalRequests > 0 ? totalHits / totalRequests : 0;
    this.emit('metrics.updated', this.getMetrics());
  }

  private handleError(error: Error): void {
    this.metrics.errorCount++;
    this.emit('error', error);
    console.error('[DataBus] Error:', error);
  }
}

// Singleton instance
let dataBusInstance: DataBus | null = null;

export function getDataBus(config?: Partial<DataBusConfig>): DataBus {
  if (!dataBusInstance) {
    dataBusInstance = new DataBus(config);
  }
  return dataBusInstance;
}

export function resetDataBus(): void {
  if (dataBusInstance) {
    dataBusInstance.destroy();
    dataBusInstance = null;
  }
}