/**
 * StreamingInfrastructure - Complete Enterprise Streaming System
 * Integrates all streaming components with performance monitoring and management
 */

import { EventEmitter } from 'eventemitter3';
import { BehaviorSubject, Subject, Observable, combineLatest, merge, interval } from 'rxjs';
import { takeUntil, filter, map, share, tap } from 'rxjs/operators';

import { MarketDataStream, MarketDataStreamConfig } from './MarketDataStream';
import { DataNormalizer, NormalizationConfig } from './DataNormalizer';
import { StreamBuffer, ThrottleManager } from './StreamBuffer';
import { ReconnectManager, ReconnectManagerConfig } from './ReconnectManager';
import { TickProcessor, TickAggregationConfig } from './TickProcessor';
import { OrderBookManager, OrderBookConfig } from './OrderBookManager';
import { CacheManager } from './CacheManager';

import type {
  StreamSource,
  StreamSubscription,
  StreamMessage,
  StreamMetrics,
  StreamError,
  StreamEvent,
  PerformanceMonitor,
  ThrottleConfig,
  BufferConfig,
  CacheConfig
} from '../types/streaming-types';

import type { MarketTick, OrderBook, Trade } from '../types/market-data';

export interface StreamingInfrastructureConfig {
  readonly marketDataStream: MarketDataStreamConfig;
  readonly normalization: NormalizationConfig;
  readonly throttle: ThrottleConfig;
  readonly buffer: BufferConfig;
  readonly reconnect: ReconnectManagerConfig;
  readonly tickAggregation: TickAggregationConfig;
  readonly orderBook: OrderBookConfig;
  readonly cache: CacheConfig;
  readonly enablePerformanceMonitoring: boolean;
  readonly enableDataPersistence: boolean;
  readonly enableQualityAnalytics: boolean;
  readonly maxConcurrentStreams: number;
  readonly globalRateLimit: number;
}

export interface SystemHealth {
  readonly timestamp: number;
  readonly status: 'healthy' | 'degraded' | 'critical' | 'offline';
  readonly uptime: number;
  readonly memoryUsage: number;
  readonly cpuUsage: number;
  readonly activeStreams: number;
  readonly errorRate: number;
  readonly latency: number;
  readonly throughput: number;
  readonly qualityScore: number;
  readonly components: ComponentHealth[];
}

export interface ComponentHealth {
  readonly name: string;
  readonly status: 'healthy' | 'degraded' | 'critical' | 'offline';
  readonly uptime: number;
  readonly errorCount: number;
  readonly lastError?: string;
  readonly performance: number;
  readonly memoryUsage: number;
}

export interface SystemMetrics {
  readonly timestamp: number;
  readonly performance: PerformanceMonitor;
  readonly streaming: StreamMetrics[];
  readonly cache: any;
  readonly orderBooks: any;
  readonly aggregation: any;
  readonly normalization: any;
  readonly reconnection: any;
}

export class StreamingInfrastructure extends EventEmitter {
  private readonly config: StreamingInfrastructureConfig;

  // Core components
  private readonly marketDataStream: MarketDataStream;
  private readonly normalizer: DataNormalizer;
  private readonly throttleManager: ThrottleManager;
  private readonly reconnectManager: ReconnectManager;
  private readonly tickProcessor: TickProcessor;
  private readonly orderBookManager: OrderBookManager;
  private readonly cacheManager: CacheManager;

  // System state
  private readonly systemHealth$ = new BehaviorSubject<SystemHealth>(this.createInitialHealth());
  private readonly systemMetrics$ = new Subject<SystemMetrics>();
  private readonly destroy$ = new Subject<void>();

  // Performance monitoring
  private startTime = Date.now();
  private performanceMonitor: PerformanceMonitor | null = null;
  private healthCheckTimer?: NodeJS.Timeout;
  private metricsTimer?: NodeJS.Timeout;

  // Stream management
  private readonly activeStreams = new Map<string, StreamSubscription>();
  private readonly streamBuffers = new Map<string, StreamBuffer>();

  constructor(config: StreamingInfrastructureConfig) {
    super();
    this.config = config;

    // Initialize all components
    this.marketDataStream = new MarketDataStream(config.marketDataStream);
    this.normalizer = new DataNormalizer(config.normalization);
    this.throttleManager = new ThrottleManager();
    this.reconnectManager = new ReconnectManager(config.reconnect);
    this.tickProcessor = new TickProcessor(config.tickAggregation);
    this.orderBookManager = new OrderBookManager(config.orderBook);
    this.cacheManager = new CacheManager(config.cache);

    this.initialize();
  }

  // ==========================================
  // INITIALIZATION
  // ==========================================

  private initialize(): void {
    this.setupEventHandlers();
    this.setupDataFlow();
    this.setupPerformanceMonitoring();
    this.setupHealthMonitoring();
    this.setupErrorHandling();

    this.emit('system_initialized', {
      timestamp: Date.now(),
      config: this.config
    });
  }

  private setupEventHandlers(): void {
    // Market data stream events
    this.marketDataStream.on('message.received', (data) => {
      this.handleIncomingMessage(data.message);
    });

    this.marketDataStream.on('subscription_added', (data) => {
      this.activeStreams.set(data.subscription.id, data.subscription);
      this.emit('stream_added', data);
    });

    this.marketDataStream.on('subscription_removed', (data) => {
      this.activeStreams.delete(data.subscription.id);
      this.emit('stream_removed', data);
    });

    // Reconnection events
    this.reconnectManager.on('reconnect_success', (data) => {
      this.emit('connection_restored', data);
    });

    this.reconnectManager.on('reconnect_failed', (data) => {
      this.emit('connection_failed', data);
    });

    // Order book events
    this.orderBookManager.on('order_book_updated', (data) => {
      this.handleOrderBookUpdate(data);
    });

    // Cache events
    this.cacheManager.on('cache_error', (data) => {
      this.handleCacheError(data);
    });
  }

  private setupDataFlow(): void {
    // Connect all data processing components
    this.marketDataStream.getMessages$().pipe(
      takeUntil(this.destroy$)
    ).subscribe(message => {
      this.processStreamMessage(message);
    });

    // Connect tick processing
    this.tickProcessor.getAggregatedStream().pipe(
      takeUntil(this.destroy$)
    ).subscribe(aggregated => {
      this.handleAggregatedData(aggregated);
    });

    // Connect pattern detection
    this.tickProcessor.getPatternStream().pipe(
      takeUntil(this.destroy$)
    ).subscribe(pattern => {
      this.emit('pattern_detected', pattern);
    });
  }

  // ==========================================
  // PUBLIC API
  // ==========================================

  public async addSource(source: StreamSource): Promise<void> {
    try {
      this.marketDataStream.addSource(source);
      this.reconnectManager.registerSource(source);

      this.emit('source_added', { source });
    } catch (error) {
      this.emit('source_error', { source, error });
      throw error;
    }
  }

  public async subscribe(
    symbol: string,
    dataType: 'tick' | 'orderbook' | 'trade',
    sourceId?: string
  ): Promise<string> {
    try {
      const subscriptionId = await this.marketDataStream.subscribe(symbol, dataType, sourceId);

      // Setup throttling if needed
      if (this.config.throttle.enabled) {
        const throttle = this.throttleManager.createThrottle(
          `${symbol}_${dataType}`,
          this.config.throttle
        );

        throttle.getOutputStream().pipe(
          takeUntil(this.destroy$)
        ).subscribe(message => {
          this.emit('throttled_message', message);
        });
      }

      // Setup buffering
      if (this.config.buffer.enabled) {
        const buffer = new StreamBuffer(`${symbol}_${dataType}`, this.config.buffer);
        this.streamBuffers.set(subscriptionId, buffer);
      }

      this.emit('subscription_created', { subscriptionId, symbol, dataType });
      return subscriptionId;

    } catch (error) {
      this.emit('subscription_error', { symbol, dataType, error });
      throw error;
    }
  }

  public async unsubscribe(subscriptionId: string): Promise<void> {
    try {
      await this.marketDataStream.unsubscribe(subscriptionId);

      // Cleanup associated resources
      const buffer = this.streamBuffers.get(subscriptionId);
      if (buffer) {
        buffer.destroy();
        this.streamBuffers.delete(subscriptionId);
      }

      this.emit('subscription_removed', { subscriptionId });

    } catch (error) {
      this.emit('unsubscription_error', { subscriptionId, error });
      throw error;
    }
  }

  public getTickStream(symbol?: string): Observable<MarketTick> {
    return this.marketDataStream.getTickStream(symbol);
  }

  public getOrderBookStream(symbol?: string): Observable<OrderBook> {
    return this.orderBookManager.getSnapshotStream(symbol);
  }

  public getTradeStream(symbol?: string): Observable<Trade> {
    return this.marketDataStream.getTradeStream(symbol);
  }

  public getSystemHealthStream(): Observable<SystemHealth> {
    return this.systemHealth$.asObservable().pipe(
      takeUntil(this.destroy$),
      share()
    );
  }

  public getSystemMetricsStream(): Observable<SystemMetrics> {
    return this.systemMetrics$.asObservable().pipe(
      takeUntil(this.destroy$),
      share()
    );
  }

  public getCurrentHealth(): SystemHealth {
    return this.systemHealth$.value;
  }

  public getPerformanceMetrics(): PerformanceMonitor | null {
    return this.performanceMonitor;
  }

  // ==========================================
  // DATA PROCESSING
  // ==========================================

  private handleIncomingMessage(rawMessage: any): void {
    try {
      // Normalize the message
      const source = this.findSourceById(rawMessage.sourceId);
      if (!source) {
        this.emit('source_not_found', { sourceId: rawMessage.sourceId });
        return;
      }

      const normalized = this.normalizer.normalize(source, rawMessage.data, rawMessage.dataType);

      if (!normalized.success) {
        this.emit('normalization_failed', {
          errors: normalized.errors,
          warnings: normalized.warnings,
          rawMessage
        });
        return;
      }

      // Process based on data type
      switch (rawMessage.dataType) {
        case 'tick':
          this.processTick(normalized.data!);
          break;
        case 'orderbook':
          this.processOrderBook(rawMessage.symbol, normalized.data!);
          break;
        case 'trade':
          this.processTrade(normalized.data!);
          break;
      }

      // Cache if enabled
      if (this.config.enableDataPersistence) {
        this.cacheMessage(rawMessage, normalized.data!);
      }

    } catch (error) {
      this.emit('message_processing_error', { error, rawMessage });
    }
  }

  private processTick(tick: MarketTick): void {
    // Send to tick processor
    this.tickProcessor.processTick(tick);

    // Apply throttling if configured
    const throttleKey = `${tick.symbol}_tick`;
    const throttle = this.throttleManager.getThrottle(throttleKey);
    if (throttle) {
      throttle.input({
        id: this.generateId(),
        type: 'data',
        symbol: tick.symbol,
        data: tick,
        timestamp: tick.timestamp,
        source: tick.source,
        quality: tick.quality
      });
    }

    this.emit('tick_processed', tick);
  }

  private processOrderBook(symbol: string, orderBookData: any): void {
    if (orderBookData.type === 'snapshot') {
      this.orderBookManager.processSnapshot(symbol, orderBookData);
    } else {
      this.orderBookManager.processUpdate(symbol, orderBookData);
    }

    this.emit('orderbook_processed', { symbol, data: orderBookData });
  }

  private processTrade(trade: Trade): void {
    // Send to tick processor for analysis
    this.tickProcessor.processTrade(trade);

    // Send to order book manager for impact analysis
    this.orderBookManager.processTrade(trade);

    this.emit('trade_processed', trade);
  }

  private handleAggregatedData(aggregated: any): void {
    // Cache aggregated data
    if (this.config.enableDataPersistence) {
      this.cacheManager.set(
        `agg_${aggregated.symbol}_${aggregated.interval}_${aggregated.timestamp}`,
        aggregated,
        'aggregated'
      );
    }

    this.emit('data_aggregated', aggregated);
  }

  private handleOrderBookUpdate(data: any): void {
    // Additional processing for order book updates
    this.emit('orderbook_updated', data);
  }

  // ==========================================
  // CACHING
  // ==========================================

  private async cacheMessage(rawMessage: any, processedData: any): Promise<void> {
    try {
      const key = this.generateCacheKey(rawMessage);
      const storeName = this.getStoreNameForDataType(rawMessage.dataType);

      await this.cacheManager.set(key, processedData, storeName);

    } catch (error) {
      this.emit('cache_error', { error, rawMessage });
    }
  }

  private generateCacheKey(message: any): string {
    return `${message.symbol}_${message.dataType}_${message.timestamp}`;
  }

  private getStoreNameForDataType(dataType: string): string {
    switch (dataType) {
      case 'tick': return 'ticks';
      case 'orderbook': return 'order_books';
      case 'trade': return 'trades';
      default: return 'ticks';
    }
  }

  // ==========================================
  // PERFORMANCE MONITORING
  // ==========================================

  private setupPerformanceMonitoring(): void {
    if (!this.config.enablePerformanceMonitoring) return;

    this.metricsTimer = setInterval(() => {
      this.collectSystemMetrics();
    }, 5000);
  }

  private setupHealthMonitoring(): void {
    this.healthCheckTimer = setInterval(() => {
      this.updateSystemHealth();
    }, 10000);
  }

  private collectSystemMetrics(): void {
    try {
      const performance = this.getPerformanceSnapshot();
      const streaming = this.marketDataStream.getStreamMetrics();
      const cache = this.cacheManager.getStats();

      const metrics: SystemMetrics = {
        timestamp: Date.now(),
        performance,
        streaming,
        cache,
        orderBooks: this.orderBookManager.getAnalytics(),
        aggregation: this.getAggregationMetrics(),
        normalization: this.getNormalizationMetrics(),
        reconnection: this.reconnectManager.getMetrics()
      };

      this.systemMetrics$.next(metrics);
      this.emit('metrics_collected', metrics);

    } catch (error) {
      this.emit('metrics_error', error);
    }
  }

  private updateSystemHealth(): void {
    try {
      const health = this.calculateSystemHealth();
      this.systemHealth$.next(health);
      this.emit('health_updated', health);

    } catch (error) {
      this.emit('health_check_error', error);
    }
  }

  private getPerformanceSnapshot(): PerformanceMonitor {
    // Update performance monitor
    this.performanceMonitor = {
      cpu: this.estimateCPUUsage(),
      memory: this.getMemoryUsage(),
      network: this.estimateNetworkUsage(),
      disk: 0, // Would need actual disk monitoring
      activeConnections: this.activeStreams.size,
      buffersInUse: this.streamBuffers.size,
      workersActive: 0, // Would track active workers
      timestamp: Date.now()
    };

    return this.performanceMonitor;
  }

  private calculateSystemHealth(): SystemHealth {
    const uptime = Date.now() - this.startTime;
    const memoryUsage = this.getMemoryUsage();
    const cpuUsage = this.estimateCPUUsage();
    const activeStreams = this.activeStreams.size;
    const errorRate = this.calculateErrorRate();
    const latency = this.calculateAverageLatency();
    const throughput = this.calculateThroughput();
    const qualityScore = this.calculateQualityScore();

    let status: SystemHealth['status'] = 'healthy';
    if (errorRate > 0.1 || memoryUsage > 0.9 || cpuUsage > 0.9) {
      status = 'critical';
    } else if (errorRate > 0.05 || memoryUsage > 0.8 || cpuUsage > 0.8) {
      status = 'degraded';
    }

    const components = this.getComponentHealth();

    return {
      timestamp: Date.now(),
      status,
      uptime,
      memoryUsage,
      cpuUsage,
      activeStreams,
      errorRate,
      latency,
      throughput,
      qualityScore,
      components
    };
  }

  private getComponentHealth(): ComponentHealth[] {
    return [
      {
        name: 'MarketDataStream',
        status: 'healthy', // Would implement actual health checks
        uptime: Date.now() - this.startTime,
        errorCount: 0,
        performance: 0.95,
        memoryUsage: 0.1
      },
      {
        name: 'OrderBookManager',
        status: 'healthy',
        uptime: Date.now() - this.startTime,
        errorCount: 0,
        performance: 0.98,
        memoryUsage: 0.05
      },
      {
        name: 'TickProcessor',
        status: 'healthy',
        uptime: Date.now() - this.startTime,
        errorCount: 0,
        performance: 0.92,
        memoryUsage: 0.08
      },
      {
        name: 'CacheManager',
        status: 'healthy',
        uptime: Date.now() - this.startTime,
        errorCount: 0,
        performance: 0.96,
        memoryUsage: 0.15
      }
    ];
  }

  // ==========================================
  // METRICS CALCULATIONS
  // ==========================================

  private getMemoryUsage(): number {
    if (typeof performance !== 'undefined' && (performance as any).memory) {
      const memory = (performance as any).memory;
      return memory.usedJSHeapSize / memory.jsHeapSizeLimit;
    }
    return 0;
  }

  private estimateCPUUsage(): number {
    // Simplified CPU estimation based on processing load
    const activeComponents = [
      this.activeStreams.size > 0,
      this.streamBuffers.size > 0,
      true // Base system load
    ].filter(Boolean).length;

    return Math.min(activeComponents * 0.2, 1);
  }

  private estimateNetworkUsage(): number {
    // Simplified network usage estimation
    return this.activeStreams.size * 10; // KB/s per stream
  }

  private calculateErrorRate(): number {
    // Would track actual errors over time window
    return 0.01; // 1% placeholder
  }

  private calculateAverageLatency(): number {
    // Would calculate from actual latency measurements
    return 50; // 50ms placeholder
  }

  private calculateThroughput(): number {
    // Would calculate from actual message counts
    return this.activeStreams.size * 100; // messages/second
  }

  private calculateQualityScore(): number {
    // Composite quality score from all components
    return 95; // 95% placeholder
  }

  private getAggregationMetrics(): any {
    // Would get actual aggregation metrics
    return {
      aggregationsPerSecond: 10,
      averageProcessingTime: 5,
      patternDetectionRate: 0.1
    };
  }

  private getNormalizationMetrics(): any {
    // Would get actual normalization metrics
    return {
      normalizationsPerSecond: 100,
      successRate: 0.99,
      averageProcessingTime: 1
    };
  }

  // ==========================================
  // ERROR HANDLING
  // ==========================================

  private setupErrorHandling(): void {
    // Global error handler
    this.on('error', (error) => {
      this.handleSystemError(error);
    });

    // Component error handlers
    this.marketDataStream.on('error', (error) => {
      this.handleComponentError('MarketDataStream', error);
    });

    this.orderBookManager.on('error', (error) => {
      this.handleComponentError('OrderBookManager', error);
    });

    this.cacheManager.on('error', (error) => {
      this.handleComponentError('CacheManager', error);
    });
  }

  private handleSystemError(error: any): void {
    this.emit('system_error', {
      timestamp: Date.now(),
      error: error.message || error,
      stack: error.stack
    });
  }

  private handleComponentError(component: string, error: any): void {
    this.emit('component_error', {
      timestamp: Date.now(),
      component,
      error: error.message || error,
      stack: error.stack
    });
  }

  private handleCacheError(error: any): void {
    // Handle cache-specific errors
    this.emit('cache_error', error);
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  private findSourceById(sourceId: string): StreamSource | undefined {
    return this.marketDataStream.getActiveSources().find(s => s.id === sourceId);
  }

  private createInitialHealth(): SystemHealth {
    return {
      timestamp: Date.now(),
      status: 'healthy',
      uptime: 0,
      memoryUsage: 0,
      cpuUsage: 0,
      activeStreams: 0,
      errorRate: 0,
      latency: 0,
      throughput: 0,
      qualityScore: 100,
      components: []
    };
  }

  private generateId(): string {
    return `stream-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }

  // ==========================================
  // CLEANUP
  // ==========================================

  public async destroy(): Promise<void> {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear timers
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    if (this.metricsTimer) {
      clearInterval(this.metricsTimer);
    }

    // Destroy all components
    this.marketDataStream.destroy();
    this.throttleManager.destroy();
    this.reconnectManager.destroy();
    this.tickProcessor.destroy();
    this.orderBookManager.destroy();
    await this.cacheManager.destroy();

    // Destroy buffers
    this.streamBuffers.forEach(buffer => buffer.destroy());
    this.streamBuffers.clear();

    // Clear state
    this.activeStreams.clear();

    // Complete observables
    this.systemHealth$.complete();
    this.systemMetrics$.complete();

    this.removeAllListeners();

    this.emit('system_destroyed', { timestamp: Date.now() });
  }
}

// ==========================================
// FACTORY FUNCTION
// ==========================================

export function createStreamingInfrastructure(
  config: Partial<StreamingInfrastructureConfig>
): StreamingInfrastructure {
  const defaultConfig: StreamingInfrastructureConfig = {
    marketDataStream: {
      throttle: {
        enabled: true,
        maxUpdatesPerSecond: 60,
        burstLimit: 100,
        windowSize: 1000,
        strategy: 'drop_oldest'
      },
      buffer: {
        enabled: true,
        maxSize: 1000,
        maxAge: 300000,
        compressionEnabled: true,
        persistToDisk: false,
        strategy: 'fifo'
      },
      reconnect: {
        enabled: true,
        maxAttempts: 5,
        initialDelay: 1000,
        maxDelay: 30000,
        backoffMultiplier: 2,
        jitter: true
      },
      maxConcurrentConnections: 10,
      enableQualityMonitoring: true,
      enablePerformanceMonitoring: true,
      enableDataPersistence: false,
      workerPoolSize: 2
    },
    normalization: {
      baseTimezone: 'UTC',
      baseCurrency: 'USD',
      precision: {
        price: 4,
        volume: 2,
        percentage: 2
      },
      validation: {
        enablePriceValidation: true,
        enableVolumeValidation: true,
        maxPriceDeviation: 1000,
        minVolume: 0
      },
      conversion: {
        enableCurrencyConversion: false,
        exchangeRates: new Map()
      }
    },
    throttle: {
      enabled: true,
      maxUpdatesPerSecond: 60,
      burstLimit: 100,
      windowSize: 1000,
      strategy: 'drop_oldest'
    },
    buffer: {
      enabled: true,
      maxSize: 1000,
      maxAge: 300000,
      compressionEnabled: true,
      persistToDisk: false,
      strategy: 'fifo'
    },
    reconnect: {
      globalRetryLimit: 10,
      globalBackoffMultiplier: 2,
      circuitBreakerThreshold: 5,
      circuitBreakerTimeout: 60000,
      healthCheckInterval: 30000,
      enableAdaptiveBackoff: true,
      enableJitter: true,
      maxJitterPercent: 20,
      qualityThreshold: 0.8
    },
    tickAggregation: {
      intervals: ['1s', '5s', '15s', '30s', '1m', '5m'],
      enableVWAP: true,
      enableVolumeDelta: true,
      enableMicrostructure: true,
      maxTicksPerSecond: 1000,
      enablePatternDetection: true,
      enableRealTimeIndicators: true
    },
    orderBook: {
      maxLevels: 50,
      enableValidation: true,
      enableGapDetection: true,
      enableAnalytics: true,
      checksumValidation: false,
      autoReconstruct: true,
      reconstructionTimeout: 5000,
      priceToleranceBps: 1000,
      volumeToleranceBps: 10000
    },
    cache: {
      enabled: true,
      maxSizeBytes: 100 * 1024 * 1024, // 100MB
      maxAge: 3600000, // 1 hour
      compression: true,
      persistToDisk: true,
      evictionStrategy: 'lru',
      indexingEnabled: true
    },
    enablePerformanceMonitoring: true,
    enableDataPersistence: true,
    enableQualityAnalytics: true,
    maxConcurrentStreams: 50,
    globalRateLimit: 10000
  };

  const mergedConfig = { ...defaultConfig, ...config };
  return new StreamingInfrastructure(mergedConfig);
}