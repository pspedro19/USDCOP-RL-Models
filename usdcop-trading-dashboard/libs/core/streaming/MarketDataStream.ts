/**
 * MarketDataStream - Elite Trading Platform Streaming Engine
 * High-performance, multi-exchange market data streaming with enterprise reliability
 */

import { EventEmitter } from 'eventemitter3';
import { BehaviorSubject, Subject, Observable, interval, merge, NEVER } from 'rxjs';
import {
  filter,
  map,
  retry,
  switchMap,
  takeUntil,
  throttleTime,
  bufferTime,
  share,
  tap,
  catchError
} from 'rxjs/operators';

import type {
  StreamSource,
  StreamSubscription,
  StreamMessage,
  StreamMetrics,
  StreamError,
  StreamEvent,
  ConnectionState,
  ThrottleConfig,
  BufferConfig,
  ReconnectConfig,
  StreamDataType,
  DataQualityMetrics,
  PerformanceMonitor
} from '../types/streaming-types';

import type { MarketTick, OrderBook, Trade } from '../types/market-data';
import { getWebSocketManager, WebSocketManager } from '../websocket/WebSocketManager';

export interface MarketDataStreamConfig {
  readonly throttle: ThrottleConfig;
  readonly buffer: BufferConfig;
  readonly reconnect: ReconnectConfig;
  readonly maxConcurrentConnections: number;
  readonly enableQualityMonitoring: boolean;
  readonly enablePerformanceMonitoring: boolean;
  readonly enableDataPersistence: boolean;
  readonly workerPoolSize: number;
}

export class MarketDataStream extends EventEmitter {
  private readonly config: MarketDataStreamConfig;
  private readonly sources = new Map<string, StreamSource>();
  private readonly subscriptions = new Map<string, StreamSubscription>();
  private readonly connections = new Map<string, WebSocketManager>();

  // RxJS Subjects for reactive streams
  private readonly connectionState$ = new BehaviorSubject<ConnectionState>('disconnected');
  private readonly rawMessages$ = new Subject<StreamMessage>();
  private readonly processedMessages$ = new Subject<StreamMessage>();
  private readonly errors$ = new Subject<StreamError>();
  private readonly events$ = new Subject<StreamEvent>();
  private readonly metrics$ = new Subject<StreamMetrics>();
  private readonly destroy$ = new Subject<void>();

  // Performance and quality tracking
  private readonly streamMetrics = new Map<string, StreamMetrics>();
  private readonly qualityMetrics = new Map<string, DataQualityMetrics>();
  private performanceMonitor: PerformanceMonitor | null = null;

  // Message buffers for throttling
  private readonly messageBuffers = new Map<string, StreamMessage[]>();
  private readonly throttleTimers = new Map<string, NodeJS.Timeout>();

  // Quality monitoring
  private readonly duplicateDetector = new Map<string, Set<string>>();
  private readonly sequenceTracker = new Map<string, number>();
  private readonly latencyTracker = new Map<string, number[]>();

  // Worker pool for background processing
  private readonly workers: Worker[] = [];
  private workerIndex = 0;

  constructor(config: MarketDataStreamConfig) {
    super();
    this.config = config;
    this.initialize();
  }

  private initialize(): void {
    this.setupPerformanceMonitoring();
    this.setupWorkerPool();
    this.setupMessageProcessing();
    this.setupQualityMonitoring();
    this.setupMetricsCollection();
  }

  // ==========================================
  // SOURCE MANAGEMENT
  // ==========================================

  public addSource(source: StreamSource): void {
    this.sources.set(source.id, source);
    this.emit('source_added', { source });
    this.log('info', `Added stream source: ${source.name} (${source.id})`);
  }

  public removeSource(sourceId: string): void {
    const source = this.sources.get(sourceId);
    if (!source) {
      this.log('warn', `Source not found: ${sourceId}`);
      return;
    }

    // Remove all subscriptions for this source
    const subscriptionsToRemove = Array.from(this.subscriptions.entries())
      .filter(([_, sub]) => sub.sourceId === sourceId)
      .map(([id, _]) => id);

    subscriptionsToRemove.forEach(subId => this.unsubscribe(subId));

    // Disconnect and clean up
    const connection = this.connections.get(sourceId);
    if (connection) {
      connection.destroy();
      this.connections.delete(sourceId);
    }

    this.sources.delete(sourceId);
    this.emit('source_removed', { sourceId, source });
    this.log('info', `Removed stream source: ${sourceId}`);
  }

  public getActiveSources(): StreamSource[] {
    return Array.from(this.sources.values()).filter(source => source.isActive);
  }

  // ==========================================
  // SUBSCRIPTION MANAGEMENT
  // ==========================================

  public async subscribe(
    symbol: string,
    dataType: StreamDataType,
    sourceId?: string
  ): Promise<string> {
    const subscriptionId = this.generateId();

    // Find best source if none specified
    const source = sourceId
      ? this.sources.get(sourceId)
      : this.findBestSource(symbol, dataType);

    if (!source) {
      throw new Error(`No suitable source found for ${symbol}:${dataType}`);
    }

    const subscription: StreamSubscription = {
      id: subscriptionId,
      symbol,
      dataType,
      sourceId: source.id,
      isActive: true,
      createdAt: Date.now()
    };

    this.subscriptions.set(subscriptionId, subscription);

    try {
      await this.establishConnection(source);
      await this.sendSubscription(source.id, subscription);

      this.emit('subscription_added', { subscription });
      this.log('info', `Subscribed to ${symbol}:${dataType} via ${source.name}`);

      return subscriptionId;
    } catch (error) {
      this.subscriptions.delete(subscriptionId);
      throw error;
    }
  }

  public async unsubscribe(subscriptionId: string): Promise<void> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) {
      this.log('warn', `Subscription not found: ${subscriptionId}`);
      return;
    }

    try {
      await this.sendUnsubscription(subscription.sourceId, subscription);
      this.subscriptions.delete(subscriptionId);

      this.emit('subscription_removed', { subscription });
      this.log('info', `Unsubscribed from ${subscription.symbol}:${subscription.dataType}`);
    } catch (error) {
      this.log('error', `Failed to unsubscribe: ${subscriptionId}`, error);
      throw error;
    }
  }

  public getActiveSubscriptions(): StreamSubscription[] {
    return Array.from(this.subscriptions.values()).filter(sub => sub.isActive);
  }

  // ==========================================
  // DATA STREAMS
  // ==========================================

  public getTickStream(symbol?: string): Observable<MarketTick> {
    return this.processedMessages$.pipe(
      filter(msg => msg.type === 'data' && (!symbol || msg.symbol === symbol)),
      map(msg => msg.data as MarketTick),
      filter(tick => this.isValidTick(tick)),
      share()
    );
  }

  public getOrderBookStream(symbol?: string): Observable<OrderBook> {
    return this.processedMessages$.pipe(
      filter(msg => msg.type === 'data' && (!symbol || msg.symbol === symbol)),
      map(msg => msg.data as OrderBook),
      filter(book => this.isValidOrderBook(book)),
      share()
    );
  }

  public getTradeStream(symbol?: string): Observable<Trade> {
    return this.processedMessages$.pipe(
      filter(msg => msg.type === 'data' && (!symbol || msg.symbol === symbol)),
      map(msg => msg.data as Trade),
      filter(trade => this.isValidTrade(trade)),
      share()
    );
  }

  public getThrottledStream(symbol: string, maxUpdatesPerSecond: number): Observable<StreamMessage> {
    return this.processedMessages$.pipe(
      filter(msg => msg.symbol === symbol),
      throttleTime(1000 / maxUpdatesPerSecond),
      share()
    );
  }

  public getBufferedStream(symbol: string, bufferTimeMs: number): Observable<StreamMessage[]> {
    return this.processedMessages$.pipe(
      filter(msg => msg.symbol === symbol),
      bufferTime(bufferTimeMs),
      filter(buffer => buffer.length > 0),
      share()
    );
  }

  // ==========================================
  // QUALITY & PERFORMANCE MONITORING
  // ==========================================

  public getMetricsStream(): Observable<StreamMetrics> {
    return this.metrics$.asObservable();
  }

  public getQualityMetrics(symbol?: string): DataQualityMetrics[] {
    const metrics = Array.from(this.qualityMetrics.values());
    return symbol ? metrics.filter(m => m.symbol === symbol) : metrics;
  }

  public getPerformanceMetrics(): PerformanceMonitor | null {
    return this.performanceMonitor;
  }

  public getStreamMetrics(streamId?: string): StreamMetrics[] {
    const metrics = Array.from(this.streamMetrics.values());
    return streamId ? metrics.filter(m => m.streamId === streamId) : metrics;
  }

  // ==========================================
  // CONNECTION MANAGEMENT
  // ==========================================

  private async establishConnection(source: StreamSource): Promise<void> {
    if (this.connections.has(source.id)) {
      return; // Already connected
    }

    const wsManager = getWebSocketManager({
      url: source.wsUrl,
      maxReconnectAttempts: source.reconnectConfig.maxAttempts,
      reconnectDelay: source.reconnectConfig.initialDelay,
      heartbeatInterval: 30000,
      connectionTimeout: 10000,
      messageTimeout: 5000,
      maxMessageQueue: 1000,
      enableCompression: true,
      enableLogging: true,
      autoReconnect: source.reconnectConfig.enabled
    });

    // Setup event handlers
    wsManager.on('message.received', ({ message }) => {
      this.handleRawMessage(source.id, message);
    });

    wsManager.on('connection.error', ({ error }) => {
      this.handleConnectionError(source.id, error);
    });

    wsManager.on('connection.closed', () => {
      this.handleConnectionClosed(source.id);
    });

    // Connect
    const connectionId = await wsManager.connect();
    this.connections.set(source.id, wsManager);

    this.emit('connection_established', { sourceId: source.id, connectionId });
    this.log('info', `Connected to ${source.name}`);
  }

  private async sendSubscription(sourceId: string, subscription: StreamSubscription): Promise<void> {
    const connection = this.connections.get(sourceId);
    if (!connection) {
      throw new Error(`No connection for source: ${sourceId}`);
    }

    const message: StreamMessage = {
      id: this.generateId(),
      type: 'subscribe',
      symbol: subscription.symbol,
      data: {
        symbol: subscription.symbol,
        dataType: subscription.dataType,
        interval: subscription.interval
      },
      timestamp: Date.now(),
      source: 'internal',
      quality: 'realtime'
    };

    connection.send('default', message);
  }

  private async sendUnsubscription(sourceId: string, subscription: StreamSubscription): Promise<void> {
    const connection = this.connections.get(sourceId);
    if (!connection) {
      return; // Connection already closed
    }

    const message: StreamMessage = {
      id: this.generateId(),
      type: 'unsubscribe',
      symbol: subscription.symbol,
      data: {
        symbol: subscription.symbol,
        dataType: subscription.dataType
      },
      timestamp: Date.now(),
      source: 'internal',
      quality: 'realtime'
    };

    connection.send('default', message);
  }

  // ==========================================
  // MESSAGE PROCESSING
  // ==========================================

  private setupMessageProcessing(): void {
    // Raw message processing pipeline
    this.rawMessages$.pipe(
      takeUntil(this.destroy$),
      // Duplicate detection
      filter(msg => this.checkDuplicate(msg)),
      // Sequence validation
      tap(msg => this.validateSequence(msg)),
      // Quality assessment
      map(msg => this.assessQuality(msg)),
      // Throttling
      switchMap(msg => this.applyThrottling(msg)),
      // Error handling
      catchError((error, caught) => {
        this.handleProcessingError(error);
        return caught;
      })
    ).subscribe(msg => {
      this.processedMessages$.next(msg);
      this.updateMetrics(msg);
    });
  }

  private handleRawMessage(sourceId: string, rawMessage: any): void {
    try {
      const message = this.normalizeMessage(sourceId, rawMessage);
      this.rawMessages$.next(message);
      this.updateLatencyTracking(message);
    } catch (error) {
      this.handleMessageError(sourceId, rawMessage, error);
    }
  }

  private normalizeMessage(sourceId: string, rawMessage: any): StreamMessage {
    const source = this.sources.get(sourceId);
    if (!source) {
      throw new Error(`Unknown source: ${sourceId}`);
    }

    // Normalize based on exchange format
    switch (source.type) {
      case 'crypto':
        return this.normalizeCryptoMessage(source, rawMessage);
      case 'forex':
        return this.normalizeForexMessage(source, rawMessage);
      default:
        return this.normalizeGenericMessage(source, rawMessage);
    }
  }

  private normalizeCryptoMessage(source: StreamSource, raw: any): StreamMessage {
    // Implement crypto-specific normalization
    return {
      id: this.generateId(),
      type: 'data',
      symbol: raw.symbol || raw.s,
      data: {
        symbol: raw.symbol || raw.s,
        timestamp: raw.timestamp || raw.T || Date.now(),
        price: parseFloat(raw.price || raw.p),
        volume: parseFloat(raw.volume || raw.v || '0'),
        bid: parseFloat(raw.bid || raw.b),
        ask: parseFloat(raw.ask || raw.a)
      },
      timestamp: Date.now(),
      source: source.name as any,
      quality: 'realtime'
    };
  }

  private normalizeForexMessage(source: StreamSource, raw: any): StreamMessage {
    // Implement forex-specific normalization
    return {
      id: this.generateId(),
      type: 'data',
      symbol: raw.symbol || 'USDCOP',
      data: {
        symbol: raw.symbol || 'USDCOP',
        timestamp: raw.timestamp || Date.now(),
        price: parseFloat(raw.price || raw.close),
        volume: parseFloat(raw.volume || '0'),
        bid: parseFloat(raw.bid),
        ask: parseFloat(raw.ask)
      },
      timestamp: Date.now(),
      source: source.name as any,
      quality: 'realtime'
    };
  }

  private normalizeGenericMessage(source: StreamSource, raw: any): StreamMessage {
    return {
      id: this.generateId(),
      type: raw.type || 'data',
      symbol: raw.symbol || 'UNKNOWN',
      data: raw.data || raw,
      timestamp: Date.now(),
      source: source.name as any,
      quality: 'realtime'
    };
  }

  // ==========================================
  // QUALITY CONTROL
  // ==========================================

  private checkDuplicate(message: StreamMessage): boolean {
    const key = `${message.symbol}_${message.timestamp}_${JSON.stringify(message.data)}`;
    const messageHash = this.hashMessage(key);

    if (!this.duplicateDetector.has(message.symbol)) {
      this.duplicateDetector.set(message.symbol, new Set());
    }

    const hashes = this.duplicateDetector.get(message.symbol)!;
    if (hashes.has(messageHash)) {
      this.updateQualityMetric(message.symbol, 'duplicates', 1);
      return false; // Duplicate detected
    }

    hashes.add(messageHash);

    // Clean old hashes (keep last 1000)
    if (hashes.size > 1000) {
      const hashArray = Array.from(hashes);
      hashes.clear();
      hashArray.slice(-500).forEach(h => hashes.add(h));
    }

    return true;
  }

  private validateSequence(message: StreamMessage): void {
    if (!message.sequence) return;

    const lastSequence = this.sequenceTracker.get(message.symbol) || 0;
    const currentSequence = message.sequence;

    if (currentSequence <= lastSequence) {
      this.updateQualityMetric(message.symbol, 'outOfOrder', 1);
    } else if (currentSequence > lastSequence + 1) {
      const gap = currentSequence - lastSequence - 1;
      this.updateQualityMetric(message.symbol, 'gaps', gap);
    }

    this.sequenceTracker.set(message.symbol, currentSequence);
  }

  private assessQuality(message: StreamMessage): StreamMessage {
    const age = Date.now() - message.timestamp;
    const latency = message.latency || 0;

    let quality: any = 'good';

    if (age > 5000 || latency > 1000) {
      quality = 'poor';
      this.updateQualityMetric(message.symbol, 'staleDataCount', 1);
    } else if (age > 1000 || latency > 500) {
      quality = 'fair';
    } else {
      quality = 'excellent';
    }

    return { ...message, quality };
  }

  // ==========================================
  // THROTTLING & BUFFERING
  // ==========================================

  private applyThrottling(message: StreamMessage): Observable<StreamMessage> {
    if (!this.config.throttle.enabled) {
      return new Observable(subscriber => {
        subscriber.next(message);
        subscriber.complete();
      });
    }

    const symbol = message.symbol;
    const maxRate = this.config.throttle.maxUpdatesPerSecond;
    const interval = 1000 / maxRate;

    // Initialize buffer if needed
    if (!this.messageBuffers.has(symbol)) {
      this.messageBuffers.set(symbol, []);
    }

    const buffer = this.messageBuffers.get(symbol)!;
    buffer.push(message);

    // Setup throttle timer if not exists
    if (!this.throttleTimers.has(symbol)) {
      const timer = setInterval(() => {
        const messages = this.messageBuffers.get(symbol);
        if (messages && messages.length > 0) {
          const messageToSend = this.selectBestMessage(messages);
          messages.length = 0; // Clear buffer
          this.processedMessages$.next(messageToSend);
        }
      }, interval);

      this.throttleTimers.set(symbol, timer);
    }

    return NEVER; // Messages are emitted via timer
  }

  private selectBestMessage(messages: StreamMessage[]): StreamMessage {
    switch (this.config.throttle.strategy) {
      case 'drop_oldest':
        return messages[messages.length - 1];
      case 'drop_newest':
        return messages[0];
      case 'merge':
        return this.mergeMessages(messages);
      case 'sample':
        return messages[Math.floor(Math.random() * messages.length)];
      default:
        return messages[messages.length - 1];
    }
  }

  private mergeMessages(messages: StreamMessage[]): StreamMessage {
    if (messages.length === 1) return messages[0];

    const latest = messages[messages.length - 1];
    const merged = { ...latest };

    // Calculate average price if multiple ticks
    if (messages.length > 1) {
      const prices = messages.map(m => m.data.price).filter(p => p);
      if (prices.length > 0) {
        merged.data = {
          ...merged.data,
          price: prices.reduce((a, b) => a + b) / prices.length,
          volume: messages.reduce((sum, m) => sum + (m.data.volume || 0), 0)
        };
      }
    }

    return merged;
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  private findBestSource(symbol: string, dataType: StreamDataType): StreamSource | undefined {
    const availableSources = Array.from(this.sources.values())
      .filter(source =>
        source.isActive &&
        source.supportedSymbols.includes(symbol) &&
        source.supportedDataTypes.includes(dataType)
      )
      .sort((a, b) => b.priority - a.priority);

    return availableSources[0];
  }

  private setupWorkerPool(): void {
    if (typeof Worker === 'undefined') return;

    for (let i = 0; i < this.config.workerPoolSize; i++) {
      try {
        const worker = new Worker('/workers/stream-processor.js');
        worker.onmessage = (event) => {
          this.handleWorkerMessage(event.data);
        };
        this.workers.push(worker);
      } catch (error) {
        this.log('warn', 'Failed to create worker', error);
      }
    }
  }

  private handleWorkerMessage(message: any): void {
    // Handle processed data from workers
    this.emit('worker_result', message);
  }

  private setupPerformanceMonitoring(): void {
    if (!this.config.enablePerformanceMonitoring) return;

    interval(5000).pipe(
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.updatePerformanceMetrics();
    });
  }

  private setupQualityMonitoring(): void {
    if (!this.config.enableQualityMonitoring) return;

    interval(10000).pipe(
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.updateQualityMetrics();
    });
  }

  private setupMetricsCollection(): void {
    interval(1000).pipe(
      takeUntil(this.destroy$)
    ).subscribe(() => {
      this.collectMetrics();
    });
  }

  private updatePerformanceMetrics(): void {
    if (typeof performance !== 'undefined' && (performance as any).memory) {
      this.performanceMonitor = {
        cpu: 0, // Would need actual CPU monitoring
        memory: (performance as any).memory.usedJSHeapSize,
        network: 0, // Would need network monitoring
        disk: 0, // Would need disk monitoring
        activeConnections: this.connections.size,
        buffersInUse: this.messageBuffers.size,
        workersActive: this.workers.length,
        timestamp: Date.now()
      };
    }
  }

  private updateQualityMetrics(): void {
    this.qualityMetrics.forEach((metrics, symbol) => {
      const totalMessages = metrics.completeness * 1000; // Estimate
      const qualityScore = this.calculateQualityScore(metrics);

      this.qualityMetrics.set(symbol, {
        ...metrics,
        qualityScore,
        timestamp: Date.now()
      } as any);
    });
  }

  private calculateQualityScore(metrics: DataQualityMetrics): number {
    const weights = {
      completeness: 0.3,
      timeliness: 0.25,
      accuracy: 0.25,
      consistency: 0.2
    };

    return Math.round(
      metrics.completeness * weights.completeness * 100 +
      metrics.timeliness * weights.timeliness * 100 +
      metrics.accuracy * weights.accuracy * 100 +
      metrics.consistency * weights.consistency * 100
    );
  }

  private collectMetrics(): void {
    this.subscriptions.forEach((subscription, id) => {
      const metrics: StreamMetrics = {
        streamId: id,
        sourceId: subscription.sourceId,
        symbol: subscription.symbol,
        startTime: subscription.createdAt,
        uptime: Date.now() - subscription.createdAt,
        totalMessages: 0, // Would track actual count
        messagesPerSecond: 0, // Would calculate actual rate
        averageLatency: 0, // Would calculate from latency tracker
        minLatency: 0,
        maxLatency: 0,
        errorCount: 0,
        errorRate: 0,
        reconnections: 0,
        droppedMessages: 0,
        bufferUtilization: 0,
        memoryUsage: 0,
        quality: 'good'
      };

      this.streamMetrics.set(id, metrics);
      this.metrics$.next(metrics);
    });
  }

  private updateLatencyTracking(message: StreamMessage): void {
    if (!message.latency) return;

    if (!this.latencyTracker.has(message.symbol)) {
      this.latencyTracker.set(message.symbol, []);
    }

    const latencies = this.latencyTracker.get(message.symbol)!;
    latencies.push(message.latency);

    // Keep only last 100 latency measurements
    if (latencies.length > 100) {
      latencies.splice(0, latencies.length - 100);
    }
  }

  private updateQualityMetric(symbol: string, metric: string, value: number): void {
    if (!this.qualityMetrics.has(symbol)) {
      this.qualityMetrics.set(symbol, {
        symbol,
        source: 'internal' as any,
        completeness: 1,
        timeliness: 1,
        accuracy: 1,
        consistency: 1,
        gaps: 0,
        duplicates: 0,
        outOfOrder: 0,
        staleDataCount: 0,
        qualityScore: 100
      });
    }

    const metrics = this.qualityMetrics.get(symbol)!;
    (metrics as any)[metric] = ((metrics as any)[metric] || 0) + value;
  }

  private updateMetrics(message: StreamMessage): void {
    // Update stream metrics based on processed message
    const subscription = Array.from(this.subscriptions.values())
      .find(sub => sub.symbol === message.symbol);

    if (subscription) {
      const metrics = this.streamMetrics.get(subscription.id);
      if (metrics) {
        (metrics as any).totalMessages++;
        this.streamMetrics.set(subscription.id, metrics);
      }
    }
  }

  // ==========================================
  // ERROR HANDLING
  // ==========================================

  private handleConnectionError(sourceId: string, error: any): void {
    const streamError: StreamError = {
      id: this.generateId(),
      type: 'connection_failed',
      message: error.message || 'Connection error',
      timestamp: Date.now(),
      source: sourceId as any,
      retryable: true,
      context: { sourceId, error }
    };

    this.errors$.next(streamError);
    this.emit('error', streamError);
    this.log('error', `Connection error for ${sourceId}`, error);
  }

  private handleConnectionClosed(sourceId: string): void {
    this.emit('connection_lost', { sourceId });
    this.log('warn', `Connection closed: ${sourceId}`);
  }

  private handleMessageError(sourceId: string, rawMessage: any, error: any): void {
    const streamError: StreamError = {
      id: this.generateId(),
      type: 'data_corruption',
      message: error.message || 'Message processing error',
      timestamp: Date.now(),
      source: sourceId as any,
      retryable: false,
      context: { sourceId, rawMessage, error }
    };

    this.errors$.next(streamError);
    this.log('error', `Message error for ${sourceId}`, error);
  }

  private handleProcessingError(error: any): void {
    const streamError: StreamError = {
      id: this.generateId(),
      type: 'protocol_error',
      message: error.message || 'Processing error',
      timestamp: Date.now(),
      source: 'internal',
      retryable: true,
      context: { error }
    };

    this.errors$.next(streamError);
    this.log('error', 'Processing error', error);
  }

  // ==========================================
  // VALIDATION METHODS
  // ==========================================

  private isValidTick(tick: MarketTick): boolean {
    return !!(
      tick &&
      tick.symbol &&
      typeof tick.timestamp === 'number' &&
      typeof tick.last === 'number' &&
      tick.last > 0
    );
  }

  private isValidOrderBook(book: OrderBook): boolean {
    return !!(
      book &&
      book.symbol &&
      Array.isArray(book.bids) &&
      Array.isArray(book.asks) &&
      book.bids.length > 0 &&
      book.asks.length > 0
    );
  }

  private isValidTrade(trade: Trade): boolean {
    return !!(
      trade &&
      trade.symbol &&
      typeof trade.timestamp === 'number' &&
      typeof trade.price === 'number' &&
      typeof trade.size === 'number' &&
      trade.price > 0 &&
      trade.size > 0
    );
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  private hashMessage(input: string): string {
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }

  private generateId(): string {
    return `stream-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }

  private log(level: string, message: string, data?: any): void {
    const logFn = level === 'error' ? console.error :
                  level === 'warn' ? console.warn :
                  level === 'debug' ? console.debug :
                  console.log;

    logFn(`[MarketDataStream] ${message}`, data || '');
  }

  // ==========================================
  // CLEANUP
  // ==========================================

  public destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    // Clean up timers
    this.throttleTimers.forEach(timer => clearInterval(timer));
    this.throttleTimers.clear();

    // Close all connections
    this.connections.forEach(connection => connection.destroy());
    this.connections.clear();

    // Terminate workers
    this.workers.forEach(worker => worker.terminate());
    this.workers.length = 0;

    // Clear data structures
    this.sources.clear();
    this.subscriptions.clear();
    this.messageBuffers.clear();
    this.streamMetrics.clear();
    this.qualityMetrics.clear();
    this.duplicateDetector.clear();
    this.sequenceTracker.clear();
    this.latencyTracker.clear();

    // Complete observables
    this.connectionState$.complete();
    this.rawMessages$.complete();
    this.processedMessages$.complete();
    this.errors$.complete();
    this.events$.complete();
    this.metrics$.complete();

    this.removeAllListeners();
    this.log('info', 'MarketDataStream destroyed');
  }
}