/**
 * Enhanced Streaming Data Types - Elite Trading Platform
 * High-performance, type-safe streaming infrastructure for market data
 */

import type { MarketTick, OrderBook, Trade, DataSource, DataQuality, TimeInterval } from './market-data';

// ==========================================
// CORE STREAMING INTERFACES
// ==========================================

export interface StreamSource {
  readonly id: string;
  readonly name: string;
  readonly type: ExchangeType;
  readonly baseUrl: string;
  readonly wsUrl: string;
  readonly apiKey?: string;
  readonly isActive: boolean;
  readonly priority: number;
  readonly rateLimit: number;
  readonly supportedSymbols: readonly string[];
  readonly supportedDataTypes: readonly StreamDataType[];
  readonly reconnectConfig: ReconnectConfig;
}

export interface StreamSubscription {
  readonly id: string;
  readonly symbol: string;
  readonly dataType: StreamDataType;
  readonly sourceId: string;
  readonly interval?: TimeInterval;
  readonly filters?: StreamFilters;
  readonly isActive: boolean;
  readonly createdAt: number;
  readonly lastUpdate?: number;
}

export interface StreamMessage<T = any> {
  readonly id: string;
  readonly type: StreamMessageType;
  readonly symbol: string;
  readonly data: T;
  readonly timestamp: number;
  readonly source: DataSource;
  readonly quality: DataQuality;
  readonly latency?: number;
  readonly sequence?: number;
}

export interface StreamBuffer<T = any> {
  readonly id: string;
  readonly maxSize: number;
  readonly currentSize: number;
  readonly data: readonly T[];
  readonly oldestTimestamp: number;
  readonly newestTimestamp: number;
  readonly isCircular: boolean;
}

// ==========================================
// ENUMS AND UNION TYPES
// ==========================================

export type ExchangeType =
  | 'forex'
  | 'crypto'
  | 'stocks'
  | 'commodities'
  | 'internal'
  | 'simulation';

export type StreamDataType =
  | 'tick'
  | 'orderbook'
  | 'trade'
  | 'ohlcv'
  | 'news'
  | 'sentiment'
  | 'economic';

export type StreamMessageType =
  | 'subscribe'
  | 'unsubscribe'
  | 'data'
  | 'error'
  | 'heartbeat'
  | 'reconnect'
  | 'status';

export type StreamQuality =
  | 'excellent'
  | 'good'
  | 'fair'
  | 'poor'
  | 'disconnected';

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'error'
  | 'suspended';

// ==========================================
// CONFIGURATION INTERFACES
// ==========================================

export interface ReconnectConfig {
  readonly enabled: boolean;
  readonly maxAttempts: number;
  readonly initialDelay: number;
  readonly maxDelay: number;
  readonly backoffMultiplier: number;
  readonly jitter: boolean;
}

export interface ThrottleConfig {
  readonly enabled: boolean;
  readonly maxUpdatesPerSecond: number;
  readonly burstLimit: number;
  readonly windowSize: number;
  readonly strategy: ThrottleStrategy;
}

export interface BufferConfig {
  readonly enabled: boolean;
  readonly maxSize: number;
  readonly maxAge: number;
  readonly compressionEnabled: boolean;
  readonly persistToDisk: boolean;
  readonly strategy: BufferStrategy;
}

export interface StreamFilters {
  readonly minPrice?: number;
  readonly maxPrice?: number;
  readonly minVolume?: number;
  readonly symbols?: readonly string[];
  readonly dataTypes?: readonly StreamDataType[];
  readonly sources?: readonly string[];
}

export type ThrottleStrategy = 'drop_oldest' | 'drop_newest' | 'merge' | 'sample';
export type BufferStrategy = 'fifo' | 'lifo' | 'priority' | 'time_based';

// ==========================================
// PERFORMANCE & MONITORING
// ==========================================

export interface StreamMetrics {
  readonly streamId: string;
  readonly sourceId: string;
  readonly symbol: string;
  readonly startTime: number;
  readonly uptime: number;
  readonly totalMessages: number;
  readonly messagesPerSecond: number;
  readonly averageLatency: number;
  readonly minLatency: number;
  readonly maxLatency: number;
  readonly errorCount: number;
  readonly errorRate: number;
  readonly reconnections: number;
  readonly droppedMessages: number;
  readonly bufferUtilization: number;
  readonly memoryUsage: number;
  readonly quality: StreamQuality;
}

export interface DataQualityMetrics {
  readonly symbol: string;
  readonly source: DataSource;
  readonly completeness: number; // 0-1
  readonly timeliness: number; // 0-1
  readonly accuracy: number; // 0-1
  readonly consistency: number; // 0-1
  readonly gaps: number;
  readonly duplicates: number;
  readonly outOfOrder: number;
  readonly staleDataCount: number;
  readonly qualityScore: number; // 0-100
}

export interface PerformanceMonitor {
  readonly cpu: number;
  readonly memory: number;
  readonly network: number;
  readonly disk: number;
  readonly activeConnections: number;
  readonly buffersInUse: number;
  readonly workersActive: number;
  readonly timestamp: number;
}

// ==========================================
// EXCHANGE-SPECIFIC TYPES
// ==========================================

export interface ExchangeConfig {
  readonly name: string;
  readonly type: ExchangeType;
  readonly endpoints: ExchangeEndpoints;
  readonly authentication?: ExchangeAuth;
  readonly rateLimit: RateLimit;
  readonly dataFormat: DataFormat;
  readonly supportedSymbols: readonly string[];
  readonly features: ExchangeFeatures;
}

export interface ExchangeEndpoints {
  readonly rest: string;
  readonly websocket: string;
  readonly streaming?: string;
  readonly historical?: string;
}

export interface ExchangeAuth {
  readonly type: 'api_key' | 'oauth' | 'jwt' | 'none';
  readonly credentials: Record<string, string>;
  readonly headers?: Record<string, string>;
}

export interface RateLimit {
  readonly requestsPerSecond: number;
  readonly requestsPerMinute: number;
  readonly burstLimit: number;
  readonly cooldownPeriod: number;
}

export interface DataFormat {
  readonly messageFormat: 'json' | 'msgpack' | 'protobuf' | 'avro';
  readonly compression: 'none' | 'gzip' | 'deflate' | 'brotli';
  readonly timestampFormat: 'unix' | 'iso' | 'custom';
  readonly priceFormat: 'string' | 'number' | 'fixed';
}

export interface ExchangeFeatures {
  readonly supportsOrderBook: boolean;
  readonly supportsFullDepth: boolean;
  readonly supportsTrades: boolean;
  readonly supportsOHLCV: boolean;
  readonly supportsNews: boolean;
  readonly supportsReconnect: boolean;
  readonly supportsHeartbeat: boolean;
  readonly maxSymbolsPerConnection: number;
}

// ==========================================
// ERROR HANDLING & EVENTS
// ==========================================

export interface StreamError {
  readonly id: string;
  readonly type: StreamErrorType;
  readonly message: string;
  readonly code?: string | number;
  readonly timestamp: number;
  readonly source: DataSource;
  readonly symbol?: string;
  readonly retryable: boolean;
  readonly context?: Record<string, any>;
}

export type StreamErrorType =
  | 'connection_failed'
  | 'authentication_failed'
  | 'rate_limit_exceeded'
  | 'invalid_symbol'
  | 'data_corruption'
  | 'timeout'
  | 'protocol_error'
  | 'server_error'
  | 'network_error'
  | 'buffer_overflow';

export interface StreamEvent {
  readonly id: string;
  readonly type: StreamEventType;
  readonly timestamp: number;
  readonly data: any;
  readonly context?: Record<string, any>;
}

export type StreamEventType =
  | 'stream_started'
  | 'stream_stopped'
  | 'connection_established'
  | 'connection_lost'
  | 'reconnection_started'
  | 'reconnection_success'
  | 'subscription_added'
  | 'subscription_removed'
  | 'data_received'
  | 'error_occurred'
  | 'buffer_full'
  | 'quality_degraded';

// ==========================================
// WORKER & PROCESSING TYPES
// ==========================================

export interface WorkerMessage {
  readonly id: string;
  readonly type: WorkerMessageType;
  readonly data: any;
  readonly timestamp: number;
}

export type WorkerMessageType =
  | 'process_tick'
  | 'aggregate_data'
  | 'calculate_indicators'
  | 'detect_patterns'
  | 'compress_buffer'
  | 'persist_data'
  | 'quality_check';

export interface ProcessingResult {
  readonly id: string;
  readonly type: WorkerMessageType;
  readonly result: any;
  readonly processingTime: number;
  readonly memoryUsed: number;
  readonly timestamp: number;
  readonly error?: string;
}

// ==========================================
// HISTORICAL DATA & REPLAY
// ==========================================

export interface ReplayConfig {
  readonly startDate: number;
  readonly endDate: number;
  readonly speed: number; // 1x = real-time, 2x = 2x speed, etc.
  readonly symbols: readonly string[];
  readonly dataTypes: readonly StreamDataType[];
  readonly filters?: StreamFilters;
  readonly loop: boolean;
}

export interface ReplayState {
  readonly isActive: boolean;
  readonly currentTime: number;
  readonly progress: number; // 0-1
  readonly speed: number;
  readonly remainingTime: number;
  readonly totalEvents: number;
  readonly processedEvents: number;
}

export interface HistoricalDataRange {
  readonly symbol: string;
  readonly dataType: StreamDataType;
  readonly startTime: number;
  readonly endTime: number;
  readonly count: number;
  readonly sizeBytes: number;
  readonly compressed: boolean;
}

// ==========================================
// CACHE & STORAGE TYPES
// ==========================================

export interface CacheConfig {
  readonly enabled: boolean;
  readonly maxSizeBytes: number;
  readonly maxAge: number;
  readonly compression: boolean;
  readonly persistToDisk: boolean;
  readonly evictionStrategy: CacheEvictionStrategy;
  readonly indexingEnabled: boolean;
}

export type CacheEvictionStrategy = 'lru' | 'lfu' | 'ttl' | 'size_based';

export interface CacheEntry<T = any> {
  readonly key: string;
  readonly data: T;
  readonly timestamp: number;
  readonly accessCount: number;
  readonly lastAccess: number;
  readonly sizeBytes: number;
  readonly compressed: boolean;
  readonly ttl?: number;
}

export interface StorageMetrics {
  readonly totalSize: number;
  readonly usedSize: number;
  readonly availableSize: number;
  readonly entryCount: number;
  readonly hitRate: number;
  readonly missRate: number;
  readonly evictions: number;
  readonly compressionRatio: number;
}