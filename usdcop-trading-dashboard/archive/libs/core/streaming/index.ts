/**
 * Streaming Infrastructure - Export Index
 * Complete high-performance streaming data infrastructure for trading platforms
 */

// Core streaming infrastructure
export { StreamingInfrastructure, createStreamingInfrastructure } from './StreamingInfrastructure';
export type { StreamingInfrastructureConfig, SystemHealth, SystemMetrics, ComponentHealth } from './StreamingInfrastructure';

// Market data streaming
export { MarketDataStream } from './MarketDataStream';
export type { MarketDataStreamConfig } from './MarketDataStream';

// Data normalization
export { DataNormalizer } from './DataNormalizer';
export type { NormalizationConfig, NormalizationResult } from './DataNormalizer';

// Buffering and throttling
export { StreamBuffer, ThrottleManager, Throttle } from './StreamBuffer';
export type { BufferMetrics, ThrottleMetrics } from './StreamBuffer';

// Reconnection management
export { ReconnectManager } from './ReconnectManager';
export type { ReconnectManagerConfig, ReconnectAttempt, ReconnectMetrics } from './ReconnectManager';

// Tick processing and aggregation
export { TickProcessor } from './TickProcessor';
export type {
  TickAggregationConfig,
  AggregatedTick,
  TickPattern,
  VolumeProfile,
  MicrostructureMetrics
} from './TickProcessor';

// Order book management
export { OrderBookManager } from './OrderBookManager';
export type {
  OrderBookConfig,
  OrderBookSnapshot,
  OrderBookUpdate,
  OrderBookAnalytics,
  OrderBookGap,
  OrderBookValidation
} from './OrderBookManager';

// Caching system
export { CacheManager } from './CacheManager';
export type { CacheQuery, CacheStats, CacheOperation } from './CacheManager';

// Type definitions
export type {
  // Core types
  StreamSource,
  StreamSubscription,
  StreamMessage,
  StreamMetrics,
  StreamError,
  StreamEvent,

  // Configuration types
  ThrottleConfig,
  BufferConfig,
  ReconnectConfig,
  CacheConfig,

  // Data types
  StreamDataType,
  StreamBuffer as IStreamBuffer,
  StreamQuality,
  ConnectionState,

  // Performance types
  PerformanceMonitor,
  DataQualityMetrics,

  // Exchange types
  ExchangeType,
  ExchangeConfig,

  // Processing types
  WorkerMessage,
  ProcessingResult,

  // Storage types
  CacheEntry,
  StorageMetrics
} from '../types/streaming-types';

// Market data types (re-export for convenience)
export type {
  MarketTick,
  OrderBook,
  OrderBookLevel,
  Trade,
  OHLCV,
  DataSource,
  DataQuality,
  TimeInterval,
  TradeSide,
  MarketStatus
} from '../types/market-data';

// Utility functions
export const StreamingUtils = {
  /**
   * Create a basic streaming configuration for forex trading
   */
  createForexConfig(): Partial<StreamingInfrastructureConfig> {
    return {
      tickAggregation: {
        intervals: ['1s', '5s', '15s', '30s', '1m', '5m', '15m', '30m', '1h'],
        enableVWAP: true,
        enableVolumeDelta: true,
        enableMicrostructure: true,
        maxTicksPerSecond: 1000,
        enablePatternDetection: true,
        enableRealTimeIndicators: true
      },
      throttle: {
        enabled: true,
        maxUpdatesPerSecond: 60,
        burstLimit: 100,
        windowSize: 1000,
        strategy: 'merge'
      },
      normalization: {
        baseTimezone: 'UTC',
        baseCurrency: 'USD',
        precision: {
          price: 4,
          volume: 0,
          percentage: 2
        },
        validation: {
          enablePriceValidation: true,
          enableVolumeValidation: false,
          maxPriceDeviation: 1000,
          minVolume: 0
        }
      }
    };
  },

  /**
   * Create a basic streaming configuration for crypto trading
   */
  createCryptoConfig(): Partial<StreamingInfrastructureConfig> {
    return {
      tickAggregation: {
        intervals: ['1s', '5s', '15s', '30s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'],
        enableVWAP: true,
        enableVolumeDelta: true,
        enableMicrostructure: true,
        maxTicksPerSecond: 2000,
        enablePatternDetection: true,
        enableRealTimeIndicators: true
      },
      throttle: {
        enabled: true,
        maxUpdatesPerSecond: 100,
        burstLimit: 200,
        windowSize: 1000,
        strategy: 'drop_oldest'
      },
      orderBook: {
        maxLevels: 100,
        enableValidation: true,
        enableGapDetection: true,
        enableAnalytics: true,
        checksumValidation: true,
        autoReconstruct: true,
        reconstructionTimeout: 3000,
        priceToleranceBps: 10000,
        volumeToleranceBps: 50000
      }
    };
  },

  /**
   * Create a high-frequency trading configuration
   */
  createHFTConfig(): Partial<StreamingInfrastructureConfig> {
    return {
      throttle: {
        enabled: false, // No throttling for HFT
        maxUpdatesPerSecond: 10000,
        burstLimit: 1000,
        windowSize: 100,
        strategy: 'drop_newest'
      },
      buffer: {
        enabled: true,
        maxSize: 10000,
        maxAge: 60000, // 1 minute
        compressionEnabled: false, // No compression for speed
        persistToDisk: false,
        strategy: 'fifo'
      },
      tickAggregation: {
        intervals: ['1s', '5s', '15s', '30s', '1m'],
        enableVWAP: true,
        enableVolumeDelta: true,
        enableMicrostructure: true,
        maxTicksPerSecond: 10000,
        enablePatternDetection: false, // Disable for performance
        enableRealTimeIndicators: false
      },
      cache: {
        enabled: false, // Disable cache for maximum speed
        maxSizeBytes: 0,
        maxAge: 0,
        compression: false,
        persistToDisk: false,
        evictionStrategy: 'lru',
        indexingEnabled: false
      }
    };
  },

  /**
   * Generate a unique stream source ID
   */
  generateSourceId(exchangeName: string, dataType: string): string {
    return `${exchangeName}_${dataType}_${Date.now()}_${Math.random().toString(36).substring(2)}`;
  },

  /**
   * Create a standard stream source configuration
   */
  createStreamSource(
    name: string,
    type: ExchangeType,
    wsUrl: string,
    supportedSymbols: string[]
  ): StreamSource {
    return {
      id: this.generateSourceId(name, type),
      name,
      type,
      baseUrl: '',
      wsUrl,
      isActive: true,
      priority: 1,
      rateLimit: 1000,
      supportedSymbols,
      supportedDataTypes: ['tick', 'orderbook', 'trade'],
      reconnectConfig: {
        enabled: true,
        maxAttempts: 5,
        initialDelay: 1000,
        maxDelay: 30000,
        backoffMultiplier: 2,
        jitter: true
      }
    };
  },

  /**
   * Validate streaming configuration
   */
  validateConfig(config: Partial<StreamingInfrastructureConfig>): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate throttle configuration
    if (config.throttle?.enabled && config.throttle.maxUpdatesPerSecond <= 0) {
      errors.push('Throttle maxUpdatesPerSecond must be positive');
    }

    // Validate buffer configuration
    if (config.buffer?.enabled && config.buffer.maxSize <= 0) {
      errors.push('Buffer maxSize must be positive');
    }

    // Validate cache configuration
    if (config.cache?.enabled && config.cache.maxSizeBytes <= 0) {
      errors.push('Cache maxSizeBytes must be positive');
    }

    // Validate order book configuration
    if (config.orderBook?.maxLevels && config.orderBook.maxLevels <= 0) {
      errors.push('OrderBook maxLevels must be positive');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  },

  /**
   * Performance optimization recommendations
   */
  getPerformanceRecommendations(currentConfig: Partial<StreamingInfrastructureConfig>): string[] {
    const recommendations: string[] = [];

    // Check throttling
    if (!currentConfig.throttle?.enabled) {
      recommendations.push('Consider enabling throttling to prevent UI overload');
    } else if (currentConfig.throttle.maxUpdatesPerSecond > 100) {
      recommendations.push('Consider reducing maxUpdatesPerSecond for better UI performance');
    }

    // Check buffering
    if (currentConfig.buffer?.maxSize && currentConfig.buffer.maxSize > 5000) {
      recommendations.push('Large buffer sizes may consume excessive memory');
    }

    // Check caching
    if (currentConfig.cache?.enabled && !currentConfig.cache.compression) {
      recommendations.push('Enable compression to reduce cache memory usage');
    }

    // Check aggregation
    if (currentConfig.tickAggregation?.intervals && currentConfig.tickAggregation.intervals.length > 10) {
      recommendations.push('Too many aggregation intervals may impact performance');
    }

    return recommendations;
  }
};

// Default export
export default StreamingInfrastructure;