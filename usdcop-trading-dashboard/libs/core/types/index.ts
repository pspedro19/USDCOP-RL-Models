/**
 * Core Types Barrel Export
 * Elite Trading Platform Type Definitions
 */

// Market Data Types
export * from './market-data';

// Trading Types
export * from './trading';

// Event Types
export * from './events';

// Common Utility Types
export interface ApiResponse<T = any> {
  readonly success: boolean;
  readonly data?: T;
  readonly error?: string;
  readonly timestamp: number;
  readonly requestId?: string;
}

export interface PaginatedResponse<T = any> extends ApiResponse<T[]> {
  readonly pagination: {
    readonly page: number;
    readonly limit: number;
    readonly total: number;
    readonly totalPages: number;
    readonly hasNext: boolean;
    readonly hasPrev: boolean;
  };
}

export interface CacheEntry<T = any> {
  readonly key: string;
  readonly value: T;
  readonly timestamp: number;
  readonly ttl: number;
  readonly hits: number;
  readonly size: number;
}

export interface WebSocketMessage<T = any> {
  readonly id: string;
  readonly type: string;
  readonly data: T;
  readonly timestamp: number;
  readonly channel?: string;
}

export interface SubscriptionRequest {
  readonly id: string;
  readonly channel: string;
  readonly symbols?: string[];
  readonly params?: Record<string, any>;
}

export interface StreamConfig {
  readonly url: string;
  readonly reconnectAttempts: number;
  readonly reconnectDelay: number;
  readonly heartbeatInterval: number;
  readonly subscriptions: SubscriptionRequest[];
  readonly authentication?: {
    readonly apiKey: string;
    readonly secret: string;
  };
}

// Configuration Types
export interface DatabaseConfig {
  readonly host: string;
  readonly port: number;
  readonly database: string;
  readonly username: string;
  readonly password: string;
  readonly ssl: boolean;
  readonly pool: {
    readonly min: number;
    readonly max: number;
    readonly idle: number;
  };
}

export interface RedisConfig {
  readonly host: string;
  readonly port: number;
  readonly password?: string;
  readonly db: number;
  readonly keyPrefix: string;
  readonly ttl: number;
}

export interface EnvironmentConfig {
  readonly nodeEnv: 'development' | 'production' | 'test';
  readonly apiUrl: string;
  readonly wsUrl: string;
  readonly database: DatabaseConfig;
  readonly redis: RedisConfig;
  readonly logging: {
    readonly level: 'debug' | 'info' | 'warn' | 'error';
    readonly format: 'json' | 'text';
  };
  readonly features: {
    readonly realTimeData: boolean;
    readonly backtesting: boolean;
    readonly analytics: boolean;
    readonly alerts: boolean;
  };
}

// Error Types
export interface TradingError extends Error {
  readonly code: string;
  readonly type: ErrorType;
  readonly context?: Record<string, any>;
  readonly timestamp: number;
  readonly severity: ErrorSeverity;
}

export type ErrorType = 'market_data' | 'trading' | 'connection' | 'validation' | 'system' | 'user';
export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

// Performance Types
export interface MemoryMetrics {
  readonly heapUsed: number;
  readonly heapTotal: number;
  readonly external: number;
  readonly rss: number;
}

export interface NetworkMetrics {
  readonly bytesReceived: number;
  readonly bytesSent: number;
  readonly packetsReceived: number;
  readonly packetsSent: number;
  readonly latency: number;
  readonly connectionCount: number;
}

export interface ApplicationMetrics {
  readonly uptime: number;
  readonly memory: MemoryMetrics;
  readonly network: NetworkMetrics;
  readonly activeConnections: number;
  readonly requestsPerSecond: number;
  readonly errorsPerMinute: number;
  readonly cacheHitRate: number;
}

export interface PerformanceAlert {
  readonly id: string;
  readonly type: 'cpu' | 'memory' | 'render' | 'network' | 'fps';
  readonly severity: 'warning' | 'critical';
  readonly message: string;
  readonly value: number;
  readonly threshold: number;
  readonly timestamp: number;
}

export interface RenderMetrics {
  readonly frameTime: number;
  readonly fps: number;
  readonly droppedFrames: number;
  readonly renderTime: number;
  readonly layoutTime: number;
  readonly paintTime: number;
}

// Utility Types
export type Mutable<T> = {
  -readonly [P in keyof T]: T[P];
};

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type RequiredKeys<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type OptionalKeys<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type Timestamp = number;
export type UUID = string;
export type Symbol = string;
export type Price = number;
export type Volume = number;
export type Percentage = number;