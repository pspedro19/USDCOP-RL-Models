/**
 * Trading Platform Configuration Management
 * Elite Trading Platform Core Configuration
 */

import type {
  EnvironmentConfig,
  DatabaseConfig,
  RedisConfig
} from '../types';

import type { DataBusConfig } from '../data-bus';
import type { EventManagerConfig } from '../event-manager';
import type { WebSocketManagerConfig } from '../websocket';
import type { PerformanceMonitorConfig } from '../performance';

export interface TradingPlatformConfig {
  readonly environment: EnvironmentConfig;
  readonly dataBus: DataBusConfig;
  readonly eventManager: EventManagerConfig;
  readonly webSocket: WebSocketManagerConfig;
  readonly performance: PerformanceMonitorConfig;
  readonly features: FeatureFlags;
  readonly ui: UIConfig;
  readonly trading: TradingConfig;
  readonly market: MarketConfig;
}

export interface FeatureFlags {
  readonly realTimeData: boolean;
  readonly backtesting: boolean;
  readonly analytics: boolean;
  readonly alerts: boolean;
  readonly paperTrading: boolean;
  readonly advancedCharting: boolean;
  readonly aiPredictions: boolean;
  readonly riskManagement: boolean;
  readonly portfolioAnalytics: boolean;
  readonly socialTrading: boolean;
}

export interface UIConfig {
  readonly theme: 'light' | 'dark' | 'auto';
  readonly chartRefreshRate: number;
  readonly maxChartHistory: number;
  readonly enableAnimations: boolean;
  readonly enableNotifications: boolean;
  readonly autoSaveLayouts: boolean;
  readonly defaultLayout: string;
  readonly maxOpenCharts: number;
  readonly enableVirtualization: boolean;
}

export interface TradingConfig {
  readonly defaultOrderType: 'market' | 'limit';
  readonly defaultTimeInForce: 'GTC' | 'IOC' | 'FOK';
  readonly enableOrderConfirmation: boolean;
  readonly maxOrderSize: number;
  readonly minOrderSize: number;
  readonly defaultLeverage: number;
  readonly maxLeverage: number;
  readonly enableStopLoss: boolean;
  readonly enableTakeProfit: boolean;
  readonly riskLimits: RiskLimitsConfig;
}

export interface RiskLimitsConfig {
  readonly maxDailyLoss: number;
  readonly maxPositionSize: number;
  readonly maxDrawdown: number;
  readonly marginCallLevel: number;
  readonly stopOutLevel: number;
  readonly maxOpenPositions: number;
}

export interface MarketConfig {
  readonly defaultSymbol: string;
  readonly subscribedSymbols: string[];
  readonly defaultTimeframe: string;
  readonly supportedTimeframes: string[];
  readonly dataProviders: DataProviderConfig[];
  readonly enableOrderBook: boolean;
  readonly orderBookDepth: number;
  readonly enableTrades: boolean;
  readonly maxTradeHistory: number;
}

export interface DataProviderConfig {
  readonly id: string;
  readonly name: string;
  readonly url: string;
  readonly apiKey?: string;
  readonly secret?: string;
  readonly enabled: boolean;
  readonly priority: number;
  readonly rateLimit: number;
  readonly timeout: number;
}

// Environment-specific configurations
const developmentConfig: Partial<TradingPlatformConfig> = {
  environment: {
    nodeEnv: 'development',
    apiUrl: '/api/proxy/trading',
    wsUrl: 'ws://localhost:8080',
    database: {
      host: 'localhost',
      port: 5432,
      database: 'trading_dev',
      username: 'dev_user',
      password: 'dev_password',
      ssl: false,
      pool: { min: 2, max: 10, idle: 10000 }
    },
    redis: {
      host: 'localhost',
      port: 6379,
      db: 0,
      keyPrefix: 'trading:dev:',
      ttl: 300
    },
    logging: {
      level: 'debug',
      format: 'text'
    },
    features: {
      realTimeData: true,
      backtesting: true,
      analytics: true,
      alerts: true
    }
  },
  dataBus: {
    maxCacheSize: 5000,
    defaultTTL: 300000,
    enablePersistence: false,
    compressionThreshold: 1024,
    batchSize: 50,
    flushInterval: 1000
  },
  eventManager: {
    maxListeners: 500,
    enableLogging: true,
    logLevel: 'debug',
    enableMetrics: true,
    metricsInterval: 5000,
    enablePersistence: false,
    maxEventHistory: 5000,
    enableDeduplication: true,
    deduplicationWindow: 1000
  },
  performance: {
    enableCPUMonitoring: true,
    enableMemoryMonitoring: true,
    enableNetworkMonitoring: true,
    enableRenderMonitoring: true,
    samplingInterval: 1000,
    alertThresholds: {
      cpuUsage: 90,
      memoryUsage: 90,
      renderTime: 20,
      fps: 45
    },
    retentionPeriod: 600000,
    enableLogging: true
  }
};

const productionConfig: Partial<TradingPlatformConfig> = {
  environment: {
    nodeEnv: 'production',
    apiUrl: process.env.NEXT_PUBLIC_API_URL || 'https://api.tradingplatform.com',
    wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'wss://ws.tradingplatform.com',
    database: {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'trading',
      username: process.env.DB_USER || 'trader',
      password: process.env.DB_PASSWORD || '',
      ssl: true,
      pool: { min: 5, max: 20, idle: 10000 }
    },
    redis: {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
      db: 0,
      keyPrefix: 'trading:prod:',
      ttl: 300
    },
    logging: {
      level: 'info',
      format: 'json'
    },
    features: {
      realTimeData: true,
      backtesting: true,
      analytics: true,
      alerts: true
    }
  },
  dataBus: {
    maxCacheSize: 20000,
    defaultTTL: 300000,
    enablePersistence: true,
    compressionThreshold: 512,
    batchSize: 100,
    flushInterval: 500
  },
  eventManager: {
    maxListeners: 2000,
    enableLogging: false,
    logLevel: 'error',
    enableMetrics: true,
    metricsInterval: 10000,
    enablePersistence: true,
    maxEventHistory: 50000,
    enableDeduplication: true,
    deduplicationWindow: 1000
  },
  performance: {
    enableCPUMonitoring: true,
    enableMemoryMonitoring: true,
    enableNetworkMonitoring: true,
    enableRenderMonitoring: false,
    samplingInterval: 5000,
    alertThresholds: {
      cpuUsage: 80,
      memoryUsage: 85,
      renderTime: 16,
      fps: 50
    },
    retentionPeriod: 1800000,
    enableLogging: false
  }
};

// Default base configuration
const baseConfig: TradingPlatformConfig = {
  environment: {
    nodeEnv: 'development',
    apiUrl: '/api/proxy/trading',
    wsUrl: 'ws://localhost:8080',
    database: {
      host: 'localhost',
      port: 5432,
      database: 'trading',
      username: 'trader',
      password: 'password',
      ssl: false,
      pool: { min: 2, max: 10, idle: 10000 }
    },
    redis: {
      host: 'localhost',
      port: 6379,
      db: 0,
      keyPrefix: 'trading:',
      ttl: 300
    },
    logging: {
      level: 'info',
      format: 'text'
    },
    features: {
      realTimeData: true,
      backtesting: true,
      analytics: true,
      alerts: true
    }
  },
  dataBus: {
    maxCacheSize: 10000,
    defaultTTL: 300000,
    enablePersistence: false,
    compressionThreshold: 1024,
    batchSize: 100,
    flushInterval: 1000
  },
  eventManager: {
    maxListeners: 1000,
    enableLogging: false,
    logLevel: 'info',
    enableMetrics: true,
    metricsInterval: 5000,
    enablePersistence: false,
    maxEventHistory: 10000,
    enableDeduplication: true,
    deduplicationWindow: 1000
  },
  webSocket: {
    url: 'ws://localhost:8080',
    protocols: ['trading-protocol'],
    maxReconnectAttempts: 10,
    reconnectDelay: 5000,
    heartbeatInterval: 30000,
    connectionTimeout: 10000,
    messageTimeout: 5000,
    maxMessageQueue: 1000,
    enableCompression: true,
    enableLogging: false,
    autoReconnect: true
  },
  performance: {
    enableCPUMonitoring: true,
    enableMemoryMonitoring: true,
    enableNetworkMonitoring: true,
    enableRenderMonitoring: true,
    samplingInterval: 1000,
    alertThresholds: {
      cpuUsage: 80,
      memoryUsage: 85,
      renderTime: 16,
      fps: 50
    },
    retentionPeriod: 300000,
    enableLogging: false
  },
  features: {
    realTimeData: true,
    backtesting: true,
    analytics: true,
    alerts: true,
    paperTrading: true,
    advancedCharting: true,
    aiPredictions: false,
    riskManagement: true,
    portfolioAnalytics: true,
    socialTrading: false
  },
  ui: {
    theme: 'dark',
    chartRefreshRate: 1000,
    maxChartHistory: 10000,
    enableAnimations: true,
    enableNotifications: true,
    autoSaveLayouts: true,
    defaultLayout: 'terminal',
    maxOpenCharts: 6,
    enableVirtualization: true
  },
  trading: {
    defaultOrderType: 'limit',
    defaultTimeInForce: 'GTC',
    enableOrderConfirmation: true,
    maxOrderSize: 1000000,
    minOrderSize: 0.001,
    defaultLeverage: 1,
    maxLeverage: 100,
    enableStopLoss: true,
    enableTakeProfit: true,
    riskLimits: {
      maxDailyLoss: 5000,
      maxPositionSize: 100000,
      maxDrawdown: 10000,
      marginCallLevel: 100,
      stopOutLevel: 50,
      maxOpenPositions: 20
    }
  },
  market: {
    defaultSymbol: 'USDCOP',
    subscribedSymbols: ['USDCOP', 'EURUSD', 'GBPUSD', 'USDJPY'],
    defaultTimeframe: '5m',
    supportedTimeframes: ['1s', '5s', '15s', '30s', '1m', '5m', '15m', '30m', '1h', '4h', '1d'],
    dataProviders: [
      {
        id: 'twelvedata',
        name: 'Twelve Data',
        url: 'https://api.twelvedata.com',
        enabled: true,
        priority: 1,
        rateLimit: 800,
        timeout: 5000
      }
    ],
    enableOrderBook: true,
    orderBookDepth: 20,
    enableTrades: true,
    maxTradeHistory: 1000
  }
};

/**
 * Configuration Manager Class
 */
export class ConfigManager {
  private static instance: ConfigManager;
  private config: TradingPlatformConfig;

  private constructor() {
    this.config = this.loadConfiguration();
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  private loadConfiguration(): TradingPlatformConfig {
    const env = process.env.NODE_ENV || 'development';

    let envConfig: Partial<TradingPlatformConfig> = {};

    switch (env) {
      case 'production':
        envConfig = productionConfig;
        break;
      case 'development':
        envConfig = developmentConfig;
        break;
      default:
        envConfig = developmentConfig;
    }

    return this.mergeConfig(baseConfig, envConfig);
  }

  private mergeConfig(
    base: TradingPlatformConfig,
    override: Partial<TradingPlatformConfig>
  ): TradingPlatformConfig {
    return {
      ...base,
      ...override,
      environment: { ...base.environment, ...override.environment },
      dataBus: { ...base.dataBus, ...override.dataBus },
      eventManager: { ...base.eventManager, ...override.eventManager },
      webSocket: { ...base.webSocket, ...override.webSocket },
      performance: { ...base.performance, ...override.performance },
      features: { ...base.features, ...override.features },
      ui: { ...base.ui, ...override.ui },
      trading: { ...base.trading, ...override.trading },
      market: { ...base.market, ...override.market }
    };
  }

  public getConfig(): TradingPlatformConfig {
    return { ...this.config };
  }

  public updateConfig(updates: Partial<TradingPlatformConfig>): void {
    this.config = this.mergeConfig(this.config, updates);
  }

  public getEnvironmentConfig(): EnvironmentConfig {
    return { ...this.config.environment };
  }

  public getDataBusConfig(): DataBusConfig {
    return { ...this.config.dataBus };
  }

  public getEventManagerConfig(): EventManagerConfig {
    return { ...this.config.eventManager };
  }

  public getWebSocketConfig(): WebSocketManagerConfig {
    return { ...this.config.webSocket };
  }

  public getPerformanceConfig(): PerformanceMonitorConfig {
    return { ...this.config.performance };
  }

  public getFeatureFlags(): FeatureFlags {
    return { ...this.config.features };
  }

  public isFeatureEnabled(feature: keyof FeatureFlags): boolean {
    return this.config.features[feature];
  }

  public validateConfig(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate required URLs
    if (!this.config.environment.apiUrl) {
      errors.push('API URL is required');
    }

    if (!this.config.environment.wsUrl) {
      errors.push('WebSocket URL is required');
    }

    // Validate database config
    if (!this.config.environment.database.host) {
      errors.push('Database host is required');
    }

    // Validate thresholds
    if (this.config.performance.alertThresholds.cpuUsage > 100) {
      errors.push('CPU usage threshold cannot exceed 100%');
    }

    if (this.config.performance.alertThresholds.memoryUsage > 100) {
      errors.push('Memory usage threshold cannot exceed 100%');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

// Export singleton instance
export const configManager = ConfigManager.getInstance();

// Export convenience functions
export function getConfig(): TradingPlatformConfig {
  return configManager.getConfig();
}

export function getEnvironmentConfig(): EnvironmentConfig {
  return configManager.getEnvironmentConfig();
}

export function isFeatureEnabled(feature: keyof FeatureFlags): boolean {
  return configManager.isFeatureEnabled(feature);
}