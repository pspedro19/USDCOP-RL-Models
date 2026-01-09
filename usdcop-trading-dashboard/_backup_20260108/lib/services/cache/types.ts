// Cache Types for USDCOP Trading System

export interface CacheConfig {
  ttl: number; // Time to live in seconds
  maxSize?: number; // Maximum number of entries
  namespace?: string; // Key prefix for namespacing
}

export interface CacheEntry<T> {
  value: T;
  timestamp: number;
  expiresAt: number;
}

export interface SignalCacheData {
  signal_id: string;
  timestamp: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  features: Record<string, number>;
  model_version: string;
  execution_latency_ms: number;
}

export interface PositionCacheData {
  position_id: string;
  symbol: string;
  entry_price: number;
  current_price: number;
  quantity: number;
  pnl: number;
  pnl_percent: number;
  entry_time: string;
  status: 'OPEN' | 'CLOSED';
}

export interface MetricsCacheData {
  timestamp: string;
  total_pnl: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  active_positions: number;
  portfolio_value: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  hit_rate: number;
  memory_usage_mb: number;
}

export interface ICacheClient {
  get<T>(key: string): Promise<T | null>;
  set<T>(key: string, value: T, ttl?: number): Promise<void>;
  delete(key: string): Promise<void>;
  exists(key: string): Promise<boolean>;
  ttl(key: string): Promise<number>;
  keys(pattern: string): Promise<string[]>;
  clear(): Promise<void>;
  getStats(): Promise<CacheStats>;
}

export interface ISignalCache {
  setLatest(signal: SignalCacheData): Promise<void>;
  getLatest(): Promise<SignalCacheData | null>;
  addToHistory(symbol: string, signal: SignalCacheData): Promise<void>;
  getHistory(symbol: string, limit?: number): Promise<SignalCacheData[]>;
  clearHistory(symbol: string): Promise<void>;
}

export interface IMetricsCache {
  setFinancialMetrics(metrics: MetricsCacheData): Promise<void>;
  getFinancialMetrics(): Promise<MetricsCacheData | null>;
  setCustomMetric(key: string, value: number | string): Promise<void>;
  getCustomMetric(key: string): Promise<number | string | null>;
  clearMetrics(): Promise<void>;
}
