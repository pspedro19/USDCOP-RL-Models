// Redis Client with In-Memory Fallback for USDCOP Trading System

import type { ICacheClient, CacheEntry, CacheStats, CacheConfig } from './types';

/**
 * Redis Client with in-memory fallback
 * Uses Map-based implementation for development when Redis is not available
 */
export class RedisClient implements ICacheClient {
  private cache: Map<string, CacheEntry<any>>;
  private stats: { hits: number; misses: number };
  private config: Required<CacheConfig>;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<CacheConfig> = {}) {
    this.cache = new Map();
    this.stats = { hits: 0, misses: 0 };
    this.config = {
      ttl: config.ttl || 300, // Default 5 minutes
      maxSize: config.maxSize || 1000,
      namespace: config.namespace || 'usdcop',
    };

    // Start cleanup interval to remove expired entries
    this.startCleanup();
  }

  private getKey(key: string): string {
    return `${this.config.namespace}:${key}`;
  }

  private isExpired(entry: CacheEntry<any>): boolean {
    return Date.now() > entry.expiresAt;
  }

  private startCleanup(): void {
    // Clean up expired entries every minute
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 60000);
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }

  private enforceMaxSize(): void {
    if (this.cache.size > this.config.maxSize) {
      // Remove oldest entries (FIFO)
      const toRemove = this.cache.size - this.config.maxSize;
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < toRemove; i++) {
        this.cache.delete(keys[i]);
      }
    }
  }

  async get<T>(key: string): Promise<T | null> {
    const fullKey = this.getKey(key);
    const entry = this.cache.get(fullKey);

    if (!entry) {
      this.stats.misses++;
      return null;
    }

    if (this.isExpired(entry)) {
      this.cache.delete(fullKey);
      this.stats.misses++;
      return null;
    }

    this.stats.hits++;
    return entry.value as T;
  }

  async set<T>(key: string, value: T, ttl?: number): Promise<void> {
    const fullKey = this.getKey(key);
    const ttlSeconds = ttl || this.config.ttl;
    const now = Date.now();

    const entry: CacheEntry<T> = {
      value,
      timestamp: now,
      expiresAt: now + ttlSeconds * 1000,
    };

    this.cache.set(fullKey, entry);
    this.enforceMaxSize();
  }

  async delete(key: string): Promise<void> {
    const fullKey = this.getKey(key);
    this.cache.delete(fullKey);
  }

  async exists(key: string): Promise<boolean> {
    const fullKey = this.getKey(key);
    const entry = this.cache.get(fullKey);

    if (!entry) return false;
    if (this.isExpired(entry)) {
      this.cache.delete(fullKey);
      return false;
    }

    return true;
  }

  async ttl(key: string): Promise<number> {
    const fullKey = this.getKey(key);
    const entry = this.cache.get(fullKey);

    if (!entry || this.isExpired(entry)) {
      return -1;
    }

    const remainingMs = entry.expiresAt - Date.now();
    return Math.ceil(remainingMs / 1000);
  }

  async keys(pattern: string): Promise<string[]> {
    const fullPattern = this.getKey(pattern);
    const regex = new RegExp(
      '^' + fullPattern.replace(/\*/g, '.*').replace(/\?/g, '.') + '$'
    );

    return Array.from(this.cache.keys())
      .filter((key) => regex.test(key))
      .filter((key) => {
        const entry = this.cache.get(key);
        return entry && !this.isExpired(entry);
      })
      .map((key) => key.replace(`${this.config.namespace}:`, ''));
  }

  async clear(): Promise<void> {
    this.cache.clear();
    this.stats = { hits: 0, misses: 0 };
  }

  async getStats(): Promise<CacheStats> {
    const totalRequests = this.stats.hits + this.stats.misses;
    const hitRate = totalRequests > 0 ? this.stats.hits / totalRequests : 0;

    // Estimate memory usage (rough approximation)
    let memoryBytes = 0;
    for (const [key, entry] of this.cache.entries()) {
      memoryBytes += key.length * 2; // UTF-16
      memoryBytes += JSON.stringify(entry.value).length * 2;
      memoryBytes += 32; // Overhead for timestamps
    }

    return {
      hits: this.stats.hits,
      misses: this.stats.misses,
      size: this.cache.size,
      hit_rate: hitRate,
      memory_usage_mb: memoryBytes / (1024 * 1024),
    };
  }

  /**
   * List operations for signal history
   */
  async lpush(key: string, value: any): Promise<void> {
    const fullKey = this.getKey(key);
    const list = (await this.get<any[]>(key)) || [];
    list.unshift(value);
    await this.set(fullKey, list);
  }

  async lrange(key: string, start: number, stop: number): Promise<any[]> {
    const list = (await this.get<any[]>(key)) || [];

    // Handle negative indices (Redis-style)
    const startIdx = start < 0 ? Math.max(0, list.length + start) : start;
    const stopIdx = stop < 0 ? list.length + stop + 1 : stop + 1;

    return list.slice(startIdx, stopIdx);
  }

  async llen(key: string): Promise<number> {
    const list = (await this.get<any[]>(key)) || [];
    return list.length;
  }

  /**
   * Cleanup on shutdown
   */
  async disconnect(): Promise<void> {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.cache.clear();
  }

  /**
   * Get singleton instance
   */
  private static instance: RedisClient | null = null;

  static getInstance(config?: Partial<CacheConfig>): RedisClient {
    if (!RedisClient.instance) {
      RedisClient.instance = new RedisClient(config);
    }
    return RedisClient.instance;
  }
}

// Export singleton instance getter
export const getRedisClient = (config?: Partial<CacheConfig>) =>
  RedisClient.getInstance(config);
