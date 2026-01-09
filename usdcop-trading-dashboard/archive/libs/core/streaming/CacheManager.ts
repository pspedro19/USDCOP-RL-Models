/**
 * CacheManager - High-Performance IndexedDB Caching System
 * Advanced caching with compression, indexing, and intelligent eviction
 */

import { openDB, IDBPDatabase, IDBPTransaction } from 'idb';
import { EventEmitter } from 'eventemitter3';
import { Subject, BehaviorSubject, Observable, interval } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import type {
  CacheConfig,
  CacheEntry,
  StorageMetrics,
  CacheEvictionStrategy
} from '../types/streaming-types';

import type { MarketTick, OrderBook, Trade } from '../types/market-data';

export interface CacheQuery {
  readonly symbol?: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly dataType?: string;
  readonly limit?: number;
  readonly orderBy?: string;
  readonly orderDirection?: 'asc' | 'desc';
}

export interface CacheStats {
  readonly totalEntries: number;
  readonly totalSize: number;
  readonly hitRate: number;
  readonly missRate: number;
  readonly evictions: number;
  readonly compressionRatio: number;
  readonly oldestEntry: number;
  readonly newestEntry: number;
  readonly avgEntrySize: number;
  readonly efficiency: number;
}

export interface CacheOperation {
  readonly type: 'get' | 'set' | 'delete' | 'clear' | 'evict';
  readonly key: string;
  readonly size?: number;
  readonly success: boolean;
  readonly duration: number;
  readonly timestamp: number;
  readonly fromCache?: boolean;
}

const DB_NAME = 'TradingDataCache';
const DB_VERSION = 1;

// Store names
const STORES = {
  TICKS: 'ticks',
  ORDER_BOOKS: 'order_books',
  TRADES: 'trades',
  AGGREGATED: 'aggregated',
  METADATA: 'metadata'
} as const;

export class CacheManager extends EventEmitter {
  private readonly config: CacheConfig;
  private db: IDBPDatabase | null = null;
  private isInitialized = false;

  // Cache statistics
  private stats: CacheStats = {
    totalEntries: 0,
    totalSize: 0,
    hitRate: 0,
    missRate: 0,
    evictions: 0,
    compressionRatio: 1,
    oldestEntry: 0,
    newestEntry: 0,
    avgEntrySize: 0,
    efficiency: 0
  };

  // Operation tracking
  private operations: CacheOperation[] = [];
  private hitCount = 0;
  private missCount = 0;

  // Observables
  private readonly operations$ = new Subject<CacheOperation>();
  private readonly stats$ = new BehaviorSubject<CacheStats>(this.stats);
  private readonly destroy$ = new Subject<void>();

  // Cleanup and maintenance
  private maintenanceTimer?: NodeJS.Timeout;
  private compressionWorker?: Worker;

  constructor(config: CacheConfig) {
    super();
    this.config = config;
    this.initialize();
  }

  // ==========================================
  // INITIALIZATION
  // ==========================================

  private async initialize(): Promise<void> {
    try {
      if (!this.config.enabled || typeof indexedDB === 'undefined') {
        this.emit('cache_disabled', { reason: 'not_supported_or_disabled' });
        return;
      }

      await this.initializeDatabase();
      await this.loadStats();
      this.setupMaintenance();
      this.setupCompression();

      this.isInitialized = true;
      this.emit('cache_initialized', { config: this.config });

    } catch (error) {
      this.emit('cache_error', { error, operation: 'initialize' });
      throw error;
    }
  }

  private async initializeDatabase(): Promise<void> {
    this.db = await openDB(DB_NAME, DB_VERSION, {
      upgrade(db, oldVersion, newVersion, transaction) {
        // Create stores with indexes
        if (!db.objectStoreNames.contains(STORES.TICKS)) {
          const tickStore = db.createObjectStore(STORES.TICKS, { keyPath: 'key' });
          tickStore.createIndex('symbol', 'symbol');
          tickStore.createIndex('timestamp', 'timestamp');
          tickStore.createIndex('symbol_timestamp', ['symbol', 'timestamp']);
        }

        if (!db.objectStoreNames.contains(STORES.ORDER_BOOKS)) {
          const bookStore = db.createObjectStore(STORES.ORDER_BOOKS, { keyPath: 'key' });
          bookStore.createIndex('symbol', 'symbol');
          bookStore.createIndex('timestamp', 'timestamp');
          bookStore.createIndex('symbol_timestamp', ['symbol', 'timestamp']);
        }

        if (!db.objectStoreNames.contains(STORES.TRADES)) {
          const tradeStore = db.createObjectStore(STORES.TRADES, { keyPath: 'key' });
          tradeStore.createIndex('symbol', 'symbol');
          tradeStore.createIndex('timestamp', 'timestamp');
          tradeStore.createIndex('symbol_timestamp', ['symbol', 'timestamp']);
        }

        if (!db.objectStoreNames.contains(STORES.AGGREGATED)) {
          const aggStore = db.createObjectStore(STORES.AGGREGATED, { keyPath: 'key' });
          aggStore.createIndex('symbol', 'symbol');
          aggStore.createIndex('interval', 'interval');
          aggStore.createIndex('timestamp', 'timestamp');
          aggStore.createIndex('symbol_interval', ['symbol', 'interval']);
        }

        if (!db.objectStoreNames.contains(STORES.METADATA)) {
          db.createObjectStore(STORES.METADATA, { keyPath: 'key' });
        }
      }
    });
  }

  // ==========================================
  // CORE CACHE OPERATIONS
  // ==========================================

  public async get<T = any>(key: string, storeName: string = STORES.TICKS): Promise<T | null> {
    if (!this.isInitialized || !this.db) {
      return null;
    }

    const startTime = performance.now();

    try {
      const transaction = this.db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const entry = await store.get(key) as CacheEntry<T> | undefined;

      const duration = performance.now() - startTime;

      if (entry) {
        // Check if entry is expired
        if (this.isExpired(entry)) {
          await this.delete(key, storeName);
          this.recordOperation('get', key, 0, false, duration, false);
          this.missCount++;
          return null;
        }

        // Update access statistics
        await this.updateAccessStats(key, storeName);

        // Decompress if needed
        const data = entry.compressed ? await this.decompress(entry.data) : entry.data;

        this.recordOperation('get', key, entry.sizeBytes, true, duration, true);
        this.hitCount++;

        return data;
      } else {
        this.recordOperation('get', key, 0, false, duration, false);
        this.missCount++;
        return null;
      }

    } catch (error) {
      this.emit('cache_error', { error, operation: 'get', key });
      return null;
    }
  }

  public async set<T = any>(
    key: string,
    data: T,
    storeName: string = STORES.TICKS,
    ttl?: number
  ): Promise<boolean> {
    if (!this.isInitialized || !this.db) {
      return false;
    }

    const startTime = performance.now();

    try {
      // Check if we need to evict entries first
      await this.checkAndEvict();

      // Prepare entry
      let processedData = data;
      let compressed = false;
      let sizeBytes = this.estimateSize(data);

      // Compress if enabled and data is large enough
      if (this.config.compression && sizeBytes > 1024) {
        processedData = await this.compress(data);
        compressed = true;
        sizeBytes = this.estimateSize(processedData);
      }

      const entry: CacheEntry<T> = {
        key,
        data: processedData,
        timestamp: Date.now(),
        accessCount: 1,
        lastAccess: Date.now(),
        sizeBytes,
        compressed,
        ttl: ttl || this.config.maxAge
      };

      const transaction = this.db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      await store.put(entry);

      const duration = performance.now() - startTime;
      this.recordOperation('set', key, sizeBytes, true, duration);

      // Update statistics
      this.updateStats();

      return true;

    } catch (error) {
      this.emit('cache_error', { error, operation: 'set', key });
      return false;
    }
  }

  public async delete(key: string, storeName: string = STORES.TICKS): Promise<boolean> {
    if (!this.isInitialized || !this.db) {
      return false;
    }

    const startTime = performance.now();

    try {
      const transaction = this.db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      await store.delete(key);

      const duration = performance.now() - startTime;
      this.recordOperation('delete', key, 0, true, duration);

      this.updateStats();
      return true;

    } catch (error) {
      this.emit('cache_error', { error, operation: 'delete', key });
      return false;
    }
  }

  public async clear(storeName?: string): Promise<boolean> {
    if (!this.isInitialized || !this.db) {
      return false;
    }

    const startTime = performance.now();

    try {
      const storeNames = storeName ? [storeName] : Object.values(STORES);
      const transaction = this.db.transaction(storeNames, 'readwrite');

      for (const name of storeNames) {
        const store = transaction.objectStore(name);
        await store.clear();
      }

      const duration = performance.now() - startTime;
      this.recordOperation('clear', 'all', 0, true, duration);

      this.resetStats();
      return true;

    } catch (error) {
      this.emit('cache_error', { error, operation: 'clear' });
      return false;
    }
  }

  // ==========================================
  // QUERY METHODS
  // ==========================================

  public async query<T = any>(
    query: CacheQuery,
    storeName: string = STORES.TICKS
  ): Promise<T[]> {
    if (!this.isInitialized || !this.db) {
      return [];
    }

    try {
      const transaction = this.db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);

      let cursor;
      let results: T[] = [];

      // Use appropriate index
      if (query.symbol && query.startTime) {
        const index = store.index('symbol_timestamp');
        const range = this.buildTimeRange(query.symbol, query.startTime, query.endTime);
        cursor = await index.openCursor(range);
      } else if (query.symbol) {
        const index = store.index('symbol');
        cursor = await index.openCursor(query.symbol);
      } else if (query.startTime) {
        const index = store.index('timestamp');
        const range = this.buildTimeRange(null, query.startTime, query.endTime);
        cursor = await index.openCursor(range);
      } else {
        cursor = await store.openCursor();
      }

      while (cursor) {
        const entry = cursor.value as CacheEntry<T>;

        if (!this.isExpired(entry)) {
          const data = entry.compressed ? await this.decompress(entry.data) : entry.data;
          results.push(data);

          if (query.limit && results.length >= query.limit) {
            break;
          }
        }

        cursor = await cursor.continue();
      }

      // Sort results if needed
      if (query.orderBy) {
        results.sort((a, b) => {
          const aVal = (a as any)[query.orderBy!];
          const bVal = (b as any)[query.orderBy!];

          if (query.orderDirection === 'desc') {
            return bVal - aVal;
          } else {
            return aVal - bVal;
          }
        });
      }

      return results;

    } catch (error) {
      this.emit('cache_error', { error, operation: 'query' });
      return [];
    }
  }

  public async getTicksInRange(
    symbol: string,
    startTime: number,
    endTime: number
  ): Promise<MarketTick[]> {
    return this.query<MarketTick>({
      symbol,
      startTime,
      endTime,
      orderBy: 'timestamp',
      orderDirection: 'asc'
    }, STORES.TICKS);
  }

  public async getOrderBooksInRange(
    symbol: string,
    startTime: number,
    endTime: number
  ): Promise<OrderBook[]> {
    return this.query<OrderBook>({
      symbol,
      startTime,
      endTime,
      orderBy: 'timestamp',
      orderDirection: 'asc'
    }, STORES.ORDER_BOOKS);
  }

  public async getTradesInRange(
    symbol: string,
    startTime: number,
    endTime: number
  ): Promise<Trade[]> {
    return this.query<Trade>({
      symbol,
      startTime,
      endTime,
      orderBy: 'timestamp',
      orderDirection: 'asc'
    }, STORES.TRADES);
  }

  // ==========================================
  // BULK OPERATIONS
  // ==========================================

  public async bulkSet<T = any>(
    entries: Array<{ key: string; data: T; storeName?: string }>,
    ttl?: number
  ): Promise<number> {
    if (!this.isInitialized || !this.db) {
      return 0;
    }

    let successCount = 0;
    const batchSize = 100; // Process in batches to avoid blocking

    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);

      try {
        const transactions = new Map<string, IDBPTransaction<unknown, string[], 'readwrite'>>();

        for (const entry of batch) {
          const storeName = entry.storeName || STORES.TICKS;

          if (!transactions.has(storeName)) {
            transactions.set(storeName, this.db.transaction(storeName, 'readwrite'));
          }

          const transaction = transactions.get(storeName)!;
          const store = transaction.objectStore(storeName);

          // Prepare cache entry
          let processedData = entry.data;
          let compressed = false;
          let sizeBytes = this.estimateSize(entry.data);

          if (this.config.compression && sizeBytes > 1024) {
            processedData = await this.compress(entry.data);
            compressed = true;
            sizeBytes = this.estimateSize(processedData);
          }

          const cacheEntry: CacheEntry<T> = {
            key: entry.key,
            data: processedData,
            timestamp: Date.now(),
            accessCount: 1,
            lastAccess: Date.now(),
            sizeBytes,
            compressed,
            ttl: ttl || this.config.maxAge
          };

          await store.put(cacheEntry);
          successCount++;
        }

        // Complete all transactions
        await Promise.all(Array.from(transactions.values()).map(tx => tx.done));

      } catch (error) {
        this.emit('cache_error', { error, operation: 'bulk_set', batch: i });
      }
    }

    this.updateStats();
    return successCount;
  }

  public async bulkDelete(keys: string[], storeName: string = STORES.TICKS): Promise<number> {
    if (!this.isInitialized || !this.db) {
      return 0;
    }

    let successCount = 0;

    try {
      const transaction = this.db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);

      for (const key of keys) {
        await store.delete(key);
        successCount++;
      }

      this.updateStats();

    } catch (error) {
      this.emit('cache_error', { error, operation: 'bulk_delete' });
    }

    return successCount;
  }

  // ==========================================
  // EVICTION STRATEGIES
  // ==========================================

  private async checkAndEvict(): Promise<void> {
    const currentSize = await this.calculateTotalSize();

    if (currentSize > this.config.maxSizeBytes) {
      const evictSize = currentSize - (this.config.maxSizeBytes * 0.8); // Evict to 80%
      await this.evictEntries(evictSize);
    }
  }

  private async evictEntries(targetSize: number): Promise<number> {
    let evictedSize = 0;
    let evictedCount = 0;

    try {
      switch (this.config.evictionStrategy) {
        case 'lru':
          evictedSize = await this.evictLRU(targetSize);
          break;
        case 'lfu':
          evictedSize = await this.evictLFU(targetSize);
          break;
        case 'ttl':
          evictedSize = await this.evictExpired();
          break;
        case 'size_based':
          evictedSize = await this.evictBySizeGreedy(targetSize);
          break;
        default:
          evictedSize = await this.evictLRU(targetSize);
      }

      this.stats.evictions += evictedCount;
      this.recordOperation('evict', 'multiple', evictedSize, true, 0);

    } catch (error) {
      this.emit('cache_error', { error, operation: 'evict' });
    }

    return evictedSize;
  }

  private async evictLRU(targetSize: number): Promise<number> {
    let evictedSize = 0;
    const stores = Object.values(STORES);

    for (const storeName of stores) {
      if (evictedSize >= targetSize) break;

      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const entries: CacheEntry[] = [];

      let cursor = await store.openCursor();
      while (cursor) {
        entries.push(cursor.value);
        cursor = await cursor.continue();
      }

      // Sort by last access time (oldest first)
      entries.sort((a, b) => a.lastAccess - b.lastAccess);

      for (const entry of entries) {
        if (evictedSize >= targetSize) break;

        await store.delete(entry.key);
        evictedSize += entry.sizeBytes;
      }
    }

    return evictedSize;
  }

  private async evictLFU(targetSize: number): Promise<number> {
    let evictedSize = 0;
    const stores = Object.values(STORES);

    for (const storeName of stores) {
      if (evictedSize >= targetSize) break;

      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const entries: CacheEntry[] = [];

      let cursor = await store.openCursor();
      while (cursor) {
        entries.push(cursor.value);
        cursor = await cursor.continue();
      }

      // Sort by access count (lowest first)
      entries.sort((a, b) => a.accessCount - b.accessCount);

      for (const entry of entries) {
        if (evictedSize >= targetSize) break;

        await store.delete(entry.key);
        evictedSize += entry.sizeBytes;
      }
    }

    return evictedSize;
  }

  private async evictExpired(): Promise<number> {
    let evictedSize = 0;
    const stores = Object.values(STORES);
    const now = Date.now();

    for (const storeName of stores) {
      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);

      let cursor = await store.openCursor();
      while (cursor) {
        const entry = cursor.value as CacheEntry;

        if (this.isExpired(entry)) {
          await store.delete(entry.key);
          evictedSize += entry.sizeBytes;
        }

        cursor = await cursor.continue();
      }
    }

    return evictedSize;
  }

  private async evictBySizeGreedy(targetSize: number): Promise<number> {
    let evictedSize = 0;
    const stores = Object.values(STORES);

    for (const storeName of stores) {
      if (evictedSize >= targetSize) break;

      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const entries: CacheEntry[] = [];

      let cursor = await store.openCursor();
      while (cursor) {
        entries.push(cursor.value);
        cursor = await cursor.continue();
      }

      // Sort by size (largest first) for greedy eviction
      entries.sort((a, b) => b.sizeBytes - a.sizeBytes);

      for (const entry of entries) {
        if (evictedSize >= targetSize) break;

        await store.delete(entry.key);
        evictedSize += entry.sizeBytes;
      }
    }

    return evictedSize;
  }

  // ==========================================
  // COMPRESSION
  // ==========================================

  private setupCompression(): void {
    if (!this.config.compression) return;

    try {
      // Setup compression worker if available
      if (typeof Worker !== 'undefined') {
        this.compressionWorker = new Worker('/workers/compression-worker.js');
        this.compressionWorker.onmessage = (event) => {
          this.handleCompressionResult(event.data);
        };
      }
    } catch (error) {
      this.emit('compression_error', error);
    }
  }

  private async compress<T>(data: T): Promise<any> {
    try {
      // Simple JSON compression (in production, use proper compression library)
      const jsonString = JSON.stringify(data);

      // Use compression worker if available
      if (this.compressionWorker) {
        return new Promise((resolve, reject) => {
          const messageId = this.generateId();
          const timeout = setTimeout(() => reject(new Error('Compression timeout')), 5000);

          const handler = (event: MessageEvent) => {
            if (event.data.id === messageId) {
              clearTimeout(timeout);
              this.compressionWorker!.removeEventListener('message', handler);
              resolve(event.data.result);
            }
          };

          this.compressionWorker!.addEventListener('message', handler);
          this.compressionWorker!.postMessage({
            id: messageId,
            type: 'compress',
            data: jsonString
          });
        });
      } else {
        // Fallback: simple encoding
        return btoa(jsonString);
      }

    } catch (error) {
      this.emit('compression_error', error);
      return data;
    }
  }

  private async decompress<T>(compressedData: any): Promise<T> {
    try {
      // Use compression worker if available
      if (this.compressionWorker && typeof compressedData === 'string') {
        return new Promise((resolve, reject) => {
          const messageId = this.generateId();
          const timeout = setTimeout(() => reject(new Error('Decompression timeout')), 5000);

          const handler = (event: MessageEvent) => {
            if (event.data.id === messageId) {
              clearTimeout(timeout);
              this.compressionWorker!.removeEventListener('message', handler);
              resolve(JSON.parse(event.data.result));
            }
          };

          this.compressionWorker!.addEventListener('message', handler);
          this.compressionWorker!.postMessage({
            id: messageId,
            type: 'decompress',
            data: compressedData
          });
        });
      } else {
        // Fallback: simple decoding
        const jsonString = atob(compressedData);
        return JSON.parse(jsonString);
      }

    } catch (error) {
      this.emit('compression_error', error);
      return compressedData;
    }
  }

  private handleCompressionResult(result: any): void {
    // Handle compression worker results
    this.emit('compression_result', result);
  }

  // ==========================================
  // STATISTICS & MONITORING
  // ==========================================

  public getStats(): CacheStats {
    return { ...this.stats };
  }

  public getStatsStream(): Observable<CacheStats> {
    return this.stats$.asObservable().pipe(
      takeUntil(this.destroy$)
    );
  }

  public getOperationsStream(): Observable<CacheOperation> {
    return this.operations$.asObservable().pipe(
      takeUntil(this.destroy$)
    );
  }

  private async updateStats(): Promise<void> {
    try {
      const totalSize = await this.calculateTotalSize();
      const totalEntries = await this.calculateTotalEntries();
      const { oldest, newest } = await this.getTimestampRange();

      const totalOperations = this.hitCount + this.missCount;
      const hitRate = totalOperations > 0 ? (this.hitCount / totalOperations) * 100 : 0;
      const missRate = totalOperations > 0 ? (this.missCount / totalOperations) * 100 : 0;

      this.stats = {
        ...this.stats,
        totalEntries,
        totalSize,
        hitRate,
        missRate,
        oldestEntry: oldest,
        newestEntry: newest,
        avgEntrySize: totalEntries > 0 ? totalSize / totalEntries : 0,
        efficiency: hitRate // Simplified efficiency calculation
      };

      this.stats$.next(this.stats);

    } catch (error) {
      this.emit('stats_error', error);
    }
  }

  private async loadStats(): Promise<void> {
    try {
      const savedStats = await this.get('cache_stats', STORES.METADATA);
      if (savedStats) {
        this.stats = { ...this.stats, ...savedStats };
        this.hitCount = savedStats.hitCount || 0;
        this.missCount = savedStats.missCount || 0;
      }
    } catch (error) {
      // Ignore errors loading stats
    }
  }

  private async saveStats(): Promise<void> {
    try {
      await this.set('cache_stats', {
        ...this.stats,
        hitCount: this.hitCount,
        missCount: this.missCount
      }, STORES.METADATA);
    } catch (error) {
      // Ignore errors saving stats
    }
  }

  private resetStats(): void {
    this.stats = {
      totalEntries: 0,
      totalSize: 0,
      hitRate: 0,
      missRate: 0,
      evictions: 0,
      compressionRatio: 1,
      oldestEntry: 0,
      newestEntry: 0,
      avgEntrySize: 0,
      efficiency: 0
    };

    this.hitCount = 0;
    this.missCount = 0;
    this.operations.length = 0;

    this.stats$.next(this.stats);
  }

  private recordOperation(
    type: CacheOperation['type'],
    key: string,
    size: number,
    success: boolean,
    duration: number,
    fromCache: boolean = false
  ): void {
    const operation: CacheOperation = {
      type,
      key,
      size,
      success,
      duration,
      timestamp: Date.now(),
      fromCache
    };

    this.operations.push(operation);

    // Keep only last 1000 operations
    if (this.operations.length > 1000) {
      this.operations.splice(0, this.operations.length - 1000);
    }

    this.operations$.next(operation);
  }

  // ==========================================
  // MAINTENANCE
  // ==========================================

  private setupMaintenance(): void {
    // Run maintenance every 5 minutes
    this.maintenanceTimer = setInterval(() => {
      this.runMaintenance();
    }, 300000);
  }

  private async runMaintenance(): Promise<void> {
    try {
      // Clean expired entries
      await this.evictExpired();

      // Update statistics
      await this.updateStats();

      // Save statistics
      await this.saveStats();

      // Check if we need to evict entries
      await this.checkAndEvict();

      this.emit('maintenance_complete', {
        timestamp: Date.now(),
        stats: this.stats
      });

    } catch (error) {
      this.emit('maintenance_error', error);
    }
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  private async updateAccessStats(key: string, storeName: string): Promise<void> {
    try {
      const transaction = this.db!.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const entry = await store.get(key) as CacheEntry;

      if (entry) {
        entry.accessCount++;
        entry.lastAccess = Date.now();
        await store.put(entry);
      }
    } catch (error) {
      // Ignore access update errors
    }
  }

  private isExpired(entry: CacheEntry): boolean {
    if (!entry.ttl) return false;
    return Date.now() - entry.timestamp > entry.ttl;
  }

  private async calculateTotalSize(): Promise<number> {
    let totalSize = 0;
    const stores = Object.values(STORES);

    for (const storeName of stores) {
      try {
        const transaction = this.db!.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);

        let cursor = await store.openCursor();
        while (cursor) {
          const entry = cursor.value as CacheEntry;
          totalSize += entry.sizeBytes;
          cursor = await cursor.continue();
        }
      } catch (error) {
        // Continue with other stores
      }
    }

    return totalSize;
  }

  private async calculateTotalEntries(): Promise<number> {
    let totalEntries = 0;
    const stores = Object.values(STORES);

    for (const storeName of stores) {
      try {
        const transaction = this.db!.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);
        const count = await store.count();
        totalEntries += count;
      } catch (error) {
        // Continue with other stores
      }
    }

    return totalEntries;
  }

  private async getTimestampRange(): Promise<{ oldest: number; newest: number }> {
    let oldest = Date.now();
    let newest = 0;
    const stores = Object.values(STORES);

    for (const storeName of stores) {
      try {
        const transaction = this.db!.transaction(storeName, 'readonly');
        const store = transaction.objectStore(storeName);

        let cursor = await store.openCursor();
        while (cursor) {
          const entry = cursor.value as CacheEntry;
          if (entry.timestamp < oldest) oldest = entry.timestamp;
          if (entry.timestamp > newest) newest = entry.timestamp;
          cursor = await cursor.continue();
        }
      } catch (error) {
        // Continue with other stores
      }
    }

    return { oldest: oldest === Date.now() ? 0 : oldest, newest };
  }

  private buildTimeRange(
    symbol: string | null,
    startTime?: number,
    endTime?: number
  ): IDBKeyRange | undefined {
    if (!startTime && !endTime) return undefined;

    if (symbol) {
      // Build range for symbol_timestamp index
      const start = startTime ? [symbol, startTime] : [symbol, 0];
      const end = endTime ? [symbol, endTime] : [symbol, Date.now()];
      return IDBKeyRange.bound(start, end);
    } else {
      // Build range for timestamp index
      if (startTime && endTime) {
        return IDBKeyRange.bound(startTime, endTime);
      } else if (startTime) {
        return IDBKeyRange.lowerBound(startTime);
      } else if (endTime) {
        return IDBKeyRange.upperBound(endTime);
      }
    }

    return undefined;
  }

  private estimateSize(data: any): number {
    try {
      // Rough estimation of object size in bytes
      return new Blob([JSON.stringify(data)]).size;
    } catch {
      // Fallback estimation
      return JSON.stringify(data).length * 2; // Rough Unicode estimate
    }
  }

  private generateId(): string {
    return `cache-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }

  // ==========================================
  // CLEANUP
  // ==========================================

  public async destroy(): Promise<void> {
    this.destroy$.next();
    this.destroy$.complete();

    // Clear timers
    if (this.maintenanceTimer) {
      clearInterval(this.maintenanceTimer);
    }

    // Terminate worker
    if (this.compressionWorker) {
      this.compressionWorker.terminate();
    }

    // Save final stats
    await this.saveStats();

    // Close database
    if (this.db) {
      this.db.close();
      this.db = null;
    }

    // Complete observables
    this.stats$.complete();
    this.operations$.complete();

    this.removeAllListeners();
    this.isInitialized = false;
  }
}