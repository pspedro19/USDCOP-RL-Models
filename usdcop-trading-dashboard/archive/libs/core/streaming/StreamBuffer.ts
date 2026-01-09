/**
 * StreamBuffer - Advanced Buffering and Throttling System
 * High-performance circular buffers with smart throttling (max 60 updates/sec)
 */

import { EventEmitter } from 'eventemitter3';
import { BehaviorSubject, Subject, Observable, interval } from 'rxjs';
import { filter, map, takeUntil } from 'rxjs/operators';

import type {
  StreamMessage,
  StreamBuffer as IStreamBuffer,
  ThrottleConfig,
  BufferConfig,
  ThrottleStrategy,
  BufferStrategy
} from '../types/streaming-types';

export interface BufferMetrics {
  readonly bufferId: string;
  readonly symbol: string;
  readonly currentSize: number;
  readonly maxSize: number;
  readonly utilization: number;
  readonly throughput: number;
  readonly droppedMessages: number;
  readonly averageAge: number;
  readonly compressionRatio: number;
  readonly memoryUsage: number;
}

export interface ThrottleMetrics {
  readonly throttleId: string;
  readonly symbol: string;
  readonly inputRate: number;
  readonly outputRate: number;
  readonly queueSize: number;
  readonly droppedCount: number;
  readonly latency: number;
  readonly efficiency: number;
}

export class StreamBuffer<T = StreamMessage> extends EventEmitter implements IStreamBuffer<T> {
  public readonly id: string;
  public readonly maxSize: number;
  public readonly isCircular: boolean;

  private readonly config: BufferConfig;
  private readonly data: T[] = [];
  private readonly timestamps: number[] = [];
  private readonly priorities: number[] = [];

  private writeIndex = 0;
  private _currentSize = 0;
  private _oldestTimestamp = 0;
  private _newestTimestamp = 0;

  private metrics: BufferMetrics;
  private compressionEnabled = false;
  private compressionWorker?: Worker;

  constructor(
    id: string,
    config: BufferConfig
  ) {
    super();

    this.id = id;
    this.config = config;
    this.maxSize = config.maxSize;
    this.isCircular = true;

    this.metrics = {
      bufferId: id,
      symbol: '',
      currentSize: 0,
      maxSize: config.maxSize,
      utilization: 0,
      throughput: 0,
      droppedMessages: 0,
      averageAge: 0,
      compressionRatio: 1,
      memoryUsage: 0
    };

    this.initialize();
  }

  // ==========================================
  // BUFFER INTERFACE IMPLEMENTATION
  // ==========================================

  public get currentSize(): number {
    return this._currentSize;
  }

  public get oldestTimestamp(): number {
    return this._oldestTimestamp;
  }

  public get newestTimestamp(): number {
    return this._newestTimestamp;
  }

  public get data(): readonly T[] {
    return this.getData();
  }

  // ==========================================
  // CORE BUFFER OPERATIONS
  // ==========================================

  public push(item: T, priority: number = 0): boolean {
    const timestamp = Date.now();

    try {
      // Check if buffer is full and handle overflow
      if (this._currentSize >= this.maxSize) {
        if (!this.handleOverflow(item, priority)) {
          this.metrics = { ...this.metrics, droppedMessages: this.metrics.droppedMessages + 1 };
          this.emit('item_dropped', { item, reason: 'buffer_full' });
          return false;
        }
      }

      // Add item to buffer
      const index = this.getNextIndex();
      this.data[index] = item;
      this.timestamps[index] = timestamp;
      this.priorities[index] = priority;

      // Update metadata
      this._currentSize++;
      this._newestTimestamp = timestamp;

      if (this._currentSize === 1) {
        this._oldestTimestamp = timestamp;
      }

      // Update metrics
      this.updateMetrics();

      this.emit('item_added', { item, index, timestamp });
      return true;

    } catch (error) {
      this.emit('error', { error, operation: 'push', item });
      return false;
    }
  }

  public pop(): T | undefined {
    if (this._currentSize === 0) {
      return undefined;
    }

    try {
      let oldestIndex = 0;
      let oldestTimestamp = this.timestamps[0];

      // Find oldest item
      for (let i = 1; i < this._currentSize; i++) {
        if (this.timestamps[i] < oldestTimestamp) {
          oldestTimestamp = this.timestamps[i];
          oldestIndex = i;
        }
      }

      const item = this.data[oldestIndex];

      // Remove item by shifting array
      this.removeAtIndex(oldestIndex);

      this.emit('item_removed', { item, timestamp: oldestTimestamp });
      return item;

    } catch (error) {
      this.emit('error', { error, operation: 'pop' });
      return undefined;
    }
  }

  public peek(count: number = 1): T[] {
    if (count <= 0 || this._currentSize === 0) {
      return [];
    }

    const result: T[] = [];
    const actualCount = Math.min(count, this._currentSize);

    // Get newest items
    const sortedIndices = this.getSortedIndices(false); // newest first

    for (let i = 0; i < actualCount; i++) {
      const index = sortedIndices[i];
      result.push(this.data[index]);
    }

    return result;
  }

  public clear(): void {
    const clearedCount = this._currentSize;

    this.data.length = 0;
    this.timestamps.length = 0;
    this.priorities.length = 0;

    this._currentSize = 0;
    this._oldestTimestamp = 0;
    this._newestTimestamp = 0;
    this.writeIndex = 0;

    this.updateMetrics();
    this.emit('buffer_cleared', { clearedCount });
  }

  public getRange(startTime: number, endTime: number): T[] {
    const result: T[] = [];

    for (let i = 0; i < this._currentSize; i++) {
      const timestamp = this.timestamps[i];
      if (timestamp >= startTime && timestamp <= endTime) {
        result.push(this.data[i]);
      }
    }

    return result.sort((a, b) => {
      const aIndex = this.data.indexOf(a);
      const bIndex = this.data.indexOf(b);
      return this.timestamps[aIndex] - this.timestamps[bIndex];
    });
  }

  public filter(predicate: (item: T, timestamp: number) => boolean): T[] {
    const result: T[] = [];

    for (let i = 0; i < this._currentSize; i++) {
      if (predicate(this.data[i], this.timestamps[i])) {
        result.push(this.data[i]);
      }
    }

    return result;
  }

  public compress(): Promise<number> {
    return new Promise((resolve, reject) => {
      if (!this.config.compressionEnabled || this._currentSize === 0) {
        resolve(1);
        return;
      }

      try {
        // Implement compression using Web Worker if available
        if (typeof Worker !== 'undefined' && !this.compressionWorker) {
          this.setupCompressionWorker();
        }

        // Simple compression: remove duplicate consecutive values
        const compressed = this.removeDuplicates();
        const originalSize = this._currentSize;
        const compressedSize = compressed.length;

        if (compressedSize < originalSize) {
          this.replaceData(compressed);
          const ratio = originalSize / compressedSize;
          this.metrics = { ...this.metrics, compressionRatio: ratio };
          this.emit('buffer_compressed', { originalSize, compressedSize, ratio });
        }

        resolve(this.metrics.compressionRatio);

      } catch (error) {
        reject(error);
      }
    });
  }

  // ==========================================
  // BUFFER STRATEGIES
  // ==========================================

  private handleOverflow(newItem: T, priority: number): boolean {
    switch (this.config.strategy) {
      case 'fifo':
        return this.handleFIFOOverflow();
      case 'lifo':
        return this.handleLIFOOverflow();
      case 'priority':
        return this.handlePriorityOverflow(newItem, priority);
      case 'time_based':
        return this.handleTimeBasedOverflow();
      default:
        return this.handleFIFOOverflow();
    }
  }

  private handleFIFOOverflow(): boolean {
    // Remove oldest item
    const oldestIndex = this.findOldestIndex();
    this.removeAtIndex(oldestIndex);
    return true;
  }

  private handleLIFOOverflow(): boolean {
    // Remove newest item
    const newestIndex = this.findNewestIndex();
    this.removeAtIndex(newestIndex);
    return true;
  }

  private handlePriorityOverflow(newItem: T, newPriority: number): boolean {
    // Find lowest priority item
    let lowestPriority = Number.MAX_SAFE_INTEGER;
    let lowestPriorityIndex = -1;

    for (let i = 0; i < this._currentSize; i++) {
      if (this.priorities[i] < lowestPriority) {
        lowestPriority = this.priorities[i];
        lowestPriorityIndex = i;
      }
    }

    // Only replace if new item has higher priority
    if (newPriority > lowestPriority) {
      this.removeAtIndex(lowestPriorityIndex);
      return true;
    }

    return false; // Don't add new item
  }

  private handleTimeBasedOverflow(): boolean {
    // Remove items older than maxAge
    const cutoffTime = Date.now() - this.config.maxAge;
    let removedAny = false;

    for (let i = this._currentSize - 1; i >= 0; i--) {
      if (this.timestamps[i] < cutoffTime) {
        this.removeAtIndex(i);
        removedAny = true;
      }
    }

    return removedAny || this.handleFIFOOverflow();
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  private initialize(): void {
    // Setup compression if enabled
    if (this.config.compressionEnabled) {
      this.setupCompression();
    }

    // Setup automatic cleanup
    if (this.config.maxAge > 0) {
      this.setupAutoCleanup();
    }
  }

  private setupCompression(): void {
    this.compressionEnabled = true;

    // Compress buffer every 30 seconds
    setInterval(() => {
      this.compress().catch(error => {
        this.emit('error', { error, operation: 'auto_compression' });
      });
    }, 30000);
  }

  private setupAutoCleanup(): void {
    // Clean old items every 10 seconds
    setInterval(() => {
      this.cleanupOldItems();
    }, 10000);
  }

  private setupCompressionWorker(): void {
    try {
      this.compressionWorker = new Worker('/workers/compression-worker.js');
      this.compressionWorker.onmessage = (event) => {
        this.handleCompressionResult(event.data);
      };
    } catch (error) {
      this.emit('error', { error, operation: 'setup_compression_worker' });
    }
  }

  private handleCompressionResult(result: any): void {
    if (result.success) {
      this.metrics = { ...this.metrics, compressionRatio: result.compressionRatio };
      this.emit('compression_complete', result);
    } else {
      this.emit('error', { error: result.error, operation: 'compression' });
    }
  }

  private cleanupOldItems(): void {
    if (this.config.maxAge <= 0) return;

    const cutoffTime = Date.now() - this.config.maxAge;
    let removedCount = 0;

    for (let i = this._currentSize - 1; i >= 0; i--) {
      if (this.timestamps[i] < cutoffTime) {
        this.removeAtIndex(i);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      this.emit('items_expired', { removedCount, cutoffTime });
    }
  }

  private getNextIndex(): number {
    if (this._currentSize < this.maxSize) {
      return this._currentSize;
    } else {
      // Circular buffer: overwrite oldest
      const index = this.writeIndex;
      this.writeIndex = (this.writeIndex + 1) % this.maxSize;
      return index;
    }
  }

  private removeAtIndex(index: number): void {
    if (index < 0 || index >= this._currentSize) return;

    // Shift arrays to remove item
    for (let i = index; i < this._currentSize - 1; i++) {
      this.data[i] = this.data[i + 1];
      this.timestamps[i] = this.timestamps[i + 1];
      this.priorities[i] = this.priorities[i + 1];
    }

    this._currentSize--;

    // Update oldest/newest timestamps
    this.updateTimestampBounds();
    this.updateMetrics();
  }

  private findOldestIndex(): number {
    if (this._currentSize === 0) return -1;

    let oldestIndex = 0;
    let oldestTimestamp = this.timestamps[0];

    for (let i = 1; i < this._currentSize; i++) {
      if (this.timestamps[i] < oldestTimestamp) {
        oldestTimestamp = this.timestamps[i];
        oldestIndex = i;
      }
    }

    return oldestIndex;
  }

  private findNewestIndex(): number {
    if (this._currentSize === 0) return -1;

    let newestIndex = 0;
    let newestTimestamp = this.timestamps[0];

    for (let i = 1; i < this._currentSize; i++) {
      if (this.timestamps[i] > newestTimestamp) {
        newestTimestamp = this.timestamps[i];
        newestIndex = i;
      }
    }

    return newestIndex;
  }

  private getSortedIndices(ascending: boolean = true): number[] {
    const indices = Array.from({ length: this._currentSize }, (_, i) => i);

    return indices.sort((a, b) => {
      const diff = this.timestamps[a] - this.timestamps[b];
      return ascending ? diff : -diff;
    });
  }

  private updateTimestampBounds(): void {
    if (this._currentSize === 0) {
      this._oldestTimestamp = 0;
      this._newestTimestamp = 0;
      return;
    }

    let oldest = this.timestamps[0];
    let newest = this.timestamps[0];

    for (let i = 1; i < this._currentSize; i++) {
      const timestamp = this.timestamps[i];
      if (timestamp < oldest) oldest = timestamp;
      if (timestamp > newest) newest = timestamp;
    }

    this._oldestTimestamp = oldest;
    this._newestTimestamp = newest;
  }

  private removeDuplicates(): T[] {
    if (this._currentSize === 0) return [];

    const result: T[] = [];
    const seen = new Set<string>();

    for (let i = 0; i < this._currentSize; i++) {
      const item = this.data[i];
      const key = this.getItemKey(item);

      if (!seen.has(key)) {
        seen.add(key);
        result.push(item);
      }
    }

    return result;
  }

  private getItemKey(item: T): string {
    // Generate unique key for deduplication
    if (typeof item === 'object' && item !== null) {
      return JSON.stringify(item);
    }
    return String(item);
  }

  private replaceData(newData: T[]): void {
    this.clear();

    newData.forEach((item, index) => {
      this.data[index] = item;
      this.timestamps[index] = Date.now();
      this.priorities[index] = 0;
    });

    this._currentSize = newData.length;
    this.updateTimestampBounds();
  }

  private getData(): T[] {
    return this.data.slice(0, this._currentSize);
  }

  private updateMetrics(): void {
    const now = Date.now();

    // Calculate utilization
    const utilization = this._currentSize / this.maxSize;

    // Calculate average age
    let totalAge = 0;
    for (let i = 0; i < this._currentSize; i++) {
      totalAge += now - this.timestamps[i];
    }
    const averageAge = this._currentSize > 0 ? totalAge / this._currentSize : 0;

    // Estimate memory usage (rough calculation)
    const estimatedItemSize = 100; // bytes per item (rough estimate)
    const memoryUsage = this._currentSize * estimatedItemSize;

    this.metrics = {
      ...this.metrics,
      currentSize: this._currentSize,
      utilization,
      averageAge,
      memoryUsage
    };
  }

  public getMetrics(): BufferMetrics {
    this.updateMetrics();
    return { ...this.metrics };
  }

  public destroy(): void {
    this.clear();

    if (this.compressionWorker) {
      this.compressionWorker.terminate();
      this.compressionWorker = undefined;
    }

    this.removeAllListeners();
  }
}

// ==========================================
// THROTTLE MANAGER
// ==========================================

export class ThrottleManager extends EventEmitter {
  private readonly throttles = new Map<string, Throttle>();
  private readonly metrics = new Map<string, ThrottleMetrics>();

  public createThrottle(
    id: string,
    config: ThrottleConfig
  ): Throttle {
    const throttle = new Throttle(id, config);

    // Setup metrics collection
    throttle.on('metrics_updated', (metrics) => {
      this.metrics.set(id, metrics);
      this.emit('throttle_metrics', metrics);
    });

    this.throttles.set(id, throttle);
    return throttle;
  }

  public getThrottle(id: string): Throttle | undefined {
    return this.throttles.get(id);
  }

  public removeThrottle(id: string): void {
    const throttle = this.throttles.get(id);
    if (throttle) {
      throttle.destroy();
      this.throttles.delete(id);
      this.metrics.delete(id);
    }
  }

  public getMetrics(): ThrottleMetrics[] {
    return Array.from(this.metrics.values());
  }

  public destroy(): void {
    this.throttles.forEach(throttle => throttle.destroy());
    this.throttles.clear();
    this.metrics.clear();
    this.removeAllListeners();
  }
}

export class Throttle extends EventEmitter {
  public readonly id: string;
  private readonly config: ThrottleConfig;

  private readonly inputQueue: StreamMessage[] = [];
  private readonly outputSubject = new Subject<StreamMessage>();
  private readonly destroy$ = new Subject<void>();

  private lastEmitTime = 0;
  private messageCount = 0;
  private droppedCount = 0;

  private throttleTimer?: NodeJS.Timeout;
  private metricsTimer?: NodeJS.Timeout;

  constructor(id: string, config: ThrottleConfig) {
    super();

    this.id = id;
    this.config = config;

    this.initialize();
  }

  // ==========================================
  // THROTTLE OPERATIONS
  // ==========================================

  public input(message: StreamMessage): void {
    this.messageCount++;

    if (!this.config.enabled) {
      this.outputSubject.next(message);
      return;
    }

    // Add to queue
    this.inputQueue.push(message);

    // Check burst limit
    if (this.inputQueue.length > this.config.burstLimit) {
      this.handleBurstOverflow();
    }

    // Process immediately if rate allows
    this.processQueue();
  }

  public getOutputStream(): Observable<StreamMessage> {
    return this.outputSubject.asObservable().pipe(
      takeUntil(this.destroy$)
    );
  }

  public getMetrics(): ThrottleMetrics {
    const now = Date.now();
    const windowStart = now - this.config.windowSize;

    // Calculate rates over window
    const inputRate = this.messageCount / (this.config.windowSize / 1000);
    const outputRate = Math.min(inputRate, this.config.maxUpdatesPerSecond);

    // Calculate latency (time messages spend in queue)
    const averageQueueTime = this.inputQueue.length > 0 ?
      this.inputQueue.reduce((sum, msg) => sum + (now - msg.timestamp), 0) / this.inputQueue.length : 0;

    const efficiency = this.messageCount > 0 ?
      (this.messageCount - this.droppedCount) / this.messageCount : 1;

    return {
      throttleId: this.id,
      symbol: this.inputQueue.length > 0 ? this.inputQueue[0].symbol : '',
      inputRate,
      outputRate,
      queueSize: this.inputQueue.length,
      droppedCount: this.droppedCount,
      latency: averageQueueTime,
      efficiency
    };
  }

  // ==========================================
  // PRIVATE METHODS
  // ==========================================

  private initialize(): void {
    // Setup throttle timer
    const interval = 1000 / this.config.maxUpdatesPerSecond;
    this.throttleTimer = setInterval(() => {
      this.processQueue();
    }, interval);

    // Setup metrics collection
    this.metricsTimer = setInterval(() => {
      const metrics = this.getMetrics();
      this.emit('metrics_updated', metrics);
    }, 1000);
  }

  private processQueue(): void {
    if (this.inputQueue.length === 0) return;

    const now = Date.now();
    const minInterval = 1000 / this.config.maxUpdatesPerSecond;

    // Check if enough time has passed
    if (now - this.lastEmitTime < minInterval) {
      return;
    }

    // Select message based on strategy
    const message = this.selectMessage();
    if (message) {
      this.outputSubject.next(message);
      this.lastEmitTime = now;
    }
  }

  private selectMessage(): StreamMessage | null {
    if (this.inputQueue.length === 0) return null;

    let selectedMessage: StreamMessage;

    switch (this.config.strategy) {
      case 'drop_oldest':
        selectedMessage = this.inputQueue.shift()!;
        // Drop remaining old messages if any
        if (this.inputQueue.length > 1) {
          const kept = this.inputQueue.pop()!;
          this.droppedCount += this.inputQueue.length;
          this.inputQueue.length = 0;
          this.inputQueue.push(kept);
        }
        break;

      case 'drop_newest':
        selectedMessage = this.inputQueue.shift()!;
        this.droppedCount += this.inputQueue.length;
        this.inputQueue.length = 0;
        break;

      case 'merge':
        selectedMessage = this.mergeMessages();
        this.inputQueue.length = 0;
        break;

      case 'sample':
        const randomIndex = Math.floor(Math.random() * this.inputQueue.length);
        selectedMessage = this.inputQueue[randomIndex];
        this.droppedCount += this.inputQueue.length - 1;
        this.inputQueue.length = 0;
        break;

      default:
        selectedMessage = this.inputQueue.shift()!;
        break;
    }

    return selectedMessage;
  }

  private mergeMessages(): StreamMessage {
    if (this.inputQueue.length === 1) {
      return this.inputQueue[0];
    }

    const latest = this.inputQueue[this.inputQueue.length - 1];
    const merged = { ...latest };

    // Merge data based on message type
    if (merged.type === 'data' && merged.data) {
      // Calculate average values for numeric fields
      const numericFields = ['price', 'bid', 'ask', 'volume'];
      const values: any = {};

      numericFields.forEach(field => {
        const fieldValues = this.inputQueue
          .map(msg => msg.data[field])
          .filter(val => typeof val === 'number' && !isNaN(val));

        if (fieldValues.length > 0) {
          if (field === 'volume') {
            // Sum volumes
            values[field] = fieldValues.reduce((sum, val) => sum + val, 0);
          } else {
            // Average prices
            values[field] = fieldValues.reduce((sum, val) => sum + val, 0) / fieldValues.length;
          }
        }
      });

      merged.data = { ...merged.data, ...values };
    }

    return merged;
  }

  private handleBurstOverflow(): void {
    const overflow = this.inputQueue.length - this.config.burstLimit;

    // Remove oldest messages
    this.inputQueue.splice(0, overflow);
    this.droppedCount += overflow;

    this.emit('burst_overflow', { dropped: overflow, remaining: this.inputQueue.length });
  }

  public destroy(): void {
    this.destroy$.next();
    this.destroy$.complete();

    if (this.throttleTimer) {
      clearInterval(this.throttleTimer);
    }

    if (this.metricsTimer) {
      clearInterval(this.metricsTimer);
    }

    this.inputQueue.length = 0;
    this.outputSubject.complete();
    this.removeAllListeners();
  }
}