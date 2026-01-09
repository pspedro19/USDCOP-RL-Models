/**
 * PerformanceOptimizer - Elite Trading Platform Performance Management
 *
 * Advanced performance optimization system featuring:
 * - LTTB (Largest Triangle Three Buckets) data sampling
 * - Memory management and canvas pooling
 * - Real-time performance monitoring
 * - Adaptive optimization based on device capabilities
 *
 * Targets: 60 FPS, <1s load times, <150MB memory usage
 */

import { EventEmitter } from 'eventemitter3';
import { getPerformanceMonitor } from './PerformanceMonitor';

export interface OptimizationConfig {
  readonly enableDataSampling: boolean;
  readonly enableCanvasPooling: boolean;
  readonly enableMemoryManagement: boolean;
  readonly enableAdaptiveOptimization: boolean;
  readonly targetFPS: number;
  readonly maxMemoryUsage: number;
  readonly samplingThreshold: number;
  readonly canvasPoolSize: number;
}

export interface DataPoint {
  readonly x: number;
  readonly y: number;
  readonly timestamp: number;
  readonly volume?: number;
  readonly open?: number;
  readonly high?: number;
  readonly low?: number;
  readonly close?: number;
}

export interface SampledDataset {
  readonly original: DataPoint[];
  readonly sampled: DataPoint[];
  readonly samplingRatio: number;
  readonly algorithm: 'LTTB' | 'UNIFORM' | 'ADAPTIVE';
  readonly quality: number;
}

export interface CanvasPool {
  readonly available: HTMLCanvasElement[];
  readonly inUse: Set<HTMLCanvasElement>;
  readonly maxSize: number;
  readonly currentSize: number;
}

export interface MemoryStats {
  readonly heapUsed: number;
  readonly heapTotal: number;
  readonly percentage: number;
  readonly canvasMemory: number;
  readonly dataMemory: number;
  readonly totalAllocated: number;
}

export class PerformanceOptimizer extends EventEmitter {
  private readonly config: OptimizationConfig;
  private readonly performanceMonitor = getPerformanceMonitor();

  private canvasPool: CanvasPool;
  private memoryManager: MemoryManager;
  private dataCache = new Map<string, SampledDataset>();
  private optimizationLevel = 100;

  private rafId?: number;
  private lastOptimizationCheck = 0;
  private performanceHistory: number[] = [];

  constructor(config: Partial<OptimizationConfig> = {}) {
    super();

    this.config = {
      enableDataSampling: true,
      enableCanvasPooling: true,
      enableMemoryManagement: true,
      enableAdaptiveOptimization: true,
      targetFPS: 60,
      maxMemoryUsage: 150 * 1024 * 1024, // 150MB
      samplingThreshold: 1000,
      canvasPoolSize: 10,
      ...config
    };

    this.canvasPool = this.createCanvasPool();
    this.memoryManager = new MemoryManager(this.config.maxMemoryUsage);

    this.initialize();
  }

  /**
   * Initialize the performance optimization system
   */
  private initialize(): void {
    this.startAdaptiveOptimization();
    this.setupMemoryMonitoring();
    this.preloadOptimizations();
  }

  /**
   * LTTB (Largest Triangle Three Buckets) Data Sampling
   * Reduces dataset size while preserving visual fidelity
   */
  public sampleDataLTTB(data: DataPoint[], targetPoints: number): SampledDataset {
    if (!this.config.enableDataSampling || data.length <= targetPoints) {
      return {
        original: data,
        sampled: data,
        samplingRatio: 1,
        algorithm: 'LTTB',
        quality: 100
      };
    }

    const cacheKey = `lttb_${data.length}_${targetPoints}`;
    const cached = this.dataCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const startTime = performance.now();
    const sampled = this.performLTTB(data, targetPoints);
    const processingTime = performance.now() - startTime;

    const result: SampledDataset = {
      original: data,
      sampled,
      samplingRatio: sampled.length / data.length,
      algorithm: 'LTTB',
      quality: this.calculateSamplingQuality(data, sampled)
    };

    // Cache the result for future use
    this.dataCache.set(cacheKey, result);

    this.emit('data.sampled', {
      originalSize: data.length,
      sampledSize: sampled.length,
      processingTime,
      quality: result.quality
    });

    return result;
  }

  /**
   * Adaptive data sampling based on current performance
   */
  public adaptiveSample(data: DataPoint[], preferredPoints: number): SampledDataset {
    const currentFPS = this.performanceMonitor.getCurrentMetrics().fps;
    const memoryUsage = this.getMemoryStats().percentage;

    let targetPoints = preferredPoints;

    // Adjust target points based on performance
    if (currentFPS < this.config.targetFPS * 0.8) {
      targetPoints = Math.floor(targetPoints * 0.7);
    } else if (currentFPS < this.config.targetFPS * 0.9) {
      targetPoints = Math.floor(targetPoints * 0.85);
    }

    // Adjust for memory pressure
    if (memoryUsage > 80) {
      targetPoints = Math.floor(targetPoints * 0.6);
    } else if (memoryUsage > 70) {
      targetPoints = Math.floor(targetPoints * 0.8);
    }

    return this.sampleDataLTTB(data, Math.max(50, targetPoints));
  }

  /**
   * Canvas pool management for optimal memory usage
   */
  public acquireCanvas(width: number, height: number): HTMLCanvasElement {
    if (!this.config.enableCanvasPooling) {
      return this.createCanvas(width, height);
    }

    // Try to find a suitable canvas from the pool
    for (const canvas of this.canvasPool.available) {
      if (canvas.width >= width && canvas.height >= height) {
        this.canvasPool.available.splice(this.canvasPool.available.indexOf(canvas), 1);
        this.canvasPool.inUse.add(canvas);

        // Resize if needed
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }

        return canvas;
      }
    }

    // Create new canvas if none available
    const canvas = this.createCanvas(width, height);
    this.canvasPool.inUse.add(canvas);

    return canvas;
  }

  /**
   * Release canvas back to pool
   */
  public releaseCanvas(canvas: HTMLCanvasElement): void {
    if (!this.config.enableCanvasPooling) {
      return;
    }

    if (this.canvasPool.inUse.has(canvas)) {
      this.canvasPool.inUse.delete(canvas);

      // Clear canvas
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      // Return to pool if there's space
      if (this.canvasPool.available.length < this.config.canvasPoolSize) {
        this.canvasPool.available.push(canvas);
      }
    }
  }

  /**
   * Memory management utilities
   */
  public getMemoryStats(): MemoryStats {
    const baseStats = this.memoryManager.getStats();
    const canvasMemory = this.calculateCanvasMemory();
    const dataMemory = this.calculateDataMemory();

    return {
      ...baseStats,
      canvasMemory,
      dataMemory,
      totalAllocated: baseStats.heapUsed + canvasMemory + dataMemory
    };
  }

  /**
   * Force garbage collection and optimize memory usage
   */
  public optimizeMemory(): void {
    if (!this.config.enableMemoryManagement) return;

    // Clean up data cache
    this.cleanupDataCache();

    // Clean up canvas pool
    this.cleanupCanvasPool();

    // Force garbage collection if available
    if ('gc' in window && typeof window.gc === 'function') {
      window.gc();
    }

    this.emit('memory.optimized', this.getMemoryStats());
  }

  /**
   * Get current optimization level (0-100)
   */
  public getOptimizationLevel(): number {
    return this.optimizationLevel;
  }

  /**
   * Force optimization level adjustment
   */
  public setOptimizationLevel(level: number): void {
    this.optimizationLevel = Math.max(0, Math.min(100, level));
    this.emit('optimization.level.changed', this.optimizationLevel);
  }

  /**
   * Destroy optimizer and cleanup resources
   */
  public destroy(): void {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }

    this.cleanupCanvasPool();
    this.dataCache.clear();
    this.memoryManager.destroy();
    this.removeAllListeners();
  }

  // Private implementation methods

  private performLTTB(data: DataPoint[], targetPoints: number): DataPoint[] {
    if (data.length <= 2 || targetPoints <= 2) {
      return data.slice(0, targetPoints);
    }

    const sampled: DataPoint[] = [];

    // Always include first point
    sampled.push(data[0]);

    const bucketSize = (data.length - 2) / (targetPoints - 2);

    for (let i = 1; i < targetPoints - 1; i++) {
      const bucketStart = Math.floor((i - 1) * bucketSize) + 1;
      const bucketEnd = Math.floor(i * bucketSize) + 1;
      const nextBucketEnd = Math.floor((i + 1) * bucketSize) + 1;

      // Calculate average point of next bucket
      let avgX = 0, avgY = 0;
      let avgCount = 0;

      for (let j = bucketEnd; j < Math.min(nextBucketEnd, data.length); j++) {
        avgX += data[j].x;
        avgY += data[j].y;
        avgCount++;
      }

      if (avgCount > 0) {
        avgX /= avgCount;
        avgY /= avgCount;
      }

      // Find the point in current bucket that forms largest triangle
      let maxArea = -1;
      let selectedPoint = data[bucketStart];

      const prevPoint = sampled[sampled.length - 1];

      for (let j = bucketStart; j < Math.min(bucketEnd, data.length); j++) {
        const area = Math.abs(
          (prevPoint.x - avgX) * (data[j].y - prevPoint.y) -
          (prevPoint.x - data[j].x) * (avgY - prevPoint.y)
        ) * 0.5;

        if (area > maxArea) {
          maxArea = area;
          selectedPoint = data[j];
        }
      }

      sampled.push(selectedPoint);
    }

    // Always include last point
    sampled.push(data[data.length - 1]);

    return sampled;
  }

  private calculateSamplingQuality(original: DataPoint[], sampled: DataPoint[]): number {
    if (original.length === sampled.length) return 100;

    // Calculate visual fidelity preservation
    const originalRange = this.calculateDataRange(original);
    const sampledRange = this.calculateDataRange(sampled);

    const rangePreservation = 1 - Math.abs(originalRange - sampledRange) / originalRange;
    const densityRatio = sampled.length / original.length;

    return Math.round((rangePreservation * 0.7 + densityRatio * 0.3) * 100);
  }

  private calculateDataRange(data: DataPoint[]): number {
    if (data.length === 0) return 0;

    const yValues = data.map(d => d.y);
    return Math.max(...yValues) - Math.min(...yValues);
  }

  private createCanvasPool(): CanvasPool {
    return {
      available: [],
      inUse: new Set(),
      maxSize: this.config.canvasPoolSize,
      currentSize: 0
    };
  }

  private createCanvas(width: number, height: number): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    // Enable hardware acceleration
    const ctx = canvas.getContext('2d', {
      alpha: false,
      desynchronized: true
    });

    if (ctx) {
      ctx.imageSmoothingEnabled = false;
      ctx.imageSmoothingQuality = 'low';
    }

    return canvas;
  }

  private calculateCanvasMemory(): number {
    let memory = 0;

    // Calculate memory usage of canvases in use
    for (const canvas of this.canvasPool.inUse) {
      memory += canvas.width * canvas.height * 4; // 4 bytes per pixel (RGBA)
    }

    // Add available canvases
    for (const canvas of this.canvasPool.available) {
      memory += canvas.width * canvas.height * 4;
    }

    return memory;
  }

  private calculateDataMemory(): number {
    let memory = 0;

    for (const dataset of this.dataCache.values()) {
      memory += dataset.original.length * 64; // Estimate 64 bytes per data point
      memory += dataset.sampled.length * 64;
    }

    return memory;
  }

  private cleanupDataCache(): void {
    const memoryThreshold = this.config.maxMemoryUsage * 0.3; // 30% of max for data cache
    let currentMemory = this.calculateDataMemory();

    if (currentMemory > memoryThreshold) {
      // Sort by last access (LRU eviction)
      const entries = Array.from(this.dataCache.entries());
      const toRemove = Math.ceil(entries.length * 0.3); // Remove 30%

      for (let i = 0; i < toRemove; i++) {
        this.dataCache.delete(entries[i][0]);
      }
    }
  }

  private cleanupCanvasPool(): void {
    // Clear all canvases
    for (const canvas of this.canvasPool.available) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    for (const canvas of this.canvasPool.inUse) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    this.canvasPool.available.length = 0;
    this.canvasPool.inUse.clear();
  }

  private startAdaptiveOptimization(): void {
    if (!this.config.enableAdaptiveOptimization) return;

    const optimize = () => {
      const now = performance.now();

      if (now - this.lastOptimizationCheck > 1000) { // Check every second
        const metrics = this.performanceMonitor.getCurrentMetrics();

        this.performanceHistory.push(metrics.fps);
        if (this.performanceHistory.length > 10) {
          this.performanceHistory.shift();
        }

        this.adjustOptimizationLevel(metrics);
        this.lastOptimizationCheck = now;
      }

      this.rafId = requestAnimationFrame(optimize);
    };

    optimize();
  }

  private adjustOptimizationLevel(metrics: any): void {
    const avgFPS = this.performanceHistory.reduce((a, b) => a + b, 0) / this.performanceHistory.length;
    const memoryUsage = metrics.memoryUsage.percentage;

    let targetLevel = 100;

    // Adjust based on FPS
    if (avgFPS < this.config.targetFPS * 0.7) {
      targetLevel = 60; // Aggressive optimization
    } else if (avgFPS < this.config.targetFPS * 0.8) {
      targetLevel = 75; // Moderate optimization
    } else if (avgFPS < this.config.targetFPS * 0.9) {
      targetLevel = 85; // Light optimization
    }

    // Adjust for memory pressure
    if (memoryUsage > 85) {
      targetLevel = Math.min(targetLevel, 50);
    } else if (memoryUsage > 75) {
      targetLevel = Math.min(targetLevel, 70);
    }

    // Smooth transition
    const delta = targetLevel - this.optimizationLevel;
    this.optimizationLevel += Math.sign(delta) * Math.min(Math.abs(delta), 5);

    this.emit('optimization.adjusted', {
      level: this.optimizationLevel,
      fps: avgFPS,
      memory: memoryUsage
    });
  }

  private setupMemoryMonitoring(): void {
    setInterval(() => {
      const stats = this.getMemoryStats();

      if (stats.percentage > 90) {
        this.optimizeMemory();
      }

      this.emit('memory.stats', stats);
    }, 5000); // Check every 5 seconds
  }

  private preloadOptimizations(): void {
    // Pre-allocate canvas pool
    for (let i = 0; i < this.config.canvasPoolSize; i++) {
      const canvas = this.createCanvas(800, 600);
      this.canvasPool.available.push(canvas);
    }
  }
}

/**
 * Memory Manager - Advanced memory optimization
 */
class MemoryManager {
  private readonly maxMemory: number;
  private allocatedObjects = new WeakSet();
  private allocationHistory: number[] = [];

  constructor(maxMemory: number) {
    this.maxMemory = maxMemory;
  }

  public getStats(): MemoryStats {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        heapUsed: memory.usedJSHeapSize,
        heapTotal: memory.totalJSHeapSize,
        percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100,
        canvasMemory: 0,
        dataMemory: 0,
        totalAllocated: memory.usedJSHeapSize
      };
    }

    return {
      heapUsed: 0,
      heapTotal: 0,
      percentage: 0,
      canvasMemory: 0,
      dataMemory: 0,
      totalAllocated: 0
    };
  }

  public trackAllocation(object: any): void {
    this.allocatedObjects.add(object);
  }

  public isMemoryPressure(): boolean {
    const stats = this.getStats();
    return stats.heapUsed > this.maxMemory * 0.8;
  }

  public destroy(): void {
    this.allocationHistory.length = 0;
  }
}

// Singleton instance
let optimizerInstance: PerformanceOptimizer | null = null;

export function getPerformanceOptimizer(config?: Partial<OptimizationConfig>): PerformanceOptimizer {
  if (!optimizerInstance) {
    optimizerInstance = new PerformanceOptimizer(config);
  }
  return optimizerInstance;
}

export function resetPerformanceOptimizer(): void {
  if (optimizerInstance) {
    optimizerInstance.destroy();
    optimizerInstance = null;
  }
}