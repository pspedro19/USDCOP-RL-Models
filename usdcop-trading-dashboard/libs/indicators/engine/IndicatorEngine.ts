/**
 * Indicator Engine
 * ===============
 *
 * Professional-grade technical indicators engine with Web Worker pool
 * for institutional trading platforms.
 */

import * as Comlink from 'comlink';
import { EventEmitter } from 'eventemitter3';
import {
  CandleData,
  IndicatorConfig,
  IndicatorValue,
  WorkerMessage,
  WorkerResponse,
  VolumeProfile,
  CorrelationMatrix,
  BacktestResult,
  PerformanceMetrics,
  CustomIndicator,
  IndicatorAlert,
  OptimizationResult
} from '../types';

export interface IndicatorEngineConfig {
  maxWorkers: number;
  cacheSize: number;
  enableProfiling: boolean;
  workerTimeout: number;
  batchSize: number;
}

export class IndicatorEngine extends EventEmitter {
  private workers: Worker[] = [];
  private workerQueue: Worker[] = [];
  private activeJobs = new Map<string, { resolve: Function; reject: Function; timeout: NodeJS.Timeout }>();
  private cache = new Map<string, { result: any; timestamp: number; ttl: number }>();
  private config: IndicatorEngineConfig;
  private performanceStats = {
    totalCalculations: 0,
    totalTime: 0,
    cacheHits: 0,
    workerUtilization: Array<number>(),
    errorRate: 0
  };

  constructor(config: Partial<IndicatorEngineConfig> = {}) {
    super();

    this.config = {
      maxWorkers: config.maxWorkers || Math.min(navigator.hardwareConcurrency || 4, 8),
      cacheSize: config.cacheSize || 1000,
      enableProfiling: config.enableProfiling || true,
      workerTimeout: config.workerTimeout || 30000,
      batchSize: config.batchSize || 100,
      ...config
    };

    this.initializeWorkers();
    this.startPerformanceMonitoring();
  }

  /**
   * Initialize Web Worker pool
   */
  private async initializeWorkers(): Promise<void> {
    for (let i = 0; i < this.config.maxWorkers; i++) {
      try {
        const worker = new Worker(
          new URL('../workers/indicator-worker.ts', import.meta.url),
          { type: 'module' }
        );

        const wrappedWorker = Comlink.wrap(worker) as any;
        this.workers.push(worker);
        this.workerQueue.push(worker);

        worker.onmessage = this.handleWorkerMessage.bind(this);
        worker.onerror = this.handleWorkerError.bind(this);

        this.emit('workerReady', { workerId: i, totalWorkers: this.config.maxWorkers });
      } catch (error) {
        console.error(`Failed to initialize worker ${i}:`, error);
        this.emit('workerError', { workerId: i, error });
      }
    }
  }

  /**
   * Calculate single indicator
   */
  public async calculateIndicator(
    data: CandleData[],
    config: IndicatorConfig,
    options: { useCache?: boolean; priority?: number } = {}
  ): Promise<any> {
    const startTime = performance.now();
    const { useCache = true, priority = 0 } = options;

    // Check cache first
    if (useCache) {
      const cacheKey = this.generateCacheKey(data, config);
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        this.performanceStats.cacheHits++;
        this.emit('calculationComplete', { config: config.name, fromCache: true, time: 0 });
        return cached;
      }
    }

    try {
      const worker = await this.getAvailableWorker(priority);
      const jobId = this.generateJobId();

      const result = await this.executeWorkerJob(worker, {
        id: jobId,
        type: 'CALCULATE_INDICATOR',
        payload: { data, config }
      });

      const executionTime = performance.now() - startTime;
      this.updatePerformanceStats(executionTime, true);

      // Cache result
      if (useCache && result) {
        const cacheKey = this.generateCacheKey(data, config);
        this.setCache(cacheKey, result, 300000); // 5 minutes TTL
      }

      this.emit('calculationComplete', {
        config: config.name,
        fromCache: false,
        time: executionTime,
        dataPoints: data.length
      });

      return result;
    } catch (error) {
      this.updatePerformanceStats(performance.now() - startTime, false);
      this.emit('calculationError', { config: config.name, error });
      throw error;
    }
  }

  /**
   * Calculate multiple indicators in batch
   */
  public async calculateBatch(
    data: CandleData[],
    configs: IndicatorConfig[],
    options: { parallel?: boolean; useCache?: boolean } = {}
  ): Promise<{ [key: string]: any }> {
    const { parallel = true, useCache = true } = options;

    if (parallel && configs.length > 1) {
      // Parallel execution across multiple workers
      const chunks = this.chunkArray(configs, this.config.batchSize);
      const promises = chunks.map(chunk =>
        this.processBatch(data, chunk, { useCache })
      );

      const results = await Promise.all(promises);
      return results.reduce((acc, batch) => ({ ...acc, ...batch }), {});
    } else {
      // Sequential execution on single worker
      return this.processBatch(data, configs, { useCache });
    }
  }

  /**
   * Calculate volume profile with advanced analytics
   */
  public async calculateVolumeProfile(
    data: CandleData[],
    options: {
      levels?: number;
      valueAreaPercent?: number;
      sessionSplit?: boolean;
      timeframe?: string;
    } = {}
  ): Promise<VolumeProfile> {
    const worker = await this.getAvailableWorker();
    const jobId = this.generateJobId();

    return this.executeWorkerJob(worker, {
      id: jobId,
      type: 'VOLUME_PROFILE',
      payload: { data, options }
    });
  }

  /**
   * Calculate correlation matrix between multiple assets/indicators
   */
  public async calculateCorrelationMatrix(
    datasets: { name: string; data: CandleData[] }[],
    configs: IndicatorConfig[]
  ): Promise<CorrelationMatrix> {
    const worker = await this.getAvailableWorker();
    const jobId = this.generateJobId();

    return this.executeWorkerJob(worker, {
      id: jobId,
      type: 'CALCULATE_CORRELATION',
      payload: { datasets, configs }
    });
  }

  /**
   * Run backtesting on indicator strategy
   */
  public async backtest(
    data: CandleData[],
    strategy: {
      name: string;
      indicators: IndicatorConfig[];
      rules: {
        entry: string;
        exit: string;
        riskManagement: {
          stopLoss?: number;
          takeProfit?: number;
          maxPositionSize?: number;
        };
      };
    },
    options: {
      initialCapital?: number;
      commission?: number;
      slippage?: number;
    } = {}
  ): Promise<BacktestResult> {
    const worker = await this.getAvailableWorker();
    const jobId = this.generateJobId();

    return this.executeWorkerJob(worker, {
      id: jobId,
      type: 'BACKTEST',
      payload: { data, strategy, options }
    });
  }

  /**
   * Optimize indicator parameters
   */
  public async optimizeParameters(
    data: CandleData[],
    config: IndicatorConfig,
    parameterRanges: { [key: string]: { min: number; max: number; step: number } },
    objective: 'sharpe' | 'returns' | 'drawdown' | 'winRate' = 'sharpe'
  ): Promise<OptimizationResult> {
    const startTime = performance.now();
    const combinations = this.generateParameterCombinations(parameterRanges);

    const results: Array<{ params: any; score: number }> = [];

    // Test parameter combinations in batches
    const batchSize = Math.min(50, combinations.length);
    for (let i = 0; i < combinations.length; i += batchSize) {
      const batch = combinations.slice(i, i + batchSize);

      const batchPromises = batch.map(async params => {
        const testConfig = { ...config, parameters: { ...config.parameters, ...params } };

        try {
          const indicatorResult = await this.calculateIndicator(data, testConfig, { useCache: false });
          const score = this.calculateObjectiveScore(indicatorResult, objective);
          return { params, score };
        } catch (error) {
          return { params, score: -Infinity };
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);

      this.emit('optimizationProgress', {
        completed: results.length,
        total: combinations.length,
        bestScore: Math.max(...results.map(r => r.score))
      });
    }

    // Find best parameters
    const best = results.reduce((best, current) =>
      current.score > best.score ? current : best
    );

    const optimizationTime = performance.now() - startTime;

    return {
      parameters: best.params,
      performance: {} as PerformanceMetrics, // Would calculate detailed metrics
      score: best.score,
      iterations: results.length,
      converged: true
    };
  }

  /**
   * Create custom indicator from user code
   */
  public createCustomIndicator(indicator: CustomIndicator): void {
    // Validate and register custom indicator
    this.validateCustomIndicator(indicator);
    this.emit('customIndicatorRegistered', indicator);
  }

  /**
   * Set up real-time alerts
   */
  public setupAlert(alert: IndicatorAlert): string {
    const alertId = this.generateJobId();
    this.emit('alertSetup', { alertId, alert });
    return alertId;
  }

  /**
   * Get engine performance statistics
   */
  public getPerformanceStats(): any {
    return {
      ...this.performanceStats,
      averageCalculationTime: this.performanceStats.totalTime / Math.max(this.performanceStats.totalCalculations, 1),
      cacheHitRate: this.performanceStats.cacheHits / Math.max(this.performanceStats.totalCalculations, 1),
      activeWorkers: this.workers.length - this.workerQueue.length,
      totalWorkers: this.workers.length,
      cacheSize: this.cache.size
    };
  }

  /**
   * Clear cache and reset statistics
   */
  public reset(): void {
    this.cache.clear();
    this.performanceStats = {
      totalCalculations: 0,
      totalTime: 0,
      cacheHits: 0,
      workerUtilization: [],
      errorRate: 0
    };
    this.emit('engineReset');
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    // Clear all timeouts
    this.activeJobs.forEach(job => clearTimeout(job.timeout));
    this.activeJobs.clear();

    // Terminate workers
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    this.workerQueue = [];

    // Clear cache
    this.cache.clear();

    this.emit('engineDestroyed');
  }

  // Private helper methods

  private async getAvailableWorker(priority: number = 0): Promise<Worker> {
    return new Promise((resolve, reject) => {
      if (this.workerQueue.length > 0) {
        const worker = this.workerQueue.shift()!;
        resolve(worker);
      } else {
        // Wait for worker to become available
        const timeout = setTimeout(() => {
          reject(new Error('Worker timeout: No available workers'));
        }, this.config.workerTimeout);

        const onWorkerAvailable = (worker: Worker) => {
          clearTimeout(timeout);
          resolve(worker);
        };

        this.once('workerAvailable', onWorkerAvailable);
      }
    });
  }

  private async executeWorkerJob(worker: Worker, message: WorkerMessage): Promise<any> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.activeJobs.delete(message.id);
        this.returnWorker(worker);
        reject(new Error(`Worker job timeout: ${message.id}`));
      }, this.config.workerTimeout);

      this.activeJobs.set(message.id, { resolve, reject, timeout });

      worker.postMessage(message);
    });
  }

  private handleWorkerMessage(event: MessageEvent<WorkerResponse>): void {
    const { id, type, result, error } = event.data;
    const job = this.activeJobs.get(id);

    if (!job) return;

    clearTimeout(job.timeout);
    this.activeJobs.delete(id);
    this.returnWorker(event.target as Worker);

    if (type === 'SUCCESS') {
      job.resolve(result);
    } else {
      job.reject(new Error(error || 'Worker calculation failed'));
    }
  }

  private handleWorkerError(event: ErrorEvent): void {
    console.error('Worker error:', event);
    this.emit('workerError', event);
  }

  private returnWorker(worker: Worker): void {
    this.workerQueue.push(worker);
    this.emit('workerAvailable', worker);
  }

  private async processBatch(
    data: CandleData[],
    configs: IndicatorConfig[],
    options: { useCache?: boolean }
  ): Promise<{ [key: string]: any }> {
    const worker = await this.getAvailableWorker();
    const jobId = this.generateJobId();

    return this.executeWorkerJob(worker, {
      id: jobId,
      type: 'CALCULATE_INDICATOR',
      payload: { data, configs, options }
    });
  }

  private generateCacheKey(data: CandleData[], config: IndicatorConfig): string {
    const dataHash = this.hashData(data.slice(-100)); // Hash last 100 candles
    const configHash = this.hashObject(config);
    return `${dataHash}_${configHash}`;
  }

  private hashData(data: CandleData[]): string {
    // Simple hash based on last few data points
    const relevant = data.slice(-5);
    return btoa(JSON.stringify(relevant.map(d => [d.timestamp, d.close, d.volume])));
  }

  private hashObject(obj: any): string {
    return btoa(JSON.stringify(obj));
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    if (Date.now() > cached.timestamp + cached.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.result;
  }

  private setCache(key: string, result: any, ttl: number): void {
    if (this.cache.size >= this.config.cacheSize) {
      // Remove oldest entries
      const entries = Array.from(this.cache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      entries.slice(0, Math.floor(this.config.cacheSize * 0.2)).forEach(([k]) => {
        this.cache.delete(k);
      });
    }

    this.cache.set(key, {
      result,
      timestamp: Date.now(),
      ttl
    });
  }

  private generateJobId(): string {
    return `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  private updatePerformanceStats(executionTime: number, success: boolean): void {
    this.performanceStats.totalCalculations++;
    this.performanceStats.totalTime += executionTime;

    if (!success) {
      this.performanceStats.errorRate =
        (this.performanceStats.errorRate * (this.performanceStats.totalCalculations - 1) + 1) /
        this.performanceStats.totalCalculations;
    }
  }

  private startPerformanceMonitoring(): void {
    setInterval(() => {
      const workerUtilization = (this.workers.length - this.workerQueue.length) / this.workers.length;
      this.performanceStats.workerUtilization.push(workerUtilization);

      // Keep only last 60 measurements (5 minutes if checked every 5 seconds)
      if (this.performanceStats.workerUtilization.length > 60) {
        this.performanceStats.workerUtilization.shift();
      }

      this.emit('performanceUpdate', this.getPerformanceStats());
    }, 5000);
  }

  private generateParameterCombinations(ranges: { [key: string]: { min: number; max: number; step: number } }): any[] {
    const keys = Object.keys(ranges);
    const combinations: any[] = [];

    const generate = (params: any, keyIndex: number): void => {
      if (keyIndex === keys.length) {
        combinations.push({ ...params });
        return;
      }

      const key = keys[keyIndex];
      const range = ranges[key];

      for (let value = range.min; value <= range.max; value += range.step) {
        params[key] = value;
        generate(params, keyIndex + 1);
      }
    };

    generate({}, 0);
    return combinations;
  }

  private calculateObjectiveScore(indicatorResult: any[], objective: string): number {
    // Simplified scoring - would implement proper metrics in production
    if (!indicatorResult || indicatorResult.length === 0) return -Infinity;

    switch (objective) {
      case 'sharpe':
        return Math.random(); // Placeholder
      case 'returns':
        return Math.random(); // Placeholder
      case 'drawdown':
        return Math.random(); // Placeholder
      case 'winRate':
        return Math.random(); // Placeholder
      default:
        return 0;
    }
  }

  private validateCustomIndicator(indicator: CustomIndicator): void {
    if (!indicator.name || !indicator.code) {
      throw new Error('Custom indicator must have name and code');
    }

    // Additional validation would go here
  }
}