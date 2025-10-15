/**
 * WorkerPool - High-Performance Web Workers Management
 *
 * Advanced web workers pool for parallel processing of:
 * - Technical indicator calculations
 * - Data transformations
 * - Heavy mathematical operations
 * - Real-time data processing
 *
 * Features:
 * - Dynamic worker allocation
 * - Load balancing
 * - Worker health monitoring
 * - Automatic failover
 * - Memory-efficient task distribution
 */

import * as Comlink from 'comlink';
import { EventEmitter } from 'eventemitter3';

export interface WorkerPoolConfig {
  readonly maxWorkers: number;
  readonly minWorkers: number;
  readonly workerIdleTimeout: number;
  readonly taskTimeout: number;
  readonly enableHealthMonitoring: boolean;
  readonly loadBalancingStrategy: 'round-robin' | 'least-loaded' | 'random';
  readonly retryAttempts: number;
}

export interface WorkerTask<T = any, R = any> {
  readonly id: string;
  readonly type: string;
  readonly data: T;
  readonly priority: number;
  readonly timeout?: number;
  readonly retries?: number;
  readonly onProgress?: (progress: number) => void;
  readonly onSuccess: (result: R) => void;
  readonly onError: (error: Error) => void;
}

export interface WorkerInstance {
  readonly id: string;
  readonly worker: Worker;
  readonly proxy: any;
  readonly isIdle: boolean;
  readonly taskCount: number;
  readonly lastUsed: number;
  readonly health: WorkerHealth;
  readonly currentTask?: string;
}

export interface WorkerHealth {
  readonly isHealthy: boolean;
  readonly errorCount: number;
  readonly lastError?: Error;
  readonly responseTime: number;
  readonly memoryUsage: number;
  readonly taskSuccessRate: number;
}

export interface PoolStats {
  readonly totalWorkers: number;
  readonly activeWorkers: number;
  readonly idleWorkers: number;
  readonly queuedTasks: number;
  readonly completedTasks: number;
  readonly failedTasks: number;
  readonly averageResponseTime: number;
  readonly memoryUsage: number;
}

export class WorkerPool extends EventEmitter {
  private readonly config: WorkerPoolConfig;
  private readonly workers = new Map<string, WorkerInstance>();
  private readonly taskQueue: WorkerTask[] = [];
  private readonly activeTasks = new Map<string, WorkerTask>();

  private workerCounter = 0;
  private taskCounter = 0;
  private completedTasks = 0;
  private failedTasks = 0;
  private totalResponseTime = 0;

  private healthCheckInterval?: NodeJS.Timeout;
  private cleanupInterval?: NodeJS.Timeout;

  constructor(config: Partial<WorkerPoolConfig> = {}) {
    super();

    this.config = {
      maxWorkers: Math.max(1, navigator.hardwareConcurrency || 4),
      minWorkers: 1,
      workerIdleTimeout: 60000, // 1 minute
      taskTimeout: 30000, // 30 seconds
      enableHealthMonitoring: true,
      loadBalancingStrategy: 'least-loaded',
      retryAttempts: 3,
      ...config
    };

    this.initialize();
  }

  /**
   * Initialize the worker pool
   */
  private async initialize(): Promise<void> {
    // Create minimum workers
    for (let i = 0; i < this.config.minWorkers; i++) {
      await this.createWorker();
    }

    this.startHealthMonitoring();
    this.startCleanupTask();

    this.emit('pool.initialized', {
      minWorkers: this.config.minWorkers,
      maxWorkers: this.config.maxWorkers
    });
  }

  /**
   * Execute a task with automatic worker allocation
   */
  public async executeTask<T, R>(
    type: string,
    data: T,
    options: Partial<WorkerTask<T, R>> = {}
  ): Promise<R> {
    return new Promise((resolve, reject) => {
      const task: WorkerTask<T, R> = {
        id: `task_${++this.taskCounter}`,
        type,
        data,
        priority: 1,
        timeout: this.config.taskTimeout,
        retries: 0,
        onSuccess: resolve,
        onError: reject,
        ...options
      };

      this.queueTask(task);
    });
  }

  /**
   * Execute multiple tasks in parallel
   */
  public async executeParallel<T, R>(
    tasks: Array<{ type: string; data: T; options?: Partial<WorkerTask<T, R>> }>
  ): Promise<R[]> {
    const promises = tasks.map(({ type, data, options }) =>
      this.executeTask<T, R>(type, data, options)
    );

    return Promise.all(promises);
  }

  /**
   * Execute tasks with custom load balancing
   */
  public async executeBatch<T, R>(
    type: string,
    dataArray: T[],
    batchSize?: number
  ): Promise<R[]> {
    const actualBatchSize = batchSize || Math.ceil(dataArray.length / this.workers.size);
    const batches: T[][] = [];

    for (let i = 0; i < dataArray.length; i += actualBatchSize) {
      batches.push(dataArray.slice(i, i + actualBatchSize));
    }

    const promises = batches.map(batch =>
      this.executeTask<T[], R[]>(`${type}_batch`, batch)
    );

    const results = await Promise.all(promises);
    return results.flat();
  }

  /**
   * Get pool statistics
   */
  public getStats(): PoolStats {
    const activeWorkers = Array.from(this.workers.values()).filter(w => !w.isIdle);
    const idleWorkers = Array.from(this.workers.values()).filter(w => w.isIdle);

    return {
      totalWorkers: this.workers.size,
      activeWorkers: activeWorkers.length,
      idleWorkers: idleWorkers.length,
      queuedTasks: this.taskQueue.length,
      completedTasks: this.completedTasks,
      failedTasks: this.failedTasks,
      averageResponseTime: this.completedTasks > 0 ? this.totalResponseTime / this.completedTasks : 0,
      memoryUsage: this.calculatePoolMemoryUsage()
    };
  }

  /**
   * Get worker health information
   */
  public getWorkerHealth(): WorkerHealth[] {
    return Array.from(this.workers.values()).map(w => w.health);
  }

  /**
   * Terminate all workers and cleanup
   */
  public async destroy(): Promise<void> {
    // Clear intervals
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }

    // Terminate all workers
    const terminationPromises = Array.from(this.workers.values()).map(worker =>
      this.terminateWorker(worker.id)
    );

    await Promise.all(terminationPromises);

    // Clear queues
    this.taskQueue.length = 0;
    this.activeTasks.clear();
    this.workers.clear();

    this.emit('pool.destroyed');
  }

  // Private implementation methods

  private async createWorker(): Promise<WorkerInstance> {
    const workerId = `worker_${++this.workerCounter}`;

    try {
      // Create worker with the indicator worker script
      const worker = new Worker('/workers/indicator-worker.js');
      const proxy = Comlink.wrap(worker);

      const workerInstance: WorkerInstance = {
        id: workerId,
        worker,
        proxy,
        isIdle: true,
        taskCount: 0,
        lastUsed: Date.now(),
        health: {
          isHealthy: true,
          errorCount: 0,
          responseTime: 0,
          memoryUsage: 0,
          taskSuccessRate: 100
        }
      };

      // Setup error handling
      worker.onerror = (error) => {
        this.handleWorkerError(workerId, error);
      };

      worker.onmessageerror = (error) => {
        this.handleWorkerError(workerId, error);
      };

      this.workers.set(workerId, workerInstance);

      this.emit('worker.created', { workerId, totalWorkers: this.workers.size });

      return workerInstance;

    } catch (error) {
      this.emit('worker.creation.failed', { workerId, error });
      throw error;
    }
  }

  private async terminateWorker(workerId: string): Promise<void> {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    try {
      worker.worker.terminate();
      this.workers.delete(workerId);

      this.emit('worker.terminated', { workerId, totalWorkers: this.workers.size });

    } catch (error) {
      this.emit('worker.termination.failed', { workerId, error });
    }
  }

  private queueTask<T, R>(task: WorkerTask<T, R>): void {
    // Add to queue with priority sorting
    this.taskQueue.push(task);
    this.taskQueue.sort((a, b) => b.priority - a.priority);

    this.emit('task.queued', { taskId: task.id, queueLength: this.taskQueue.length });

    // Try to process the queue
    this.processQueue();
  }

  private async processQueue(): Promise<void> {
    while (this.taskQueue.length > 0) {
      const worker = await this.getAvailableWorker();
      if (!worker) break;

      const task = this.taskQueue.shift()!;
      await this.assignTaskToWorker(task, worker);
    }
  }

  private async getAvailableWorker(): Promise<WorkerInstance | null> {
    // First, try to find an idle worker
    const idleWorkers = Array.from(this.workers.values()).filter(w => w.isIdle && w.health.isHealthy);

    if (idleWorkers.length > 0) {
      return this.selectWorkerByStrategy(idleWorkers);
    }

    // If no idle workers and we can create more
    if (this.workers.size < this.config.maxWorkers) {
      try {
        return await this.createWorker();
      } catch (error) {
        this.emit('worker.scaling.failed', { error });
      }
    }

    // Wait for a worker to become available
    return null;
  }

  private selectWorkerByStrategy(workers: WorkerInstance[]): WorkerInstance {
    switch (this.config.loadBalancingStrategy) {
      case 'round-robin':
        return workers[this.taskCounter % workers.length];

      case 'least-loaded':
        return workers.reduce((min, worker) =>
          worker.taskCount < min.taskCount ? worker : min
        );

      case 'random':
        return workers[Math.floor(Math.random() * workers.length)];

      default:
        return workers[0];
    }
  }

  private async assignTaskToWorker<T, R>(task: WorkerTask<T, R>, worker: WorkerInstance): Promise<void> {
    const startTime = performance.now();

    // Update worker state
    worker.isIdle = false;
    worker.currentTask = task.id;
    worker.taskCount++;
    worker.lastUsed = Date.now();

    // Add to active tasks
    this.activeTasks.set(task.id, task);

    this.emit('task.started', { taskId: task.id, workerId: worker.id });

    try {
      // Setup timeout
      const timeoutId = setTimeout(() => {
        this.handleTaskTimeout(task.id);
      }, task.timeout || this.config.taskTimeout);

      // Execute task
      const result = await worker.proxy.calculateIndicator(task.data.data, task.data.config);

      clearTimeout(timeoutId);

      // Update metrics
      const responseTime = performance.now() - startTime;
      this.updateWorkerHealth(worker, true, responseTime);
      this.completedTasks++;
      this.totalResponseTime += responseTime;

      // Cleanup
      this.completeTask(task.id, worker);

      this.emit('task.completed', { taskId: task.id, workerId: worker.id, responseTime });

      // Call success callback
      task.onSuccess(result);

    } catch (error) {
      this.handleTaskError(task, worker, error as Error);
    }
  }

  private completeTask(taskId: string, worker: WorkerInstance): void {
    worker.isIdle = true;
    worker.currentTask = undefined;
    this.activeTasks.delete(taskId);

    // Process next queued task
    this.processQueue();
  }

  private handleTaskError<T, R>(task: WorkerTask<T, R>, worker: WorkerInstance, error: Error): void {
    this.updateWorkerHealth(worker, false, 0, error);

    const shouldRetry = (task.retries || 0) < this.config.retryAttempts;

    if (shouldRetry) {
      // Retry task
      task.retries = (task.retries || 0) + 1;
      this.completeTask(task.id, worker);
      this.queueTask(task);

      this.emit('task.retried', { taskId: task.id, attempt: task.retries });

    } else {
      // Task failed permanently
      this.failedTasks++;
      this.completeTask(task.id, worker);

      this.emit('task.failed', { taskId: task.id, error });

      task.onError(error);
    }
  }

  private handleTaskTimeout(taskId: string): void {
    const task = this.activeTasks.get(taskId);
    if (!task) return;

    const worker = Array.from(this.workers.values()).find(w => w.currentTask === taskId);
    if (!worker) return;

    const timeoutError = new Error(`Task ${taskId} timed out`);
    this.handleTaskError(task, worker, timeoutError);
  }

  private handleWorkerError(workerId: string, error: ErrorEvent): void {
    const worker = this.workers.get(workerId);
    if (!worker) return;

    this.updateWorkerHealth(worker, false, 0, new Error(error.message));

    this.emit('worker.error', { workerId, error: error.message });

    // If worker is consistently unhealthy, terminate and recreate
    if (worker.health.errorCount > 5) {
      this.terminateWorker(workerId);
      this.createWorker(); // Replace with new worker
    }
  }

  private updateWorkerHealth(
    worker: WorkerInstance,
    success: boolean,
    responseTime: number,
    error?: Error
  ): void {
    const health = worker.health;

    if (success) {
      health.responseTime = (health.responseTime * 0.9) + (responseTime * 0.1); // Moving average
      health.taskSuccessRate = Math.min(100, health.taskSuccessRate + 1);
    } else {
      health.errorCount++;
      health.lastError = error;
      health.taskSuccessRate = Math.max(0, health.taskSuccessRate - 5);
    }

    health.isHealthy = health.errorCount < 3 && health.taskSuccessRate > 70;
  }

  private calculatePoolMemoryUsage(): number {
    // Estimate memory usage based on active workers and tasks
    const baseWorkerMemory = 10 * 1024 * 1024; // 10MB per worker estimate
    const taskMemory = this.activeTasks.size * 1024 * 1024; // 1MB per active task estimate

    return (this.workers.size * baseWorkerMemory) + taskMemory;
  }

  private startHealthMonitoring(): void {
    if (!this.config.enableHealthMonitoring) return;

    this.healthCheckInterval = setInterval(() => {
      this.performHealthCheck();
    }, 10000); // Check every 10 seconds
  }

  private performHealthCheck(): void {
    const unhealthyWorkers: string[] = [];

    for (const [workerId, worker] of this.workers) {
      if (!worker.health.isHealthy) {
        unhealthyWorkers.push(workerId);
      }
    }

    if (unhealthyWorkers.length > 0) {
      this.emit('health.check.failed', { unhealthyWorkers });

      // Restart unhealthy workers
      unhealthyWorkers.forEach(workerId => {
        this.terminateWorker(workerId);
        this.createWorker();
      });
    }

    this.emit('health.check.completed', {
      totalWorkers: this.workers.size,
      healthyWorkers: this.workers.size - unhealthyWorkers.length,
      unhealthyWorkers: unhealthyWorkers.length
    });
  }

  private startCleanupTask(): void {
    this.cleanupInterval = setInterval(() => {
      this.cleanupIdleWorkers();
    }, 30000); // Cleanup every 30 seconds
  }

  private cleanupIdleWorkers(): void {
    const now = Date.now();
    const workersToTerminate: string[] = [];

    for (const [workerId, worker] of this.workers) {
      const idleTime = now - worker.lastUsed;

      if (
        worker.isIdle &&
        idleTime > this.config.workerIdleTimeout &&
        this.workers.size > this.config.minWorkers
      ) {
        workersToTerminate.push(workerId);
      }
    }

    workersToTerminate.forEach(workerId => {
      this.terminateWorker(workerId);
    });

    if (workersToTerminate.length > 0) {
      this.emit('workers.cleaned', { terminatedWorkers: workersToTerminate.length });
    }
  }
}

// Singleton instance
let workerPoolInstance: WorkerPool | null = null;

export function getWorkerPool(config?: Partial<WorkerPoolConfig>): WorkerPool {
  if (!workerPoolInstance) {
    workerPoolInstance = new WorkerPool(config);
  }
  return workerPoolInstance;
}

export function resetWorkerPool(): void {
  if (workerPoolInstance) {
    workerPoolInstance.destroy();
    workerPoolInstance = null;
  }
}