// Performance Logger for USDCOP Trading System

import type { IPerformanceLogger, PerformanceLogEntry, LatencyMetrics } from './types';
import { getLogger } from './StructuredLogger';

interface ActiveOperation {
  operation: string;
  startTime: number;
}

/**
 * Performance logger for tracking operation latency
 * Provides detailed performance metrics and SLA monitoring
 */
export class PerformanceLogger implements IPerformanceLogger {
  private operations: Map<string, ActiveOperation> = new Map();
  private metrics: Map<string, PerformanceLogEntry[]> = new Map();
  private logger = getLogger({ service: 'PerformanceLogger' });
  private maxEntriesPerOperation: number;

  constructor(maxEntriesPerOperation: number = 1000) {
    this.maxEntriesPerOperation = maxEntriesPerOperation;
  }

  startOperation(operation: string): string {
    const operationId = `${operation}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    this.operations.set(operationId, {
      operation,
      startTime: Date.now(),
    });

    this.logger.debug(`Started operation: ${operation}`, { operation_id: operationId });

    return operationId;
  }

  endOperation(
    operationId: string,
    success: boolean = true,
    metadata?: Record<string, any>
  ): void {
    const activeOp = this.operations.get(operationId);

    if (!activeOp) {
      this.logger.warn(`Operation not found: ${operationId}`);
      return;
    }

    const durationMs = Date.now() - activeOp.startTime;
    this.operations.delete(operationId);

    const entry: PerformanceLogEntry = {
      operation: activeOp.operation,
      duration_ms: durationMs,
      timestamp: new Date().toISOString(),
      success,
      metadata,
    };

    this.addMetric(entry);

    const logLevel = success ? 'debug' : 'warn';
    this.logger[logLevel](
      `Completed operation: ${activeOp.operation}`,
      { operation_id: operationId },
      { duration_ms: durationMs, success, ...metadata }
    );
  }

  logLatency(
    operation: string,
    durationMs: number,
    metadata?: Record<string, any>
  ): void {
    const entry: PerformanceLogEntry = {
      operation,
      duration_ms: durationMs,
      timestamp: new Date().toISOString(),
      success: true,
      metadata,
    };

    this.addMetric(entry);

    this.logger.debug(
      `Logged latency for: ${operation}`,
      undefined,
      { duration_ms: durationMs, ...metadata }
    );
  }

  private addMetric(entry: PerformanceLogEntry): void {
    const entries = this.metrics.get(entry.operation) || [];
    entries.unshift(entry);

    // Enforce max size
    if (entries.length > this.maxEntriesPerOperation) {
      entries.pop();
    }

    this.metrics.set(entry.operation, entries);
  }

  getMetrics(operation?: string): LatencyMetrics | LatencyMetrics[] {
    if (operation) {
      return this.calculateMetrics(operation);
    }

    // Return metrics for all operations
    const allMetrics: LatencyMetrics[] = [];
    for (const op of this.metrics.keys()) {
      allMetrics.push(this.calculateMetrics(op));
    }

    return allMetrics;
  }

  private calculateMetrics(operation: string): LatencyMetrics {
    const entries = this.metrics.get(operation) || [];

    if (entries.length === 0) {
      return {
        operation,
        count: 0,
        total_ms: 0,
        avg_ms: 0,
        min_ms: 0,
        max_ms: 0,
        p50_ms: 0,
        p95_ms: 0,
        p99_ms: 0,
      };
    }

    const durations = entries.map((e) => e.duration_ms).sort((a, b) => a - b);
    const total = durations.reduce((sum, d) => sum + d, 0);

    return {
      operation,
      count: entries.length,
      total_ms: total,
      avg_ms: total / entries.length,
      min_ms: durations[0],
      max_ms: durations[durations.length - 1],
      p50_ms: this.percentile(durations, 0.5),
      p95_ms: this.percentile(durations, 0.95),
      p99_ms: this.percentile(durations, 0.99),
    };
  }

  private percentile(sortedArray: number[], p: number): number {
    if (sortedArray.length === 0) return 0;

    const index = Math.ceil(sortedArray.length * p) - 1;
    return sortedArray[Math.max(0, index)];
  }

  clearMetrics(): void {
    this.logger.info('Clearing performance metrics', undefined, {
      operations_cleared: this.metrics.size,
      total_entries: Array.from(this.metrics.values()).reduce(
        (sum, entries) => sum + entries.length,
        0
      ),
    });

    this.metrics.clear();
  }

  /**
   * Get operations exceeding latency threshold
   */
  getSlowOperations(thresholdMs: number): PerformanceLogEntry[] {
    const slowOps: PerformanceLogEntry[] = [];

    for (const entries of this.metrics.values()) {
      for (const entry of entries) {
        if (entry.duration_ms > thresholdMs) {
          slowOps.push(entry);
        }
      }
    }

    return slowOps.sort((a, b) => b.duration_ms - a.duration_ms);
  }

  /**
   * Get failed operations
   */
  getFailedOperations(): PerformanceLogEntry[] {
    const failed: PerformanceLogEntry[] = [];

    for (const entries of this.metrics.values()) {
      for (const entry of entries) {
        if (!entry.success) {
          failed.push(entry);
        }
      }
    }

    return failed.sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }

  /**
   * Get performance summary
   */
  getSummary(): {
    total_operations: number;
    total_entries: number;
    avg_latency_ms: number;
    operations_over_100ms: number;
    operations_over_1000ms: number;
    failed_operations: number;
  } {
    let totalEntries = 0;
    let totalLatency = 0;
    let over100ms = 0;
    let over1000ms = 0;
    let failed = 0;

    for (const entries of this.metrics.values()) {
      for (const entry of entries) {
        totalEntries++;
        totalLatency += entry.duration_ms;

        if (entry.duration_ms > 100) over100ms++;
        if (entry.duration_ms > 1000) over1000ms++;
        if (!entry.success) failed++;
      }
    }

    return {
      total_operations: this.metrics.size,
      total_entries: totalEntries,
      avg_latency_ms: totalEntries > 0 ? totalLatency / totalEntries : 0,
      operations_over_100ms: over100ms,
      operations_over_1000ms: over1000ms,
      failed_operations: failed,
    };
  }

  /**
   * Measure async function execution
   */
  async measure<T>(
    operation: string,
    fn: () => Promise<T>,
    metadata?: Record<string, any>
  ): Promise<T> {
    const operationId = this.startOperation(operation);

    try {
      const result = await fn();
      this.endOperation(operationId, true, metadata);
      return result;
    } catch (error) {
      this.endOperation(operationId, false, {
        ...metadata,
        error: (error as Error).message,
      });
      throw error;
    }
  }
}

// Export singleton instance
let performanceLoggerInstance: PerformanceLogger | null = null;

export const getPerformanceLogger = (): PerformanceLogger => {
  if (!performanceLoggerInstance) {
    performanceLoggerInstance = new PerformanceLogger();
  }
  return performanceLoggerInstance;
};
