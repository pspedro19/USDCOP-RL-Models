// Latency Monitor for USDCOP Trading System

import type { ILatencyMonitor, LatencyMeasurement, LatencyStats } from './types';
import { getLogger } from '../logging';

/**
 * Latency monitor for tracking service response times
 * Provides detailed latency metrics for performance analysis
 */
export class LatencyMonitor implements ILatencyMonitor {
  private measurements: LatencyMeasurement[] = [];
  private logger = getLogger({ service: 'LatencyMonitor' });
  private maxMeasurements: number;

  constructor(maxMeasurements: number = 10000) {
    this.maxMeasurements = maxMeasurements;
  }

  recordLatency(
    service: string,
    operation: string,
    latencyMs: number,
    success: boolean,
    metadata?: Record<string, any>
  ): void {
    const measurement: LatencyMeasurement = {
      timestamp: new Date().toISOString(),
      service,
      operation,
      latency_ms: latencyMs,
      success,
      metadata,
    };

    this.measurements.unshift(measurement);

    // Enforce max size
    if (this.measurements.length > this.maxMeasurements) {
      this.measurements.pop();
    }

    // Log warning for high latency
    if (latencyMs > 1000) {
      this.logger.warn(
        `High latency detected: ${service}.${operation}`,
        { service, operation },
        { latency_ms: latencyMs, success, ...metadata }
      );
    }
  }

  getLatencyStats(service: string, operation?: string): LatencyStats | LatencyStats[] {
    if (operation) {
      return this.calculateStats(service, operation);
    }

    // Get stats for all operations of the service
    const operations = new Set(
      this.measurements
        .filter((m) => m.service === service)
        .map((m) => m.operation)
    );

    return Array.from(operations).map((op) => this.calculateStats(service, op));
  }

  private calculateStats(service: string, operation: string): LatencyStats {
    const filtered = this.measurements.filter(
      (m) => m.service === service && m.operation === operation
    );

    if (filtered.length === 0) {
      return {
        service,
        operation,
        count: 0,
        avg_ms: 0,
        min_ms: 0,
        max_ms: 0,
        p50_ms: 0,
        p95_ms: 0,
        p99_ms: 0,
        success_rate: 0,
      };
    }

    const latencies = filtered.map((m) => m.latency_ms).sort((a, b) => a - b);
    const total = latencies.reduce((sum, l) => sum + l, 0);
    const successCount = filtered.filter((m) => m.success).length;

    return {
      service,
      operation,
      count: filtered.length,
      avg_ms: total / filtered.length,
      min_ms: latencies[0],
      max_ms: latencies[latencies.length - 1],
      p50_ms: this.percentile(latencies, 0.5),
      p95_ms: this.percentile(latencies, 0.95),
      p99_ms: this.percentile(latencies, 0.99),
      success_rate: successCount / filtered.length,
    };
  }

  private percentile(sortedArray: number[], p: number): number {
    if (sortedArray.length === 0) return 0;

    const index = Math.ceil(sortedArray.length * p) - 1;
    return sortedArray[Math.max(0, index)];
  }

  getRecentMeasurements(limit: number = 100): LatencyMeasurement[] {
    return this.measurements.slice(0, limit);
  }

  clearMeasurements(service?: string): void {
    if (service) {
      this.measurements = this.measurements.filter((m) => m.service !== service);
      this.logger.info(`Cleared latency measurements for service: ${service}`);
    } else {
      this.measurements = [];
      this.logger.info('Cleared all latency measurements');
    }
  }

  getAverageLatency(service: string, minutes: number = 5): number {
    const cutoffTime = new Date(Date.now() - minutes * 60 * 1000);

    const recentMeasurements = this.measurements.filter(
      (m) => m.service === service && new Date(m.timestamp) >= cutoffTime
    );

    if (recentMeasurements.length === 0) {
      return 0;
    }

    const total = recentMeasurements.reduce((sum, m) => sum + m.latency_ms, 0);
    return total / recentMeasurements.length;
  }

  /**
   * Get latency trend over time
   */
  getLatencyTrend(
    service: string,
    operation: string,
    intervalMinutes: number = 5
  ): Array<{ timestamp: string; avg_latency_ms: number; count: number }> {
    const filtered = this.measurements.filter(
      (m) => m.service === service && m.operation === operation
    );

    const intervals = new Map<
      string,
      { total: number; count: number; timestamp: Date }
    >();

    for (const measurement of filtered) {
      const timestamp = new Date(measurement.timestamp);
      const intervalKey = this.getIntervalKey(timestamp, intervalMinutes);

      const existing = intervals.get(intervalKey) || {
        total: 0,
        count: 0,
        timestamp,
      };

      existing.total += measurement.latency_ms;
      existing.count++;

      intervals.set(intervalKey, existing);
    }

    return Array.from(intervals.entries())
      .map(([_, data]) => ({
        timestamp: data.timestamp.toISOString(),
        avg_latency_ms: data.total / data.count,
        count: data.count,
      }))
      .sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
  }

  private getIntervalKey(timestamp: Date, intervalMinutes: number): string {
    const intervalMs = intervalMinutes * 60 * 1000;
    const intervalStart = Math.floor(timestamp.getTime() / intervalMs) * intervalMs;
    return new Date(intervalStart).toISOString();
  }

  /**
   * Get services with highest latency
   */
  getHighLatencyServices(limit: number = 5): Array<{
    service: string;
    operation: string;
    avg_latency_ms: number;
  }> {
    const serviceOps = new Map<string, { total: number; count: number }>();

    for (const measurement of this.measurements) {
      const key = `${measurement.service}:${measurement.operation}`;
      const existing = serviceOps.get(key) || { total: 0, count: 0 };

      existing.total += measurement.latency_ms;
      existing.count++;

      serviceOps.set(key, existing);
    }

    return Array.from(serviceOps.entries())
      .map(([key, data]) => {
        const [service, operation] = key.split(':');
        return {
          service,
          operation,
          avg_latency_ms: data.total / data.count,
        };
      })
      .sort((a, b) => b.avg_latency_ms - a.avg_latency_ms)
      .slice(0, limit);
  }

  /**
   * Get model-to-frontend latency (end-to-end)
   */
  getEndToEndLatency(minutes: number = 5): {
    avg_ms: number;
    min_ms: number;
    max_ms: number;
    p95_ms: number;
  } {
    const cutoffTime = new Date(Date.now() - minutes * 60 * 1000);

    const recentMeasurements = this.measurements.filter(
      (m) => new Date(m.timestamp) >= cutoffTime
    );

    if (recentMeasurements.length === 0) {
      return { avg_ms: 0, min_ms: 0, max_ms: 0, p95_ms: 0 };
    }

    const latencies = recentMeasurements
      .map((m) => m.latency_ms)
      .sort((a, b) => a - b);

    const total = latencies.reduce((sum, l) => sum + l, 0);

    return {
      avg_ms: total / latencies.length,
      min_ms: latencies[0],
      max_ms: latencies[latencies.length - 1],
      p95_ms: this.percentile(latencies, 0.95),
    };
  }
}

// Export singleton instance
let latencyMonitorInstance: LatencyMonitor | null = null;

export const getLatencyMonitor = (): LatencyMonitor => {
  if (!latencyMonitorInstance) {
    latencyMonitorInstance = new LatencyMonitor();
  }
  return latencyMonitorInstance;
};
