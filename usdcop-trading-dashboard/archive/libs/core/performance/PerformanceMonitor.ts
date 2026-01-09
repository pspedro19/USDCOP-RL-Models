/**
 * PerformanceMonitor - Elite Trading Platform Performance Monitoring
 * Real-time performance tracking and optimization
 */

import { EventEmitter } from 'eventemitter3';
import type { PerformanceMetrics, MemoryMetrics, NetworkMetrics, ApplicationMetrics } from '../types';

export interface PerformanceMonitorConfig {
  readonly enableCPUMonitoring: boolean;
  readonly enableMemoryMonitoring: boolean;
  readonly enableNetworkMonitoring: boolean;
  readonly enableRenderMonitoring: boolean;
  readonly samplingInterval: number;
  readonly alertThresholds: {
    readonly cpuUsage: number;
    readonly memoryUsage: number;
    readonly renderTime: number;
    readonly fps: number;
  };
  readonly retentionPeriod: number;
  readonly enableLogging: boolean;
}

export interface PerformanceAlert {
  readonly id: string;
  readonly type: 'cpu' | 'memory' | 'render' | 'network' | 'fps';
  readonly severity: 'warning' | 'critical';
  readonly message: string;
  readonly value: number;
  readonly threshold: number;
  readonly timestamp: number;
}

export interface RenderMetrics {
  readonly frameTime: number;
  readonly fps: number;
  readonly droppedFrames: number;
  readonly renderTime: number;
  readonly layoutTime: number;
  readonly paintTime: number;
}

export class PerformanceMonitor extends EventEmitter {
  private readonly config: PerformanceMonitorConfig;
  private readonly metricsHistory: PerformanceMetrics[] = [];
  private readonly alerts: PerformanceAlert[] = [];

  private samplingTimer?: NodeJS.Timeout;
  private lastFrameTime = 0;
  private frameCount = 0;
  private droppedFrames = 0;
  private renderStartTime = 0;

  // Performance observers
  private performanceObserver?: PerformanceObserver;
  private resizeObserver?: ResizeObserver;

  constructor(config: Partial<PerformanceMonitorConfig> = {}) {
    super();

    this.config = {
      enableCPUMonitoring: true,
      enableMemoryMonitoring: true,
      enableNetworkMonitoring: true,
      enableRenderMonitoring: true,
      samplingInterval: 1000,
      alertThresholds: {
        cpuUsage: 80,
        memoryUsage: 85,
        renderTime: 16, // 60fps target
        fps: 50
      },
      retentionPeriod: 300000, // 5 minutes
      enableLogging: false,
      ...config
    };

    this.initialize();
  }

  private initialize(): void {
    this.startPerformanceMonitoring();
    this.setupPerformanceObserver();
    this.setupRenderMonitoring();
  }

  /**
   * Start performance monitoring
   */
  public start(): void {
    if (this.samplingTimer) {
      this.stop();
    }

    this.samplingTimer = setInterval(() => {
      this.collectMetrics();
    }, this.config.samplingInterval);

    this.emit('monitoring.started');
    this.log('info', 'Performance monitoring started');
  }

  /**
   * Stop performance monitoring
   */
  public stop(): void {
    if (this.samplingTimer) {
      clearInterval(this.samplingTimer);
      this.samplingTimer = undefined;
    }

    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }

    this.emit('monitoring.stopped');
    this.log('info', 'Performance monitoring stopped');
  }

  /**
   * Get current performance metrics
   */
  public getCurrentMetrics(): PerformanceMetrics {
    return this.collectMetrics();
  }

  /**
   * Get performance metrics history
   */
  public getMetricsHistory(duration?: number): PerformanceMetrics[] {
    const cutoff = duration ? Date.now() - duration : 0;
    return this.metricsHistory.filter(m => m.timestamp >= cutoff);
  }

  /**
   * Get performance alerts
   */
  public getAlerts(severity?: 'warning' | 'critical'): PerformanceAlert[] {
    return severity ? this.alerts.filter(a => a.severity === severity) : [...this.alerts];
  }

  /**
   * Clear alerts
   */
  public clearAlerts(): void {
    this.alerts.length = 0;
    this.emit('alerts.cleared');
  }

  /**
   * Mark the start of a render frame
   */
  public markRenderStart(): void {
    this.renderStartTime = performance.now();
  }

  /**
   * Mark the end of a render frame
   */
  public markRenderEnd(): void {
    if (this.renderStartTime) {
      const renderTime = performance.now() - this.renderStartTime;
      this.updateRenderMetrics(renderTime);
      this.renderStartTime = 0;
    }
  }

  /**
   * Measure function execution time
   */
  public measureFunction<T>(name: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;

    this.emit('function.measured', { name, duration });
    this.log('debug', `Function ${name} took ${duration.toFixed(2)}ms`);

    return result;
  }

  /**
   * Measure async function execution time
   */
  public async measureAsyncFunction<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;

    this.emit('async.function.measured', { name, duration });
    this.log('debug', `Async function ${name} took ${duration.toFixed(2)}ms`);

    return result;
  }

  /**
   * Create a performance marker
   */
  public mark(name: string): void {
    performance.mark(name);
    this.emit('marker.created', { name, timestamp: Date.now() });
  }

  /**
   * Measure time between two markers
   */
  public measure(name: string, startMark: string, endMark?: string): number {
    performance.measure(name, startMark, endMark);

    const entries = performance.getEntriesByName(name, 'measure');
    const duration = entries.length > 0 ? entries[entries.length - 1].duration : 0;

    this.emit('measurement.created', { name, duration });
    return duration;
  }

  /**
   * Get application metrics summary
   */
  public getApplicationMetrics(): ApplicationMetrics {
    const memory = this.getMemoryMetrics();
    const network = this.getNetworkMetrics();

    return {
      uptime: performance.now(),
      memory,
      network,
      activeConnections: 0, // To be updated by WebSocketManager
      requestsPerSecond: 0, // To be updated by application
      errorsPerMinute: 0, // To be updated by error tracking
      cacheHitRate: 0 // To be updated by cache manager
    };
  }

  /**
   * Destroy performance monitor
   */
  public destroy(): void {
    this.stop();
    this.metricsHistory.length = 0;
    this.alerts.length = 0;
    this.removeAllListeners();
    this.log('info', 'PerformanceMonitor destroyed');
  }

  // Private helper methods
  private startPerformanceMonitoring(): void {
    this.start();
  }

  private collectMetrics(): PerformanceMetrics {
    const timestamp = Date.now();

    const metrics: PerformanceMetrics = {
      timestamp,
      memoryUsage: this.getMemoryUsage(),
      cpuUsage: this.getCPUUsage(),
      networkLatency: this.getNetworkLatency(),
      renderTime: this.getRenderTime(),
      fps: this.getFPS(),
      eventsPerSecond: 0, // To be updated by EventManager
      websocketLatency: 0, // To be updated by WebSocketManager
      dataProcessingTime: 0 // To be updated by DataBus
    };

    // Add to history
    this.metricsHistory.push(metrics);

    // Clean old metrics
    this.cleanupMetricsHistory();

    // Check for alerts
    this.checkAlerts(metrics);

    this.emit('metrics.collected', metrics);
    return metrics;
  }

  private getMemoryUsage(): PerformanceMetrics['memoryUsage'] {
    if (!this.config.enableMemoryMonitoring) {
      return { used: 0, total: 0, percentage: 0 };
    }

    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100
      };
    }

    return { used: 0, total: 0, percentage: 0 };
  }

  private getCPUUsage(): number {
    if (!this.config.enableCPUMonitoring) {
      return 0;
    }

    // CPU usage estimation based on frame timing
    const now = performance.now();
    if (this.lastFrameTime) {
      const frameTime = now - this.lastFrameTime;
      const targetFrameTime = 16.67; // 60fps
      return Math.min(100, (frameTime / targetFrameTime) * 100);
    }

    this.lastFrameTime = now;
    return 0;
  }

  private getNetworkLatency(): number {
    if (!this.config.enableNetworkMonitoring) {
      return 0;
    }

    // Get navigation timing for network latency estimation
    const navTiming = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    if (navTiming) {
      return navTiming.responseEnd - navTiming.requestStart;
    }

    return 0;
  }

  private getRenderTime(): number {
    if (!this.config.enableRenderMonitoring) {
      return 0;
    }

    const paintEntries = performance.getEntriesByType('paint');
    if (paintEntries.length > 0) {
      const lastPaint = paintEntries[paintEntries.length - 1];
      return lastPaint.duration || 0;
    }

    return 0;
  }

  private getFPS(): number {
    if (!this.config.enableRenderMonitoring) {
      return 0;
    }

    const now = performance.now();
    this.frameCount++;

    if (this.lastFrameTime) {
      const elapsed = now - this.lastFrameTime;
      if (elapsed >= 1000) {
        const fps = (this.frameCount * 1000) / elapsed;
        this.frameCount = 0;
        this.lastFrameTime = now;
        return fps;
      }
    } else {
      this.lastFrameTime = now;
    }

    return 60; // Default assumption
  }

  private updateRenderMetrics(renderTime: number): void {
    const targetFrameTime = 16.67; // 60fps

    if (renderTime > targetFrameTime) {
      this.droppedFrames++;
    }

    this.emit('render.metrics', {
      renderTime,
      droppedFrames: this.droppedFrames,
      fps: this.getFPS()
    });
  }

  private setupPerformanceObserver(): void {
    if (typeof PerformanceObserver === 'undefined') return;

    try {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();

        for (const entry of entries) {
          this.emit('performance.entry', {
            name: entry.name,
            type: entry.entryType,
            duration: entry.duration,
            startTime: entry.startTime
          });
        }
      });

      this.performanceObserver.observe({
        entryTypes: ['measure', 'navigation', 'resource', 'paint']
      });

    } catch (error) {
      this.log('warn', 'Failed to setup PerformanceObserver', error);
    }
  }

  private setupRenderMonitoring(): void {
    if (!this.config.enableRenderMonitoring) return;

    // Setup animation frame monitoring
    const monitorFrames = () => {
      this.markRenderStart();

      requestAnimationFrame(() => {
        this.markRenderEnd();
        monitorFrames();
      });
    };

    monitorFrames();
  }

  private getMemoryMetrics(): MemoryMetrics {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        heapUsed: memory.usedJSHeapSize,
        heapTotal: memory.totalJSHeapSize,
        external: 0,
        rss: 0
      };
    }

    return { heapUsed: 0, heapTotal: 0, external: 0, rss: 0 };
  }

  private getNetworkMetrics(): NetworkMetrics {
    // Basic network metrics - can be enhanced with actual network monitoring
    return {
      bytesReceived: 0,
      bytesSent: 0,
      packetsReceived: 0,
      packetsSent: 0,
      latency: this.getNetworkLatency(),
      connectionCount: 0
    };
  }

  private checkAlerts(metrics: PerformanceMetrics): void {
    const { alertThresholds } = this.config;

    // CPU usage alert
    if (metrics.cpuUsage > alertThresholds.cpuUsage) {
      this.createAlert('cpu', 'critical', `High CPU usage: ${metrics.cpuUsage.toFixed(1)}%`,
        metrics.cpuUsage, alertThresholds.cpuUsage);
    }

    // Memory usage alert
    if (metrics.memoryUsage.percentage > alertThresholds.memoryUsage) {
      this.createAlert('memory', 'critical',
        `High memory usage: ${metrics.memoryUsage.percentage.toFixed(1)}%`,
        metrics.memoryUsage.percentage, alertThresholds.memoryUsage);
    }

    // Render time alert
    if (metrics.renderTime > alertThresholds.renderTime) {
      this.createAlert('render', 'warning',
        `Slow render time: ${metrics.renderTime.toFixed(1)}ms`,
        metrics.renderTime, alertThresholds.renderTime);
    }

    // FPS alert
    if (metrics.fps < alertThresholds.fps) {
      this.createAlert('fps', 'warning',
        `Low FPS: ${metrics.fps.toFixed(1)}`,
        metrics.fps, alertThresholds.fps);
    }
  }

  private createAlert(
    type: PerformanceAlert['type'],
    severity: PerformanceAlert['severity'],
    message: string,
    value: number,
    threshold: number
  ): void {
    const alert: PerformanceAlert = {
      id: this.generateId(),
      type,
      severity,
      message,
      value,
      threshold,
      timestamp: Date.now()
    };

    this.alerts.push(alert);

    // Limit alerts history
    if (this.alerts.length > 100) {
      this.alerts.splice(0, this.alerts.length - 100);
    }

    this.emit('alert.created', alert);
    this.log('warn', `Performance alert: ${message}`);
  }

  private cleanupMetricsHistory(): void {
    const cutoff = Date.now() - this.config.retentionPeriod;
    const startLength = this.metricsHistory.length;

    let i = 0;
    while (i < this.metricsHistory.length && this.metricsHistory[i].timestamp < cutoff) {
      i++;
    }

    if (i > 0) {
      this.metricsHistory.splice(0, i);
      this.log('debug', `Cleaned ${i} old metrics entries`);
    }
  }

  private log(level: string, message: string, data?: any): void {
    if (!this.config.enableLogging) return;

    const logFn = level === 'error' ? console.error :
                  level === 'warn' ? console.warn :
                  level === 'debug' ? console.debug :
                  console.log;

    logFn(`[PerformanceMonitor] ${message}`, data || '');
  }

  private generateId(): string {
    return `perf-${Date.now()}-${Math.random().toString(36).substring(2)}`;
  }
}

// Singleton instance
let performanceMonitorInstance: PerformanceMonitor | null = null;

export function getPerformanceMonitor(config?: Partial<PerformanceMonitorConfig>): PerformanceMonitor {
  if (!performanceMonitorInstance) {
    performanceMonitorInstance = new PerformanceMonitor(config);
  }
  return performanceMonitorInstance;
}

export function resetPerformanceMonitor(): void {
  if (performanceMonitorInstance) {
    performanceMonitorInstance.destroy();
    performanceMonitorInstance = null;
  }
}