/**
 * Replay Performance Monitoring
 *
 * Provides performance tracking, adaptive quality settings,
 * and frame budget management for smooth replay experience.
 */

// ═══════════════════════════════════════════════════════════════════════════
// PERFORMANCE THRESHOLDS
// ═══════════════════════════════════════════════════════════════════════════

export const PERF_THRESHOLDS = {
  FRAME_BUDGET_MS: 16.67,    // 60fps target
  FRAME_WARNING_MS: 33.33,   // 30fps warning
  FRAME_CRITICAL_MS: 100,    // Very slow - degraded experience
  METRICS_CALC_BUDGET_MS: 5, // Max time for metrics calculation
  FILTER_BUDGET_MS: 10,      // Max time for data filtering
  RENDER_BUDGET_MS: 8,       // Max time for render operations

  // P0-5 FIX: Add MIN_TICK_INTERVAL_MS (was undefined)
  MIN_TICK_INTERVAL_MS: 16,  // Minimum time between ticks (60fps)
  MAX_BATCH_SIZE: 100,       // Max items to process per batch
  DEBOUNCE_MS: 50,           // Debounce interval for updates
};

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface PerformanceSample {
  timestamp: number;
  duration: number;
  operation: string;
}

export interface PerformanceStats {
  avgFrameTime: number;
  maxFrameTime: number;
  minFrameTime: number;
  p95FrameTime: number;
  p99FrameTime: number;
  sampleCount: number;
  droppedFrames: number;
}

export interface QualitySettings {
  maxVisibleCandles: number;
  maxVisibleTrades: number;
  animationEnabled: boolean;
  highlightDuration: number;
  chartUpdateFrequency: number; // Every N ticks
  metricsUpdateFrequency: number; // Every N ticks
  smoothScrolling: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// QUALITY PRESETS
// ═══════════════════════════════════════════════════════════════════════════

export const QUALITY_PRESETS: Record<'high' | 'medium' | 'low', QualitySettings> = {
  high: {
    maxVisibleCandles: 500,
    maxVisibleTrades: 100,
    animationEnabled: true,
    highlightDuration: 2000,
    chartUpdateFrequency: 1,
    metricsUpdateFrequency: 1,
    smoothScrolling: true,
  },
  medium: {
    maxVisibleCandles: 300,
    maxVisibleTrades: 50,
    animationEnabled: true,
    highlightDuration: 1500,
    chartUpdateFrequency: 2,
    metricsUpdateFrequency: 2,
    smoothScrolling: true,
  },
  low: {
    maxVisibleCandles: 150,
    maxVisibleTrades: 25,
    animationEnabled: false,
    highlightDuration: 1000,
    chartUpdateFrequency: 4,
    metricsUpdateFrequency: 4,
    smoothScrolling: false,
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// PERFORMANCE MONITOR CLASS
// ═══════════════════════════════════════════════════════════════════════════

type WarningCallback = (message: string, stats: PerformanceStats) => void;

class ReplayPerformanceMonitor {
  private samples: PerformanceSample[] = [];
  private readonly maxSamples = 100;
  private warningCallbacks: WarningCallback[] = [];
  private droppedFrames = 0;
  private lastFrameTime = 0;

  /**
   * Measure the duration of a synchronous operation
   */
  measure<T>(operation: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;

    this.addSample({ timestamp: Date.now(), duration, operation });
    this.checkThresholds(operation, duration);

    return result;
  }

  /**
   * Measure the duration of an async operation
   */
  async measureAsync<T>(operation: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;

    this.addSample({ timestamp: Date.now(), duration, operation });
    this.checkThresholds(operation, duration);

    return result;
  }

  /**
   * Record a frame timing
   */
  recordFrame(): void {
    const now = performance.now();
    if (this.lastFrameTime > 0) {
      const frameDuration = now - this.lastFrameTime;
      this.addSample({ timestamp: Date.now(), duration: frameDuration, operation: 'frame' });

      if (frameDuration > PERF_THRESHOLDS.FRAME_WARNING_MS) {
        this.droppedFrames++;
      }
    }
    this.lastFrameTime = now;
  }

  /**
   * Start timing a frame (returns a function to end timing)
   */
  startFrame(): () => void {
    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this.addSample({ timestamp: Date.now(), duration, operation: 'frame' });
      this.checkThresholds('frame', duration);
    };
  }

  private addSample(sample: PerformanceSample): void {
    this.samples.push(sample);
    if (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
  }

  private checkThresholds(operation: string, duration: number): void {
    if (duration > PERF_THRESHOLDS.FRAME_CRITICAL_MS) {
      this.warn(`Critical: ${operation} took ${duration.toFixed(1)}ms`);
    } else if (duration > PERF_THRESHOLDS.FRAME_WARNING_MS) {
      this.warn(`Warning: ${operation} took ${duration.toFixed(1)}ms`);
    }
  }

  private warn(message: string): void {
    const stats = this.getStats();
    console.warn(`[ReplayPerf] ${message}`);
    this.warningCallbacks.forEach(cb => cb(message, stats));
  }

  /**
   * Register a warning callback
   */
  onWarning(callback: WarningCallback): () => void {
    this.warningCallbacks.push(callback);
    return () => {
      this.warningCallbacks = this.warningCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Get performance statistics
   */
  getStats(): PerformanceStats {
    const frameSamples = this.samples
      .filter(s => s.operation === 'frame')
      .map(s => s.duration);

    if (frameSamples.length === 0) {
      return {
        avgFrameTime: 0,
        maxFrameTime: 0,
        minFrameTime: 0,
        p95FrameTime: 0,
        p99FrameTime: 0,
        sampleCount: 0,
        droppedFrames: this.droppedFrames,
      };
    }

    const sorted = [...frameSamples].sort((a, b) => a - b);
    const p95Index = Math.floor(sorted.length * 0.95);
    const p99Index = Math.floor(sorted.length * 0.99);

    return {
      avgFrameTime: frameSamples.reduce((a, b) => a + b, 0) / frameSamples.length,
      maxFrameTime: Math.max(...frameSamples),
      minFrameTime: Math.min(...frameSamples),
      p95FrameTime: sorted[p95Index] || sorted[sorted.length - 1],
      p99FrameTime: sorted[p99Index] || sorted[sorted.length - 1],
      sampleCount: frameSamples.length,
      droppedFrames: this.droppedFrames,
    };
  }

  /**
   * Get samples for a specific operation
   */
  getSamplesForOperation(operation: string): PerformanceSample[] {
    return this.samples.filter(s => s.operation === operation);
  }

  /**
   * Reset all samples and counters
   */
  reset(): void {
    this.samples = [];
    this.droppedFrames = 0;
    this.lastFrameTime = 0;
  }

  /**
   * Get current FPS estimate
   */
  getCurrentFPS(): number {
    const stats = this.getStats();
    if (stats.avgFrameTime === 0) return 60;
    return Math.min(60, 1000 / stats.avgFrameTime);
  }
}

// Singleton instance
export const replayPerfMonitor = new ReplayPerformanceMonitor();

// ═══════════════════════════════════════════════════════════════════════════
// ADAPTIVE QUALITY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get adaptive quality settings based on current performance
 */
export function getAdaptiveQuality(avgFrameTime: number): QualitySettings {
  // High quality (smooth 60fps)
  if (avgFrameTime < PERF_THRESHOLDS.FRAME_BUDGET_MS) {
    return { ...QUALITY_PRESETS.high };
  }

  // Medium quality (30fps acceptable)
  if (avgFrameTime < PERF_THRESHOLDS.FRAME_WARNING_MS) {
    return { ...QUALITY_PRESETS.medium };
  }

  // Low quality (performance mode)
  return { ...QUALITY_PRESETS.low };
}

/**
 * Get quality level name
 */
export function getQualityLevel(avgFrameTime: number): 'high' | 'medium' | 'low' {
  if (avgFrameTime < PERF_THRESHOLDS.FRAME_BUDGET_MS) return 'high';
  if (avgFrameTime < PERF_THRESHOLDS.FRAME_WARNING_MS) return 'medium';
  return 'low';
}

// ═══════════════════════════════════════════════════════════════════════════
// THROTTLE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Calculate optimal tick interval based on speed and performance
 */
export function calculateTickInterval(
  speed: number,
  avgFrameTime: number
): number {
  const baseInterval = 100; // 100ms base interval at 1x speed
  const speedAdjusted = baseInterval / speed;

  // If performance is poor, increase interval to reduce load
  if (avgFrameTime > PERF_THRESHOLDS.FRAME_WARNING_MS) {
    return Math.max(speedAdjusted, PERF_THRESHOLDS.FRAME_WARNING_MS * 2);
  }

  return Math.max(speedAdjusted, PERF_THRESHOLDS.MIN_TICK_INTERVAL_MS);
}

/**
 * Should skip this frame based on quality settings
 */
export function shouldSkipFrame(
  frameNumber: number,
  quality: QualitySettings
): boolean {
  return frameNumber % quality.chartUpdateFrequency !== 0;
}

/**
 * Should skip metrics update based on quality settings
 */
export function shouldSkipMetricsUpdate(
  frameNumber: number,
  quality: QualitySettings
): boolean {
  return frameNumber % quality.metricsUpdateFrequency !== 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Estimate memory usage for replay data
 */
export function estimateMemoryUsage(
  tradeCount: number,
  equityPointCount: number,
  candlestickCount: number
): number {
  // Rough estimates in bytes
  const tradeSize = 200; // ~200 bytes per trade object
  const equityPointSize = 80; // ~80 bytes per equity point
  const candlestickSize = 60; // ~60 bytes per candlestick

  return (
    tradeCount * tradeSize +
    equityPointCount * equityPointSize +
    candlestickCount * candlestickSize
  );
}

/**
 * Check if data size is within acceptable limits
 */
export function isDataSizeAcceptable(
  tradeCount: number,
  equityPointCount: number,
  candlestickCount: number,
  maxMemoryMB: number = 50
): boolean {
  const estimatedBytes = estimateMemoryUsage(tradeCount, equityPointCount, candlestickCount);
  const maxBytes = maxMemoryMB * 1024 * 1024;
  return estimatedBytes <= maxBytes;
}

// ═══════════════════════════════════════════════════════════════════════════
// DEBUG UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format performance stats for display
 */
export function formatPerformanceStats(stats: PerformanceStats): string {
  return [
    `FPS: ${(1000 / stats.avgFrameTime).toFixed(1)}`,
    `Avg: ${stats.avgFrameTime.toFixed(2)}ms`,
    `Max: ${stats.maxFrameTime.toFixed(2)}ms`,
    `P95: ${stats.p95FrameTime.toFixed(2)}ms`,
    `Dropped: ${stats.droppedFrames}`,
  ].join(' | ');
}

/**
 * Log performance summary to console
 */
export function logPerformanceSummary(): void {
  const stats = replayPerfMonitor.getStats();
  console.log('[ReplayPerf] Summary:', formatPerformanceStats(stats));
}
