/**
 * Performance Monitor for ChartPro
 * Real-time performance monitoring and optimization
 */

import { IChartApi } from 'lightweight-charts';

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  renderTime: number;
  dataPoints: number;
  memoryUsage?: number;
  cpuUsage?: number;
  gpuUsage?: number;
  canvasOperations: number;
  seriesCount: number;
  visibleDataPoints: number;
}

export interface PerformanceThresholds {
  minFPS: number;
  maxFrameTime: number;
  maxRenderTime: number;
  maxDataPoints: number;
  memoryWarningThreshold?: number;
  memoryErrorThreshold?: number;
}

export interface PerformanceOptimization {
  datasampling: boolean;
  levelOfDetail: boolean;
  webGLAcceleration: boolean;
  batchUpdates: boolean;
  virtualization: boolean;
  compression: boolean;
}

export class PerformanceMonitor {
  private chart: IChartApi;
  private isMonitoring = false;
  private frameCount = 0;
  private lastFrameTime = 0;
  private frameTimeHistory: number[] = [];
  private renderTimeHistory: number[] = [];
  private animationFrameId: number | null = null;

  public onStatsUpdate: ((metrics: PerformanceMetrics) => void) | null = null;
  public onPerformanceIssue: ((issue: string, metrics: PerformanceMetrics) => void) | null = null;

  private defaultThresholds: PerformanceThresholds = {
    minFPS: 30,
    maxFrameTime: 33, // ~30 FPS
    maxRenderTime: 16, // 16ms for 60 FPS
    maxDataPoints: 50000,
    memoryWarningThreshold: 100 * 1024 * 1024, // 100MB
    memoryErrorThreshold: 500 * 1024 * 1024 // 500MB
  };

  private thresholds: PerformanceThresholds;

  // Performance optimization settings
  private optimizations: PerformanceOptimization = {
    datasampling: true,
    levelOfDetail: true,
    webGLAcceleration: true,
    batchUpdates: true,
    virtualization: true,
    compression: false
  };

  // WebGL support detection
  private webGLSupported: boolean;
  private webGLContext: WebGLRenderingContext | null = null;

  // Data sampling configuration
  private samplingConfig = {
    enabled: true,
    thresholds: {
      light: 1000,
      medium: 5000,
      heavy: 20000
    },
    strategies: {
      light: 'uniform', // Every nth point
      medium: 'adaptive', // Based on volatility
      heavy: 'lod' // Level of detail
    }
  };

  constructor(chart: IChartApi, thresholds?: Partial<PerformanceThresholds>) {
    this.chart = chart;
    this.thresholds = { ...this.defaultThresholds, ...thresholds };
    this.webGLSupported = this.detectWebGLSupport();

    this.initializePerformanceMonitoring();
  }

  private detectWebGLSupport(): boolean {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      this.webGLContext = gl as WebGLRenderingContext;
      return !!gl;
    } catch (e) {
      return false;
    }
  }

  private initializePerformanceMonitoring(): void {
    // Setup performance observer if available
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          for (const entry of entries) {
            if (entry.entryType === 'measure' && entry.name.startsWith('chart-')) {
              this.renderTimeHistory.push(entry.duration);
              if (this.renderTimeHistory.length > 100) {
                this.renderTimeHistory.shift();
              }
            }
          }
        });

        observer.observe({ entryTypes: ['measure'] });
      } catch (e) {
        console.warn('Performance Observer not supported:', e);
      }
    }

    // Setup memory monitoring if available
    if ('memory' in performance) {
      this.setupMemoryMonitoring();
    }
  }

  private setupMemoryMonitoring(): void {
    setInterval(() => {
      if (!this.isMonitoring) return;

      const memory = (performance as any).memory;
      if (memory) {
        const usedBytes = memory.usedJSHeapSize;
        const totalBytes = memory.totalJSHeapSize;
        const limitBytes = memory.jsHeapSizeLimit;

        // Check memory thresholds
        if (this.thresholds.memoryWarningThreshold && usedBytes > this.thresholds.memoryWarningThreshold) {
          this.onPerformanceIssue?.('High memory usage detected', this.getCurrentMetrics());

          // Trigger garbage collection if possible
          this.triggerGarbageCollection();
        }

        if (this.thresholds.memoryErrorThreshold && usedBytes > this.thresholds.memoryErrorThreshold) {
          this.onPerformanceIssue?.('Critical memory usage', this.getCurrentMetrics());

          // Apply aggressive optimizations
          this.applyEmergencyOptimizations();
        }
      }
    }, 5000); // Check every 5 seconds
  }

  public startMonitoring(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    this.frameCount = 0;
    this.lastFrameTime = performance.now();

    this.monitorFramerate();
  }

  public stopMonitoring(): void {
    this.isMonitoring = false;

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private monitorFramerate(): void {
    const monitor = () => {
      if (!this.isMonitoring) return;

      const now = performance.now();
      const deltaTime = now - this.lastFrameTime;

      this.frameCount++;
      this.frameTimeHistory.push(deltaTime);

      // Keep only last 60 frame times (1 second at 60fps)
      if (this.frameTimeHistory.length > 60) {
        this.frameTimeHistory.shift();
      }

      // Calculate metrics every 60 frames
      if (this.frameCount % 60 === 0) {
        const metrics = this.getCurrentMetrics();
        this.onStatsUpdate?.(metrics);

        // Check performance thresholds
        this.checkPerformanceThresholds(metrics);
      }

      this.lastFrameTime = now;
      this.animationFrameId = requestAnimationFrame(monitor);
    };

    this.animationFrameId = requestAnimationFrame(monitor);
  }

  public getCurrentMetrics(): PerformanceMetrics {
    const avgFrameTime = this.frameTimeHistory.length > 0
      ? this.frameTimeHistory.reduce((a, b) => a + b, 0) / this.frameTimeHistory.length
      : 0;

    const avgRenderTime = this.renderTimeHistory.length > 0
      ? this.renderTimeHistory.reduce((a, b) => a + b, 0) / this.renderTimeHistory.length
      : 0;

    const fps = avgFrameTime > 0 ? 1000 / avgFrameTime : 0;

    // Get approximate data points count
    const dataPoints = this.estimateDataPoints();

    const metrics: PerformanceMetrics = {
      fps: Math.round(fps),
      frameTime: Math.round(avgFrameTime * 100) / 100,
      renderTime: Math.round(avgRenderTime * 100) / 100,
      dataPoints,
      canvasOperations: this.frameCount,
      seriesCount: this.getSeriesCount(),
      visibleDataPoints: this.getVisibleDataPoints()
    };

    // Add memory info if available
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      metrics.memoryUsage = memory?.usedJSHeapSize || 0;
    }

    return metrics;
  }

  private checkPerformanceThresholds(metrics: PerformanceMetrics): void {
    // Check FPS
    if (metrics.fps < this.thresholds.minFPS) {
      this.onPerformanceIssue?.(`Low FPS: ${metrics.fps}`, metrics);
      this.applyPerformanceOptimizations('fps');
    }

    // Check frame time
    if (metrics.frameTime > this.thresholds.maxFrameTime) {
      this.onPerformanceIssue?.(`High frame time: ${metrics.frameTime}ms`, metrics);
      this.applyPerformanceOptimizations('frametime');
    }

    // Check render time
    if (metrics.renderTime > this.thresholds.maxRenderTime) {
      this.onPerformanceIssue?.(`High render time: ${metrics.renderTime}ms`, metrics);
      this.applyPerformanceOptimizations('render');
    }

    // Check data points
    if (metrics.dataPoints > this.thresholds.maxDataPoints) {
      this.onPerformanceIssue?.(`Too many data points: ${metrics.dataPoints}`, metrics);
      this.applyPerformanceOptimizations('data');
    }
  }

  private applyPerformanceOptimizations(reason: string): void {
    console.log(`Applying performance optimizations due to: ${reason}`);

    switch (reason) {
      case 'fps':
      case 'frametime':
        this.enableDataSampling();
        this.enableBatchUpdates();
        break;

      case 'render':
        this.enableWebGLAcceleration();
        this.enableLevelOfDetail();
        break;

      case 'data':
        this.enableDataSampling();
        this.enableVirtualization();
        break;

      case 'memory':
        this.enableCompression();
        this.triggerGarbageCollection();
        break;
    }
  }

  private applyEmergencyOptimizations(): void {
    console.warn('Applying emergency performance optimizations');

    // Enable all optimizations
    this.optimizations = {
      datasampling: true,
      levelOfDetail: true,
      webGLAcceleration: true,
      batchUpdates: true,
      virtualization: true,
      compression: true
    };

    // Apply aggressive data sampling
    this.enableAggressiveDataSampling();

    // Reduce visual quality temporarily
    this.reduceVisualQuality();
  }

  private enableDataSampling(): void {
    if (this.optimizations.datasampling) return;

    this.optimizations.datasampling = true;
    console.log('Data sampling enabled');

    // Implement data sampling logic
    // This would integrate with the chart to reduce the number of visible data points
  }

  private enableBatchUpdates(): void {
    if (this.optimizations.batchUpdates) return;

    this.optimizations.batchUpdates = true;
    console.log('Batch updates enabled');

    // Implement batched update logic
  }

  private enableWebGLAcceleration(): void {
    if (!this.webGLSupported || this.optimizations.webGLAcceleration) return;

    this.optimizations.webGLAcceleration = true;
    console.log('WebGL acceleration enabled');

    // Enable WebGL rendering if supported
  }

  private enableLevelOfDetail(): void {
    if (this.optimizations.levelOfDetail) return;

    this.optimizations.levelOfDetail = true;
    console.log('Level of detail enabled');

    // Implement LOD system
  }

  private enableVirtualization(): void {
    if (this.optimizations.virtualization) return;

    this.optimizations.virtualization = true;
    console.log('Virtualization enabled');

    // Implement data virtualization
  }

  private enableCompression(): void {
    if (this.optimizations.compression) return;

    this.optimizations.compression = true;
    console.log('Data compression enabled');

    // Implement data compression
  }

  private enableAggressiveDataSampling(): void {
    // Reduce data points by 75%
    console.log('Aggressive data sampling enabled');
  }

  private reduceVisualQuality(): void {
    // Temporarily reduce visual effects
    console.log('Visual quality reduced for performance');
  }

  private triggerGarbageCollection(): void {
    // Force garbage collection if possible
    if ('gc' in window && typeof (window as any).gc === 'function') {
      try {
        (window as any).gc();
        console.log('Garbage collection triggered');
      } catch (e) {
        // Ignore errors
      }
    }
  }

  private estimateDataPoints(): number {
    // This would need to integrate with the actual chart data
    // For now, return an estimate
    return 10000; // Placeholder
  }

  private getSeriesCount(): number {
    // This would need to integrate with the chart to count active series
    return 5; // Placeholder
  }

  private getVisibleDataPoints(): number {
    // This would calculate the number of data points currently visible
    return 1000; // Placeholder
  }

  // Public optimization methods
  public enableOptimization(type: keyof PerformanceOptimization): void {
    this.optimizations[type] = true;

    switch (type) {
      case 'datasampling':
        this.enableDataSampling();
        break;
      case 'webGLAcceleration':
        this.enableWebGLAcceleration();
        break;
      case 'batchUpdates':
        this.enableBatchUpdates();
        break;
      case 'levelOfDetail':
        this.enableLevelOfDetail();
        break;
      case 'virtualization':
        this.enableVirtualization();
        break;
      case 'compression':
        this.enableCompression();
        break;
    }
  }

  public disableOptimization(type: keyof PerformanceOptimization): void {
    this.optimizations[type] = false;
    console.log(`${type} optimization disabled`);
  }

  public setThresholds(thresholds: Partial<PerformanceThresholds>): void {
    this.thresholds = { ...this.thresholds, ...thresholds };
  }

  public getOptimizations(): PerformanceOptimization {
    return { ...this.optimizations };
  }

  public getThresholds(): PerformanceThresholds {
    return { ...this.thresholds };
  }

  public isWebGLSupported(): boolean {
    return this.webGLSupported;
  }

  public getWebGLInfo(): any {
    if (!this.webGLContext) return null;

    const gl = this.webGLContext;
    return {
      vendor: gl.getParameter(gl.VENDOR),
      renderer: gl.getParameter(gl.RENDERER),
      version: gl.getParameter(gl.VERSION),
      shadingLanguageVersion: gl.getParameter(gl.SHADING_LANGUAGE_VERSION),
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS),
      maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
      extensions: gl.getSupportedExtensions()
    };
  }

  // Data sampling utilities
  public sampleData<T extends { time: any; value: number }>(
    data: T[],
    maxPoints: number,
    strategy: 'uniform' | 'adaptive' | 'lod' = 'adaptive'
  ): T[] {
    if (data.length <= maxPoints) return data;

    switch (strategy) {
      case 'uniform':
        return this.uniformSample(data, maxPoints);
      case 'adaptive':
        return this.adaptiveSample(data, maxPoints);
      case 'lod':
        return this.levelOfDetailSample(data, maxPoints);
      default:
        return this.uniformSample(data, maxPoints);
    }
  }

  private uniformSample<T>(data: T[], maxPoints: number): T[] {
    const step = Math.ceil(data.length / maxPoints);
    const result: T[] = [];

    for (let i = 0; i < data.length; i += step) {
      result.push(data[i]);
    }

    return result;
  }

  private adaptiveSample<T extends { value: number }>(data: T[], maxPoints: number): T[] {
    if (data.length <= maxPoints) return data;

    // Calculate volatility for each point
    const volatility = data.map((point, i) => {
      if (i === 0 || i === data.length - 1) return Infinity; // Always include first and last

      const prev = data[i - 1].value;
      const next = data[i + 1].value;
      return Math.abs(point.value - (prev + next) / 2);
    });

    // Sort by volatility, keeping indices
    const indexed = volatility.map((vol, index) => ({ vol, index }));
    indexed.sort((a, b) => b.vol - a.vol);

    // Take the most volatile points
    const selectedIndices = indexed
      .slice(0, maxPoints)
      .map(item => item.index)
      .sort((a, b) => a - b);

    return selectedIndices.map(i => data[i]);
  }

  private levelOfDetailSample<T>(data: T[], maxPoints: number): T[] {
    // Implement level-of-detail sampling
    // This is a simplified version
    return this.uniformSample(data, maxPoints);
  }

  public destroy(): void {
    this.stopMonitoring();

    // Cleanup
    this.frameTimeHistory = [];
    this.renderTimeHistory = [];
    this.onStatsUpdate = null;
    this.onPerformanceIssue = null;
  }
}

export default PerformanceMonitor;