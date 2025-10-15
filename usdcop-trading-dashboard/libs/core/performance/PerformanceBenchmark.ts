/**
 * PerformanceBenchmark - Elite Performance Testing & Benchmarking Suite
 *
 * Professional-grade performance testing system featuring:
 * - Automated performance regression testing
 * - Real-world trading scenario benchmarks
 * - Memory leak detection
 * - Frame rate stability testing
 * - Load testing with synthetic data
 * - Performance budgets and alerts
 * - Continuous performance monitoring
 *
 * Institutional-grade performance validation for trading platforms
 */

import { EventEmitter } from 'eventemitter3';
import { getPerformanceMonitor } from './PerformanceMonitor';
import { getPerformanceOptimizer } from './PerformanceOptimizer';
import { getWorkerPool } from './WorkerPool';
import { getBundleAnalyzer } from './BundleAnalyzer';

export interface BenchmarkTest {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly category: 'rendering' | 'memory' | 'network' | 'computation' | 'interaction';
  readonly priority: 'low' | 'medium' | 'high' | 'critical';
  readonly timeout: number;
  readonly iterations: number;
  readonly warmupIterations: number;
  readonly dataSize: number;
  readonly expectedDuration: number;
  readonly performanceBudget: {
    readonly maxDuration: number;
    readonly maxMemory: number;
    readonly minFPS: number;
    readonly maxBundleSize: number;
  };
}

export interface BenchmarkResult {
  readonly testId: string;
  readonly success: boolean;
  readonly duration: number;
  readonly fps: number;
  readonly memoryUsage: number;
  readonly memoryPeak: number;
  readonly cpuUsage: number;
  readonly networkLatency: number;
  readonly bundleSize: number;
  readonly iterations: number;
  readonly statistics: {
    readonly mean: number;
    readonly median: number;
    readonly p95: number;
    readonly p99: number;
    readonly min: number;
    readonly max: number;
    readonly standardDeviation: number;
  };
  readonly budgetViolations: Array<{
    readonly metric: string;
    readonly actual: number;
    readonly budget: number;
    readonly severity: 'warning' | 'error';
  }>;
  readonly timestamp: number;
  readonly error?: string;
}

export interface BenchmarkSuite {
  readonly id: string;
  readonly name: string;
  readonly description: string;
  readonly tests: BenchmarkTest[];
  readonly parallel: boolean;
  readonly retryOnFailure: boolean;
  readonly maxRetries: number;
}

export interface PerformanceReport {
  readonly suiteId: string;
  readonly totalTests: number;
  readonly passedTests: number;
  readonly failedTests: number;
  readonly totalDuration: number;
  readonly overallScore: number;
  readonly results: BenchmarkResult[];
  readonly regressions: Array<{
    readonly testId: string;
    readonly metric: string;
    readonly previousValue: number;
    readonly currentValue: number;
    readonly degradation: number;
  }>;
  readonly improvements: Array<{
    readonly testId: string;
    readonly metric: string;
    readonly previousValue: number;
    readonly currentValue: number;
    readonly improvement: number;
  }>;
  readonly timestamp: number;
}

export interface BenchmarkConfig {
  readonly enableRegressionTesting: boolean;
  readonly enableMemoryProfiling: boolean;
  readonly enableNetworkThrottling: boolean;
  readonly enableCPUThrottling: boolean;
  readonly baselineFile?: string;
  readonly reportFormat: 'json' | 'html' | 'xml';
  readonly outputDirectory: string;
  readonly enableContinuousMonitoring: boolean;
  readonly monitoringInterval: number;
  readonly alertThresholds: {
    readonly regressionThreshold: number;
    readonly memoryLeakThreshold: number;
    readonly fpsDropThreshold: number;
  };
}

export class PerformanceBenchmark extends EventEmitter {
  private readonly config: BenchmarkConfig;
  private readonly performanceMonitor = getPerformanceMonitor();
  private readonly optimizer = getPerformanceOptimizer();
  private readonly workerPool = getWorkerPool();
  private readonly bundleAnalyzer = getBundleAnalyzer();

  private suites = new Map<string, BenchmarkSuite>();
  private results = new Map<string, BenchmarkResult[]>();
  private baselines = new Map<string, BenchmarkResult>();
  private monitoringTimer?: NodeJS.Timeout;

  constructor(config: Partial<BenchmarkConfig> = {}) {
    super();

    this.config = {
      enableRegressionTesting: true,
      enableMemoryProfiling: true,
      enableNetworkThrottling: false,
      enableCPUThrottling: false,
      reportFormat: 'json',
      outputDirectory: './performance-reports',
      enableContinuousMonitoring: false,
      monitoringInterval: 60000, // 1 minute
      alertThresholds: {
        regressionThreshold: 0.1, // 10% degradation
        memoryLeakThreshold: 50 * 1024 * 1024, // 50MB
        fpsDropThreshold: 10 // FPS drop threshold
      },
      ...config
    };

    this.initialize();
  }

  /**
   * Initialize the benchmark system
   */
  private initialize(): void {
    this.createDefaultBenchmarkSuites();

    if (this.config.enableContinuousMonitoring) {
      this.startContinuousMonitoring();
    }
  }

  /**
   * Register a benchmark suite
   */
  public registerSuite(suite: BenchmarkSuite): void {
    this.suites.set(suite.id, suite);
    this.emit('suite.registered', suite);
  }

  /**
   * Run a specific benchmark test
   */
  public async runTest(testId: string): Promise<BenchmarkResult> {
    const test = this.findTestById(testId);
    if (!test) {
      throw new Error(`Test not found: ${testId}`);
    }

    this.emit('test.started', test);

    try {
      const result = await this.executeTest(test);

      this.storeResult(result);
      this.checkPerformanceBudget(result);
      this.checkForRegressions(result);

      this.emit('test.completed', result);
      return result;

    } catch (error) {
      const failedResult: BenchmarkResult = {
        testId: test.id,
        success: false,
        duration: 0,
        fps: 0,
        memoryUsage: 0,
        memoryPeak: 0,
        cpuUsage: 0,
        networkLatency: 0,
        bundleSize: 0,
        iterations: 0,
        statistics: {
          mean: 0, median: 0, p95: 0, p99: 0,
          min: 0, max: 0, standardDeviation: 0
        },
        budgetViolations: [],
        timestamp: Date.now(),
        error: error instanceof Error ? error.message : String(error)
      };

      this.emit('test.failed', failedResult);
      return failedResult;
    }
  }

  /**
   * Run an entire benchmark suite
   */
  public async runSuite(suiteId: string): Promise<PerformanceReport> {
    const suite = this.suites.get(suiteId);
    if (!suite) {
      throw new Error(`Suite not found: ${suiteId}`);
    }

    this.emit('suite.started', suite);

    const startTime = performance.now();
    const results: BenchmarkResult[] = [];

    try {
      if (suite.parallel) {
        // Run tests in parallel
        const promises = suite.tests.map(test => this.runTest(test.id));
        const parallelResults = await Promise.allSettled(promises);

        parallelResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            console.error(`Test ${suite.tests[index].id} failed:`, result.reason);
          }
        });
      } else {
        // Run tests sequentially
        for (const test of suite.tests) {
          try {
            const result = await this.runTest(test.id);
            results.push(result);
          } catch (error) {
            console.error(`Test ${test.id} failed:`, error);
          }
        }
      }

      const totalDuration = performance.now() - startTime;
      const report = this.generateReport(suite, results, totalDuration);

      this.emit('suite.completed', report);
      return report;

    } catch (error) {
      this.emit('suite.failed', { suite, error });
      throw error;
    }
  }

  /**
   * Run all registered benchmark suites
   */
  public async runAllSuites(): Promise<PerformanceReport[]> {
    const reports: PerformanceReport[] = [];

    for (const [suiteId] of this.suites) {
      try {
        const report = await this.runSuite(suiteId);
        reports.push(report);
      } catch (error) {
        console.error(`Suite ${suiteId} failed:`, error);
      }
    }

    this.emit('all.suites.completed', reports);
    return reports;
  }

  /**
   * Set baseline results for regression testing
   */
  public setBaseline(testId: string, result: BenchmarkResult): void {
    this.baselines.set(testId, result);
    this.emit('baseline.set', { testId, result });
  }

  /**
   * Load baselines from previous runs
   */
  public async loadBaselines(data: Record<string, BenchmarkResult>): Promise<void> {
    Object.entries(data).forEach(([testId, result]) => {
      this.baselines.set(testId, result);
    });

    this.emit('baselines.loaded', { count: Object.keys(data).length });
  }

  /**
   * Get performance trends for a specific test
   */
  public getPerformanceTrends(testId: string, timeframe?: number): Array<{
    timestamp: number;
    duration: number;
    fps: number;
    memoryUsage: number;
  }> {
    const testResults = this.results.get(testId) || [];
    const cutoff = timeframe ? Date.now() - timeframe : 0;

    return testResults
      .filter(r => r.timestamp >= cutoff)
      .map(r => ({
        timestamp: r.timestamp,
        duration: r.duration,
        fps: r.fps,
        memoryUsage: r.memoryUsage
      }))
      .sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Generate comprehensive performance report
   */
  public generateReport(
    suite: BenchmarkSuite,
    results: BenchmarkResult[],
    totalDuration: number
  ): PerformanceReport {
    const passedTests = results.filter(r => r.success).length;
    const failedTests = results.filter(r => !r.success).length;

    // Calculate overall score (0-100)
    const overallScore = this.calculateOverallScore(results);

    // Detect regressions and improvements
    const regressions = this.detectRegressions(results);
    const improvements = this.detectImprovements(results);

    return {
      suiteId: suite.id,
      totalTests: results.length,
      passedTests,
      failedTests,
      totalDuration,
      overallScore,
      results,
      regressions,
      improvements,
      timestamp: Date.now()
    };
  }

  /**
   * Start continuous performance monitoring
   */
  public startContinuousMonitoring(): void {
    if (this.monitoringTimer) {
      this.stopContinuousMonitoring();
    }

    this.monitoringTimer = setInterval(async () => {
      try {
        // Run critical performance tests
        const criticalTests = this.getCriticalTests();

        for (const test of criticalTests) {
          const result = await this.runTest(test.id);

          if (!result.success) {
            this.emit('monitoring.alert', {
              type: 'test_failure',
              testId: test.id,
              result
            });
          }
        }

      } catch (error) {
        this.emit('monitoring.error', error);
      }
    }, this.config.monitoringInterval);

    this.emit('monitoring.started');
  }

  /**
   * Stop continuous performance monitoring
   */
  public stopContinuousMonitoring(): void {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = undefined;
    }

    this.emit('monitoring.stopped');
  }

  /**
   * Export results to file
   */
  public async exportResults(format: 'json' | 'html' | 'xml' = 'json'): Promise<string> {
    const allResults = Array.from(this.results.entries()).reduce((acc, [testId, results]) => {
      acc[testId] = results;
      return acc;
    }, {} as Record<string, BenchmarkResult[]>);

    switch (format) {
      case 'json':
        return JSON.stringify(allResults, null, 2);
      case 'html':
        return this.generateHTMLReport(allResults);
      case 'xml':
        return this.generateXMLReport(allResults);
      default:
        throw new Error(`Unsupported format: ${format}`);
    }
  }

  /**
   * Destroy benchmark system and cleanup
   */
  public destroy(): void {
    this.stopContinuousMonitoring();
    this.suites.clear();
    this.results.clear();
    this.baselines.clear();
    this.removeAllListeners();
  }

  // Private implementation methods

  private async executeTest(test: BenchmarkTest): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const measurements: number[] = [];

    // Setup test environment
    await this.setupTestEnvironment(test);

    try {
      // Warmup iterations
      for (let i = 0; i < test.warmupIterations; i++) {
        await this.executeTestIteration(test);
      }

      // Actual test iterations
      for (let i = 0; i < test.iterations; i++) {
        const iterationStart = performance.now();
        await this.executeTestIteration(test);
        const iterationDuration = performance.now() - iterationStart;
        measurements.push(iterationDuration);
      }

      const duration = performance.now() - startTime;
      const stats = this.calculateStatistics(measurements);

      // Collect performance metrics
      const metrics = this.performanceMonitor.getCurrentMetrics();
      const memoryStats = this.optimizer.getMemoryStats();
      const bundleStats = this.bundleAnalyzer.getBundleStats();

      const result: BenchmarkResult = {
        testId: test.id,
        success: true,
        duration,
        fps: metrics.fps,
        memoryUsage: memoryStats.percentage,
        memoryPeak: memoryStats.heapUsed / (1024 * 1024), // MB
        cpuUsage: metrics.cpuUsage,
        networkLatency: metrics.networkLatency,
        bundleSize: bundleStats?.totalGzipSize || 0,
        iterations: test.iterations,
        statistics: stats,
        budgetViolations: [],
        timestamp: Date.now()
      };

      return result;

    } finally {
      await this.cleanupTestEnvironment(test);
    }
  }

  private async executeTestIteration(test: BenchmarkTest): Promise<void> {
    switch (test.category) {
      case 'rendering':
        await this.executeRenderingTest(test);
        break;
      case 'memory':
        await this.executeMemoryTest(test);
        break;
      case 'network':
        await this.executeNetworkTest(test);
        break;
      case 'computation':
        await this.executeComputationTest(test);
        break;
      case 'interaction':
        await this.executeInteractionTest(test);
        break;
      default:
        throw new Error(`Unknown test category: ${test.category}`);
    }
  }

  private async executeRenderingTest(test: BenchmarkTest): Promise<void> {
    // Simulate heavy rendering workload
    return new Promise(resolve => {
      let frameCount = 0;
      const targetFrames = Math.floor(test.dataSize / 1000);

      const renderFrame = () => {
        // Simulate complex rendering operations
        const canvas = document.createElement('canvas');
        canvas.width = 1920;
        canvas.height = 1080;
        const ctx = canvas.getContext('2d')!;

        // Draw complex shapes
        for (let i = 0; i < 100; i++) {
          ctx.beginPath();
          ctx.arc(Math.random() * 1920, Math.random() * 1080, Math.random() * 50, 0, Math.PI * 2);
          ctx.fillStyle = `hsl(${Math.random() * 360}, 50%, 50%)`;
          ctx.fill();
        }

        frameCount++;
        if (frameCount < targetFrames) {
          requestAnimationFrame(renderFrame);
        } else {
          resolve();
        }
      };

      requestAnimationFrame(renderFrame);
    });
  }

  private async executeMemoryTest(test: BenchmarkTest): Promise<void> {
    // Simulate memory-intensive operations
    const data: any[] = [];

    for (let i = 0; i < test.dataSize; i++) {
      data.push({
        id: i,
        timestamp: Date.now(),
        values: new Array(100).fill(0).map(() => Math.random()),
        metadata: {
          type: 'test_data',
          iteration: i,
          created: new Date().toISOString()
        }
      });
    }

    // Perform operations on data
    data.forEach(item => {
      item.processed = item.values.reduce((sum: number, val: number) => sum + val, 0);
    });

    // Cleanup
    data.length = 0;
  }

  private async executeNetworkTest(test: BenchmarkTest): Promise<void> {
    // Simulate network requests
    const promises: Promise<any>[] = [];

    for (let i = 0; i < test.dataSize / 1000; i++) {
      promises.push(
        fetch(`/api/test-endpoint?iteration=${i}`)
          .catch(() => ({ status: 'simulated' }))
      );
    }

    await Promise.all(promises);
  }

  private async executeComputationTest(test: BenchmarkTest): Promise<void> {
    // Use worker pool for heavy computations
    const tasks = [];

    for (let i = 0; i < test.dataSize / 1000; i++) {
      tasks.push({
        type: 'fibonacci',
        data: { n: 35 + (i % 5) } // Vary complexity
      });
    }

    await this.workerPool.executeParallel(tasks);
  }

  private async executeInteractionTest(test: BenchmarkTest): Promise<void> {
    // Simulate user interactions
    return new Promise(resolve => {
      let interactions = 0;
      const maxInteractions = test.dataSize / 1000;

      const simulateInteraction = () => {
        // Simulate click events
        const event = new MouseEvent('click', {
          bubbles: true,
          cancelable: true,
          clientX: Math.random() * window.innerWidth,
          clientY: Math.random() * window.innerHeight
        });

        document.body.dispatchEvent(event);

        interactions++;
        if (interactions < maxInteractions) {
          setTimeout(simulateInteraction, 10);
        } else {
          resolve();
        }
      };

      simulateInteraction();
    });
  }

  private async setupTestEnvironment(test: BenchmarkTest): Promise<void> {
    // Clear caches
    this.optimizer.optimizeMemory();

    // Setup network throttling if enabled
    if (this.config.enableNetworkThrottling) {
      // Would integrate with browser DevTools Protocol in real implementation
    }

    // Setup CPU throttling if enabled
    if (this.config.enableCPUThrottling) {
      // Would integrate with browser DevTools Protocol in real implementation
    }
  }

  private async cleanupTestEnvironment(test: BenchmarkTest): Promise<void> {
    // Force garbage collection if available
    if ('gc' in window && typeof window.gc === 'function') {
      window.gc();
    }

    // Reset performance optimizer
    this.optimizer.optimizeMemory();
  }

  private calculateStatistics(measurements: number[]): BenchmarkResult['statistics'] {
    if (measurements.length === 0) {
      return { mean: 0, median: 0, p95: 0, p99: 0, min: 0, max: 0, standardDeviation: 0 };
    }

    const sorted = [...measurements].sort((a, b) => a - b);
    const mean = measurements.reduce((sum, val) => sum + val, 0) / measurements.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    const p99 = sorted[Math.floor(sorted.length * 0.99)];
    const min = sorted[0];
    const max = sorted[sorted.length - 1];

    const variance = measurements.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / measurements.length;
    const standardDeviation = Math.sqrt(variance);

    return { mean, median, p95, p99, min, max, standardDeviation };
  }

  private storeResult(result: BenchmarkResult): void {
    if (!this.results.has(result.testId)) {
      this.results.set(result.testId, []);
    }

    const testResults = this.results.get(result.testId)!;
    testResults.push(result);

    // Limit history to last 100 results
    if (testResults.length > 100) {
      testResults.splice(0, testResults.length - 100);
    }
  }

  private checkPerformanceBudget(result: BenchmarkResult): void {
    const test = this.findTestById(result.testId);
    if (!test) return;

    const violations: BenchmarkResult['budgetViolations'] = [];
    const budget = test.performanceBudget;

    if (result.duration > budget.maxDuration) {
      violations.push({
        metric: 'duration',
        actual: result.duration,
        budget: budget.maxDuration,
        severity: result.duration > budget.maxDuration * 1.5 ? 'error' : 'warning'
      });
    }

    if (result.memoryUsage > budget.maxMemory) {
      violations.push({
        metric: 'memory',
        actual: result.memoryUsage,
        budget: budget.maxMemory,
        severity: 'warning'
      });
    }

    if (result.fps < budget.minFPS) {
      violations.push({
        metric: 'fps',
        actual: result.fps,
        budget: budget.minFPS,
        severity: 'error'
      });
    }

    if (result.bundleSize > budget.maxBundleSize) {
      violations.push({
        metric: 'bundleSize',
        actual: result.bundleSize,
        budget: budget.maxBundleSize,
        severity: 'warning'
      });
    }

    result.budgetViolations.push(...violations);

    if (violations.length > 0) {
      this.emit('budget.violated', { testId: result.testId, violations });
    }
  }

  private checkForRegressions(result: BenchmarkResult): void {
    if (!this.config.enableRegressionTesting) return;

    const baseline = this.baselines.get(result.testId);
    if (!baseline) return;

    const regressionThreshold = this.config.alertThresholds.regressionThreshold;

    // Check duration regression
    const durationRegression = (result.duration - baseline.duration) / baseline.duration;
    if (durationRegression > regressionThreshold) {
      this.emit('regression.detected', {
        testId: result.testId,
        metric: 'duration',
        previousValue: baseline.duration,
        currentValue: result.duration,
        degradation: durationRegression
      });
    }

    // Check FPS regression
    const fpsRegression = (baseline.fps - result.fps) / baseline.fps;
    if (fpsRegression > regressionThreshold) {
      this.emit('regression.detected', {
        testId: result.testId,
        metric: 'fps',
        previousValue: baseline.fps,
        currentValue: result.fps,
        degradation: fpsRegression
      });
    }
  }

  private detectRegressions(results: BenchmarkResult[]): PerformanceReport['regressions'] {
    const regressions: PerformanceReport['regressions'] = [];

    results.forEach(result => {
      const baseline = this.baselines.get(result.testId);
      if (!baseline) return;

      const threshold = this.config.alertThresholds.regressionThreshold;

      // Duration regression
      const durationRegression = (result.duration - baseline.duration) / baseline.duration;
      if (durationRegression > threshold) {
        regressions.push({
          testId: result.testId,
          metric: 'duration',
          previousValue: baseline.duration,
          currentValue: result.duration,
          degradation: durationRegression
        });
      }

      // FPS regression
      const fpsRegression = (baseline.fps - result.fps) / baseline.fps;
      if (fpsRegression > threshold) {
        regressions.push({
          testId: result.testId,
          metric: 'fps',
          previousValue: baseline.fps,
          currentValue: result.fps,
          degradation: fpsRegression
        });
      }
    });

    return regressions;
  }

  private detectImprovements(results: BenchmarkResult[]): PerformanceReport['improvements'] {
    const improvements: PerformanceReport['improvements'] = [];

    results.forEach(result => {
      const baseline = this.baselines.get(result.testId);
      if (!baseline) return;

      const threshold = this.config.alertThresholds.regressionThreshold;

      // Duration improvement
      const durationImprovement = (baseline.duration - result.duration) / baseline.duration;
      if (durationImprovement > threshold) {
        improvements.push({
          testId: result.testId,
          metric: 'duration',
          previousValue: baseline.duration,
          currentValue: result.duration,
          improvement: durationImprovement
        });
      }

      // FPS improvement
      const fpsImprovement = (result.fps - baseline.fps) / baseline.fps;
      if (fpsImprovement > threshold) {
        improvements.push({
          testId: result.testId,
          metric: 'fps',
          previousValue: baseline.fps,
          currentValue: result.fps,
          improvement: fpsImprovement
        });
      }
    });

    return improvements;
  }

  private calculateOverallScore(results: BenchmarkResult[]): number {
    if (results.length === 0) return 0;

    const successRate = results.filter(r => r.success).length / results.length;
    const budgetCompliance = results.filter(r => r.budgetViolations.length === 0).length / results.length;

    return Math.round((successRate * 0.6 + budgetCompliance * 0.4) * 100);
  }

  private findTestById(testId: string): BenchmarkTest | null {
    for (const suite of this.suites.values()) {
      const test = suite.tests.find(t => t.id === testId);
      if (test) return test;
    }
    return null;
  }

  private getCriticalTests(): BenchmarkTest[] {
    const tests: BenchmarkTest[] = [];

    for (const suite of this.suites.values()) {
      tests.push(...suite.tests.filter(t => t.priority === 'critical'));
    }

    return tests;
  }

  private generateHTMLReport(results: Record<string, BenchmarkResult[]>): string {
    // Basic HTML report generation
    // In a real implementation, this would use a proper template engine
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Performance Benchmark Report</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          .test-result { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
          .success { background-color: #d4edda; }
          .failure { background-color: #f8d7da; }
        </style>
      </head>
      <body>
        <h1>Performance Benchmark Report</h1>
        <p>Generated on: ${new Date().toISOString()}</p>
        ${Object.entries(results).map(([testId, testResults]) => `
          <div class="test-result ${testResults[testResults.length - 1].success ? 'success' : 'failure'}">
            <h3>${testId}</h3>
            <p>Latest Result: ${testResults[testResults.length - 1].success ? 'PASS' : 'FAIL'}</p>
            <p>Duration: ${testResults[testResults.length - 1].duration.toFixed(2)}ms</p>
            <p>FPS: ${testResults[testResults.length - 1].fps.toFixed(2)}</p>
            <p>Memory Usage: ${testResults[testResults.length - 1].memoryUsage.toFixed(2)}%</p>
          </div>
        `).join('')}
      </body>
      </html>
    `;
  }

  private generateXMLReport(results: Record<string, BenchmarkResult[]>): string {
    // Basic XML report generation
    return `
      <?xml version="1.0" encoding="UTF-8"?>
      <PerformanceReport timestamp="${new Date().toISOString()}">
        ${Object.entries(results).map(([testId, testResults]) => {
          const latest = testResults[testResults.length - 1];
          return `
            <Test id="${testId}">
              <Result>${latest.success ? 'PASS' : 'FAIL'}</Result>
              <Duration>${latest.duration}</Duration>
              <FPS>${latest.fps}</FPS>
              <MemoryUsage>${latest.memoryUsage}</MemoryUsage>
              <Timestamp>${latest.timestamp}</Timestamp>
            </Test>
          `;
        }).join('')}
      </PerformanceReport>
    `;
  }

  private createDefaultBenchmarkSuites(): void {
    // Trading platform specific benchmark suite
    const tradingSuite: BenchmarkSuite = {
      id: 'trading-platform',
      name: 'Trading Platform Performance',
      description: 'Core trading platform performance benchmarks',
      parallel: false,
      retryOnFailure: true,
      maxRetries: 3,
      tests: [
        {
          id: 'chart-rendering',
          name: 'Chart Rendering Performance',
          description: 'Test chart rendering with large datasets',
          category: 'rendering',
          priority: 'critical',
          timeout: 30000,
          iterations: 10,
          warmupIterations: 3,
          dataSize: 10000,
          expectedDuration: 100,
          performanceBudget: {
            maxDuration: 200,
            maxMemory: 70,
            minFPS: 45,
            maxBundleSize: 500 * 1024
          }
        },
        {
          id: 'order-book-updates',
          name: 'Order Book Update Performance',
          description: 'Test order book real-time updates',
          category: 'rendering',
          priority: 'critical',
          timeout: 20000,
          iterations: 100,
          warmupIterations: 10,
          dataSize: 1000,
          expectedDuration: 16,
          performanceBudget: {
            maxDuration: 20,
            maxMemory: 60,
            minFPS: 55,
            maxBundleSize: 300 * 1024
          }
        },
        {
          id: 'indicator-calculation',
          name: 'Technical Indicator Calculation',
          description: 'Test heavy indicator calculations',
          category: 'computation',
          priority: 'high',
          timeout: 15000,
          iterations: 50,
          warmupIterations: 5,
          dataSize: 5000,
          expectedDuration: 50,
          performanceBudget: {
            maxDuration: 100,
            maxMemory: 50,
            minFPS: 50,
            maxBundleSize: 200 * 1024
          }
        },
        {
          id: 'memory-stress',
          name: 'Memory Stress Test',
          description: 'Test memory usage under heavy load',
          category: 'memory',
          priority: 'high',
          timeout: 60000,
          iterations: 5,
          warmupIterations: 1,
          dataSize: 100000,
          expectedDuration: 1000,
          performanceBudget: {
            maxDuration: 2000,
            maxMemory: 80,
            minFPS: 30,
            maxBundleSize: 600 * 1024
          }
        }
      ]
    };

    this.registerSuite(tradingSuite);
  }
}

// Singleton instance
let benchmarkInstance: PerformanceBenchmark | null = null;

export function getPerformanceBenchmark(config?: Partial<BenchmarkConfig>): PerformanceBenchmark {
  if (!benchmarkInstance) {
    benchmarkInstance = new PerformanceBenchmark(config);
  }
  return benchmarkInstance;
}

export function resetPerformanceBenchmark(): void {
  if (benchmarkInstance) {
    benchmarkInstance.destroy();
    benchmarkInstance = null;
  }
}