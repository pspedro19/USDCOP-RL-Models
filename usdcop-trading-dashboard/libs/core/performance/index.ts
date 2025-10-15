/**
 * Elite Performance Module - Institutional Trading Platform Optimization
 *
 * Complete performance optimization suite featuring:
 * - Real-time monitoring and alerting
 * - LTTB data sampling for smooth 60 FPS
 * - Web Workers pool for parallel processing
 * - Virtual scrolling for large datasets
 * - Canvas pooling and memory management
 * - Advanced lazy loading and code splitting
 * - Bundle analysis and optimization
 * - Automated performance testing
 */

// Core Performance Monitoring
export { PerformanceMonitor, getPerformanceMonitor, resetPerformanceMonitor } from './PerformanceMonitor';
export type { PerformanceMonitorConfig, PerformanceAlert, RenderMetrics } from './PerformanceMonitor';

// Performance Optimization Engine
export { PerformanceOptimizer, getPerformanceOptimizer, resetPerformanceOptimizer } from './PerformanceOptimizer';
export type { OptimizationConfig, DataPoint, SampledDataset, CanvasPool, MemoryStats } from './PerformanceOptimizer';

// Web Workers Pool Management
export { WorkerPool, getWorkerPool, resetWorkerPool } from './WorkerPool';
export type { WorkerPoolConfig, WorkerTask, WorkerInstance, PoolStats } from './WorkerPool';

// Bundle Analysis and Optimization
export { BundleAnalyzer, getBundleAnalyzer, resetBundleAnalyzer } from './BundleAnalyzer';
export type { BundleModule, BundleChunk, BundleStats, OptimizationRecommendation } from './BundleAnalyzer';

// Performance Benchmarking
export { PerformanceBenchmark, getPerformanceBenchmark, resetPerformanceBenchmark } from './PerformanceBenchmark';
export type { BenchmarkTest, BenchmarkResult, BenchmarkSuite, PerformanceReport } from './PerformanceBenchmark';

// Lazy Loading and Code Splitting
export {
  LazyLoad,
  withLazyLoading,
  createLazyComponent,
  useLazyLoad,
  LoadingSkeleton,
  SmartLoadingIndicator,
  LazyLoadErrorBoundary
} from './LazyLoadManager';
export type { LazyLoadProps } from './LazyLoadManager';

// React Hooks for Performance
export {
  usePerformance,
  useMeasureFunction,
  useRenderTime,
  useRenderCount,
  useFPS,
  useMemoryUsage,
  useNetworkLatency,
  usePerformanceAlerts
} from './hooks/usePerformance';
export type { UsePerformanceOptions, PerformanceHookResult } from './hooks/usePerformance';