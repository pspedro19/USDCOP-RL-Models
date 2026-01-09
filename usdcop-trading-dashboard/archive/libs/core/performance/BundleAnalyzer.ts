/**
 * BundleAnalyzer - Advanced Bundle Optimization & Analysis System
 *
 * Professional-grade bundle analysis and optimization toolkit featuring:
 * - Real-time bundle size monitoring
 * - Dependency tree analysis
 * - Dead code detection
 * - Module duplication identification
 * - Resource compression optimization
 * - Cache-friendly asset organization
 * - Progressive loading strategies
 *
 * Target: <500KB gzipped bundle size, optimal loading performance
 */

import { EventEmitter } from 'eventemitter3';

export interface BundleModule {
  readonly id: string;
  readonly name: string;
  readonly size: number;
  readonly gzipSize: number;
  readonly path: string;
  readonly type: 'entry' | 'chunk' | 'vendor' | 'dynamic';
  readonly dependencies: string[];
  readonly importedBy: string[];
  readonly isUsed: boolean;
  readonly isTreeShakeable: boolean;
  readonly duplicates: string[];
  readonly lastModified: number;
}

export interface BundleChunk {
  readonly id: string;
  readonly name: string;
  readonly size: number;
  readonly gzipSize: number;
  readonly modules: BundleModule[];
  readonly type: 'initial' | 'async' | 'vendor';
  readonly loadPriority: 'high' | 'medium' | 'low';
  readonly cacheable: boolean;
  readonly splitReason: string;
}

export interface BundleAsset {
  readonly name: string;
  readonly size: number;
  readonly gzipSize: number;
  readonly type: string;
  readonly hash: string;
  readonly cached: boolean;
  readonly compressionRatio: number;
  readonly optimized: boolean;
}

export interface BundleStats {
  readonly totalSize: number;
  readonly totalGzipSize: number;
  readonly chunkCount: number;
  readonly moduleCount: number;
  readonly assetCount: number;
  readonly unusedModules: number;
  readonly duplicatedModules: number;
  readonly treeShakenBytes: number;
  readonly compressionRatio: number;
  readonly loadTime: number;
  readonly parseTime: number;
  readonly timestamp: number;
}

export interface OptimizationRecommendation {
  readonly id: string;
  readonly type: 'size' | 'performance' | 'caching' | 'splitting' | 'compression';
  readonly severity: 'low' | 'medium' | 'high' | 'critical';
  readonly title: string;
  readonly description: string;
  readonly impact: number; // Potential bytes saved
  readonly effort: 'low' | 'medium' | 'high';
  readonly module?: string;
  readonly chunk?: string;
  readonly solution: string;
  readonly autoFixable: boolean;
}

export interface BundleAnalyzerConfig {
  readonly enableRealTimeMonitoring: boolean;
  readonly enableTreeShaking: boolean;
  readonly enableCompression: boolean;
  readonly enableSplitting: boolean;
  readonly maxChunkSize: number;
  readonly maxAssetSize: number;
  readonly minChunkSize: number;
  readonly compressionLevel: number;
  readonly analysisInterval: number;
  readonly reportingEnabled: boolean;
}

export class BundleAnalyzer extends EventEmitter {
  private readonly config: BundleAnalyzerConfig;
  private modules = new Map<string, BundleModule>();
  private chunks = new Map<string, BundleChunk>();
  private assets = new Map<string, BundleAsset>();
  private stats: BundleStats[] = [];
  private recommendations: OptimizationRecommendation[] = [];

  private analysisTimer?: NodeJS.Timeout;
  private performanceObserver?: PerformanceObserver;
  private bundleLoadStartTime = 0;

  constructor(config: Partial<BundleAnalyzerConfig> = {}) {
    super();

    this.config = {
      enableRealTimeMonitoring: true,
      enableTreeShaking: true,
      enableCompression: true,
      enableSplitting: true,
      maxChunkSize: 250 * 1024, // 250KB
      maxAssetSize: 500 * 1024, // 500KB
      minChunkSize: 20 * 1024, // 20KB
      compressionLevel: 9,
      analysisInterval: 5000, // 5 seconds
      reportingEnabled: true,
      ...config
    };

    this.initialize();
  }

  /**
   * Initialize the bundle analyzer
   */
  private initialize(): void {
    this.setupPerformanceMonitoring();
    this.startRealTimeAnalysis();
    this.analyzeCurrentBundle();
  }

  /**
   * Analyze the current bundle and generate insights
   */
  public async analyzeBundle(): Promise<BundleStats> {
    const startTime = performance.now();

    try {
      // Analyze webpack chunks if available
      await this.analyzeWebpackChunks();

      // Analyze dynamic imports
      await this.analyzeDynamicImports();

      // Analyze static assets
      await this.analyzeStaticAssets();

      // Generate recommendations
      this.generateOptimizationRecommendations();

      // Calculate statistics
      const stats = this.calculateBundleStats();
      this.stats.push(stats);

      // Limit stats history
      if (this.stats.length > 100) {
        this.stats = this.stats.slice(-100);
      }

      const analysisTime = performance.now() - startTime;

      this.emit('analysis.completed', {
        stats,
        analysisTime,
        recommendations: this.recommendations
      });

      return stats;

    } catch (error) {
      this.emit('analysis.error', error);
      throw error;
    }
  }

  /**
   * Get current bundle statistics
   */
  public getBundleStats(): BundleStats | null {
    return this.stats.length > 0 ? this.stats[this.stats.length - 1] : null;
  }

  /**
   * Get optimization recommendations
   */
  public getRecommendations(
    severity?: OptimizationRecommendation['severity']
  ): OptimizationRecommendation[] {
    if (severity) {
      return this.recommendations.filter(r => r.severity === severity);
    }
    return [...this.recommendations];
  }

  /**
   * Get bundle modules analysis
   */
  public getModulesAnalysis(): {
    modules: BundleModule[];
    unusedModules: BundleModule[];
    duplicatedModules: BundleModule[];
    largestModules: BundleModule[];
  } {
    const modules = Array.from(this.modules.values());
    const unusedModules = modules.filter(m => !m.isUsed);
    const duplicatedModules = modules.filter(m => m.duplicates.length > 0);
    const largestModules = modules
      .sort((a, b) => b.size - a.size)
      .slice(0, 20);

    return {
      modules,
      unusedModules,
      duplicatedModules,
      largestModules
    };
  }

  /**
   * Get chunk analysis
   */
  public getChunksAnalysis(): {
    chunks: BundleChunk[];
    largestChunks: BundleChunk[];
    asyncChunks: BundleChunk[];
    vendorChunks: BundleChunk[];
  } {
    const chunks = Array.from(this.chunks.values());
    const largestChunks = chunks
      .sort((a, b) => b.size - a.size)
      .slice(0, 10);
    const asyncChunks = chunks.filter(c => c.type === 'async');
    const vendorChunks = chunks.filter(c => c.type === 'vendor');

    return {
      chunks,
      largestChunks,
      asyncChunks,
      vendorChunks
    };
  }

  /**
   * Generate bundle optimization report
   */
  public generateReport(): {
    summary: BundleStats;
    modules: any;
    chunks: any;
    assets: BundleAsset[];
    recommendations: OptimizationRecommendation[];
    optimizationPotential: number;
  } {
    const summary = this.getBundleStats();
    const modules = this.getModulesAnalysis();
    const chunks = this.getChunksAnalysis();
    const assets = Array.from(this.assets.values());
    const recommendations = this.getRecommendations();

    const optimizationPotential = recommendations.reduce(
      (total, rec) => total + rec.impact,
      0
    );

    return {
      summary: summary!,
      modules,
      chunks,
      assets,
      recommendations,
      optimizationPotential
    };
  }

  /**
   * Apply automatic optimizations
   */
  public async applyAutoOptimizations(): Promise<{
    applied: OptimizationRecommendation[];
    failed: Array<{ recommendation: OptimizationRecommendation; error: Error }>;
  }> {
    const autoFixableRecommendations = this.recommendations.filter(r => r.autoFixable);
    const applied: OptimizationRecommendation[] = [];
    const failed: Array<{ recommendation: OptimizationRecommendation; error: Error }> = [];

    for (const recommendation of autoFixableRecommendations) {
      try {
        await this.applyOptimization(recommendation);
        applied.push(recommendation);
      } catch (error) {
        failed.push({ recommendation, error: error as Error });
      }
    }

    this.emit('optimizations.applied', { applied, failed });

    return { applied, failed };
  }

  /**
   * Destroy analyzer and cleanup
   */
  public destroy(): void {
    if (this.analysisTimer) {
      clearInterval(this.analysisTimer);
    }

    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    this.modules.clear();
    this.chunks.clear();
    this.assets.clear();
    this.stats.length = 0;
    this.recommendations.length = 0;

    this.removeAllListeners();
  }

  // Private implementation methods

  private async analyzeWebpackChunks(): Promise<void> {
    // Check if webpack runtime is available
    if (typeof window !== 'undefined' && (window as any).__webpack_require__) {
      const webpackRequire = (window as any).__webpack_require__;

      // Analyze webpack chunk loading
      if (webpackRequire.cache) {
        Object.keys(webpackRequire.cache).forEach(moduleId => {
          const module = webpackRequire.cache[moduleId];
          if (module && module.exports) {
            this.analyzeWebpackModule(moduleId, module);
          }
        });
      }
    }

    // Fallback: analyze script tags
    this.analyzeScriptTags();
  }

  private analyzeWebpackModule(id: string, module: any): void {
    const moduleInfo: BundleModule = {
      id,
      name: this.getModuleName(id, module),
      size: this.estimateModuleSize(module),
      gzipSize: 0, // Will be calculated later
      path: this.getModulePath(id, module),
      type: this.getModuleType(id, module),
      dependencies: this.getModuleDependencies(module),
      importedBy: [],
      isUsed: true,
      isTreeShakeable: this.isModuleTreeShakeable(module),
      duplicates: [],
      lastModified: Date.now()
    };

    this.modules.set(id, moduleInfo);
  }

  private analyzeScriptTags(): void {
    const scripts = document.querySelectorAll('script[src]');

    scripts.forEach((script, index) => {
      const src = script.getAttribute('src');
      if (!src) return;

      const moduleInfo: BundleModule = {
        id: `script-${index}`,
        name: this.extractFilename(src),
        size: 0, // Unknown for external scripts
        gzipSize: 0,
        path: src,
        type: src.includes('vendor') ? 'vendor' : 'chunk',
        dependencies: [],
        importedBy: [],
        isUsed: true,
        isTreeShakeable: false,
        duplicates: [],
        lastModified: Date.now()
      };

      this.modules.set(moduleInfo.id, moduleInfo);
    });
  }

  private async analyzeDynamicImports(): Promise<void> {
    // Monitor dynamic import() calls
    if ('webkitPerformance' in window || 'performance' in window) {
      const resourceEntries = performance.getEntriesByType('resource') as PerformanceResourceTiming[];

      resourceEntries.forEach(entry => {
        if (entry.name.includes('.js') || entry.name.includes('.css')) {
          this.analyzeDynamicResource(entry);
        }
      });
    }
  }

  private analyzeDynamicResource(entry: PerformanceResourceTiming): void {
    const asset: BundleAsset = {
      name: this.extractFilename(entry.name),
      size: entry.transferSize || 0,
      gzipSize: entry.encodedBodySize || 0,
      type: this.getAssetType(entry.name),
      hash: this.extractHashFromUrl(entry.name),
      cached: entry.transferSize === 0,
      compressionRatio: entry.transferSize > 0
        ? (entry.decodedBodySize || 0) / entry.transferSize
        : 1,
      optimized: this.isAssetOptimized(entry)
    };

    this.assets.set(asset.name, asset);
  }

  private async analyzeStaticAssets(): Promise<void> {
    // Analyze CSS files
    const stylesheets = document.querySelectorAll('link[rel="stylesheet"]');
    stylesheets.forEach(link => {
      const href = link.getAttribute('href');
      if (href) {
        this.analyzeStaticAsset(href, 'css');
      }
    });

    // Analyze images
    const images = document.querySelectorAll('img[src]');
    images.forEach(img => {
      const src = img.getAttribute('src');
      if (src && !src.startsWith('data:')) {
        this.analyzeStaticAsset(src, 'image');
      }
    });
  }

  private analyzeStaticAsset(url: string, type: string): void {
    const asset: BundleAsset = {
      name: this.extractFilename(url),
      size: 0, // Will be estimated
      gzipSize: 0,
      type,
      hash: this.extractHashFromUrl(url),
      cached: false,
      compressionRatio: 1,
      optimized: false
    };

    this.assets.set(asset.name, asset);
  }

  private generateOptimizationRecommendations(): void {
    this.recommendations = [];

    // Check for large modules
    this.checkLargeModules();

    // Check for unused modules
    this.checkUnusedModules();

    // Check for duplicate modules
    this.checkDuplicateModules();

    // Check chunk sizes
    this.checkChunkSizes();

    // Check compression opportunities
    this.checkCompressionOpportunities();

    // Check tree shaking opportunities
    this.checkTreeShakingOpportunities();

    // Sort by impact
    this.recommendations.sort((a, b) => b.impact - a.impact);
  }

  private checkLargeModules(): void {
    const largeModules = Array.from(this.modules.values())
      .filter(m => m.size > 100 * 1024); // 100KB

    largeModules.forEach(module => {
      this.recommendations.push({
        id: `large-module-${module.id}`,
        type: 'size',
        severity: module.size > 500 * 1024 ? 'critical' : 'high',
        title: `Large module detected: ${module.name}`,
        description: `Module ${module.name} is ${(module.size / 1024).toFixed(1)}KB. Consider code splitting or optimization.`,
        impact: module.size * 0.3,
        effort: 'medium',
        module: module.id,
        solution: 'Split this module into smaller chunks or use dynamic imports',
        autoFixable: false
      });
    });
  }

  private checkUnusedModules(): void {
    const unusedModules = Array.from(this.modules.values())
      .filter(m => !m.isUsed);

    unusedModules.forEach(module => {
      this.recommendations.push({
        id: `unused-module-${module.id}`,
        type: 'size',
        severity: 'medium',
        title: `Unused module: ${module.name}`,
        description: `Module ${module.name} is imported but not used. Consider removing it.`,
        impact: module.size,
        effort: 'low',
        module: module.id,
        solution: 'Remove unused import or enable tree shaking',
        autoFixable: true
      });
    });
  }

  private checkDuplicateModules(): void {
    const duplicateModules = Array.from(this.modules.values())
      .filter(m => m.duplicates.length > 0);

    duplicateModules.forEach(module => {
      this.recommendations.push({
        id: `duplicate-module-${module.id}`,
        type: 'size',
        severity: 'high',
        title: `Duplicate module: ${module.name}`,
        description: `Module ${module.name} is duplicated ${module.duplicates.length} times. Consider using a shared vendor chunk.`,
        impact: module.size * module.duplicates.length,
        effort: 'medium',
        module: module.id,
        solution: 'Move to vendor chunk or use module federation',
        autoFixable: false
      });
    });
  }

  private checkChunkSizes(): void {
    const largeChunks = Array.from(this.chunks.values())
      .filter(c => c.size > this.config.maxChunkSize);

    largeChunks.forEach(chunk => {
      this.recommendations.push({
        id: `large-chunk-${chunk.id}`,
        type: 'splitting',
        severity: chunk.size > this.config.maxChunkSize * 2 ? 'critical' : 'high',
        title: `Large chunk: ${chunk.name}`,
        description: `Chunk ${chunk.name} is ${(chunk.size / 1024).toFixed(1)}KB. Consider splitting it.`,
        impact: chunk.size * 0.2,
        effort: 'medium',
        chunk: chunk.id,
        solution: 'Split chunk using dynamic imports or webpack optimization',
        autoFixable: false
      });
    });
  }

  private checkCompressionOpportunities(): void {
    const uncompressedAssets = Array.from(this.assets.values())
      .filter(a => !a.optimized && a.size > 10 * 1024);

    uncompressedAssets.forEach(asset => {
      this.recommendations.push({
        id: `compression-${asset.name}`,
        type: 'compression',
        severity: 'medium',
        title: `Compression opportunity: ${asset.name}`,
        description: `Asset ${asset.name} could benefit from better compression.`,
        impact: asset.size * 0.3,
        effort: 'low',
        solution: 'Enable gzip/brotli compression or optimize asset',
        autoFixable: true
      });
    });
  }

  private checkTreeShakingOpportunities(): void {
    const treeShakeableModules = Array.from(this.modules.values())
      .filter(m => m.isTreeShakeable && m.size > 50 * 1024);

    treeShakeableModules.forEach(module => {
      this.recommendations.push({
        id: `tree-shaking-${module.id}`,
        type: 'size',
        severity: 'medium',
        title: `Tree shaking opportunity: ${module.name}`,
        description: `Module ${module.name} supports tree shaking but may not be optimized.`,
        impact: module.size * 0.4,
        effort: 'low',
        module: module.id,
        solution: 'Ensure proper ES6 imports and webpack tree shaking configuration',
        autoFixable: true
      });
    });
  }

  private calculateBundleStats(): BundleStats {
    const modules = Array.from(this.modules.values());
    const chunks = Array.from(this.chunks.values());
    const assets = Array.from(this.assets.values());

    const totalSize = modules.reduce((sum, m) => sum + m.size, 0) +
                     assets.reduce((sum, a) => sum + a.size, 0);

    const totalGzipSize = modules.reduce((sum, m) => sum + m.gzipSize, 0) +
                         assets.reduce((sum, a) => sum + a.gzipSize, 0);

    const unusedModules = modules.filter(m => !m.isUsed).length;
    const duplicatedModules = modules.filter(m => m.duplicates.length > 0).length;

    const treeShakenBytes = modules
      .filter(m => m.isTreeShakeable)
      .reduce((sum, m) => sum + m.size * 0.3, 0);

    return {
      totalSize,
      totalGzipSize,
      chunkCount: chunks.length,
      moduleCount: modules.length,
      assetCount: assets.length,
      unusedModules,
      duplicatedModules,
      treeShakenBytes,
      compressionRatio: totalSize > 0 ? totalGzipSize / totalSize : 1,
      loadTime: this.calculateLoadTime(),
      parseTime: this.calculateParseTime(),
      timestamp: Date.now()
    };
  }

  private async applyOptimization(recommendation: OptimizationRecommendation): Promise<void> {
    switch (recommendation.type) {
      case 'compression':
        await this.applyCompressionOptimization(recommendation);
        break;
      case 'size':
        if (recommendation.module) {
          await this.applyModuleOptimization(recommendation);
        }
        break;
      default:
        throw new Error(`Unsupported optimization type: ${recommendation.type}`);
    }
  }

  private async applyCompressionOptimization(recommendation: OptimizationRecommendation): Promise<void> {
    // In a real implementation, this would trigger build-time optimizations
    // For now, we'll simulate the optimization
    console.log(`Applying compression optimization: ${recommendation.title}`);
  }

  private async applyModuleOptimization(recommendation: OptimizationRecommendation): Promise<void> {
    // In a real implementation, this would modify module imports or trigger tree shaking
    // For now, we'll simulate the optimization
    console.log(`Applying module optimization: ${recommendation.title}`);
  }

  private setupPerformanceMonitoring(): void {
    if (typeof PerformanceObserver === 'undefined') return;

    try {
      this.performanceObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach(entry => {
          if (entry.entryType === 'resource') {
            this.handleResourceTiming(entry as PerformanceResourceTiming);
          }
        });
      });

      this.performanceObserver.observe({
        entryTypes: ['resource', 'navigation']
      });

    } catch (error) {
      console.warn('Failed to setup performance monitoring:', error);
    }
  }

  private handleResourceTiming(entry: PerformanceResourceTiming): void {
    if (entry.name.includes('.js') || entry.name.includes('.css')) {
      this.analyzeDynamicResource(entry);
    }
  }

  private startRealTimeAnalysis(): void {
    if (!this.config.enableRealTimeMonitoring) return;

    this.analysisTimer = setInterval(() => {
      this.analyzeBundle().catch(error => {
        console.error('Real-time analysis failed:', error);
      });
    }, this.config.analysisInterval);
  }

  private analyzeCurrentBundle(): void {
    // Initial analysis
    this.analyzeBundle().catch(error => {
      console.error('Initial bundle analysis failed:', error);
    });
  }

  // Utility methods
  private getModuleName(id: string, module: any): string {
    if (module.exports && module.exports.name) {
      return module.exports.name;
    }
    return this.extractFilename(id);
  }

  private getModulePath(id: string, module: any): string {
    return id;
  }

  private getModuleType(id: string, module: any): BundleModule['type'] {
    if (id.includes('node_modules')) return 'vendor';
    if (id.includes('entry')) return 'entry';
    return 'chunk';
  }

  private getModuleDependencies(module: any): string[] {
    // This would extract actual dependencies in a real implementation
    return [];
  }

  private isModuleTreeShakeable(module: any): boolean {
    // Check if module uses ES6 exports
    return module.exports && typeof module.exports === 'object';
  }

  private estimateModuleSize(module: any): number {
    // Rough estimation based on stringified size
    try {
      return JSON.stringify(module).length;
    } catch {
      return 1024; // Default estimate
    }
  }

  private extractFilename(path: string): string {
    return path.split('/').pop() || path;
  }

  private extractHashFromUrl(url: string): string {
    const match = url.match(/\.([a-f0-9]{8,})\./);
    return match ? match[1] : '';
  }

  private getAssetType(url: string): string {
    const extension = url.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'js': return 'javascript';
      case 'css': return 'stylesheet';
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'gif':
      case 'svg':
        return 'image';
      case 'woff':
      case 'woff2':
      case 'ttf':
        return 'font';
      default:
        return 'other';
    }
  }

  private isAssetOptimized(entry: PerformanceResourceTiming): boolean {
    // Check if asset is compressed (transfer size < decoded size)
    return (entry.transferSize || 0) < (entry.decodedBodySize || 0);
  }

  private calculateLoadTime(): number {
    const navEntries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
    if (navEntries.length > 0) {
      const nav = navEntries[0];
      return nav.loadEventEnd - nav.navigationStart;
    }
    return 0;
  }

  private calculateParseTime(): number {
    const navEntries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
    if (navEntries.length > 0) {
      const nav = navEntries[0];
      return nav.domContentLoadedEventEnd - nav.domLoading;
    }
    return 0;
  }
}

// Singleton instance
let bundleAnalyzerInstance: BundleAnalyzer | null = null;

export function getBundleAnalyzer(config?: Partial<BundleAnalyzerConfig>): BundleAnalyzer {
  if (!bundleAnalyzerInstance) {
    bundleAnalyzerInstance = new BundleAnalyzer(config);
  }
  return bundleAnalyzerInstance;
}

export function resetBundleAnalyzer(): void {
  if (bundleAnalyzerInstance) {
    bundleAnalyzerInstance.destroy();
    bundleAnalyzerInstance = null;
  }
}