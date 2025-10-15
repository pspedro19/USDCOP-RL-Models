/**
 * ChartPro - Institutional Grade Trading Charts
 * Export index for all chart engine components
 */

// Core Chart Component
export { default as ChartPro } from './ChartPro';
export type { ChartProData, ChartProProps } from './ChartPro';

// Demo Component
export { default as ChartProDemo } from './ChartProDemo';

// Core Configuration
export {
  INSTITUTIONAL_DARK_THEME,
  PROFESSIONAL_LIGHT_THEME,
  TRADING_COLORS,
  PERFORMANCE_CONFIG,
  DEFAULT_CHART_CONFIG,
  CHART_DIMENSIONS,
  mergeChartConfig
} from './core/ChartConfig';
export type { ProfessionalChartConfig } from './core/ChartConfig';

// Drawing Tools
export { default as DrawingToolsManager } from './tools/DrawingToolsManager';
export type {
  DrawingTool,
  DrawingToolType,
  DrawingObject,
  DrawingStyle
} from './tools/DrawingToolsManager';

// Volume Profile
export { default as VolumeProfileManager } from './volume/VolumeProfileManager';
export type {
  VolumeProfileLevel,
  VolumeProfileData,
  VolumeProfileConfig
} from './volume/VolumeProfileManager';

// Indicators
export { default as IndicatorManager } from './indicators/IndicatorManager';
export type {
  IndicatorConfig,
  IndicatorStyle,
  IndicatorType,
  IndicatorPlugin,
  IndicatorParameter,
  IndicatorResult
} from './indicators/IndicatorManager';

// Export Manager
export { default as ExportManager } from './export/ExportManager';
export type {
  ExportOptions,
  ExportMetadata,
  ExportResult
} from './export/ExportManager';

// Performance Monitor
export { default as PerformanceMonitor } from './performance/PerformanceMonitor';
export type {
  PerformanceMetrics,
  PerformanceThresholds,
  PerformanceOptimization
} from './performance/PerformanceMonitor';

// Utility Functions
export const ChartEngineUtils = {
  // Color utilities
  adjustColor: (color: string, factor: number): string => {
    // Simple color adjustment utility
    return color;
  },

  // Format utilities
  formatPrice: (price: number, precision: number = 4): string => {
    return price.toFixed(precision);
  },

  formatVolume: (volume: number): string => {
    if (volume >= 1e9) {
      return `${(volume / 1e9).toFixed(2)}B`;
    } else if (volume >= 1e6) {
      return `${(volume / 1e6).toFixed(2)}M`;
    } else if (volume >= 1e3) {
      return `${(volume / 1e3).toFixed(2)}K`;
    }
    return volume.toFixed(0);
  },

  formatPercentage: (value: number, precision: number = 2): string => {
    return `${(value * 100).toFixed(precision)}%`;
  },

  // Time utilities
  formatTime: (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleString();
  },

  formatTimeShort: (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  },

  // Data validation utilities
  validateCandleData: (data: any[]): boolean => {
    return data.every(candle =>
      typeof candle.time === 'number' &&
      typeof candle.open === 'number' &&
      typeof candle.high === 'number' &&
      typeof candle.low === 'number' &&
      typeof candle.close === 'number' &&
      candle.high >= candle.low &&
      candle.high >= Math.max(candle.open, candle.close) &&
      candle.low <= Math.min(candle.open, candle.close)
    );
  },

  validateVolumeData: (data: any[]): boolean => {
    return data.every(volume =>
      typeof volume.time === 'number' &&
      typeof volume.value === 'number' &&
      volume.value >= 0
    );
  },

  // Performance utilities
  measurePerformance: <T>(fn: () => T, label?: string): T => {
    const start = performance.now();
    const result = fn();
    const end = performance.now();

    if (label) {
      console.log(`${label}: ${(end - start).toFixed(2)}ms`);
    }

    return result;
  },

  // Browser capability detection
  detectCapabilities: () => {
    return {
      webGL: (() => {
        try {
          const canvas = document.createElement('canvas');
          return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
          return false;
        }
      })(),

      offscreenCanvas: typeof OffscreenCanvas !== 'undefined',

      workers: typeof Worker !== 'undefined',

      performanceAPI: 'performance' in window && 'now' in performance,

      clipboardAPI: 'clipboard' in navigator,

      fullscreenAPI: 'requestFullscreen' in document.documentElement,

      memory: 'memory' in performance
    };
  }
};

// Chart Engine Version
export const CHART_ENGINE_VERSION = '1.0.0';

// Feature Detection
export const detectFeatureSupport = () => {
  const capabilities = ChartEngineUtils.detectCapabilities();

  return {
    ...capabilities,

    // Recommended configuration based on capabilities
    recommendedConfig: {
      enableWebGL: capabilities.webGL,
      enableWorkers: capabilities.workers,
      enablePerformanceMonitoring: capabilities.performanceAPI,
      maxDataPoints: capabilities.webGL ? 100000 : 25000,
      updateFrequency: capabilities.performanceAPI ? 16 : 33
    }
  };
};

// Quick Setup Function
export const createOptimalChartConfig = (customConfig: any = {}) => {
  const support = detectFeatureSupport();

  return {
    ...INSTITUTIONAL_DARK_THEME,
    performance: {
      enableWebGL: support.enableWebGL,
      maxDataPoints: support.recommendedConfig.maxDataPoints,
      updateFrequency: support.recommendedConfig.updateFrequency
    },
    features: {
      enableDrawingTools: true,
      enableTechnicalIndicators: true,
      enableVolumeProfile: true,
      enableOrderBook: false
    },
    ...customConfig
  };
};