/**
 * Professional Chart Configuration for TradingView Lightweight Charts
 * Bloomberg Terminal Level Styling
 */

import {
  DeepPartial,
  ChartOptions,
  LayoutOptions,
  GridOptions,
  CrosshairOptions,
  PriceScaleOptions,
  TimeScaleOptions,
  CrosshairMode,
  PriceScaleMode,
  ColorType
} from 'lightweight-charts';

export interface ProfessionalChartConfig extends DeepPartial<ChartOptions> {
  theme: 'dark' | 'light';
  performance: {
    enableWebGL: boolean;
    maxDataPoints: number;
    updateFrequency: number;
  };
  features: {
    enableDrawingTools: boolean;
    enableTechnicalIndicators: boolean;
    enableVolumeProfile: boolean;
    enableOrderBook: boolean;
  };
}

// Bloomberg-style Dark Theme
export const INSTITUTIONAL_DARK_THEME: ProfessionalChartConfig = {
  theme: 'dark',
  performance: {
    enableWebGL: true,
    maxDataPoints: 50000,
    updateFrequency: 16 // 60 FPS
  },
  features: {
    enableDrawingTools: true,
    enableTechnicalIndicators: true,
    enableVolumeProfile: true,
    enableOrderBook: false
  },
  layout: {
    background: {
      type: ColorType.VerticalGradient,
      topColor: '#0a0e1a',
      bottomColor: '#1a1f2e'
    } as any,
    textColor: '#e6edf3',
    fontSize: 12,
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
  },
  grid: {
    vertLines: {
      color: 'rgba(31, 41, 55, 0.4)',
      style: 0,
      visible: true
    },
    horzLines: {
      color: 'rgba(31, 41, 55, 0.4)',
      style: 0,
      visible: true
    }
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      color: '#3b82f6',
      width: 1,
      style: 3,
      visible: true,
      labelVisible: true,
      labelBackgroundColor: '#1f2937'
    },
    horzLine: {
      color: '#3b82f6',
      width: 1,
      style: 3,
      visible: true,
      labelVisible: true,
      labelBackgroundColor: '#1f2937'
    }
  },
  rightPriceScale: {
    mode: PriceScaleMode.Normal,
    autoScale: true,
    invertScale: false,
    alignLabels: true,
    borderVisible: true,
    borderColor: '#2d3548',
    textColor: '#8b949e',
    entireTextOnly: false,
    visible: true,
    ticksVisible: true,
    scaleMargins: {
      top: 0.1,
      bottom: 0.1
    }
  },
  leftPriceScale: {
    visible: false
  },
  timeScale: {
    rightOffset: 12,
    barSpacing: 3,
    fixLeftEdge: false,
    lockVisibleTimeRangeOnResize: true,
    rightBarStaysOnScroll: true,
    borderVisible: true,
    borderColor: '#2d3548',
    visible: true,
    timeVisible: true,
    secondsVisible: false,
    shiftVisibleRangeOnNewBar: true,
    allowShiftVisibleRangeOnWhitespaceClick: true,
    ticksVisible: true,
    uniformDistribution: false
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: true
  },
  handleScale: {
    axisPressedMouseMove: {
      time: true,
      price: true
    },
    axisDoubleClickReset: {
      time: true,
      price: true
    },
    mouseWheel: true,
    pinch: true
  },
  kineticScroll: {
    touch: false,
    mouse: false
  }
};

// Professional Light Theme (Alternative)
export const PROFESSIONAL_LIGHT_THEME: ProfessionalChartConfig = {
  ...INSTITUTIONAL_DARK_THEME,
  theme: 'light',
  layout: {
    background: {
      type: ColorType.VerticalGradient,
      topColor: '#f8fafc',
      bottomColor: '#e2e8f0'
    } as any,
    textColor: '#1e293b',
    fontSize: 12,
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
  },
  grid: {
    vertLines: {
      color: 'rgba(148, 163, 184, 0.3)',
      style: 0,
      visible: true
    },
    horzLines: {
      color: 'rgba(148, 163, 184, 0.3)',
      style: 0,
      visible: true
    }
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      color: '#2563eb',
      width: 1,
      style: 3,
      visible: true,
      labelVisible: true,
      labelBackgroundColor: '#f1f5f9'
    },
    horzLine: {
      color: '#2563eb',
      width: 1,
      style: 3,
      visible: true,
      labelVisible: true,
      labelBackgroundColor: '#f1f5f9'
    }
  },
  rightPriceScale: {
    ...INSTITUTIONAL_DARK_THEME.rightPriceScale,
    borderColor: '#cbd5e1',
    textColor: '#475569'
  },
  timeScale: {
    ...INSTITUTIONAL_DARK_THEME.timeScale,
    borderColor: '#cbd5e1'
  }
};

// Trading Colors (Professional Grade)
export const TRADING_COLORS = {
  bullish: {
    primary: '#00d395',
    secondary: '#00b894',
    gradient: 'linear-gradient(180deg, #00d395 0%, #00b894 100%)',
    glow: '0 0 8px rgba(0, 211, 149, 0.3)'
  },
  bearish: {
    primary: '#ff4757',
    secondary: '#e84393',
    gradient: 'linear-gradient(180deg, #ff4757 0%, #e84393 100%)',
    glow: '0 0 8px rgba(255, 71, 87, 0.3)'
  },
  neutral: {
    primary: '#74b9ff',
    secondary: '#0984e3',
    gradient: 'linear-gradient(180deg, #74b9ff 0%, #0984e3 100%)',
    glow: '0 0 8px rgba(116, 185, 255, 0.3)'
  },
  volume: {
    primary: '#a29bfe',
    secondary: '#6c5ce7',
    gradient: 'linear-gradient(180deg, #a29bfe 0%, #6c5ce7 100%)',
    glow: '0 0 8px rgba(162, 155, 254, 0.3)'
  },
  indicators: {
    ema20: '#74b9ff',
    ema50: '#fdcb6e',
    rsi: '#fd79a8',
    macd: '#55a3ff',
    bollinger: '#00cec9'
  }
};

// Performance Optimization Settings
export const PERFORMANCE_CONFIG = {
  // Enable WebGL for hardware acceleration
  webgl: {
    enabled: true,
    fallbackToCanvas: true
  },

  // Data sampling for large datasets
  sampling: {
    enabled: true,
    thresholds: {
      light: 1000,    // Start light sampling
      medium: 5000,   // Medium sampling
      heavy: 20000    // Heavy sampling
    },
    algorithms: {
      light: 'every_nth',     // Show every nth point
      medium: 'local_maxima', // Show local maxima/minima
      heavy: 'adaptive'       // Adaptive sampling based on volatility
    }
  },

  // Update throttling
  updates: {
    maxFPS: 60,
    throttleMs: 16,
    batchUpdates: true
  },

  // Memory management
  memory: {
    maxCandlesticks: 50000,
    autoCleanup: true,
    cleanupThreshold: 0.8
  }
};

// Export default configuration
export const DEFAULT_CHART_CONFIG = INSTITUTIONAL_DARK_THEME;

// Utility function to merge configurations
export function mergeChartConfig(
  base: ProfessionalChartConfig,
  override: DeepPartial<ProfessionalChartConfig>
): ProfessionalChartConfig {
  return {
    ...base,
    ...override,
    layout: { ...base.layout, ...override.layout },
    grid: { ...base.grid, ...override.grid },
    crosshair: { ...base.crosshair, ...override.crosshair },
    rightPriceScale: { ...base.rightPriceScale, ...override.rightPriceScale },
    timeScale: { ...base.timeScale, ...override.timeScale },
    performance: { ...base.performance, ...override.performance },
    features: { ...base.features, ...override.features }
  };
}

// Chart sizing utilities
export const CHART_DIMENSIONS = {
  mobile: {
    width: '100%',
    height: 400,
    candleWidth: 2
  },
  tablet: {
    width: '100%',
    height: 500,
    candleWidth: 4
  },
  desktop: {
    width: '100%',
    height: 700,
    candleWidth: 6
  },
  fullscreen: {
    width: '100vw',
    height: '100vh',
    candleWidth: 8
  }
};