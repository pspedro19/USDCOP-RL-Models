/**
 * UI Configuration
 * =================
 * User interface defaults, chart settings, and visual preferences
 */

/**
 * Chart Configuration
 */
export const chartConfig = {
  /**
   * Default chart dimensions
   */
  dimensions: {
    /**
     * Default chart height (pixels)
     */
    defaultHeight: 700,

    /**
     * Minimum chart height
     */
    minHeight: 400,

    /**
     * Maximum chart height
     */
    maxHeight: 1200,

    /**
     * Default chart width
     */
    defaultWidth: '100%',

    /**
     * Chart heights for different contexts
     */
    heights: {
      compact: 400,
      standard: 500,
      large: 700,
      fullscreen: 1000,
    },
  },

  /**
   * Chart type defaults
   */
  types: {
    /**
     * Default chart type
     */
    default: 'candlestick' as 'candlestick' | 'line' | 'area' | 'ohlc',

    /**
     * Available chart types
     */
    available: ['candlestick', 'line', 'area', 'ohlc'] as const,
  },

  /**
   * Chart settings defaults
   */
  settings: {
    /**
     * Show volume by default
     */
    showVolume: true,

    /**
     * Show indicators by default
     */
    showIndicators: true,

    /**
     * Show grid by default
     */
    showGrid: true,

    /**
     * Show crosshair by default
     */
    showCrosshair: true,

    /**
     * Auto-scroll to latest data
     */
    autoScroll: true,

    /**
     * Auto-scale chart
     */
    autoScale: true,

    /**
     * Show legend
     */
    showLegend: true,

    /**
     * Enable zoom
     */
    enableZoom: true,

    /**
     * Enable pan
     */
    enablePan: true,

    /**
     * Default visible candles
     */
    defaultVisibleCandles: 100,

    /**
     * Maximum visible candles
     */
    maxVisibleCandles: 1000,
  },

  /**
   * Chart colors and themes
   */
  colors: {
    /**
     * Default theme
     */
    defaultTheme: 'dark' as 'light' | 'dark',

    /**
     * Dark theme colors
     */
    dark: {
      background: '#0a0e27',
      gridLine: '#1e293b40',
      crosshair: '#94a3b8',
      text: '#e2e8f0',
      textMuted: '#94a3b8',
      border: '#1e293b',
      candlestick: {
        up: '#10b981',
        down: '#ef4444',
        upBorder: '#059669',
        downBorder: '#dc2626',
        wick: '#94a3b8',
      },
      volume: {
        up: '#10b98140',
        down: '#ef444440',
      },
      indicators: {
        sma: '#3b82f6',
        ema: '#8b5cf6',
        bollinger: '#f59e0b',
        rsi: '#ec4899',
        macd: '#06b6d4',
      },
    },

    /**
     * Light theme colors
     */
    light: {
      background: '#ffffff',
      gridLine: '#e2e8f040',
      crosshair: '#64748b',
      text: '#1e293b',
      textMuted: '#64748b',
      border: '#e2e8f0',
      candlestick: {
        up: '#10b981',
        down: '#ef4444',
        upBorder: '#059669',
        downBorder: '#dc2626',
        wick: '#64748b',
      },
      volume: {
        up: '#10b98140',
        down: '#ef444440',
      },
      indicators: {
        sma: '#3b82f6',
        ema: '#8b5cf6',
        bollinger: '#f59e0b',
        rsi: '#ec4899',
        macd: '#06b6d4',
      },
    },
  },

  /**
   * Grid configuration
   */
  grid: {
    /**
     * Number of horizontal grid lines
     */
    horizontalLines: 10,

    /**
     * Number of vertical grid lines
     */
    verticalLines: 10,

    /**
     * Grid line opacity
     */
    opacity: 0.25,

    /**
     * Grid line style
     */
    lineStyle: 'solid' as 'solid' | 'dashed' | 'dotted',
  },

  /**
   * Volume bar configuration
   */
  volume: {
    /**
     * Volume bar height as percentage of chart
     */
    heightPercentage: 0.25,

    /**
     * Volume opacity
     */
    opacity: 0.5,
  },

  /**
   * Indicator defaults
   */
  indicators: {
    /**
     * Moving Averages
     */
    ma: {
      sma20: { period: 20, color: '#3b82f6', enabled: true },
      sma50: { period: 50, color: '#8b5cf6', enabled: true },
      ema12: { period: 12, color: '#ec4899', enabled: false },
      ema26: { period: 26, color: '#f59e0b', enabled: false },
    },

    /**
     * Bollinger Bands
     */
    bollinger: {
      period: 20,
      stdDev: 2,
      color: '#f59e0b',
      enabled: false,
    },

    /**
     * RSI
     */
    rsi: {
      period: 14,
      overbought: 70,
      oversold: 30,
      color: '#ec4899',
      enabled: false,
    },

    /**
     * MACD
     */
    macd: {
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      color: '#06b6d4',
      enabled: false,
    },
  },
} as const;

/**
 * Table Configuration
 */
export const tableConfig = {
  /**
   * Default page size for tables
   */
  defaultPageSize: 25,

  /**
   * Available page sizes
   */
  pageSizes: [10, 25, 50, 100] as const,

  /**
   * Row height
   */
  rowHeight: 48,

  /**
   * Header height
   */
  headerHeight: 56,

  /**
   * Enable virtualization for tables with more than X rows
   */
  virtualizationThreshold: 100,
} as const;

/**
 * Layout Configuration
 */
export const layoutConfig = {
  /**
   * Sidebar configuration
   */
  sidebar: {
    /**
     * Default width (pixels)
     */
    defaultWidth: 280,

    /**
     * Collapsed width
     */
    collapsedWidth: 64,

    /**
     * Default state
     */
    defaultCollapsed: false,

    /**
     * Mobile breakpoint
     */
    mobileBreakpoint: 768,
  },

  /**
   * Header configuration
   */
  header: {
    /**
     * Height (pixels)
     */
    height: 64,

    /**
     * Show on scroll
     */
    stickyOnScroll: true,
  },

  /**
   * Content padding
   */
  content: {
    /**
     * Default padding (pixels)
     */
    padding: 24,

    /**
     * Mobile padding
     */
    mobilePadding: 16,
  },

  /**
   * Responsive breakpoints (pixels)
   */
  breakpoints: {
    mobile: 640,
    tablet: 768,
    laptop: 1024,
    desktop: 1280,
    wide: 1536,
  },
} as const;

/**
 * Animation Configuration
 */
export const animationConfig = {
  /**
   * Default animation duration (milliseconds)
   */
  duration: {
    fast: 150,
    normal: 300,
    slow: 500,
  },

  /**
   * Default easing function
   */
  easing: 'ease-in-out' as const,

  /**
   * Enable animations
   */
  enabled: true,

  /**
   * Reduce motion for accessibility
   */
  respectReducedMotion: true,
} as const;

/**
 * Toast/Notification Configuration
 */
export const notificationConfig = {
  /**
   * Default position
   */
  position: 'top-right' as 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right',

  /**
   * Auto-dismiss duration (milliseconds)
   */
  autoDismiss: {
    success: 3000,
    error: 5000,
    warning: 4000,
    info: 3000,
  },

  /**
   * Maximum visible notifications
   */
  maxVisible: 3,

  /**
   * Enable sound
   */
  enableSound: false,
} as const;

/**
 * Loading State Configuration
 */
export const loadingConfig = {
  /**
   * Skeleton animation
   */
  skeleton: {
    /**
     * Animation duration (milliseconds)
     */
    duration: 1500,

    /**
     * Base color
     */
    baseColor: '#1e293b',

    /**
     * Highlight color
     */
    highlightColor: '#334155',
  },

  /**
   * Spinner configuration
   */
  spinner: {
    /**
     * Default size (pixels)
     */
    size: 40,

    /**
     * Available sizes
     */
    sizes: {
      small: 20,
      medium: 40,
      large: 60,
    },
  },

  /**
   * Minimum display time (milliseconds)
   * Prevents flashing for very fast loads
   */
  minDisplayTime: 300,
} as const;

/**
 * Form Configuration
 */
export const formConfig = {
  /**
   * Validation modes
   */
  validationMode: 'onBlur' as 'onChange' | 'onBlur' | 'onSubmit',

  /**
   * Show validation errors immediately
   */
  showErrorsOnTouch: true,

  /**
   * Debounce delay for validation (milliseconds)
   */
  validationDebounce: 300,
} as const;

/**
 * Accessibility Configuration
 */
export const a11yConfig = {
  /**
   * Enable keyboard navigation
   */
  enableKeyboardNav: true,

  /**
   * Enable screen reader announcements
   */
  enableAriaLive: true,

  /**
   * Focus visible outline width
   */
  focusOutlineWidth: 2,

  /**
   * Focus visible outline color
   */
  focusOutlineColor: '#3b82f6',

  /**
   * Skip to main content link
   */
  showSkipLink: true,
} as const;

/**
 * Performance Configuration
 */
export const performanceConfig = {
  /**
   * Enable performance monitoring
   */
  enableMonitoring: process.env.NODE_ENV === 'development',

  /**
   * Log slow renders (milliseconds)
   */
  slowRenderThreshold: 100,

  /**
   * Enable React DevTools Profiler
   */
  enableProfiler: process.env.NODE_ENV === 'development',

  /**
   * Virtualization settings
   */
  virtualization: {
    /**
     * Overscan count for virtualized lists
     */
    overscanCount: 5,

    /**
     * Estimated item size
     */
    estimatedItemSize: 50,
  },
} as const;

/**
 * Data Display Configuration
 */
export const dataDisplayConfig = {
  /**
   * Number formatting
   */
  numberFormat: {
    /**
     * Default locale
     */
    locale: 'en-US',

    /**
     * Currency formatting
     */
    currency: {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    },

    /**
     * Percentage formatting
     */
    percentage: {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    },

    /**
     * Large number abbreviation
     */
    abbreviate: {
      enabled: true,
      threshold: 1000000, // 1M
    },
  },

  /**
   * Date/Time formatting
   */
  dateFormat: {
    /**
     * Default locale
     */
    locale: 'en-US',

    /**
     * Format presets
     */
    formats: {
      short: 'MM/DD/YYYY',
      medium: 'MMM DD, YYYY',
      long: 'MMMM DD, YYYY',
      full: 'dddd, MMMM DD, YYYY',
      time: 'HH:mm:ss',
      datetime: 'MM/DD/YYYY HH:mm:ss',
    },

    /**
     * Relative time configuration
     */
    relativeTime: {
      enabled: true,
      threshold: 86400000, // 24 hours in ms
    },
  },
} as const;

/**
 * Helper function to get chart height
 */
export function getChartHeight(size: keyof typeof chartConfig.dimensions.heights): number {
  return chartConfig.dimensions.heights[size];
}

/**
 * Helper function to get theme colors
 */
export function getThemeColors(theme: 'light' | 'dark' = 'dark') {
  return chartConfig.colors[theme];
}

export default {
  chart: chartConfig,
  table: tableConfig,
  layout: layoutConfig,
  animation: animationConfig,
  notification: notificationConfig,
  loading: loadingConfig,
  form: formConfig,
  a11y: a11yConfig,
  performance: performanceConfig,
  dataDisplay: dataDisplayConfig,
};
