/**
 * ChartPro - Institutional Grade Trading Chart
 * Built on TradingView Lightweight Charts v5 with professional features
 */

'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  LineData,
  HistogramData,
  Time,
  UTCTimestamp,
  DeepPartial,
  ChartOptions,
  SeriesType
} from 'lightweight-charts';
import { Card } from '../../ui/card';
import { Button } from '../../ui/button';
import {
  INSTITUTIONAL_DARK_THEME,
  PROFESSIONAL_LIGHT_THEME,
  TRADING_COLORS,
  PERFORMANCE_CONFIG,
  ProfessionalChartConfig,
  mergeChartConfig
} from './core/ChartConfig';
import { DrawingToolsManager } from './tools/DrawingToolsManager';
import { VolumeProfileManager } from './volume/VolumeProfileManager';
import { IndicatorManager } from './indicators/IndicatorManager';
import { ExportManager } from './export/ExportManager';
import { PerformanceMonitor } from './performance/PerformanceMonitor';
import {
  ZoomIn,
  ZoomOut,
  Download,
  Settings,
  TrendingUp,
  BarChart3,
  Crosshair,
  Maximize,
  Moon,
  Sun
} from 'lucide-react';

export interface ChartProData {
  candlesticks: CandlestickData[];
  volume?: HistogramData[];
  indicators?: {
    [key: string]: LineData[];
  };
}

export interface ChartProProps {
  data: ChartProData;
  height?: number;
  width?: string | number;
  theme?: 'dark' | 'light';
  enableDrawingTools?: boolean;
  enableVolumeProfile?: boolean;
  enableIndicators?: boolean;
  enableExport?: boolean;
  onCrosshairMove?: (price: number | null, time: Time | null) => void;
  onVisibleRangeChange?: (range: { from: Time; to: Time } | null) => void;
  className?: string;
  config?: DeepPartial<ProfessionalChartConfig>;
}

export const ChartPro: React.FC<ChartProProps> = ({
  data,
  height = 600,
  width = '100%',
  theme = 'dark',
  enableDrawingTools = true,
  enableVolumeProfile = true,
  enableIndicators = true,
  enableExport = true,
  onCrosshairMove,
  onVisibleRangeChange,
  className = '',
  config = {}
}) => {
  // Refs
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  // Managers
  const drawingToolsRef = useRef<DrawingToolsManager | null>(null);
  const volumeProfileRef = useRef<VolumeProfileManager | null>(null);
  const indicatorManagerRef = useRef<IndicatorManager | null>(null);
  const exportManagerRef = useRef<ExportManager | null>(null);
  const performanceMonitorRef = useRef<PerformanceMonitor | null>(null);

  // State
  const [currentTheme, setCurrentTheme] = useState<'dark' | 'light'>(theme);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showVolumeProfile, setShowVolumeProfile] = useState(false);
  const [activeDrawingTool, setActiveDrawingTool] = useState<string>('crosshair');
  const [performanceStats, setPerformanceStats] = useState({
    fps: 0,
    renderTime: 0,
    dataPoints: 0
  });

  // Chart configuration
  const chartConfig = useMemo(() => {
    const baseConfig = currentTheme === 'dark'
      ? INSTITUTIONAL_DARK_THEME
      : PROFESSIONAL_LIGHT_THEME;

    return mergeChartConfig(baseConfig, {
      ...config,
      width: typeof width === 'number' ? width : undefined,
      height
    });
  }, [currentTheme, config, width, height]);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart instance
    const chart = createChart(chartContainerRef.current, chartConfig);
    chartRef.current = chart;

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: TRADING_COLORS.bullish.primary,
      downColor: TRADING_COLORS.bearish.primary,
      borderUpColor: TRADING_COLORS.bullish.primary,
      borderDownColor: TRADING_COLORS.bearish.primary,
      wickUpColor: TRADING_COLORS.bullish.secondary,
      wickDownColor: TRADING_COLORS.bearish.secondary,
      priceLineVisible: false,
      lastValueVisible: true,
      title: 'Price'
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create volume series if data available
    if (data.volume && data.volume.length > 0) {
      const volumeSeries = chart.addHistogramSeries({
        color: TRADING_COLORS.volume.primary,
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
          top: 0.7,
          bottom: 0,
        },
        title: 'Volume'
      });
      volumeSeriesRef.current = volumeSeries;
    }

    // Initialize managers
    if (enableDrawingTools) {
      drawingToolsRef.current = new DrawingToolsManager(chart, chartContainerRef.current);
    }

    if (enableVolumeProfile) {
      volumeProfileRef.current = new VolumeProfileManager(chart, chartContainerRef.current);
    }

    if (enableIndicators) {
      indicatorManagerRef.current = new IndicatorManager(chart);
    }

    if (enableExport) {
      exportManagerRef.current = new ExportManager(chart, chartContainerRef.current);
    }

    // Performance monitoring
    performanceMonitorRef.current = new PerformanceMonitor(chart);
    performanceMonitorRef.current.onStatsUpdate = setPerformanceStats;

    // Event handlers
    chart.subscribeCrosshairMove((param) => {
      if (onCrosshairMove) {
        const price = param.seriesData.get(candlestickSeries) as CandlestickData | undefined;
        onCrosshairMove(price?.close ?? null, param.time ?? null);
      }
    });

    chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
      if (onVisibleRangeChange) {
        onVisibleRangeChange(range);
      }
    });

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);

      // Cleanup managers
      drawingToolsRef.current?.destroy();
      volumeProfileRef.current?.destroy();
      indicatorManagerRef.current?.destroy();
      exportManagerRef.current?.destroy();
      performanceMonitorRef.current?.destroy();

      chart.remove();
    };
  }, [chartConfig, enableDrawingTools, enableVolumeProfile, enableIndicators, enableExport]);

  // Update data
  useEffect(() => {
    if (!candlestickSeriesRef.current) return;

    // Update candlestick data
    candlestickSeriesRef.current.setData(data.candlesticks);

    // Update volume data
    if (volumeSeriesRef.current && data.volume) {
      volumeSeriesRef.current.setData(data.volume);
    }

    // Update indicators
    if (indicatorManagerRef.current && data.indicators) {
      Object.entries(data.indicators).forEach(([name, lineData]) => {
        indicatorManagerRef.current?.updateIndicator(name, lineData);
      });
    }

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [data]);

  // Update theme
  useEffect(() => {
    if (!chartRef.current) return;

    const newConfig = currentTheme === 'dark'
      ? INSTITUTIONAL_DARK_THEME
      : PROFESSIONAL_LIGHT_THEME;

    chartRef.current.applyOptions(mergeChartConfig(newConfig, config));
  }, [currentTheme, config]);

  // Toolbar handlers
  const handleZoomIn = useCallback(() => {
    if (!chartRef.current) return;
    const timeScale = chartRef.current.timeScale();
    const range = timeScale.getVisibleRange();
    if (range) {
      const diff = range.to - range.from;
      timeScale.setVisibleRange({
        from: (range.from + diff * 0.1) as Time,
        to: (range.to - diff * 0.1) as Time
      });
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (!chartRef.current) return;
    const timeScale = chartRef.current.timeScale();
    const range = timeScale.getVisibleRange();
    if (range) {
      const diff = range.to - range.from;
      timeScale.setVisibleRange({
        from: (range.from - diff * 0.1) as Time,
        to: (range.to + diff * 0.1) as Time
      });
    }
  }, []);

  const handleFitContent = useCallback(() => {
    if (!chartRef.current) return;
    chartRef.current.timeScale().fitContent();
  }, []);

  const handleToggleTheme = useCallback(() => {
    setCurrentTheme(prev => prev === 'dark' ? 'light' : 'dark');
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      chartContainerRef.current?.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const handleExport = useCallback(async (format: 'png' | 'svg' | 'pdf') => {
    if (!exportManagerRef.current) return;

    try {
      await exportManagerRef.current.export(format);
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, []);

  const handleDrawingTool = useCallback((tool: string) => {
    setActiveDrawingTool(tool);
    if (drawingToolsRef.current) {
      drawingToolsRef.current.setActiveTool(tool);
    }
  }, []);

  const handleToggleVolumeProfile = useCallback(() => {
    setShowVolumeProfile(prev => !prev);
    if (volumeProfileRef.current) {
      volumeProfileRef.current.toggle();
    }
  }, []);

  return (
    <Card className={`relative bg-gradient-to-br ${
      currentTheme === 'dark'
        ? 'from-slate-900 to-slate-800 border-slate-700'
        : 'from-slate-50 to-white border-slate-200'
    } ${className}`}>
      {/* Toolbar */}
      <div className={`absolute top-4 left-4 z-10 flex items-center gap-2 p-2 rounded-lg backdrop-blur-sm ${
        currentTheme === 'dark'
          ? 'bg-slate-800/80 border border-slate-700'
          : 'bg-white/80 border border-slate-200'
      }`}>
        {/* Zoom Controls */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleZoomIn}
          className="h-8 w-8 p-0"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleZoomOut}
          className="h-8 w-8 p-0"
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleFitContent}
          className="h-8 w-8 p-0"
        >
          <Maximize className="h-4 w-4" />
        </Button>

        <div className="w-px h-6 bg-slate-400" />

        {/* Drawing Tools */}
        {enableDrawingTools && (
          <>
            <Button
              variant={activeDrawingTool === 'crosshair' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => handleDrawingTool('crosshair')}
              className="h-8 w-8 p-0"
            >
              <Crosshair className="h-4 w-4" />
            </Button>
            <Button
              variant={activeDrawingTool === 'trendline' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => handleDrawingTool('trendline')}
              className="h-8 w-8 p-0"
            >
              <TrendingUp className="h-4 w-4" />
            </Button>
          </>
        )}

        {/* Volume Profile */}
        {enableVolumeProfile && (
          <>
            <div className="w-px h-6 bg-slate-400" />
            <Button
              variant={showVolumeProfile ? 'default' : 'ghost'}
              size="sm"
              onClick={handleToggleVolumeProfile}
              className="h-8 w-8 p-0"
            >
              <BarChart3 className="h-4 w-4" />
            </Button>
          </>
        )}

        <div className="w-px h-6 bg-slate-400" />

        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="sm"
          onClick={handleToggleTheme}
          className="h-8 w-8 p-0"
        >
          {currentTheme === 'dark' ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
        </Button>

        {/* Export */}
        {enableExport && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleExport('png')}
            className="h-8 w-8 p-0"
          >
            <Download className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Performance Stats */}
      <div className={`absolute top-4 right-4 z-10 p-2 rounded-lg backdrop-blur-sm text-xs ${
        currentTheme === 'dark'
          ? 'bg-slate-800/80 border border-slate-700 text-slate-300'
          : 'bg-white/80 border border-slate-200 text-slate-600'
      }`}>
        <div>FPS: {performanceStats.fps}</div>
        <div>Render: {performanceStats.renderTime}ms</div>
        <div>Points: {performanceStats.dataPoints.toLocaleString()}</div>
      </div>

      {/* Chart Container */}
      <div
        ref={chartContainerRef}
        className="w-full relative"
        style={{
          height: `${height}px`,
          width: typeof width === 'string' ? width : `${width}px`
        }}
      />
    </Card>
  );
};

export default ChartPro;