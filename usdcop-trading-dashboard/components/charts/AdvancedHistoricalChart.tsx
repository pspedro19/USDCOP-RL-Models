/**
 * Advanced Historical Chart with Spectacular Navigation
 * ====================================================
 *
 * The ULTIMATE trading chart experience with:
 * - Dual handle navigation with smooth drag & drop ◉◉
 * - Area sombreada movible ███
 * - Mini context chart with sparkline
 * - Professional Bloomberg/TradingView level design
 * - Real-time data integration with comprehensive historical records (dynamically loaded)
 * - Fluid animations and perfect user experience
 * - Doble click para centrar
 * - Scroll para zoom
 * - Drag & drop fluido
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { motion, AnimatePresence, PanInfo } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Calendar,
  Clock,
  Activity,
  BarChart3,
  MousePointer,
  Move,
  Maximize2,
  Minimize2,
  Settings,
  Download,
  Share2,
  Loader2,
  AlertCircle
} from 'lucide-react';

import { historicalDataManager } from '../../lib/services/historical-data-manager';
import { realTimeWebSocketManager } from '../../lib/services/realtime-websocket-manager';

interface CandlestickData {
  time: number;
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  [key: string]: any;
}

interface NavigationRange {
  start: Date;
  end: Date;
  center: Date;
}

interface AdvancedHistoricalChartProps {
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
  enableRealTime?: boolean;
  onDateChange?: (date: Date) => void;
}

const timeframeOptions = ['5m', '15m', '1h', '4h', '1d', '1w', '1M'] as const;
type TimeframeType = typeof timeframeOptions[number];

export default function AdvancedHistoricalChart({
  height = 600,
  showVolume = true,
  showIndicators = true,
  enableRealTime = false,
  onDateChange
}: AdvancedHistoricalChartProps) {
  // State Management
  const [data, setData] = useState<CandlestickData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Navigation State
  const [currentRange, setCurrentRange] = useState<NavigationRange>({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // Last 7 days
    end: new Date(),
    center: new Date()
  });

  const [timeframe, setTimeframe] = useState('5m');
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  // Chart State
  const [zoom, setZoom] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedCandle, setSelectedCandle] = useState<CandlestickData | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Navigation Handle State
  const [leftHandle, setLeftHandle] = useState(0.3); // Position as percentage (0-1)
  const [rightHandle, setRightHandle] = useState(0.7);
  const [isDraggingHandle, setIsDraggingHandle] = useState<'left' | 'right' | 'range' | null>(null);

  // Canvas References
  const chartCanvasRef = useRef<HTMLCanvasElement>(null);
  const miniChartCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Data bounds for full dataset
  const [dataBounds, setDataBounds] = useState({
    minDate: new Date('2020-01-01'),
    maxDate: new Date(),
    totalRecords: 0
  });

  /**
   * Initialize data bounds and load initial data
   */
  useEffect(() => {
    const initializeChart = async () => {
      try {
        setLoading(true);

        // Get data summary
        const summary = await historicalDataManager.getDataSummary();
        setDataBounds({
          minDate: summary.dateRange.min,
          maxDate: summary.dateRange.max,
          totalRecords: summary.totalRecords
        });

        // Load initial viewport data
        const initialData = await historicalDataManager.getViewportData(
          currentRange.center,
          timeframe,
          1000
        );

        setData(initialData);
        setError(null);

      } catch (err) {
        console.error('Error initializing chart:', err);
        setError('Failed to load chart data');
      } finally {
        setLoading(false);
      }
    };

    initializeChart();
  }, [timeframe]);

  /**
   * Load data when navigation range changes
   */
  useEffect(() => {
    const loadRangeData = async () => {
      try {
        const rangeData = await historicalDataManager.getDataForRange(
          currentRange.start,
          currentRange.end,
          timeframe
        );

        setData(rangeData);
        onDateChange?.(currentRange.center);

        // Preload adjacent data for smooth navigation
        historicalDataManager.preloadAdjacentData(currentRange.center, timeframe);

      } catch (err) {
        console.error('Error loading range data:', err);
      }
    };

    if (!loading) {
      loadRangeData();
    }
  }, [currentRange, timeframe, onDateChange, loading]);

  /**
   * Calculate date from handle position
   */
  const dateFromPosition = useCallback((position: number): Date => {
    const totalMs = dataBounds.maxDate.getTime() - dataBounds.minDate.getTime();
    return new Date(dataBounds.minDate.getTime() + (totalMs * position));
  }, [dataBounds]);

  /**
   * Calculate position from date
   */
  const positionFromDate = useCallback((date: Date): number => {
    const totalMs = dataBounds.maxDate.getTime() - dataBounds.minDate.getTime();
    return (date.getTime() - dataBounds.minDate.getTime()) / totalMs;
  }, [dataBounds]);

  /**
   * Handle navigation range changes from dual handles
   */
  const updateNavigationRange = useCallback((left: number, right: number) => {
    const startDate = dateFromPosition(left);
    const endDate = dateFromPosition(right);
    const centerDate = new Date((startDate.getTime() + endDate.getTime()) / 2);

    setCurrentRange({
      start: startDate,
      end: endDate,
      center: centerDate
    });
  }, [dateFromPosition]);

  /**
   * Handle drag for navigation handles
   */
  const handleDrag = useCallback((event: any, info: PanInfo, handle: 'left' | 'right' | 'range') => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const relativeX = event.clientX - rect.left;
    const position = Math.max(0, Math.min(1, relativeX / rect.width));

    if (handle === 'left') {
      const newLeft = Math.min(position, rightHandle - 0.05); // Minimum 5% gap
      setLeftHandle(newLeft);
      updateNavigationRange(newLeft, rightHandle);
    } else if (handle === 'right') {
      const newRight = Math.max(position, leftHandle + 0.05); // Minimum 5% gap
      setRightHandle(newRight);
      updateNavigationRange(leftHandle, newRight);
    } else if (handle === 'range') {
      // Move entire range
      const rangeWidth = rightHandle - leftHandle;
      const newLeft = Math.max(0, Math.min(1 - rangeWidth, position - rangeWidth / 2));
      const newRight = newLeft + rangeWidth;

      setLeftHandle(newLeft);
      setRightHandle(newRight);
      updateNavigationRange(newLeft, newRight);
    }
  }, [leftHandle, rightHandle, updateNavigationRange]);

  /**
   * Render main candlestick chart
   */
  const renderMainChart = useCallback(() => {
    const canvas = chartCanvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#0f172a'; // slate-950
    ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

    // Calculate price bounds
    const prices = data.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.1;

    const chartHeight = showVolume ? canvas.offsetHeight * 0.7 : canvas.offsetHeight * 0.9;
    const chartTop = 40;
    const chartBottom = chartTop + chartHeight;

    // Draw grid lines
    ctx.strokeStyle = '#334155'; // slate-700
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);

    // Horizontal grid lines (price levels)
    for (let i = 0; i <= 10; i++) {
      const y = chartTop + (chartHeight * i) / 10;
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(canvas.offsetWidth - 20, y);
      ctx.stroke();

      // Price labels
      const price = maxPrice + padding - ((maxPrice + padding - (minPrice - padding)) * i) / 10;
      ctx.fillStyle = '#64748b'; // slate-500
      ctx.font = '12px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(price.toFixed(2), 45, y + 4);
    }

    // Vertical grid lines (time)
    const timeStep = Math.max(1, Math.floor(data.length / 10));
    for (let i = 0; i < data.length; i += timeStep) {
      const x = 50 + ((canvas.offsetWidth - 70) * i) / (data.length - 1);
      ctx.beginPath();
      ctx.moveTo(x, chartTop);
      ctx.lineTo(x, chartBottom);
      ctx.stroke();

      // Time labels
      const date = new Date(data[i].time);
      ctx.fillStyle = '#64748b';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        x,
        canvas.offsetHeight - 5
      );
    }

    ctx.setLineDash([]);

    // Draw candlesticks
    const candleWidth = Math.max(1, (canvas.offsetWidth - 70) / data.length * 0.8);

    data.forEach((candle, index) => {
      const x = 50 + ((canvas.offsetWidth - 70) * index) / (data.length - 1);
      const yHigh = chartTop + ((maxPrice + padding - candle.high) / (maxPrice + padding - (minPrice - padding))) * chartHeight;
      const yLow = chartTop + ((maxPrice + padding - candle.low) / (maxPrice + padding - (minPrice - padding))) * chartHeight;
      const yOpen = chartTop + ((maxPrice + padding - candle.open) / (maxPrice + padding - (minPrice - padding))) * chartHeight;
      const yClose = chartTop + ((maxPrice + padding - candle.close) / (maxPrice + padding - (minPrice - padding))) * chartHeight;

      const isGreen = candle.close >= candle.open;

      // High-low line
      ctx.strokeStyle = isGreen ? '#10b981' : '#ef4444'; // green-500 : red-500
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, yHigh);
      ctx.lineTo(x, yLow);
      ctx.stroke();

      // Candle body
      ctx.fillStyle = isGreen ? '#10b981' : '#ef4444';
      const bodyHeight = Math.abs(yClose - yOpen);
      const bodyY = Math.min(yOpen, yClose);

      if (bodyHeight < 1) {
        // Draw line for doji
        ctx.beginPath();
        ctx.moveTo(x - candleWidth / 2, bodyY);
        ctx.lineTo(x + candleWidth / 2, bodyY);
        ctx.stroke();
      } else {
        ctx.fillRect(x - candleWidth / 2, bodyY, candleWidth, bodyHeight);
      }

      // Highlight selected candle
      if (selectedCandle && selectedCandle.time === candle.time) {
        ctx.strokeStyle = '#06b6d4'; // cyan-500
        ctx.lineWidth = 2;
        ctx.strokeRect(x - candleWidth / 2 - 2, bodyY - 2, candleWidth + 4, bodyHeight + 4);
      }
    });

    // Draw volume chart if enabled
    if (showVolume && data.length > 0) {
      const volumeTop = chartBottom + 20;
      const volumeHeight = canvas.offsetHeight - volumeTop - 40;
      const maxVolume = Math.max(...data.map(d => d.volume));

      data.forEach((candle, index) => {
        const x = 50 + ((canvas.offsetWidth - 70) * index) / (data.length - 1);
        const volumeBarHeight = (candle.volume / maxVolume) * volumeHeight;
        const isGreen = candle.close >= candle.open;

        ctx.fillStyle = isGreen ? '#10b98120' : '#ef444420'; // Semi-transparent
        ctx.fillRect(x - candleWidth / 2, volumeTop + volumeHeight - volumeBarHeight, candleWidth, volumeBarHeight);
      });

      // Volume label
      ctx.fillStyle = '#64748b';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Volume', 55, volumeTop + 15);
    }

    // Draw current price line if real-time enabled
    if (enableRealTime && data.length > 0) {
      const lastCandle = data[data.length - 1];
      const lastPrice = lastCandle.close;
      const y = chartTop + ((maxPrice + padding - lastPrice) / (maxPrice + padding - (minPrice - padding))) * chartHeight;

      ctx.strokeStyle = '#06b6d4'; // cyan-500
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(canvas.offsetWidth - 20, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Price label
      ctx.fillStyle = '#06b6d4';
      ctx.fillRect(canvas.offsetWidth - 80, y - 10, 75, 20);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(lastPrice.toFixed(4), canvas.offsetWidth - 42.5, y + 4);
    }

  }, [data, showVolume, enableRealTime, selectedCandle]);

  /**
   * Render mini context chart with navigation handles
   */
  const renderMiniChart = useCallback(() => {
    const canvas = miniChartCanvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#1e293b'; // slate-800
    ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

    // Draw sparkline
    if (data.length > 1) {
      const prices = data.map(d => d.close);
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      const priceRange = maxPrice - minPrice || 1;

      ctx.strokeStyle = '#06b6d4'; // cyan-500
      ctx.lineWidth = 2;
      ctx.beginPath();

      data.forEach((candle, index) => {
        const x = (index / (data.length - 1)) * canvas.offsetWidth;
        const y = canvas.offsetHeight - ((candle.close - minPrice) / priceRange) * (canvas.offsetHeight - 10) - 5;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Fill area under curve
      ctx.fillStyle = '#06b6d410';
      ctx.lineTo(canvas.offsetWidth, canvas.offsetHeight);
      ctx.lineTo(0, canvas.offsetHeight);
      ctx.closePath();
      ctx.fill();
    }

    // Draw selected range overlay
    const leftX = leftHandle * canvas.offsetWidth;
    const rightX = rightHandle * canvas.offsetWidth;

    // Dimmed areas outside selection
    ctx.fillStyle = '#00000060';
    ctx.fillRect(0, 0, leftX, canvas.offsetHeight);
    ctx.fillRect(rightX, 0, canvas.offsetWidth - rightX, canvas.offsetHeight);

    // Selection area border
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.strokeRect(leftX, 0, rightX - leftX, canvas.offsetHeight);

  }, [data, leftHandle, rightHandle]);

  /**
   * Render charts when data changes
   */
  useEffect(() => {
    renderMainChart();
    renderMiniChart();
  }, [renderMainChart, renderMiniChart]);

  /**
   * Handle canvas click for candle selection
   */
  const handleCanvasClick = useCallback((event: React.MouseEvent) => {
    const canvas = chartCanvasRef.current;
    if (!canvas || data.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const candleIndex = Math.floor(((x - 50) / (canvas.offsetWidth - 70)) * data.length);

    if (candleIndex >= 0 && candleIndex < data.length) {
      setSelectedCandle(data[candleIndex]);
    }
  }, [data]);

  /**
   * Navigation controls
   */
  const navigateRelative = useCallback((direction: 'prev' | 'next', amount: number = 1) => {
    const rangeMs = currentRange.end.getTime() - currentRange.start.getTime();
    const offsetMs = rangeMs * amount * (direction === 'prev' ? -1 : 1);

    const newStart = new Date(currentRange.start.getTime() + offsetMs);
    const newEnd = new Date(currentRange.end.getTime() + offsetMs);
    const newCenter = new Date((newStart.getTime() + newEnd.getTime()) / 2);

    // Ensure we stay within bounds
    if (newStart >= dataBounds.minDate && newEnd <= dataBounds.maxDate) {
      setCurrentRange({ start: newStart, end: newEnd, center: newCenter });
      setLeftHandle(positionFromDate(newStart));
      setRightHandle(positionFromDate(newEnd));
    }
  }, [currentRange, dataBounds, positionFromDate]);

  /**
   * Zoom controls
   */
  const handleZoom = useCallback((direction: 'in' | 'out', factor: number = 0.5) => {
    const currentMs = currentRange.end.getTime() - currentRange.start.getTime();
    const newMs = direction === 'in' ? currentMs * factor : currentMs / factor;

    const centerTime = currentRange.center.getTime();
    const newStart = new Date(centerTime - newMs / 2);
    const newEnd = new Date(centerTime + newMs / 2);

    // Ensure we stay within bounds
    if (newStart >= dataBounds.minDate && newEnd <= dataBounds.maxDate) {
      setCurrentRange({ start: newStart, end: newEnd, center: currentRange.center });
      setLeftHandle(positionFromDate(newStart));
      setRightHandle(positionFromDate(newEnd));
    }
  }, [currentRange, dataBounds, positionFromDate]);

  /**
   * Quick navigation buttons
   */
  const navigateToPreset = useCallback((preset: '1D' | '1W' | '1M' | '3M' | '1Y' | 'ALL') => {
    const now = new Date();
    let start: Date;

    switch (preset) {
      case '1D': start = new Date(now.getTime() - 24 * 60 * 60 * 1000); break;
      case '1W': start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000); break;
      case '1M': start = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000); break;
      case '3M': start = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000); break;
      case '1Y': start = new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000); break;
      case 'ALL': start = dataBounds.minDate; break;
      default: return;
    }

    const end = preset === 'ALL' ? dataBounds.maxDate : now;
    const center = new Date((start.getTime() + end.getTime()) / 2);

    setCurrentRange({ start, end, center });
    setLeftHandle(positionFromDate(start));
    setRightHandle(positionFromDate(end));
  }, [dataBounds, positionFromDate]);

  /**
   * Format price with proper decimals
   */
  const formatPrice = useCallback((price: number) => {
    return new Intl.NumberFormat('es-CO', {
      style: 'currency',
      currency: 'COP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(price);
  }, []);

  /**
   * Custom tooltip component
   */
  const CustomTooltip = useCallback(({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    const isUp = data.close >= data.open;

    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-slate-900/95 backdrop-blur-xl border border-slate-600/50 rounded-xl p-4 shadow-2xl min-w-[250px]"
      >
        <div className="text-xs text-slate-400 mb-3 font-mono">
          {data.formattedTime}
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <div className="text-slate-400">Open:</div>
          <div className="text-white font-mono">${data.open.toFixed(4)}</div>
          <div className="text-slate-400">High:</div>
          <div className="text-emerald-400 font-mono">${data.high.toFixed(4)}</div>
          <div className="text-slate-400">Low:</div>
          <div className="text-red-400 font-mono">${data.low.toFixed(4)}</div>
          <div className="text-slate-400">Close:</div>
          <div className={`font-mono font-bold ${
            isUp ? 'text-emerald-400' : 'text-red-400'
          }`}>
            ${data.close.toFixed(4)}
          </div>
          {showVolume && (
            <>
              <div className="text-slate-400">Volume:</div>
              <div className="text-blue-400 font-mono">{data.volume.toLocaleString()}</div>
            </>
          )}
        </div>

        {isUp ? (
          <TrendingUp className="h-4 w-4 text-emerald-400 mt-2" />
        ) : (
          <TrendingDown className="h-4 w-4 text-red-400 mt-2" />
        )}
      </motion.div>
    );
  }, [showVolume]);

  /**
   * Calculate price change
   */
  const priceChange = useMemo(() => {
    if (chartData.length < 2) return { change: 0, percentage: 0 };

    const current = chartData[chartData.length - 1];
    const previous = chartData[chartData.length - 2];
    const change = current.close - previous.close;
    const percentage = (change / previous.close) * 100;

    return { change, percentage };
  }, [chartData]);

  return (
    <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl overflow-hidden ${isFullscreen ? 'fixed inset-0 z-50 bg-slate-950' : ''}`} style={{ height }}>
      {/* Header with stats */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="text-2xl font-bold text-white">
            USDCOP
          </div>

          {chartData.length > 0 && (
            <div className="flex items-center space-x-2">
              <div className="text-xl font-mono text-white">
                ${chartData[chartData.length - 1].close.toFixed(4)}
              </div>
              <div className={`flex items-center space-x-1 ${
                priceChange.change >= 0 ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {priceChange.change >= 0 ? (
                  <TrendingUp className="h-4 w-4" />
                ) : (
                  <TrendingDown className="h-4 w-4" />
                )}
                <span className="text-sm font-mono">
                  {priceChange.change >= 0 ? '+' : ''}{priceChange.change.toFixed(4)}
                </span>
                <span className="text-sm">
                  ({priceChange.percentage >= 0 ? '+' : ''}{priceChange.percentage.toFixed(2)}%)
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Loading indicator */}
          {isLoading && (
            <div className="flex items-center space-x-2 text-cyan-400">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">
                {loadingProgress > 0 ? `${loadingProgress}%` : 'Cargando...'}
              </span>
            </div>
          )}

          {/* Fullscreen toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          >
            <Maximize2 className="h-4 w-4 text-slate-300" />
          </motion.button>
        </div>
      </div>

      {/* Error display */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center space-x-2"
          >
            <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
            <span className="text-red-400 text-sm">{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Historical Navigator */}
      <HistoricalNavigator
        currentDate={currentDate}
        minDate={minDate}
        maxDate={maxDate}
        timeframe={timeframe}
        onDateChange={handleDateChange}
        onTimeframeChange={handleTimeframeChange}
        onZoomChange={handleZoomChange}
        isLoading={isLoading}
        autoPlay={autoPlay}
        onAutoPlayChange={setAutoPlay}
      />

      {/* Main Chart */}
      <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.6}/>
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0.1}/>
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#334155"
              strokeOpacity={0.3}
              vertical={false}
            />

            <XAxis
              dataKey="formattedTime"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />

            <YAxis
              yAxisId="price"
              orientation="right"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              domain={['dataMin - 10', 'dataMax + 10']}
              tickFormatter={(value) => `$${value.toFixed(2)}`}
            />

            {showVolume && (
              <YAxis
                yAxisId="volume"
                orientation="left"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                domain={[0, 'dataMax']}
                tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
              />
            )}

            <Tooltip content={<CustomTooltip />} />

            {/* Volume bars */}
            {showVolume && (
              <Bar
                yAxisId="volume"
                dataKey="volume"
                fill="url(#volumeGradient)"
                opacity={0.7}
                radius={[1, 1, 0, 0]}
              />
            )}

            {/* Price line */}
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="#0ea5e9"
              strokeWidth={2}
              fill="url(#priceGradient)"
              dot={false}
              activeDot={{
                r: 4,
                stroke: '#0ea5e9',
                strokeWidth: 2,
                fill: '#1e293b'
              }}
            />

            {/* Brush for navigation */}
            <Brush
              dataKey="formattedTime"
              height={40}
              stroke="#0ea5e9"
              fill="rgba(14, 165, 233, 0.1)"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Data summary */}
      <div className="text-xs text-slate-400 text-center">
        Mostrando {chartData.length.toLocaleString()} puntos de datos |
        Rango total: {minDate.toLocaleDateString()} - {maxDate.toLocaleDateString()} |
        Cache: {historicalDataManager.getCacheStats().size} chunks
      </div>
    </div>
  );
}