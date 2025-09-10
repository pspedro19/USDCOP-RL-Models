'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi, 
  CandlestickData, 
  Time, 
  ColorType, 
  CrosshairMode, 
  LineStyle,
  CandlestickSeries,
  HistogramSeries,
  LineSeries
} from 'lightweight-charts';
// Custom date formatter to replace date-fns
const formatDate = (date: Date, pattern: string): string => {
  const months = [
    'ene', 'feb', 'mar', 'abr', 'may', 'jun',
    'jul', 'ago', 'sep', 'oct', 'nov', 'dic'
  ];
  
  const monthsFull = [
    'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
    'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
  ];

  const day = date.getDate();
  const month = date.getMonth();
  const year = date.getFullYear();
  const hours = date.getHours();
  const minutes = date.getMinutes();

  const pad = (num: number) => num.toString().padStart(2, '0');

  switch (pattern) {
    case 'MMM dd':
      return `${months[month]} ${pad(day)}`;
    case 'yyyy-MM-dd-HHmm':
      return `${year}-${pad(month + 1)}-${pad(day)}-${pad(hours)}${pad(minutes)}`;
    case 'MMM dd, yyyy':
      return `${months[month]} ${pad(day)}, ${year}`;
    case 'dd MMM yyyy HH:mm':
      return `${pad(day)} ${months[month]} ${year} ${pad(hours)}:${pad(minutes)}`;
    default:
      return date.toLocaleDateString('es-ES');
  }
};
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Calendar, ZoomIn, ZoomOut, Maximize2, RotateCcw, Download, TrendingUp, Activity, Target, Settings, Eye, EyeOff } from 'lucide-react';
import { Slider } from '@/components/ui/slider';

interface InteractiveTradingChartProps {
  data: any[];
  isRealtime?: boolean;
  onRangeChange?: (start: Date, end: Date) => void;
}

export const InteractiveTradingChart: React.FC<InteractiveTradingChartProps> = ({
  data,
  isRealtime = false,
  onRangeChange
}) => {
  // Log incoming data with details
  console.log(`[InteractiveTradingChart] Component mounted with ${data ? data.length : 0} data points`);
  console.log(`[InteractiveTradingChart] First few data points:`, data?.slice(0, 3));
  console.log(`[InteractiveTradingChart] Component props:`, { isRealtime, onRangeChange: !!onRangeChange });
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const bbUpperRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bbMiddleRef = useRef<ISeriesApi<'Line'> | null>(null);
  const bbLowerRef = useRef<ISeriesApi<'Line'> | null>(null);
  const emaRef = useRef<ISeriesApi<'Line'> | null>(null);
  
  const [selectedRange, setSelectedRange] = useState<[number, number]>([0, 100]);
  const [timeframe, setTimeframe] = useState<'5M' | '15M' | '30M' | '1H' | '4H' | '1D' | '1W' | 'ALL'>('ALL'); // Default to ALL to show full historical data
  const [showVolume, setShowVolume] = useState(true);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [percentChange, setPercentChange] = useState<number>(0);
  const [hoveredCandle, setHoveredCandle] = useState<CandlestickData | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [showFloatingPanel, setShowFloatingPanel] = useState(true);
  const [crosshairVisible, setCrosshairVisible] = useState(true);
  const [indicators, setIndicators] = useState({
    bb: true,
    rsi: false,
    macd: false,
    ema: true
  });

  // Initialize chart
  useEffect(() => {
    console.log(`[Chart] === CHART INITIALIZATION STARTED ===`);
    console.log(`[Chart] Data length: ${data ? data.length : 0}`);
    console.log(`[Chart] Container ref exists: ${!!chartContainerRef.current}`);
    
    if (!chartContainerRef.current) {
      console.error('[Chart] ❌ NO CHART CONTAINER REF - This is a critical error!');
      return;
    }
    
    // Check container dimensions
    const containerWidth = chartContainerRef.current.clientWidth || 800;
    const containerHeight = chartContainerRef.current.clientHeight || 500;
    console.log(`[Chart] Container dimensions: ${containerWidth}x${containerHeight}`);
    
    if (containerWidth === 0 || containerHeight === 0) {
      console.warn(`[Chart] Container has zero dimensions: ${containerWidth}x${containerHeight}`);
      // Force dimensions if container is not properly sized
      chartContainerRef.current.style.width = '100%';
      chartContainerRef.current.style.height = '500px';
      chartContainerRef.current.style.minHeight = '500px';
    }
    
    if (!data || data.length === 0) {
      console.warn('[Chart] No data available for chart initialization');
      return;
    }

    // Create chart with enhanced visual styling
    const chartHeight = 500;
    
    console.log(`[Chart] Creating chart with dimensions: ${containerWidth}x${chartHeight}`);
    
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height: chartHeight,
      layout: {
        background: {
          type: ColorType.VerticalGradient,
          topColor: '#0a0e1a',
          bottomColor: '#111827'
        },
        textColor: '#e5e7eb',
        fontSize: 12,
        fontFamily: 'Inter, system-ui, sans-serif'
      },
      grid: {
        vertLines: { 
          color: 'rgba(59, 130, 246, 0.1)', 
          style: LineStyle.Solid,
          visible: true
        },
        horzLines: { 
          color: 'rgba(59, 130, 246, 0.1)', 
          style: LineStyle.Solid,
          visible: true
        }
      },
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: {
          width: 2,
          color: 'rgba(6, 182, 212, 0.8)',
          style: LineStyle.Solid,
          labelBackgroundColor: 'rgba(6, 182, 212, 0.9)',
          visible: crosshairVisible
        },
        horzLine: {
          width: 2,
          color: 'rgba(6, 182, 212, 0.8)',
          style: LineStyle.Solid,
          labelBackgroundColor: 'rgba(6, 182, 212, 0.9)',
          visible: crosshairVisible
        }
      },
      rightPriceScale: {
        borderColor: '#2B5CE6',
        scaleMargins: {
          top: 0.1,
          bottom: 0.3 // Always leave space for volume
        }
      },
      timeScale: {
        borderColor: '#2B5CE6',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: any) => {
          const date = new Date(time * 1000);
          return formatDate(date, 'MMM dd');
        }
      },
      watermark: {
        visible: true,
        fontSize: 56,
        horzAlign: 'center',
        vertAlign: 'center',
        color: 'rgba(6, 182, 212, 0.08)',
        text: 'USD/COP Pro'
      }
    });

    chartRef.current = chart;
    console.log(`[Chart] ✅ Chart created successfully!`);

    // Add candlestick series with gradient effects
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: 'rgba(16, 185, 129, 0.9)',
      downColor: 'rgba(239, 68, 68, 0.9)',
      borderVisible: true,
      borderUpColor: 'rgba(16, 185, 129, 1)',
      borderDownColor: 'rgba(239, 68, 68, 1)',
      wickUpColor: 'rgba(16, 185, 129, 0.8)',
      wickDownColor: 'rgba(239, 68, 68, 0.8)',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01
      }
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Always create volume series with gradient
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(59, 130, 246, 0.6)',
      priceFormat: {
        type: 'volume'
      },
      priceScaleId: '',
      scaleMargins: {
        top: 0.8,
        bottom: 0
      }
    });
    volumeSeriesRef.current = volumeSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth
        });
      }
    };
    window.addEventListener('resize', handleResize);

    // Subscribe to crosshair move for price tracking and tooltip
    chart.subscribeCrosshairMove((param) => {
      if (param.seriesData && param.seriesData.size > 0) {
        const iterator = param.seriesData.values();
        const firstSeries = iterator.next().value;
        if (firstSeries && typeof firstSeries === 'object' && 'close' in firstSeries) {
          setCurrentPrice(Number(firstSeries.close));
          setHoveredCandle(firstSeries as CandlestickData);
          if (param.point) {
            setTooltipPosition({ x: param.point.x, y: param.point.y });
          }
        }
      } else {
        setHoveredCandle(null);
      }
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []); // Remove showVolume dependency to avoid recreating chart

  // Update data based on selected range and timeframe
  useEffect(() => {
    console.log(`[Chart] Update effect - Chart: ${!!chartRef.current}, Series: ${!!candlestickSeriesRef.current}, Data: ${data ? data.length : 0}`);
    if (!chartRef.current || !candlestickSeriesRef.current) {
      console.warn('[Chart] Chart or series not initialized');
      return;
    }
    if (!data || data.length === 0) {
      console.warn('[Chart] No data to update');
      return;
    }

    // First, aggregate data based on selected timeframe
    const aggregatedData = getTimeframeData(data, timeframe);
    console.log(`[Chart] Aggregated ${aggregatedData.length} data points for timeframe ${timeframe}`);
    
    if (!aggregatedData || aggregatedData.length === 0) {
      console.warn('No aggregated data available');
      return;
    }
    
    // Remove duplicates by using a Map with timestamp as key
    const uniqueDataMap = new Map();
    aggregatedData.forEach(item => {
      if (item && item.datetime) {
        const timestamp = Math.floor(new Date(item.datetime).getTime() / 1000);
        if (!uniqueDataMap.has(timestamp)) {
          uniqueDataMap.set(timestamp, item);
        }
      }
    });
    
    // Convert back to array and sort
    const uniqueData = Array.from(uniqueDataMap.values())
      .sort((a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime());
    
    if (uniqueData.length === 0) {
      console.warn('No unique data after deduplication');
      return;
    }
    
    // For ALL timeframe, show everything; otherwise apply range selection
    let displayData;
    if (timeframe === 'ALL') {
      displayData = uniqueData; // Show ALL data for full historical view
      console.log(`[Chart] Showing ALL data: ${uniqueData.length} points`);
    } else {
      const startIdx = Math.floor((selectedRange[0] / 100) * uniqueData.length);
      const endIdx = Math.floor((selectedRange[1] / 100) * uniqueData.length);
      displayData = uniqueData.slice(startIdx, Math.max(endIdx, startIdx + 1)); // Ensure at least 1 data point
      console.log(`[Chart] Showing range ${selectedRange[0]}%-${selectedRange[1]}%: ${displayData.length} points`);
    }

    // Convert data to TradingView format with validation
    const candleData = displayData
      .filter(item => {
        // Validate each data point
        return item && 
               item.datetime && 
               !isNaN(item.open) && 
               !isNaN(item.high) && 
               !isNaN(item.low) && 
               !isNaN(item.close) &&
               item.open > 0 &&
               item.high > 0 &&
               item.low > 0 &&
               item.close > 0;
      })
      .map(item => ({
        time: Math.floor(new Date(item.datetime).getTime() / 1000) as Time,
        open: Number(item.open),
        high: Number(item.high),
        low: Number(item.low),
        close: Number(item.close)
      }));

    console.log(`[Chart] Setting ${candleData.length} candles from ${displayData.length} display data`);
    
    if (candleData.length > 0) {
      // Log the date range being displayed
      const firstCandle = candleData[0];
      const lastCandle = candleData[candleData.length - 1];
      const firstDate = new Date(firstCandle.time * 1000);
      const lastDate = new Date(lastCandle.time * 1000);
      console.log(`[Chart] Date range: ${firstDate.toISOString().split('T')[0]} to ${lastDate.toISOString().split('T')[0]}`);
      console.log(`[Chart] First candle time: ${firstCandle.time}, Last candle time: ${lastCandle.time}`);
      
      candlestickSeriesRef.current.setData(candleData);
    } else {
      console.warn('[Chart] No valid candle data to display');
    }

    // Update volume series
    if (volumeSeriesRef.current) {
      if (showVolume) {
        const volumeData = displayData
          .filter(item => item && item.datetime)
          .map(item => ({
            time: Math.floor(new Date(item.datetime).getTime() / 1000) as Time,
            value: item.volume || 0,
            color: item.close >= item.open ? '#26a69a' : '#ef5350'
          }));
        volumeSeriesRef.current.setData(volumeData);
        volumeSeriesRef.current.applyOptions({ visible: true });
      } else {
        volumeSeriesRef.current.applyOptions({ visible: false });
      }
    }

    // Calculate and add Bollinger Bands if enabled
    if (indicators.bb && displayData.length > 20) {
      const bbData = calculateBollingerBands(displayData);
      
      if (!bbUpperRef.current && chartRef.current) {
        bbUpperRef.current = chartRef.current.addSeries(LineSeries, {
          color: 'rgba(255, 176, 0, 0.5)',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false
        });
        bbMiddleRef.current = chartRef.current.addSeries(LineSeries, {
          color: 'rgba(255, 176, 0, 0.3)',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false
        });
        bbLowerRef.current = chartRef.current.addSeries(LineSeries, {
          color: 'rgba(255, 176, 0, 0.5)',
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false
        });
      }
      
      if (bbUpperRef.current) bbUpperRef.current.setData(bbData.upper);
      if (bbMiddleRef.current) bbMiddleRef.current.setData(bbData.middle);
      if (bbLowerRef.current) bbLowerRef.current.setData(bbData.lower);
    }

    // Calculate and add EMA if enabled
    if (indicators.ema && displayData.length > 20) {
      const emaData = calculateEMA(displayData, 20);
      
      if (!emaRef.current && chartRef.current) {
        emaRef.current = chartRef.current.addSeries(LineSeries, {
          color: 'rgba(33, 150, 243, 0.8)',
          lineWidth: 2,
          crosshairMarkerVisible: false,
          lastValueVisible: false,
          priceLineVisible: false
        });
      }
      
      if (emaRef.current) emaRef.current.setData(emaData);
    }

    // Calculate price change
    if (displayData.length > 1) {
      const first = displayData[0];
      const last = displayData[displayData.length - 1];
      if (first && last && first.close > 0 && last.close > 0) {
        const change = last.close - first.close;
        const percent = (change / first.close) * 100;
        setPriceChange(change);
        setPercentChange(percent);
        setCurrentPrice(Number(last.close));
      }
    }

    // Fit content
    if (chartRef.current && candleData.length > 0) {
      try {
        chartRef.current.timeScale().fitContent();
      } catch (err) {
        console.warn('[Chart] Error fitting content:', err);
      }
    }

    // Notify parent of range change
    if (onRangeChange && displayData.length > 0) {
      onRangeChange(
        new Date(displayData[0].datetime),
        new Date(displayData[displayData.length - 1].datetime)
      );
    }
  }, [data, selectedRange, showVolume, onRangeChange, indicators, timeframe]);

  // Calculate Bollinger Bands
  const calculateBollingerBands = (data: any[]) => {
    const period = 20;
    const stdDev = 2;
    const upper: any[] = [];
    const middle: any[] = [];
    const lower: any[] = [];

    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const avg = slice.reduce((sum, item) => sum + item.close, 0) / period;
      const variance = slice.reduce((sum, item) => sum + Math.pow(item.close - avg, 2), 0) / period;
      const std = Math.sqrt(variance);
      
      const time = (new Date(data[i].datetime).getTime() / 1000) as Time;
      upper.push({ time, value: avg + std * stdDev });
      middle.push({ time, value: avg });
      lower.push({ time, value: avg - std * stdDev });
    }

    return { upper, middle, lower };
  };

  // Calculate EMA
  const calculateEMA = (data: any[], period: number) => {
    const ema: any[] = [];
    const multiplier = 2 / (period + 1);
    let prevEMA = data[0].close;

    for (let i = 0; i < data.length; i++) {
      if (i < period - 1) {
        prevEMA = (prevEMA * i + data[i].close) / (i + 1);
      } else {
        prevEMA = (data[i].close - prevEMA) * multiplier + prevEMA;
      }
      ema.push({
        time: (new Date(data[i].datetime).getTime() / 1000) as Time,
        value: prevEMA
      });
    }

    return ema;
  };

  // Aggregate OHLC data for different timeframes
  const aggregateOHLC = (data: any[], periodMinutes: number) => {
    if (!data || data.length === 0) return [];
    
    const aggregated: any[] = [];
    const candlesPerPeriod = periodMinutes / 5; // Since base data is 5-minute candles
    
    for (let i = 0; i < data.length; i += candlesPerPeriod) {
      const slice = data.slice(i, Math.min(i + candlesPerPeriod, data.length));
      
      if (slice.length === 0) continue;
      
      // Calculate OHLC for this period
      const open = slice[0].open;
      const high = Math.max(...slice.map(d => d.high));
      const low = Math.min(...slice.map(d => d.low));
      const close = slice[slice.length - 1].close;
      const volume = slice.reduce((sum, d) => sum + (d.volume || 0), 0);
      
      aggregated.push({
        datetime: slice[0].datetime,
        open,
        high,
        low,
        close,
        volume
      });
    }
    
    return aggregated;
  };

  // Get aggregated data based on timeframe
  const getTimeframeData = (allData: any[], tf: typeof timeframe) => {
    if (!allData || allData.length === 0) return [];
    
    switch (tf) {
      case '5M':
        // Original 5-minute data (no aggregation needed)
        return allData;
      case '15M':
        // Aggregate every 3 candles
        return aggregateOHLC(allData, 15);
      case '30M':
        // Aggregate every 6 candles
        return aggregateOHLC(allData, 30);
      case '1H':
        // Aggregate every 12 candles
        return aggregateOHLC(allData, 60);
      case '4H':
        // Aggregate every 48 candles
        return aggregateOHLC(allData, 240);
      case '1D':
        // For daily, we aggregate all candles within each trading day
        // Assuming ~59 candles per day (295 minutes / 5)
        return aggregateOHLC(allData, 295);
      case '1W':
        // Weekly aggregation (5 trading days)
        return aggregateOHLC(allData, 1475); // 295 * 5 minutes
      case 'ALL':
        // Show all data with 5-minute granularity
        return allData;
      default:
        return allData;
    }
  };

  // Timeframe presets (now with proper aggregation)
  const handleTimeframeChange = (tf: typeof timeframe) => {
    setTimeframe(tf);
    
    // For ALL, show everything
    if (tf === 'ALL') {
      setSelectedRange([0, 100]);
      return;
    }
    
    // For other timeframes, we'll aggregate the data in the useEffect
    // and show the appropriate range
    setSelectedRange([0, 100]);
  };

  // Zoom controls
  const handleZoom = (direction: 'in' | 'out') => {
    const [start, end] = selectedRange;
    const range = end - start;
    const delta = range * 0.1;

    if (direction === 'in' && range > 5) {
      setSelectedRange([start + delta, end - delta]);
    } else if (direction === 'out') {
      setSelectedRange([
        Math.max(0, start - delta),
        Math.min(100, end + delta)
      ]);
    }
  };

  // Reset view
  const handleReset = () => {
    setSelectedRange([0, 100]);
    setTimeframe('ALL');
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  };

  // Export chart
  const handleExport = () => {
    if (chartRef.current) {
      const screenshot = chartRef.current.takeScreenshot();
      screenshot.then((canvas) => {
        const link = document.createElement('a');
        link.download = `usdcop-chart-${formatDate(new Date(), 'yyyy-MM-dd-HHmm')}.png`;
        link.href = canvas.toDataURL();
        link.click();
      });
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
    <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6 relative overflow-hidden">
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/5 animate-pulse" />
      {/* Floating Tools Panel */}
      <AnimatePresence>
        {showFloatingPanel && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="absolute top-4 right-4 z-20 bg-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-xl p-3 shadow-2xl"
          >
            <div className="flex items-center gap-2">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleZoom('in')}
                className="p-2 rounded-lg bg-slate-700/50 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-400 transition-all duration-200"
              >
                <ZoomIn className="w-4 h-4" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleZoom('out')}
                className="p-2 rounded-lg bg-slate-700/50 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-400 transition-all duration-200"
              >
                <ZoomOut className="w-4 h-4" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setCrosshairVisible(!crosshairVisible)}
                className={`p-2 rounded-lg bg-slate-700/50 transition-all duration-200 ${
                  crosshairVisible 
                    ? 'text-cyan-400 bg-cyan-500/20' 
                    : 'text-slate-400 hover:bg-slate-600/50'
                }`}
              >
                <Target className="w-4 h-4" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleReset}
                className="p-2 rounded-lg bg-slate-700/50 hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-400 transition-all duration-200"
              >
                <RotateCcw className="w-4 h-4" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={handleExport}
                className="p-2 rounded-lg bg-slate-700/50 hover:bg-purple-500/20 text-slate-300 hover:text-purple-400 transition-all duration-200"
              >
                <Download className="w-4 h-4" />
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowFloatingPanel(false)}
                className="p-2 rounded-lg bg-slate-700/50 hover:bg-red-500/20 text-slate-300 hover:text-red-400 transition-all duration-200"
              >
                <EyeOff className="w-4 h-4" />
              </motion.button>
            </div></motion.div>
        )}
      </AnimatePresence>
      {!showFloatingPanel && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setShowFloatingPanel(true)}
          className="absolute top-4 right-4 z-20 p-2 rounded-lg bg-slate-800/90 backdrop-blur-xl border border-slate-700/50 text-slate-400 hover:text-cyan-400 transition-all duration-200"
        >
          <Settings className="w-4 h-4" />
        </motion.button>
      )}
      {/* Header with controls */}
      <div className="flex items-center justify-between mb-6 relative z-10">
        <div className="flex items-center gap-4">
          <motion.h2
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
          >
            USD/COP
          </motion.h2>
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-3"
          >
            <span className="text-3xl font-mono font-bold text-white drop-shadow-lg">
              ${(typeof currentPrice === 'number' ? currentPrice : 0).toFixed(2)}
            </span>
            <motion.span
              animate={{ 
                color: priceChange >= 0 ? '#10b981' : '#ef4444',
                scale: [1, 1.05, 1]
              }}
              transition={{ duration: 0.3 }}
              className={`text-sm font-semibold px-3 py-1 rounded-full backdrop-blur-sm ${
                priceChange >= 0 
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                  : 'bg-red-500/20 text-red-400 border border-red-500/30'
              }`}
            >
              {priceChange >= 0 ? '+' : ''}{(typeof priceChange === 'number' ? priceChange : 0).toFixed(2)} 
              ({(typeof percentChange === 'number' ? percentChange : 0).toFixed(2)}%)
            </motion.span>
          </motion.div>
          {isRealtime && (
            <motion.div
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-full border border-emerald-500/30 backdrop-blur-sm"
            >
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <Activity className="w-4 h-4" />
              </motion.div>
              <span className="text-sm font-medium">Live</span>
            </motion.div>
          )}
        </div>
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex items-center gap-2 bg-slate-800/50 backdrop-blur-sm rounded-xl p-1 border border-slate-700/50"
        >
          {(['5M', '15M', '30M', '1H', '4H', '1D', '1W', 'ALL'] as const).map((tf, index) => (
            <motion.button
              key={tf}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => handleTimeframeChange(tf)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                timeframe === tf
                  ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg'
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              {tf}
            </motion.button>
          ))}
        </motion.div>
      </div>
      {/* Chart container with enhanced visuals */}
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="relative"
      >
        <div 
          ref={chartContainerRef} 
          className="w-full h-[500px] rounded-xl overflow-hidden border border-slate-700/50 bg-gradient-to-br from-slate-900/50 to-slate-800/50 backdrop-blur-sm relative"
          style={{ minWidth: '600px', minHeight: '500px', width: '100%', height: '500px', display: 'block', position: 'relative' }}
          onLoad={() => console.log('[Chart] Container loaded')}
        >
          {/* Glowing border effect */}
          <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-500/20 via-transparent to-purple-500/20 opacity-50 blur-sm" />
        </div>
        {(!data || data.length === 0) && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 flex items-center justify-center bg-slate-900/90 backdrop-blur-sm rounded-xl z-50"
          >
            <div className="flex flex-col items-center gap-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full"
              />
              <span className="text-slate-400 font-medium">Loading chart data...</span>
              <span className="text-red-400 font-mono text-xs">Debug: {data ? data.length : 0} data points</span>
            </div>
          </motion.div>
        )}
      </motion.div>
      <div className="mt-4 px-4">
        <div className="flex items-center gap-4">
          <Calendar className="w-4 h-4 text-gray-400" />
          <Slider
            value={selectedRange}
            onValueChange={setSelectedRange}
            min={0}
            max={100}
            step={0.1}
            className="flex-1"
          />
          <span className="text-sm text-gray-400 min-w-[120px]">
            {data.length > 0 && (
              <>
                {formatDate(new Date(data[Math.floor((selectedRange[0] / 100) * data.length)]?.datetime || data[0].datetime), 'MMM dd, yyyy')}
                {' - '}
                {formatDate(new Date(data[Math.floor((selectedRange[1] / 100) * data.length - 1)]?.datetime || data[data.length - 1].datetime), 'MMM dd, yyyy')}
              </>
            )}
          </span>
        </div>
      </div>
      {/* Additional controls */}
      <div className="mt-4 flex items-center gap-4 text-sm">
        <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={showVolume}
            onChange={(e) => setShowVolume(e.target.checked)}
            className="rounded"
          />
          Volume
        </label>
        <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={indicators.bb}
            onChange={(e) => setIndicators(prev => ({ ...prev, bb: e.target.checked }))}
            className="rounded"
          />
          Bollinger Bands
        </label>
        <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={indicators.ema}
            onChange={(e) => setIndicators(prev => ({ ...prev, ema: e.target.checked }))}
            className="rounded"
          />
          EMA(20)
        </label>
      </div>
      {/* Enhanced Interactive Tooltip with Glassmorphism */}
      <AnimatePresence>
        {hoveredCandle && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 10 }}
            transition={{ duration: 0.2 }}
            className="absolute z-50 pointer-events-none"
            style={{
              left: `${tooltipPosition.x + 10}px`,
              top: `${tooltipPosition.y - 120}px`,
            }}
          >
            <div className="bg-slate-900/90 backdrop-blur-xl border border-slate-600/50 rounded-2xl p-4 shadow-2xl min-w-[220px] relative overflow-hidden">
              {/* Glassmorphism background */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-purple-500/10 rounded-2xl" />
              <div className="absolute inset-0 bg-gradient-to-t from-slate-800/20 to-transparent rounded-2xl" />
              <div className="relative z-10">
                <div className="text-xs text-slate-400 mb-3 font-mono bg-slate-800/50 px-2 py-1 rounded-lg">
                  {formatDate(new Date((hoveredCandle.time as number) * 1000), 'dd MMM yyyy HH:mm')}
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                  <div className="text-slate-400 font-medium">Open:</div>
                  <div className="text-white font-mono font-bold">${hoveredCandle.open.toFixed(2)}</div>
                  <div className="text-slate-400 font-medium">High:</div>
                  <div className="text-emerald-400 font-mono font-bold">${hoveredCandle.high.toFixed(2)}</div>
                  <div className="text-slate-400 font-medium">Low:</div>
                  <div className="text-red-400 font-mono font-bold">${hoveredCandle.low.toFixed(2)}</div>
                  <div className="text-slate-400 font-medium">Close:</div>
                  <div className={`font-mono font-bold text-lg ${
                    hoveredCandle.close >= hoveredCandle.open 
                      ? 'text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.3)]' 
                      : 'text-red-400 drop-shadow-[0_0_8px_rgba(239,68,68,0.3)]'
                  }`}>
                    ${hoveredCandle.close.toFixed(2)}
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-slate-700/50">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 text-xs">Change:</span>
                    <div className={`font-mono text-sm font-bold px-2 py-1 rounded-lg ${
                      hoveredCandle.close >= hoveredCandle.open 
                        ? 'text-emerald-400 bg-emerald-500/20 border border-emerald-500/30' 
                        : 'text-red-400 bg-red-500/20 border border-red-500/30'
                    }`}>
                      {hoveredCandle.close >= hoveredCandle.open ? '+' : ''}
                      {(hoveredCandle.close - hoveredCandle.open).toFixed(2)}
                      ({((hoveredCandle.close - hoveredCandle.open) / hoveredCandle.open * 100).toFixed(2)}%)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
    </motion.div>
  );
};