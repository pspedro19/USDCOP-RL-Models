'use client';

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi, 
  CandlestickData, 
  Time, 
  ColorType, 
  CrosshairMode, 
  LineStyle
} from 'lightweight-charts';
// Custom date formatting function with Spanish month names
const formatDate = (date: Date, formatStr: string) => {
  const months = [
    'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'
  ];
  
  const year = date.getFullYear();
  const month = date.getMonth();
  const day = date.getDate();
  const hours = date.getHours();
  const minutes = date.getMinutes();
  
  switch (formatStr) {
    case 'MMM dd':
      return `${months[month]} ${String(day).padStart(2, '0')}`;
    case 'yyyy-MM-dd-HHmm':
      return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}-${String(hours).padStart(2, '0')}${String(minutes).padStart(2, '0')}`;
    default:
      return date.toLocaleDateString();
  }
};

// Use the format function instead of date-fns format
const format = formatDate;
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ZoomIn, ZoomOut, RotateCcw, Download, Activity } from 'lucide-react';

interface VirtualizedChartProps {
  data: any[];
  isRealtime?: boolean;
  onRangeChange?: (start: Date, end: Date) => void;
  chunkSize?: number; // Number of data points to load per chunk
  initialViewSize?: number; // Initial number of data points to show
}

export const VirtualizedChart: React.FC<VirtualizedChartProps> = ({
  data,
  isRealtime = false,
  onRangeChange,
  chunkSize = 5000, // Load 5k points per chunk
  initialViewSize = 2000 // Show last 2k points initially
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const loadingRef = useRef<boolean>(false);
  const observerRef = useRef<IntersectionObserver | null>(null);
  
  const [visibleRange, setVisibleRange] = useState<[number, number]>([0, initialViewSize]);
  const [loadedChunks, setLoadedChunks] = useState<Set<number>>(new Set([0]));
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);

  // Memoize data processing to avoid recalculation on every render
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // Remove duplicates and sort
    const uniqueDataMap = new Map();
    data.forEach(item => {
      if (item?.datetime) {
        const timestamp = Math.floor(new Date(item.datetime).getTime() / 1000);
        if (!uniqueDataMap.has(timestamp)) {
          uniqueDataMap.set(timestamp, {
            ...item,
            timestamp
          });
        }
      }
    });
    
    return Array.from(uniqueDataMap.values())
      .sort((a, b) => a.timestamp - b.timestamp);
  }, [data]);

  // Get visible data chunk based on current range
  const visibleData = useMemo(() => {
    const [start, end] = visibleRange;
    return processedData.slice(
      Math.max(0, start - 100), // Load a bit extra for smooth scrolling
      Math.min(processedData.length, end + 100)
    );
  }, [processedData, visibleRange]);

  // Convert to chart format with validation
  const chartData = useMemo(() => {
    return visibleData
      .filter(item => 
        item?.open > 0 && 
        item?.high > 0 && 
        item?.low > 0 && 
        item?.close > 0 &&
        !isNaN(item.open) &&
        !isNaN(item.high) &&
        !isNaN(item.low) &&
        !isNaN(item.close)
      )
      .map(item => ({
        time: item.timestamp as Time,
        open: Number(item.open),
        high: Number(item.high),
        low: Number(item.low),
        close: Number(item.close),
        volume: item.volume || 0
      }));
  }, [visibleData]);

  // Initialize chart with performance optimizations
  useEffect(() => {
    if (!chartContainerRef.current || !processedData.length) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
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
        vertLines: { color: 'rgba(59, 130, 246, 0.1)', style: LineStyle.Solid },
        horzLines: { color: 'rgba(59, 130, 246, 0.1)', style: LineStyle.Solid }
      },
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: {
          color: 'rgba(6, 182, 212, 0.8)',
          style: LineStyle.Solid,
          labelBackgroundColor: 'rgba(6, 182, 212, 0.9)'
        },
        horzLine: {
          color: 'rgba(6, 182, 212, 0.8)',
          style: LineStyle.Solid,
          labelBackgroundColor: 'rgba(6, 182, 212, 0.9)'
        }
      },
      rightPriceScale: {
        borderColor: '#2B5CE6',
        scaleMargins: { top: 0.1, bottom: 0.3 }
      },
      timeScale: {
        borderColor: '#2B5CE6',
        timeVisible: true,
        secondsVisible: false,
        // Optimize time scale for large datasets
        tickMarkFormatter: (time: any) => {
          const date = new Date(time * 1000);
          return format(date, 'MMM dd');
        }
      },
      // Disable watermark for better performance
      watermark: { visible: false }
    });

    chartRef.current = chart;

    // Add candlestick series
    const candlestickSeries = chart.addSeries('Candlestick', {
      upColor: 'rgba(16, 185, 129, 0.9)',
      downColor: 'rgba(239, 68, 68, 0.9)',
      borderUpColor: 'rgba(16, 185, 129, 1)',
      borderDownColor: 'rgba(239, 68, 68, 1)',
      wickUpColor: 'rgba(16, 185, 129, 0.8)',
      wickDownColor: 'rgba(239, 68, 68, 0.8)',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 }
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Add volume series
    const volumeSeries = chart.addSeries('Histogram', {
      color: 'rgba(59, 130, 246, 0.6)',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
      scaleMargins: { top: 0.8, bottom: 0 }
    });
    volumeSeriesRef.current = volumeSeries;

    // Optimize resize handler
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        requestAnimationFrame(() => {
          chart.applyOptions({
            width: chartContainerRef.current!.clientWidth
          });
        });
      }
    };

    // Use passive listeners for better performance
    window.addEventListener('resize', handleResize, { passive: true });

    // Subscribe to visible range changes for virtualization
    chart.timeScale().subscribeVisibleTimeRangeChange((visibleTimeRange) => {
      if (!visibleTimeRange || loadingRef.current) return;
      
      // Throttle range updates
      requestAnimationFrame(() => {
        const startTime = visibleTimeRange.from as number;
        const endTime = visibleTimeRange.to as number;
        
        // Find corresponding data indices
        let startIndex = 0;
        let endIndex = processedData.length - 1;
        
        for (let i = 0; i < processedData.length; i++) {
          if (processedData[i].timestamp >= startTime) {
            startIndex = i;
            break;
          }
        }
        
        for (let i = processedData.length - 1; i >= 0; i--) {
          if (processedData[i].timestamp <= endTime) {
            endIndex = i;
            break;
          }
        }
        
        // Update visible range if significantly different
        const currentRange = visibleRange;
        const rangeDiff = Math.abs(startIndex - currentRange[0]) + Math.abs(endIndex - currentRange[1]);
        
        if (rangeDiff > chunkSize * 0.1) { // 10% threshold
          setVisibleRange([startIndex, endIndex]);
        }
        
        // Notify parent of range change
        if (onRangeChange && processedData[startIndex] && processedData[endIndex]) {
          onRangeChange(
            new Date(processedData[startIndex].datetime),
            new Date(processedData[endIndex].datetime)
          );
        }
      });
    });

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [processedData.length > 0]); // Only recreate when data is available

  // Update chart data efficiently
  useEffect(() => {
    if (!chartRef.current || !candlestickSeriesRef.current || !volumeSeriesRef.current) return;
    
    if (chartData.length === 0) {
      candlestickSeriesRef.current.setData([]);
      volumeSeriesRef.current.setData([]);
      return;
    }

    // Use requestAnimationFrame for smooth updates
    requestAnimationFrame(() => {
      if (candlestickSeriesRef.current && volumeSeriesRef.current) {
        // Batch data updates
        const candleData = chartData.map(item => ({
          time: item.time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close
        }));

        const volumeData = chartData.map(item => ({
          time: item.time,
          value: item.volume,
          color: item.close >= item.open ? '#26a69a' : '#ef5350'
        }));

        candlestickSeriesRef.current.setData(candleData);
        volumeSeriesRef.current.setData(volumeData);

        // Update current price
        if (chartData.length > 0) {
          const latest = chartData[chartData.length - 1];
          setCurrentPrice(latest.close);
          
          if (chartData.length > 1) {
            const previous = chartData[0];
            setPriceChange(latest.close - previous.close);
          }
        }

        // Fit content only for initial load or significant changes
        if (visibleRange[0] === 0) {
          try {
            chartRef.current?.timeScale().fitContent();
          } catch (error) {
            console.warn('[VirtualizedChart] Error fitting content:', error);
          }
        }
      }
    });
  }, [chartData]);

  // Lazy load adjacent chunks when approaching boundaries
  useEffect(() => {
    if (!processedData.length) return;

    const currentChunk = Math.floor(visibleRange[1] / chunkSize);
    const shouldLoadNext = (visibleRange[1] % chunkSize) > (chunkSize * 0.8);
    const shouldLoadPrev = (visibleRange[0] % chunkSize) < (chunkSize * 0.2) && visibleRange[0] > 0;

    if (shouldLoadNext && !loadedChunks.has(currentChunk + 1)) {
      loadChunk(currentChunk + 1);
    }

    if (shouldLoadPrev && !loadedChunks.has(currentChunk - 1)) {
      loadChunk(currentChunk - 1);
    }
  }, [visibleRange, loadedChunks]);

  const loadChunk = useCallback(async (chunkIndex: number) => {
    if (loadingRef.current || loadedChunks.has(chunkIndex)) return;
    
    loadingRef.current = true;
    setIsLoading(true);
    
    try {
      // Simulate chunk loading (replace with actual API call if needed)
      await new Promise(resolve => setTimeout(resolve, 100));
      
      setLoadedChunks(prev => new Set([...prev, chunkIndex]));
    } catch (error) {
      console.error('[VirtualizedChart] Error loading chunk:', error);
    } finally {
      loadingRef.current = false;
      setIsLoading(false);
    }
  }, [loadedChunks]);

  // Zoom controls optimized for large datasets
  const handleZoom = useCallback((direction: 'in' | 'out') => {
    if (!chartRef.current) return;

    const timeScale = chartRef.current.timeScale();
    const visibleRange = timeScale.getVisibleRange();
    
    if (!visibleRange) return;
    
    const rangeDuration = (visibleRange.to as number) - (visibleRange.from as number);
    const center = (visibleRange.from as number) + rangeDuration / 2;
    
    const zoomFactor = direction === 'in' ? 0.5 : 2;
    const newDuration = rangeDuration * zoomFactor;
    
    timeScale.setVisibleRange({
      from: (center - newDuration / 2) as Time,
      to: (center + newDuration / 2) as Time
    });
  }, []);

  const handleReset = useCallback(() => {
    if (!chartRef.current) return;
    
    // Reset to show last portion of data
    const endIndex = processedData.length - 1;
    const startIndex = Math.max(0, endIndex - initialViewSize);
    
    setVisibleRange([startIndex, endIndex]);
    
    requestAnimationFrame(() => {
      chartRef.current?.timeScale().fitContent();
    });
  }, [processedData.length, initialViewSize]);

  const handleExport = useCallback(() => {
    if (!chartRef.current) return;
    
    chartRef.current.takeScreenshot().then(canvas => {
      const link = document.createElement('a');
      link.download = `usdcop-chart-${format(new Date(), 'yyyy-MM-dd-HHmm')}.png`;
      link.href = canvas.toDataURL();
      link.click();
    });
  }, []);

  return (
    <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6 relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/5" />
      
      {/* Controls */}
      <div className="absolute top-4 right-4 z-20 bg-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-xl p-3 shadow-2xl">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleZoom('in')}
            className="p-2 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-400"
          >
            <ZoomIn className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleZoom('out')}
            className="p-2 hover:bg-cyan-500/20 text-slate-300 hover:text-cyan-400"
          >
            <ZoomOut className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="p-2 hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-400"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleExport}
            className="p-2 hover:bg-purple-500/20 text-slate-300 hover:text-purple-400"
          >
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-6 relative z-10">
        <div className="flex items-center gap-4">
          <motion.h2
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
          >
            USD/COP
          </motion.h2>
          <div className="flex items-center gap-3">
            <span className="text-3xl font-mono font-bold text-white drop-shadow-lg">
              ${currentPrice.toFixed(2)}
            </span>
            <span className={`text-sm font-semibold px-3 py-1 rounded-full backdrop-blur-sm ${
              priceChange >= 0 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                : 'bg-red-500/20 text-red-400 border border-red-500/30'
            }`}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}
            </span>
          </div>
          {isRealtime && (
            <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-full border border-emerald-500/30 backdrop-blur-sm">
              <Activity className="w-4 h-4" />
              <span className="text-sm font-medium">Live</span>
            </div>
          )}
        </div>

        {/* Data info */}
        <div className="flex items-center gap-4 text-sm text-slate-400">
          <span>{processedData.length.toLocaleString()} total points</span>
          <span>{chartData.length.toLocaleString()} visible</span>
          {isLoading && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-slate-600 border-t-cyan-400 rounded-full animate-spin" />
              <span>Loading...</span>
            </div>
          )}
        </div>
      </div>

      {/* Chart container */}
      <div className="relative">
        <div 
          ref={chartContainerRef} 
          className="w-full h-[500px] rounded-xl overflow-hidden border border-slate-700/50 bg-gradient-to-br from-slate-900/50 to-slate-800/50 backdrop-blur-sm"
        />
        {processedData.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90 backdrop-blur-sm rounded-xl">
            <div className="flex flex-col items-center gap-4">
              <div className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
              <span className="text-slate-400 font-medium">Loading chart data...</span>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};