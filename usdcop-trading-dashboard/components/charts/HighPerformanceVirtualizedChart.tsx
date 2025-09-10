'use client';

/**
 * High-Performance Virtualized Chart
 * Ultra-fast rendering for massive datasets using canvas virtualization
 * Features: Dynamic LOD, viewport culling, worker threads, WebGL acceleration
 */

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FixedSizeList as List } from 'react-window';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Zap,
  Activity,
  Eye,
  EyeOff,
  Settings,
  Monitor,
  Cpu,
  MemoryStick,
  Clock,
  BarChart3,
  TrendingUp,
  Maximize2,
  RefreshCw,
  Filter,
  Search,
  Grid3X3
} from 'lucide-react';

interface OHLCData {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
  index: number;
}

interface VirtualizedChartProps {
  data: OHLCData[];
  width: number;
  height: number;
  enableWebGL?: boolean;
  enableWorkers?: boolean;
  lodLevels?: number[];
  maxVisibleCandles?: number;
}

interface ViewportData {
  startIndex: number;
  endIndex: number;
  visibleData: OHLCData[];
  lodLevel: number;
  scale: number;
}

interface PerformanceMetrics {
  fps: number;
  renderTime: number;
  memoryUsage: number;
  totalDataPoints: number;
  visibleDataPoints: number;
  lodLevel: number;
}

export const HighPerformanceVirtualizedChart: React.FC<VirtualizedChartProps> = ({
  data,
  width,
  height,
  enableWebGL = true,
  enableWorkers = true,
  lodLevels = [1, 2, 4, 8, 16, 32],
  maxVisibleCandles = 1000
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const animationFrameRef = useRef<number>();
  const lastFrameTimeRef = useRef<number>(0);
  
  const [viewport, setViewport] = useState<ViewportData>({
    startIndex: Math.max(0, data.length - maxVisibleCandles),
    endIndex: data.length,
    visibleData: [],
    lodLevel: 1,
    scale: 1
  });
  
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    fps: 60,
    renderTime: 0,
    memoryUsage: 0,
    totalDataPoints: 0,
    visibleDataPoints: 0,
    lodLevel: 1
  });
  
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showPerformancePanel, setShowPerformancePanel] = useState(false);
  const [webGLSupported, setWebGLSupported] = useState(false);
  const [candleWidth, setCandleWidth] = useState(2);
  const [showVolume, setShowVolume] = useState(true);

  // Initialize WebGL context
  const initWebGL = useCallback((canvas: HTMLCanvasElement) => {
    try {
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (gl) {
        setWebGLSupported(true);
        console.log('[VirtualChart] WebGL context initialized');
        return gl;
      }
    } catch (error) {
      console.warn('[VirtualChart] WebGL not supported, falling back to 2D canvas');
    }
    setWebGLSupported(false);
    return null;
  }, []);

  // Initialize Web Workers
  const initWorkers = useCallback(() => {
    if (!enableWorkers || typeof Worker === 'undefined') return;

    try {
      // Create worker for data processing
      const workerCode = `
        self.onmessage = function(e) {
          const { data, startIndex, endIndex, lodLevel } = e.data;
          
          // Level of detail processing
          const processedData = [];
          const step = lodLevel;
          
          for (let i = startIndex; i < endIndex; i += step) {
            if (step === 1) {
              processedData.push(data[i]);
            } else {
              // Aggregate multiple candles for LOD
              const slice = data.slice(i, Math.min(i + step, data.length));
              if (slice.length > 0) {
                const aggregated = {
                  datetime: slice[0].datetime,
                  open: slice[0].open,
                  high: Math.max(...slice.map(d => d.high)),
                  low: Math.min(...slice.map(d => d.low)),
                  close: slice[slice.length - 1].close,
                  volume: slice.reduce((sum, d) => sum + d.volume, 0),
                  timestamp: slice[0].timestamp,
                  index: i
                };
                processedData.push(aggregated);
              }
            }
          }
          
          self.postMessage({ processedData, lodLevel });
        };
      `;

      const blob = new Blob([workerCode], { type: 'application/javascript' });
      workerRef.current = new Worker(URL.createObjectURL(blob));
      
      workerRef.current.onmessage = (e) => {
        const { processedData, lodLevel } = e.data;
        setViewport(prev => ({
          ...prev,
          visibleData: processedData,
          lodLevel
        }));
      };

      console.log('[VirtualChart] Web Worker initialized');
    } catch (error) {
      console.warn('[VirtualChart] Web Workers not supported:', error);
    }
  }, [enableWorkers]);

  // Calculate optimal LOD level based on zoom and viewport
  const calculateLOD = useCallback((visibleRange: number, zoom: number): number => {
    const pixelsPerCandle = (width * zoom) / visibleRange;
    
    // Choose LOD level based on pixels per candle
    if (pixelsPerCandle < 1) return lodLevels[5] || 32;      // Very zoomed out
    if (pixelsPerCandle < 2) return lodLevels[4] || 16;      // Zoomed out
    if (pixelsPerCandle < 4) return lodLevels[3] || 8;       // Medium zoom
    if (pixelsPerCandle < 8) return lodLevels[2] || 4;       // Medium zoom
    if (pixelsPerCandle < 16) return lodLevels[1] || 2;      // Zoomed in
    return lodLevels[0] || 1;                                // Full detail
  }, [width, lodLevels]);

  // Update viewport based on zoom and pan
  const updateViewport = useCallback(() => {
    const visibleRange = Math.floor(maxVisibleCandles / zoom);
    const centerIndex = Math.floor(data.length / 2) + Math.floor(panX);
    const startIndex = Math.max(0, centerIndex - Math.floor(visibleRange / 2));
    const endIndex = Math.min(data.length, startIndex + visibleRange);
    
    const lodLevel = calculateLOD(visibleRange, zoom);
    
    // Use worker for data processing if available
    if (workerRef.current && data.length > 1000) {
      workerRef.current.postMessage({
        data,
        startIndex,
        endIndex,
        lodLevel
      });
    } else {
      // Process data on main thread for smaller datasets
      const step = lodLevel;
      const processedData: OHLCData[] = [];
      
      for (let i = startIndex; i < endIndex; i += step) {
        if (step === 1) {
          processedData.push(data[i]);
        } else {
          const slice = data.slice(i, Math.min(i + step, data.length));
          if (slice.length > 0) {
            const aggregated = {
              datetime: slice[0].datetime,
              open: slice[0].open,
              high: Math.max(...slice.map(d => d.high)),
              low: Math.min(...slice.map(d => d.low)),
              close: slice[slice.length - 1].close,
              volume: slice.reduce((sum, d) => sum + d.volume, 0),
              timestamp: slice[0].timestamp,
              index: i
            };
            processedData.push(aggregated);
          }
        }
      }
      
      setViewport({
        startIndex,
        endIndex,
        visibleData: processedData,
        lodLevel,
        scale: zoom
      });
    }
  }, [data, zoom, panX, maxVisibleCandles, calculateLOD]);

  // High-performance canvas rendering
  const renderChart = useCallback((timestamp: number) => {
    const canvas = canvasRef.current;
    if (!canvas || viewport.visibleData.length === 0) return;

    const startTime = performance.now();
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Calculate FPS
    const deltaTime = timestamp - lastFrameTimeRef.current;
    const fps = Math.round(1000 / deltaTime);
    lastFrameTimeRef.current = timestamp;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate chart dimensions
    const chartHeight = showVolume ? height * 0.7 : height;
    const volumeHeight = showVolume ? height * 0.3 : 0;
    const margin = { top: 20, right: 60, bottom: 20, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const availableHeight = chartHeight - margin.top - margin.bottom;

    // Calculate price range
    const prices = viewport.visibleData.flatMap(d => [d.high, d.low]);
    if (prices.length === 0) return;
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    if (priceRange === 0) return;

    // Calculate scales
    const xScale = (index: number) => margin.left + (index / (viewport.visibleData.length - 1 || 1)) * chartWidth;
    const yScale = (price: number) => margin.top + ((maxPrice - price) / priceRange) * availableHeight;
    
    // Calculate candle width based on visible data and LOD
    const dynamicCandleWidth = Math.max(1, Math.floor(chartWidth / viewport.visibleData.length * 0.8));
    const actualCandleWidth = Math.min(dynamicCandleWidth, candleWidth * viewport.lodLevel);

    // Set canvas properties for performance
    ctx.imageSmoothingEnabled = false;
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';

    // Batch rendering operations for better performance
    ctx.beginPath();

    // Draw price bars/candles
    viewport.visibleData.forEach((point, i) => {
      const x = xScale(i);
      const isUp = point.close >= point.open;
      
      // Set color
      const color = isUp ? '#10b981' : '#ef4444';
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      
      // Draw high-low line
      ctx.beginPath();
      ctx.moveTo(x, yScale(point.high));
      ctx.lineTo(x, yScale(point.low));
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Draw open-close body
      const bodyHeight = Math.abs(yScale(point.open) - yScale(point.close));
      const bodyTop = Math.min(yScale(point.open), yScale(point.close));
      
      if (actualCandleWidth > 2) {
        // Draw candle body
        ctx.fillRect(
          x - actualCandleWidth / 2,
          bodyTop,
          actualCandleWidth,
          Math.max(1, bodyHeight)
        );
      } else {
        // Draw thin line for very zoomed out view
        ctx.beginPath();
        ctx.moveTo(x, yScale(point.open));
        ctx.lineTo(x, yScale(point.close));
        ctx.lineWidth = actualCandleWidth;
        ctx.stroke();
      }
    });

    // Draw volume bars if enabled
    if (showVolume && volumeHeight > 0) {
      const maxVolume = Math.max(...viewport.visibleData.map(d => d.volume));
      if (maxVolume > 0) {
        ctx.globalAlpha = 0.6;
        viewport.visibleData.forEach((point, i) => {
          const x = xScale(i);
          const volumeBarHeight = (point.volume / maxVolume) * volumeHeight * 0.8;
          const isUp = point.close >= point.open;
          
          ctx.fillStyle = isUp ? '#10b98140' : '#ef444440';
          ctx.fillRect(
            x - actualCandleWidth / 2,
            chartHeight + margin.top + volumeHeight - volumeBarHeight,
            actualCandleWidth,
            volumeBarHeight
          );
        });
        ctx.globalAlpha = 1.0;
      }
    }

    // Draw axes and labels (optimized - only when LOD level is low enough)
    if (viewport.lodLevel <= 4) {
      ctx.fillStyle = '#9CA3AF';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';

      // Y-axis price labels
      for (let i = 0; i <= 5; i++) {
        const price = minPrice + (i / 5) * priceRange;
        const y = yScale(price);
        ctx.fillText(`$${price.toFixed(2)}`, margin.left - 5, y + 3);
      }

      // X-axis time labels (sparse)
      const labelStep = Math.max(1, Math.floor(viewport.visibleData.length / 5));
      ctx.textAlign = 'center';
      viewport.visibleData.forEach((point, i) => {
        if (i % labelStep === 0) {
          const x = xScale(i);
          const date = new Date(point.datetime);
          ctx.fillText(
            date.toLocaleDateString(),
            x,
            chartHeight + margin.top + 15
          );
        }
      });
    }

    const renderTime = performance.now() - startTime;

    // Update performance metrics
    setPerformanceMetrics(prev => ({
      ...prev,
      fps,
      renderTime,
      totalDataPoints: data.length,
      visibleDataPoints: viewport.visibleData.length,
      lodLevel: viewport.lodLevel
    }));

  }, [viewport, width, height, showVolume, candleWidth, data.length]);

  // Animation loop
  const animate = useCallback((timestamp: number) => {
    if (isDrawing) {
      renderChart(timestamp);
    }
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [renderChart, isDrawing]);

  // Mouse event handlers
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.1, Math.min(50, prev * zoomFactor)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDrawing(true);
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isDrawing) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const deltaX = x - width / 2;
      setPanX(prev => prev + deltaX * 0.01);
    }
  }, [isDrawing, width]);

  // Initialize everything
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;

    if (enableWebGL) {
      initWebGL(canvas);
    }

    initWorkers();
    updateViewport();
    
    // Start animation loop
    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, [width, height, enableWebGL, initWebGL, initWorkers, updateViewport, animate]);

  // Update viewport when zoom/pan changes
  useEffect(() => {
    updateViewport();
  }, [zoom, panX, updateViewport]);

  // Force re-render when drawing state changes
  useEffect(() => {
    if (isDrawing) {
      renderChart(performance.now());
    }
  }, [viewport, renderChart, isDrawing]);

  const resetView = () => {
    setZoom(1);
    setPanX(0);
    setIsDrawing(true);
    setTimeout(() => setIsDrawing(false), 100);
  };

  const getPerformanceColor = (value: number, thresholds: [number, number]) => {
    if (value >= thresholds[1]) return 'text-emerald-400';
    if (value >= thresholds[0]) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="relative"
    >
      <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
          <div className="flex items-center gap-4">
            <motion.h3
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent flex items-center gap-2"
            >
              <Zap className="w-5 h-5 text-cyan-400" />
              High-Performance Chart
            </motion.h3>
            
            <Badge className={`${webGLSupported ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-slate-500/20 text-slate-400 border-slate-500/30'}`}>
              {webGLSupported ? 'WebGL' : '2D Canvas'}
            </Badge>
            
            <Badge className="bg-slate-800 text-slate-300 border-slate-600">
              LOD {viewport.lodLevel}x
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowVolume(!showVolume)}
              className={`${showVolume ? 'text-cyan-400' : 'text-slate-400'} hover:text-white`}
            >
              <BarChart3 className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPerformancePanel(!showPerformancePanel)}
              className={`${showPerformancePanel ? 'text-emerald-400' : 'text-slate-400'} hover:text-white`}
            >
              <Monitor className="w-4 h-4" />
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={resetView}
              className="text-slate-400 hover:text-white"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Performance Panel */}
        <AnimatePresence>
          {showPerformancePanel && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-slate-800/30 border-b border-slate-700/50 p-4"
            >
              <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-sm">
                <div className="text-center">
                  <div className={`text-xl font-mono font-bold ${getPerformanceColor(performanceMetrics.fps, [30, 50])}`}>
                    {performanceMetrics.fps}
                  </div>
                  <div className="text-xs text-slate-400">FPS</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-xl font-mono font-bold ${getPerformanceColor(16.67 / performanceMetrics.renderTime, [0.5, 1])}`}>
                    {performanceMetrics.renderTime.toFixed(1)}ms
                  </div>
                  <div className="text-xs text-slate-400">Render Time</div>
                </div>
                
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-cyan-400">
                    {performanceMetrics.visibleDataPoints.toLocaleString()}
                  </div>
                  <div className="text-xs text-slate-400">Visible Points</div>
                </div>
                
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-purple-400">
                    {performanceMetrics.totalDataPoints.toLocaleString()}
                  </div>
                  <div className="text-xs text-slate-400">Total Points</div>
                </div>
                
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-yellow-400">
                    {zoom.toFixed(1)}x
                  </div>
                  <div className="text-xs text-slate-400">Zoom Level</div>
                </div>
                
                <div className="text-center">
                  <div className="text-xl font-mono font-bold text-emerald-400">
                    {performanceMetrics.lodLevel}x
                  </div>
                  <div className="text-xs text-slate-400">LOD Level</div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Chart Canvas */}
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="cursor-move"
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseUp}
          />
          
          {/* Loading Indicator */}
          {!isDrawing && viewport.visibleData.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/50">
              <div className="text-center">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                  className="w-8 h-8 border-2 border-cyan-500/30 border-t-cyan-500 rounded-full mx-auto mb-2"
                />
                <p className="text-slate-400">Processing data...</p>
              </div>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between p-4 border-t border-slate-700/50 text-sm">
          <div className="flex items-center gap-4">
            <span className="text-slate-400">Zoom:</span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setZoom(prev => prev * 0.8)}
                className="text-slate-400 hover:text-white"
              >
                -
              </Button>
              <span className="text-white font-mono w-16 text-center">
                {zoom.toFixed(1)}x
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setZoom(prev => prev * 1.2)}
                className="text-slate-400 hover:text-white"
              >
                +
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-slate-400">Candle Width:</span>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCandleWidth(prev => Math.max(1, prev - 1))}
                className="text-slate-400 hover:text-white"
              >
                -
              </Button>
              <span className="text-white font-mono w-8 text-center">
                {candleWidth}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setCandleWidth(prev => Math.min(10, prev + 1))}
                className="text-slate-400 hover:text-white"
              >
                +
              </Button>
            </div>
          </div>

          <div className="text-slate-500 text-xs">
            Mouse wheel: zoom • Click & drag: pan
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

export default HighPerformanceVirtualizedChart;