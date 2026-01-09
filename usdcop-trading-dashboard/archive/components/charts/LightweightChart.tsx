/**
 * Lightweight Canvas-based Trading Chart
 * Ultra-fast rendering for large datasets using HTML5 Canvas
 */

import React, { useRef, useEffect, useState, useCallback, memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ZoomIn, ZoomOut, Maximize2, Eye, EyeOff } from 'lucide-react';

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
    case 'HH:mm':
      return `${pad(hours)}:${pad(minutes)}`;
    default:
      return date.toLocaleDateString('es-ES');
  }
};

interface ChartDataPoint {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface LightweightChartProps {
  data: ChartDataPoint[];
  isRealtime?: boolean;
  replayProgress?: number;
  height?: number;
  showVolume?: boolean;
}

const LightweightChart: React.FC<LightweightChartProps> = memo(({
  data,
  isRealtime = false,
  replayProgress = 0,
  height = 500,
  showVolume = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height });
  const [hoveredCandle, setHoveredCandle] = useState<ChartDataPoint | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [showLegend, setShowLegend] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [height]);

  // Draw chart using Canvas API
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);

    // Calculate margins
    const margin = { top: 20, right: 60, bottom: 60, left: 60 };
    const chartWidth = dimensions.width - margin.left - margin.right;
    const chartHeight = dimensions.height - margin.top - margin.bottom;

    // Show ALL historical data from 2020-2025 (no limit)
    // For performance, use intelligent sampling when data is very large
    const sampledData = data; // Show ALL data points, no slicing

    // Calculate price range
    const prices = sampledData.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices) * 0.995;
    const maxPrice = Math.max(...prices) * 1.005;
    const priceRange = maxPrice - minPrice;

    // Calculate volume range if showing volume
    const maxVolume = showVolume ? Math.max(...sampledData.map(d => d.volume)) : 0;

    // Helper functions
    const xScale = (index: number) => margin.left + (index / (sampledData.length - 1)) * chartWidth;
    const yScale = (price: number) => margin.top + ((maxPrice - price) / priceRange) * (showVolume ? chartHeight * 0.7 : chartHeight);
    const volumeScale = (volume: number) => showVolume ? (volume / maxVolume) * chartHeight * 0.25 : 0;

    // Set styles
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;

    // Draw grid lines
    ctx.beginPath();
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (i / 5) * (showVolume ? chartHeight * 0.7 : chartHeight);
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + chartWidth, y);
    }
    for (let i = 0; i <= 10; i++) {
      const x = margin.left + (i / 10) * chartWidth;
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + chartHeight);
    }
    ctx.stroke();

    // Draw price line
    ctx.beginPath();
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    sampledData.forEach((point, i) => {
      const x = xScale(i);
      const y = yScale(point.close);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw candles (simplified)
    sampledData.forEach((point, i) => {
      const x = xScale(i);
      const isUp = point.close >= point.open;
      
      // Candle color
      ctx.strokeStyle = isUp ? '#00D395' : '#FF3B69';
      ctx.fillStyle = isUp ? '#00D39520' : '#FF3B6920';
      ctx.lineWidth = 1;

      // Draw high-low line
      ctx.beginPath();
      ctx.moveTo(x, yScale(point.high));
      ctx.lineTo(x, yScale(point.low));
      ctx.stroke();

      // Draw open-close box (simplified as line for performance)
      const boxHeight = Math.abs(yScale(point.open) - yScale(point.close));
      if (boxHeight > 1) {
        const candleWidth = Math.max(1, chartWidth / sampledData.length * 0.6);
        ctx.fillRect(
          x - candleWidth / 2,
          Math.min(yScale(point.open), yScale(point.close)),
          candleWidth,
          boxHeight
        );
      }
    });

    // Draw volume bars if enabled
    if (showVolume) {
      ctx.fillStyle = '#3B82F640';
      sampledData.forEach((point, i) => {
        const x = xScale(i);
        const barHeight = volumeScale(point.volume);
        const barWidth = Math.max(1, chartWidth / sampledData.length * 0.8);
        ctx.fillRect(
          x - barWidth / 2,
          dimensions.height - margin.bottom - barHeight,
          barWidth,
          barHeight
        );
      });
    }

    // Draw axes labels
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';

    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (i / 5) * priceRange;
      const y = margin.top + ((5 - i) / 5) * (showVolume ? chartHeight * 0.7 : chartHeight);
      ctx.fillText(`$${price.toFixed(0)}`, margin.left - 5, y + 3);
    }

    // X-axis labels (show every 5th point)
    ctx.textAlign = 'center';
    const labelStep = Math.ceil(sampledData.length / 5);
    sampledData.forEach((point, i) => {
      if (i % labelStep === 0) {
        const x = xScale(i);
        const date = new Date(point.datetime);
        ctx.fillText(formatDate(date, 'HH:mm'), x, dimensions.height - margin.bottom + 15);
      }
    });

  }, [data, dimensions, showVolume]);

  // Redraw on data or dimension changes
  useEffect(() => {
    if (data.length > 0) {
      setIsLoading(false);
    }
    drawChart();
  }, [drawChart, data]);

  // Handle mouse move for tooltip
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setMousePos({ x, y });

    // Find closest data point
    const margin = { left: 60, right: 60 };
    const chartWidth = dimensions.width - margin.left - margin.right;
    const index = Math.round(((x - margin.left) / chartWidth) * (data.length - 1));
    
    if (index >= 0 && index < data.length) {
      setHoveredCandle(data[index]);
    }
  }, [data, dimensions]);

  const handleMouseLeave = useCallback(() => {
    setHoveredCandle(null);
  }, []);

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
      className="relative w-full h-full bg-slate-900 rounded-2xl overflow-hidden border-2 border-transparent bg-gradient-to-r from-slate-800 to-slate-900 p-[2px]"
    >
      {/* Glowing border container */}
      <div className="relative w-full h-full bg-slate-900 rounded-2xl overflow-hidden">
        {/* Animated glowing border */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-cyan-500/30 via-purple-500/30 to-emerald-500/30 opacity-50 blur-sm animate-pulse" />
        <div className="absolute inset-[1px] rounded-2xl bg-slate-900" />
        
        {/* Enhanced Status indicators */}
        <div className="absolute top-4 right-4 z-20 flex items-center gap-3">
        {isRealtime && (
          <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-full border border-emerald-500/30 backdrop-blur-xl shadow-lg"
          >
            <motion.div
              animate={{ 
                scale: [1, 1.3, 1],
                boxShadow: ['0 0 0 0 rgba(16, 185, 129, 0.4)', '0 0 0 10px rgba(16, 185, 129, 0)', '0 0 0 0 rgba(16, 185, 129, 0)']
              }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="w-2 h-2 bg-emerald-400 rounded-full"
            />
            <span className="text-xs font-semibold">Live</span>
          </motion.div>
        )}
        {replayProgress > 0 && replayProgress < 100 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-full border border-yellow-500/30 backdrop-blur-xl"
          >
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-bounce" />
              <span className="text-xs font-semibold">Replay {replayProgress.toFixed(0)}%</span>
            </div>
          </motion.div>
        )}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="px-4 py-2 bg-slate-800/90 rounded-full border border-slate-700/50 backdrop-blur-xl"
        >
          <span className="text-xs text-slate-400 font-mono">{data.length.toLocaleString()} points</span>
        </motion.div>
        
        {/* Zoom Controls */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="flex items-center gap-2 bg-slate-800/90 backdrop-blur-xl rounded-full p-1 border border-slate-700/50"
        >
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setZoomLevel(prev => Math.min(prev * 1.2, 5))}
            className="p-2 rounded-full text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/20 transition-all duration-200"
          >
            <ZoomIn className="w-3 h-3" />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setZoomLevel(prev => Math.max(prev * 0.8, 0.5))}
            className="p-2 rounded-full text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/20 transition-all duration-200"
          >
            <ZoomOut className="w-3 h-3" />
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setShowLegend(!showLegend)}
            className={`p-2 rounded-full transition-all duration-200 ${
              showLegend 
                ? 'text-purple-400 bg-purple-500/20' 
                : 'text-slate-400 hover:text-purple-400 hover:bg-purple-500/20'
            }`}
          >
            {showLegend ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
          </motion.button>
        </motion.div>
      </div>

      {/* Loading Skeleton */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-4 z-10 bg-slate-900/95 backdrop-blur-sm rounded-xl flex items-center justify-center"
          >
            <div className="flex flex-col items-center gap-6">
              <div className="relative">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                  className="w-12 h-12 border-3 border-cyan-500/30 border-t-cyan-500 rounded-full"
                />
                <motion.div
                  animate={{ scale: [0.8, 1.2, 0.8] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                  className="absolute inset-2 bg-cyan-500/20 rounded-full"
                />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-semibold text-white mb-2">Loading Chart</h3>
                <div className="flex items-center gap-2 text-slate-400">
                  <motion.div
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 1.5, delay: 0 }}
                    className="w-2 h-2 bg-cyan-500 rounded-full"
                  />
                  <motion.div
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 1.5, delay: 0.2 }}
                    className="w-2 h-2 bg-purple-500 rounded-full"
                  />
                  <motion.div
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 1.5, delay: 0.4 }}
                    className="w-2 h-2 bg-emerald-500 rounded-full"
                  />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="cursor-crosshair relative z-10"
        style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'center' }}
      />

      {/* Floating Legend */}
      <AnimatePresence>
        {showLegend && hoveredCandle && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            className="absolute z-30 pointer-events-none"
            style={{
              left: Math.min(mousePos.x + 15, dimensions.width - 250),
              top: Math.min(mousePos.y - 60, dimensions.height - 180)
            }}
          >
            <div className="bg-slate-900/95 backdrop-blur-xl border border-slate-600/50 rounded-2xl p-4 shadow-2xl min-w-[240px] relative overflow-hidden">
              {/* Enhanced glassmorphism background */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-purple-500/5 to-emerald-500/10 rounded-2xl" />
              <div className="absolute inset-0 bg-gradient-to-t from-slate-800/30 to-transparent rounded-2xl" />
              <div className="relative z-10">
                <div className="text-xs text-slate-400 mb-3 font-mono bg-slate-800/50 px-3 py-1.5 rounded-lg backdrop-blur">
                  {formatDate(new Date(hoveredCandle.datetime), "dd MMM yyyy HH:mm")}
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
                {hoveredCandle.volume && (
                  <div className="mt-3 pt-3 border-t border-slate-700/50">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 text-sm font-medium">Volume:</span>
                      <span className="text-cyan-400 font-mono font-bold bg-cyan-500/20 px-2 py-1 rounded-lg border border-cyan-500/30">
                        {(hoveredCandle.volume / 1000000).toFixed(2)}M
                      </span>
                    </div>
                  </div>
                )}
                <div className="mt-3 pt-3 border-t border-slate-700/50">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 text-xs">Change:</span>
                    <div className={`font-mono text-sm font-bold px-2 py-1 rounded-lg ${
                      hoveredCandle.close >= hoveredCandle.open 
                        ? 'text-emerald-400 bg-emerald-500/20 border border-emerald-500/30' 
                        : 'text-red-400 bg-red-500/20 border border-red-500/30'
                    }`}>
                      {hoveredCandle.close >= hoveredCandle.open ? '+' : ''}
                      {((hoveredCandle.close - hoveredCandle.open) / hoveredCandle.open * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      </div>
    </motion.div>
  );
});

LightweightChart.displayName = 'LightweightChart';

export default LightweightChart;