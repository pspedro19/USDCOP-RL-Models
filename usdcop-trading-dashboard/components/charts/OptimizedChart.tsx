/**
 * Optimized Trading Chart with Better Performance
 * Uses React.memo, useMemo, and data sampling for smooth 60 FPS
 */

import React, { useState, useMemo, useCallback, memo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush
} from 'recharts';
// Custom date formatting function with Spanish month names
const formatDate = (date: Date, formatStr: string, options?: any) => {
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
    case 'd MMM yyyy HH:mm':
      return `${day} ${months[month]} ${year} ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
    case 'HH:mm':
      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
    default:
      return date.toLocaleDateString();
  }
};

// Use the format function instead of date-fns format
const format = formatDate;
import { TrendingUp, TrendingDown, Zap, Activity, Gauge, Target } from 'lucide-react';

interface ChartDataPoint {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  [key: string]: any;
}

interface OptimizedChartProps {
  data: ChartDataPoint[];
  isRealtime?: boolean;
  replayProgress?: number;
  height?: number;
  showVolume?: boolean;
}

// Enhanced tooltip with glassmorphism
const CustomTooltip = memo(({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0].payload;
  const isUp = data.close >= data.open;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-slate-900/95 backdrop-blur-xl border border-slate-600/50 rounded-2xl p-4 shadow-2xl min-w-[220px] relative overflow-hidden"
    >
      {/* Glassmorphism background */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-purple-500/5 to-emerald-500/10 rounded-2xl" />
      <div className="absolute inset-0 bg-gradient-to-t from-slate-800/30 to-transparent rounded-2xl" />
      
      <div className="relative z-10">
        <div className="text-xs text-slate-400 mb-3 font-mono bg-slate-800/50 px-3 py-1.5 rounded-lg backdrop-blur">
          {format(new Date(label), "d MMM yyyy HH:mm", { locale: es })}
        </div>
        
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <div className="text-slate-400 font-medium">Open:</div>
          <div className="text-white font-mono font-bold">${data.open.toFixed(2)}</div>
          <div className="text-slate-400 font-medium">High:</div>
          <div className="text-emerald-400 font-mono font-bold">${data.high.toFixed(2)}</div>
          <div className="text-slate-400 font-medium">Low:</div>
          <div className="text-red-400 font-mono font-bold">${data.low.toFixed(2)}</div>
          <div className="text-slate-400 font-medium">Close:</div>
          <div className={`font-mono font-bold text-lg ${
            isUp 
              ? 'text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.3)]' 
              : 'text-red-400 drop-shadow-[0_0_8px_rgba(239,68,68,0.3)]'
          }`}>
            ${data.close.toFixed(2)}
          </div>
        </div>
        
        {data.volume && (
          <div className="mt-3 pt-3 border-t border-slate-700/50">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 text-sm font-medium">Volume:</span>
              <span className="text-cyan-400 font-mono font-bold bg-cyan-500/20 px-2 py-1 rounded-lg border border-cyan-500/30">
                {(data.volume / 1000000).toFixed(2)}M
              </span>
            </div>
          </div>
        )}
        
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-700/50">
          <div className="flex items-center gap-2">
            <motion.div
              animate={{ rotate: isUp ? 0 : 180 }}
              transition={{ duration: 0.3 }}
            >
              {isUp ? (
                <TrendingUp className="w-4 h-4 text-emerald-400" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-400" />
              )}
            </motion.div>
            <span className="text-slate-400 text-xs">Change:</span>
          </div>
          <div className={`font-mono text-sm font-bold px-2 py-1 rounded-lg ${
            isUp 
              ? 'text-emerald-400 bg-emerald-500/20 border border-emerald-500/30' 
              : 'text-red-400 bg-red-500/20 border border-red-500/30'
          }`}>
            {isUp ? '+' : ''}{((data.close - data.open) / data.open * 100).toFixed(2)}%
          </div>
        </div>
      </div>
    </motion.div>
  );
});

CustomTooltip.displayName = 'CustomTooltip';

// Custom dot component for candles (simplified for performance)
const CandleDot = memo(({ cx, cy, payload }: any) => {
  if (!payload) return null;
  const isUp = payload.close >= payload.open;
  return (
    <circle
      cx={cx}
      cy={cy}
      r={2}
      fill={isUp ? '#00D395' : '#FF3B69'}
      stroke="none"
    />
  );
});

CandleDot.displayName = 'CandleDot';

const OptimizedChart: React.FC<OptimizedChartProps> = memo(({
  data,
  isRealtime = false,
  replayProgress = 0,
  height = 500,
  showVolume = true
}) => {
  const [brushIndex, setBrushIndex] = useState<[number, number]>([
    Math.max(0, data.length - 100),
    data.length - 1
  ]);
  const [fps, setFps] = useState(60);
  const [renderTime, setRenderTime] = useState(0);
  const [optimizationLevel, setOptimizationLevel] = useState(100);
  const [showPerformancePanel, setShowPerformancePanel] = useState(true);
  
  // Performance monitoring
  useEffect(() => {
    let frameCount = 0;
    let startTime = performance.now();
    const updateFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      if (currentTime - startTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (currentTime - startTime)));
        frameCount = 0;
        startTime = currentTime;
      }
      requestAnimationFrame(updateFPS);
    };
    updateFPS();
  }, []);
  
  // Measure render performance
  useEffect(() => {
    const startRender = performance.now();
    const timeout = setTimeout(() => {
      setRenderTime(performance.now() - startRender);
    }, 0);
    return () => clearTimeout(timeout);
  }, [data, brushIndex]);

  // Sample data for better performance (show every nth point based on data size)
  const sampledData = useMemo(() => {
    if (data.length <= 200) return data;
    
    // For large datasets, sample more aggressively
    const targetPoints = 200; // Reduced from 500 for better performance
    const sampleRate = Math.ceil(data.length / targetPoints);
    const sampled = [];
    
    for (let i = 0; i < data.length; i += sampleRate) {
      // Get the highest high and lowest low in the sample period
      const slice = data.slice(i, Math.min(i + sampleRate, data.length));
      if (slice.length > 0) {
        const high = Math.max(...slice.map(d => d.high));
        const low = Math.min(...slice.map(d => d.low));
        const last = slice[slice.length - 1];
        
        sampled.push({
          ...last,
          high,
          low,
          volume: slice.reduce((sum, d) => sum + d.volume, 0)
        });
      }
    }
    
    return sampled;
  }, [data]);

  // Get visible data based on brush
  const visibleData = useMemo(() => {
    return sampledData.slice(brushIndex[0], brushIndex[1] + 1);
  }, [sampledData, brushIndex]);

  // Calculate domain for better scaling
  const yDomain = useMemo(() => {
    if (visibleData.length === 0) return [4000, 4200];
    
    const prices = visibleData.flatMap(d => [d.high, d.low]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const padding = (max - min) * 0.1;
    
    return [min - padding, max + padding];
  }, [visibleData]);

  // Format tick for X axis
  const formatXTick = useCallback((value: string) => {
    const date = new Date(value);
    return format(date, 'HH:mm');
  }, []);

  // Handle brush change
  const handleBrushChange = useCallback((e: any) => {
    if (e && e.startIndex !== undefined && e.endIndex !== undefined) {
      setBrushIndex([e.startIndex, e.endIndex]);
    }
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="relative w-full h-full bg-gradient-to-br from-slate-900 to-zinc-900 rounded-2xl overflow-hidden border border-slate-700/50"
    >
      {/* Background gradient animation */}
      <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/5 animate-pulse opacity-50" />
      {/* Performance Panel */}
      <AnimatePresence>
        {showPerformancePanel && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            transition={{ type: "spring", damping: 20 }}
            className="absolute top-4 right-4 z-30 bg-slate-900/95 backdrop-blur-xl border border-slate-600/50 rounded-2xl p-4 min-w-[200px]"
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Gauge className="w-4 h-4 text-cyan-400" />
                Performance
              </h3>
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setShowPerformancePanel(false)}
                className="text-slate-400 hover:text-red-400 transition-colors"
              >
                ×
              </motion.button>
            </div>
            
            {/* FPS Counter */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 flex items-center gap-1">
                  <Activity className="w-3 h-3" />
                  FPS
                </span>
                <motion.span
                  animate={{ 
                    color: fps >= 50 ? '#10b981' : fps >= 30 ? '#f59e0b' : '#ef4444',
                    textShadow: fps >= 50 ? '0 0 8px rgba(16, 185, 129, 0.3)' : 'none'
                  }}
                  className="text-sm font-mono font-bold"
                >
                  {fps}
                </motion.span>
              </div>
              
              {/* Render Time */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  Render
                </span>
                <span className="text-sm font-mono text-purple-400">
                  {renderTime.toFixed(1)}ms
                </span>
              </div>
              
              {/* Optimization Level */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 flex items-center gap-1">
                  <Target className="w-3 h-3" />
                  Optimize
                </span>
                <motion.span
                  animate={{ 
                    scale: optimizationLevel === 100 ? [1, 1.1, 1] : 1
                  }}
                  transition={{ duration: 2, repeat: optimizationLevel === 100 ? Infinity : 0 }}
                  className={`text-sm font-mono font-bold ${
                    optimizationLevel === 100 ? 'text-emerald-400' : 'text-yellow-400'
                  }`}
                >
                  {optimizationLevel}%
                </motion.span>
              </div>
              
              {/* Visual performance indicator */}
              <div className="mt-3 space-y-2">
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <motion.div
                      animate={{ width: `${fps * 100 / 60}%` }}
                      className={`h-full rounded-full ${
                        fps >= 50 ? 'bg-emerald-400' : fps >= 30 ? 'bg-yellow-400' : 'bg-red-400'
                      }`}
                    />
                  </div>
                  <span className="text-xs text-slate-500">60</span>
                </div>
                
                <div className="text-xs text-slate-500 text-center">
                  {data.length.toLocaleString()} data points • {visibleData.length} visible
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {!showPerformancePanel && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setShowPerformancePanel(true)}
          className="absolute top-4 right-4 z-20 p-2 bg-slate-800/90 backdrop-blur-xl border border-slate-700/50 rounded-lg text-slate-400 hover:text-cyan-400 transition-all duration-200"
        >
          <Gauge className="w-4 h-4" />
        </motion.button>
      )}

      {/* Status indicators */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-3">
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
              <div className="w-12 bg-yellow-400/20 rounded-full h-1 overflow-hidden">
                <motion.div
                  animate={{ width: `${replayProgress}%` }}
                  className="h-full bg-yellow-400 rounded-full"
                />
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Main Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={visibleData}
          margin={{ top: 20, right: 20, bottom: showVolume ? 100 : 60, left: 60 }}
        >
          <defs>
            <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#3B82F6" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke="#374151" 
            strokeOpacity={0.3}
          />
          
          <XAxis
            dataKey="datetime"
            tick={{ fontSize: 10, fill: '#9CA3AF' }}
            tickFormatter={formatXTick}
            stroke="#4B5563"
          />
          
          <YAxis
            domain={yDomain}
            tick={{ fontSize: 10, fill: '#9CA3AF' }}
            stroke="#4B5563"
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          
          <Tooltip content={<CustomTooltip />} />
          
          {/* Price line */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            animationDuration={300}
          />
          
          {/* Volume bars (if enabled) */}
          {showVolume && (
            <Bar
              dataKey="volume"
              fill="url(#volumeGradient)"
              yAxisId="volume"
              opacity={0.5}
            />
          )}
          
          {/* Hidden volume axis */}
          {showVolume && (
            <YAxis
              yAxisId="volume"
              orientation="right"
              tick={false}
              axisLine={false}
              domain={[0, 'dataMax']}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
      
      {/* Brush for navigation */}
      {data.length > 100 && (
        <div className="px-6">
          <ResponsiveContainer width="100%" height={60}>
            <ComposedChart
              data={sampledData}
              margin={{ top: 10, right: 20, bottom: 10, left: 60 }}
            >
              <Line
                type="monotone"
                dataKey="close"
                stroke="#3B82F6"
                strokeWidth={1}
                dot={false}
              />
              <Brush
                dataKey="datetime"
                height={30}
                stroke="#4B5563"
                fill="#1F2937"
                startIndex={brushIndex[0]}
                endIndex={brushIndex[1]}
                onChange={handleBrushChange}
                tickFormatter={formatXTick}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </motion.div>
  );
});

OptimizedChart.displayName = 'OptimizedChart';

export default OptimizedChart;