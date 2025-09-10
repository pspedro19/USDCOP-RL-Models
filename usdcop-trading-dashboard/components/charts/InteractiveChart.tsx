/**
 * Interactive Trading Chart with OHLC Tooltips
 * Ultra-dynamic chart with hover effects, animations, and real-time updates
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
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
  ReferenceArea,
  Brush,
  Area
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { useMarketStore } from '@/lib/store/market-store';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Calendar,
  Clock,
  DollarSign,
  BarChart3,
  Maximize2,
  ZoomIn,
  ZoomOut
} from 'lucide-react';
import { cn } from '@/lib/utils';

// Custom date formatting function
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
  const seconds = date.getSeconds();
  const pad = (num: number) => num.toString().padStart(2, '0');
  
  switch (pattern) {
    case 'dd MMM yyyy':
      return `${pad(day)} ${months[month]} ${year}`;
    case 'HH:mm:ss':
      return `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
    case 'dd/MM':
      return `${pad(day)}/${pad(month + 1)}`;
    case 'HH:mm':
      return `${pad(hours)}:${pad(minutes)}`;
    case 'yyyy-MM-dd':
      return `${year}-${pad(month + 1)}-${pad(day)}`;
    default:
      return date.toLocaleDateString('es-ES');
  }
};

interface ChartDataPoint {
  datetime: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
  changePercent: number;
  isNewDay?: boolean;
  isNewEpisode?: boolean;
  episodeNumber?: number;
}

interface InteractiveChartProps {
  data: any[];
  isPlaying?: boolean;
  replaySpeed?: number;
  showVolume?: boolean;
  showMarkers?: boolean;
  height?: number;
}

// Custom animated tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload[0]) return null;
  
  const data = payload[0].payload;
  const isUp = data.close >= data.open;
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8, y: -10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.2 }}
      className="bg-terminal-surface/95 backdrop-blur-md border border-terminal-accent/50 rounded-lg p-4 shadow-2xl"
      style={{
        boxShadow: `0 0 30px ${isUp ? 'var(--positive-dim)' : 'var(--negative-dim)'}`
      }}
    >
      {/* Header with time and trend */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-terminal-border">
        <div className="flex items-center space-x-2">
          <Calendar className="w-4 h-4 text-terminal-accent" />
          <span className="text-xs font-mono text-terminal-text-dim">
            {formatDate(new Date(data.datetime), 'dd MMM yyyy')}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <Clock className="w-4 h-4 text-terminal-accent" />
          <span className="text-xs font-mono text-terminal-text">
            {formatDate(new Date(data.datetime), 'HH:mm:ss')}
          </span>
        </div>
      </div>
      
      {/* OHLC Values with animations */}
      <div className="grid grid-cols-2 gap-3">
        <motion.div
          initial={{ x: -10, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.05 }}
        >
          <div className="text-xs text-terminal-text-dim mb-1">OPEN</div>
          <div className="text-sm font-mono font-bold text-terminal-text">
            ${data.open.toLocaleString('es-CO', { minimumFractionDigits: 2 })}
          </div>
        </motion.div>
        
        <motion.div
          initial={{ x: 10, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
        >
          <div className="text-xs text-terminal-text-dim mb-1">HIGH</div>
          <div className="text-sm font-mono font-bold text-positive">
            ${data.high.toLocaleString('es-CO', { minimumFractionDigits: 2 })}
          </div>
        </motion.div>
        
        <motion.div
          initial={{ x: -10, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.15 }}
        >
          <div className="text-xs text-terminal-text-dim mb-1">LOW</div>
          <div className="text-sm font-mono font-bold text-negative">
            ${data.low.toLocaleString('es-CO', { minimumFractionDigits: 2 })}
          </div>
        </motion.div>
        
        <motion.div
          initial={{ x: 10, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <div className="text-xs text-terminal-text-dim mb-1">CLOSE</div>
          <div className={cn(
            "text-sm font-mono font-bold",
            isUp ? "text-positive" : "text-negative"
          )}>
            ${data.close.toLocaleString('es-CO', { minimumFractionDigits: 2 })}
          </div>
        </motion.div>
      </div>
      
      {/* Change indicator with animation */}
      <motion.div
        initial={{ y: 10, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.25 }}
        className="mt-3 pt-3 border-t border-terminal-border"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {isUp ? (
              <TrendingUp className="w-4 h-4 text-positive" />
            ) : (
              <TrendingDown className="w-4 h-4 text-negative" />
            )}
            <span className={cn(
              "text-sm font-mono font-bold",
              isUp ? "text-positive" : "text-negative"
            )}>
              {isUp ? '+' : ''}{data.change?.toFixed(2) || 0}
            </span>
          </div>
          <div className={cn(
            "px-2 py-1 rounded text-xs font-mono",
            isUp ? "bg-up-10 text-positive" : "bg-down-10 text-negative"
          )}>
            {isUp ? '+' : ''}{data.changePercent?.toFixed(2) || 0}%
          </div>
        </div>
      </motion.div>
      
      {/* Volume */}
      {data.volume > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-2 text-xs text-terminal-text-muted"
        >
          Vol: {data.volume.toLocaleString()}
        </motion.div>
      )}
      
      {/* Episode/Day marker */}
      {data.isNewDay && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", delay: 0.35 }}
          className="mt-2 px-2 py-1 bg-terminal-accent/20 rounded text-xs text-terminal-accent text-center"
        >
          🌅 Nuevo Día
        </motion.div>
      )}
    </motion.div>
  );
};

// Custom dot for data points with animation
const CustomDot = (props: any) => {
  const { cx, cy, payload } = props;
  const isUp = payload.close >= payload.open;
  
  return (
    <motion.circle
      cx={cx}
      cy={cy}
      r={3}
      fill={isUp ? 'var(--positive)' : 'var(--negative)'}
      initial={{ scale: 0 }}
      animate={{ scale: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
      className="hover:r-5 transition-all cursor-pointer"
      style={{
        filter: `drop-shadow(0 0 4px ${isUp ? 'var(--positive-dim)' : 'var(--negative-dim)'})`
      }}
    />
  );
};

// Day separator line
const DayMarker = ({ x, index, data }: any) => {
  if (!data[index]?.isNewDay) return null;
  
  return (
    <motion.g
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.01 }}
    >
      <line
        x1={x}
        y1={0}
        x2={x}
        y2={400}
        stroke="var(--terminal-accent)"
        strokeWidth={1}
        strokeDasharray="5 5"
        opacity={0.3}
      />
      <text
        x={x}
        y={-5}
        fill="var(--terminal-accent)"
        fontSize={10}
        textAnchor="middle"
        className="font-mono"
      >
        {formatDate(new Date(data[index].datetime), 'dd/MM')}
      </text>
    </motion.g>
  );
};

const InteractiveChart: React.FC<InteractiveChartProps> = ({
  data,
  isPlaying = false,
  replaySpeed = 1,
  showVolume = true,
  showMarkers = true,
  height = 500
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<any>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 100 });
  const chartRef = useRef<any>(null);
  
  // Process data with change calculations and day markers
  const processedData = useMemo(() => {
    let lastDay = '';
    let episodeNumber = 1;
    
    return data.map((item, index) => {
      const prev = data[index - 1];
      const change = prev ? item.close - prev.close : 0;
      const changePercent = prev ? (change / prev.close) * 100 : 0;
      
      const currentDay = formatDate(new Date(item.datetime), 'yyyy-MM-dd');
      const isNewDay = currentDay !== lastDay;
      if (isNewDay) {
        lastDay = currentDay;
        episodeNumber++;
      }
      
      return {
        ...item,
        timestamp: new Date(item.datetime).getTime(),
        change,
        changePercent,
        isNewDay,
        isNewEpisode: index % 78 === 0, // Every trading day (~78 5-min bars)
        episodeNumber
      };
    });
  }, [data]);
  
  // Auto-scroll during replay
  useEffect(() => {
    if (isPlaying && processedData.length > 0) {
      const interval = setInterval(() => {
        setVisibleRange(prev => {
          const newEnd = Math.min(prev.end + 1, processedData.length);
          const newStart = Math.max(0, newEnd - 100);
          return { start: newStart, end: newEnd };
        });
      }, 5000 / replaySpeed); // 5 seconds per candle adjusted by speed
      
      return () => clearInterval(interval);
    }
  }, [isPlaying, replaySpeed, processedData.length]);
  
  // Zoom handlers
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.2, 5));
  };
  
  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.2, 0.5));
  };
  
  // Get visible data slice
  const visibleData = processedData.slice(visibleRange.start, visibleRange.end);
  
  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (visibleData.length === 0) return [0, 100];
    const lows = visibleData.map(d => d.low);
    const highs = visibleData.map(d => d.high);
    const min = Math.min(...lows) * 0.999;
    const max = Math.max(...highs) * 1.001;
    return [min, max];
  }, [visibleData]);
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative w-full"
      ref={chartRef}
    >
      {/* Zoom Controls */}
      <div className="absolute top-2 right-2 z-10 flex space-x-2">
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleZoomIn}
          className="p-2 bg-terminal-surface/80 backdrop-blur rounded hover:bg-terminal-accent/20 transition-colors"
        >
          <ZoomIn className="w-4 h-4 text-terminal-text" />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleZoomOut}
          className="p-2 bg-terminal-surface/80 backdrop-blur rounded hover:bg-terminal-accent/20 transition-colors"
        >
          <ZoomOut className="w-4 h-4 text-terminal-text" />
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
          className="p-2 bg-terminal-surface/80 backdrop-blur rounded hover:bg-terminal-accent/20 transition-colors"
        >
          <Maximize2 className="w-4 h-4 text-terminal-text" />
        </motion.button>
      </div>
      
      {/* Current Price Display */}
      {visibleData.length > 0 && (
        <motion.div
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="absolute top-2 left-2 z-10 bg-terminal-surface/90 backdrop-blur rounded-lg p-3"
        >
          <div className="text-xs text-terminal-text-dim mb-1">Current Price</div>
          <div className={cn(
            "text-2xl font-mono font-bold",
            visibleData[visibleData.length - 1].close >= visibleData[visibleData.length - 1].open
              ? "text-positive" : "text-negative"
          )}>
            ${visibleData[visibleData.length - 1].close.toLocaleString('es-CO')}
          </div>
        </motion.div>
      )}
      
      {/* Main Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={visibleData}
          margin={{ top: 20, right: 20, bottom: showVolume ? 100 : 60, left: 60 }}
        >
          {/* Background gradient */}
          <defs>
            <linearGradient id="colorUp" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--positive)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--positive)" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorDown" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--negative)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--negative)" stopOpacity={0}/>
            </linearGradient>
          </defs>
          
          {/* Grid with subtle animation */}
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--terminal-border)"
            opacity={0.3}
            className="animate-pulse"
          />
          
          {/* X-Axis with custom formatting */}
          <XAxis
            dataKey="datetime"
            stroke="var(--terminal-text-muted)"
            tick={{ fontSize: 10, fill: 'var(--terminal-text-muted)' }}
            tickFormatter={(value) => formatDate(new Date(value), 'HH:mm')}
          />
          
          {/* Y-Axis with price formatting */}
          <YAxis
            domain={yDomain}
            stroke="var(--terminal-text-muted)"
            tick={{ fontSize: 10, fill: 'var(--terminal-text-muted)' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          
          {/* Animated tooltip */}
          <Tooltip content={<CustomTooltip />} />
          
          {/* Day/Episode markers */}
          {showMarkers && visibleData.map((item, index) => (
            item.isNewDay && (
              <ReferenceLine
                key={`day-${index}`}
                x={item.datetime}
                stroke="var(--terminal-accent)"
                strokeDasharray="5 5"
                opacity={0.5}
                label={{
                  value: `Día ${item.episodeNumber}`,
                  position: "top",
                  fill: "var(--terminal-accent)",
                  fontSize: 10
                }}
              />
            )
          ))}
          
          {/* Price area fill */}
          <Area
            type="monotone"
            dataKey="close"
            stroke="none"
            fill="url(#colorUp)"
            fillOpacity={0.3}
          />
          
          {/* Main price line with gradient */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="var(--terminal-accent)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6, fill: 'var(--terminal-accent)' }}
            animationDuration={500}
            animationEasing="ease-in-out"
          />
          
          {/* High/Low lines */}
          <Line
            type="monotone"
            dataKey="high"
            stroke="var(--positive)"
            strokeWidth={1}
            strokeOpacity={0.5}
            dot={false}
            strokeDasharray="2 2"
          />
          <Line
            type="monotone"
            dataKey="low"
            stroke="var(--negative)"
            strokeWidth={1}
            strokeOpacity={0.5}
            dot={false}
            strokeDasharray="2 2"
          />
          
          {/* Volume bars */}
          {showVolume && (
            <Bar
              dataKey="volume"
              fill="var(--terminal-text-muted)"
              opacity={0.3}
              yAxisId="volume"
            />
          )}
          
          {/* Brush for navigation */}
          <Brush
            dataKey="datetime"
            height={30}
            stroke="var(--terminal-accent)"
            fill="var(--terminal-surface-variant)"
            tickFormatter={(value) => formatDate(new Date(value), 'dd/MM')}
          />
        </ComposedChart>
      </ResponsiveContainer>
      
      {/* Replay progress indicator */}
      {isPlaying && (
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: visibleRange.end / processedData.length }}
          className="absolute bottom-0 left-0 right-0 h-1 bg-terminal-accent origin-left"
          style={{
            boxShadow: '0 0 10px var(--terminal-accent)'
          }}
        />
      )}
    </motion.div>
  );
};

export default InteractiveChart;