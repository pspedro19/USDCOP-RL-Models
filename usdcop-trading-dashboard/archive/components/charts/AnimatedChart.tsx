'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, useAnimation } from 'framer-motion';
import { motionLibrary } from '@/lib/motion';
import { Card } from '@/components/ui/card';

/**
 * Professional AnimatedChart component for financial data visualization
 * Provides smooth animations for chart data updates and interactions
 */

interface ChartDataPoint {
  x: number | string;
  y: number;
  timestamp?: number;
  volume?: number;
  change?: number;
}

interface AnimatedChartProps {
  data: ChartDataPoint[];
  title?: string;
  subtitle?: string;
  type?: 'line' | 'candlestick' | 'area' | 'bar';
  height?: number;
  loading?: boolean;
  realTime?: boolean;
  showAnimation?: boolean;
  gridLines?: boolean;
  showTooltip?: boolean;
  colorScheme?: 'default' | 'trading' | 'analytics';
  className?: string;
  onDataPointHover?: (point: ChartDataPoint | null) => void;
  onChartClick?: (point: ChartDataPoint) => void;
}

export function AnimatedChart({
  data,
  title = 'Chart',
  subtitle,
  type = 'line',
  height = 300,
  loading = false,
  realTime = false,
  showAnimation = true,
  gridLines = true,
  showTooltip = true,
  colorScheme = 'trading',
  className = '',
  onDataPointHover,
  onChartClick,
}: AnimatedChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<ChartDataPoint | null>(null);
  const [animationComplete, setAnimationComplete] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const controls = useAnimation();

  const colorSchemes = {
    default: {
      line: '#06B6D4',
      area: 'rgba(6, 182, 212, 0.2)',
      positive: '#10B981',
      negative: '#EF4444',
      grid: 'rgba(100, 116, 139, 0.2)',
    },
    trading: {
      line: '#00D395',
      area: 'rgba(0, 211, 149, 0.15)',
      positive: '#00D395',
      negative: '#FF3B69',
      grid: 'rgba(6, 182, 212, 0.15)',
    },
    analytics: {
      line: '#8B5CF6',
      area: 'rgba(139, 92, 246, 0.2)',
      positive: '#8B5CF6',
      negative: '#F59E0B',
      grid: 'rgba(139, 92, 246, 0.1)',
    },
  };

  const colors = colorSchemes[colorScheme];

  // Calculate chart dimensions and scales
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const chartWidth = 800 - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  const xScale = (index: number) => (index / (data.length - 1)) * chartWidth;
  const yMin = Math.min(...data.map(d => d.y));
  const yMax = Math.max(...data.map(d => d.y));
  const yRange = yMax - yMin || 1;
  const yScale = (value: number) => chartHeight - ((value - yMin) / yRange) * chartHeight;

  // Generate path for line chart
  const generatePath = (animate: boolean = false) => {
    if (data.length === 0) return '';
    
    const pathData = data.map((point, index) => {
      const x = xScale(index);
      const y = animate ? chartHeight : yScale(point.y);
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
    
    return pathData;
  };

  // Generate area path
  const generateAreaPath = (animate: boolean = false) => {
    if (data.length === 0) return '';
    
    const linePath = generatePath(animate);
    const baseY = chartHeight;
    const firstX = xScale(0);
    const lastX = xScale(data.length - 1);
    
    return `${linePath} L ${lastX} ${baseY} L ${firstX} ${baseY} Z`;
  };

  // Animation variants
  const pathVariants = {
    hidden: { 
      pathLength: 0, 
      opacity: 0 
    },
    visible: {
      pathLength: 1,
      opacity: 1,
      transition: {
        pathLength: {
          duration: motionLibrary.presets.duration.chart,
          ease: motionLibrary.presets.easing.tradingView,
        },
        opacity: {
          duration: 0.3,
        }
      }
    }
  };

  const pointVariants = {
    hidden: { 
      scale: 0, 
      opacity: 0 
    },
    visible: (i: number) => ({
      scale: 1,
      opacity: 1,
      transition: {
        delay: i * 0.05 + 0.5,
        duration: 0.3,
        ease: motionLibrary.presets.easing.bounce,
      }
    }),
    hover: {
      scale: 1.5,
      transition: { duration: 0.2 }
    }
  };

  // Handle real-time data updates
  useEffect(() => {
    if (realTime && showAnimation) {
      controls.start('visible');
    }
  }, [data, realTime, controls, showAnimation]);

  // Loading skeleton
  if (loading) {
    return (
      <Card variant="professional" className={`p-6 ${className}`}>
        <motion.div
          variants={motionLibrary.lists.container}
          initial="initial"
          animate="animate"
        >
          <motion.div variants={motionLibrary.lists.item} className="mb-4">
            <div className="h-6 bg-slate-700/50 rounded w-48 loading-shimmer-glass" />
          </motion.div>
          <motion.div variants={motionLibrary.lists.item}>
            <div className={`bg-slate-800/30 rounded-xl relative overflow-hidden`} style={{ height }}>
              <div className="absolute inset-4 space-y-4">
                {/* Grid lines skeleton */}
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className="h-px bg-slate-700/30 loading-shimmer-glass"
                    style={{ marginTop: `${(i / 4) * 80}%` }}
                  />
                ))}
              </div>
              {/* Chart line skeleton */}
              <motion.div
                className="absolute bottom-4 left-4 right-4 h-0.5 bg-cyan-400/30 rounded-full"
                animate={{
                  opacity: [0.3, 0.8, 0.3],
                  scale: [1, 1.02, 1],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              />
            </div>
          </motion.div>
        </motion.div>
      </Card>
    );
  }

  return (
    <Card variant="professional" className={`p-6 ${className}`}>
      {/* Chart Header */}
      <motion.div
        className="mb-6"
        variants={motionLibrary.utils.createSlideAnimation('up', 10)}
        initial="initial"
        animate="animate"
      >
        <h3 className="text-xl font-bold text-cyan-300 text-glow mb-1">
          {title}
        </h3>
        {subtitle && (
          <p className="text-sm text-slate-400 font-mono">
            {subtitle}
          </p>
        )}
      </motion.div>

      {/* Chart Container */}
      <motion.div
        className="relative"
        variants={motionLibrary.charts.container}
        initial="initial"
        animate="animate"
        style={{ height }}
      >
        <svg
          ref={svgRef}
          width="100%"
          height={height}
          viewBox={`0 0 800 ${height}`}
          className="overflow-visible"
        >
          <defs>
            {/* Gradient definitions */}
            <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={colors.line} stopOpacity={0.4} />
              <stop offset="100%" stopColor={colors.line} stopOpacity={0.05} />
            </linearGradient>
            
            <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={colors.line} />
              <stop offset="50%" stopColor={colors.positive} />
              <stop offset="100%" stopColor={colors.line} />
            </linearGradient>

            {/* Glow filter */}
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* Grid Lines */}
          {gridLines && (
            <g className="grid-lines" opacity={0.3}>
              {Array.from({ length: 6 }).map((_, i) => (
                <motion.line
                  key={`grid-${i}`}
                  x1={margin.left}
                  y1={margin.top + (i * chartHeight) / 5}
                  x2={margin.left + chartWidth}
                  y2={margin.top + (i * chartHeight) / 5}
                  stroke={colors.grid}
                  strokeWidth={1}
                  strokeDasharray="2,2"
                  initial={{ opacity: 0, pathLength: 0 }}
                  animate={{ opacity: 0.3, pathLength: 1 }}
                  transition={{
                    delay: i * 0.1,
                    duration: 0.5,
                  }}
                />
              ))}
            </g>
          )}

          {/* Chart Area (for area chart) */}
          {type === 'area' && (
            <motion.path
              d={generateAreaPath()}
              fill="url(#areaGradient)"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                duration: motionLibrary.presets.duration.chart,
                ease: motionLibrary.presets.easing.smooth,
              }}
              transform={`translate(${margin.left}, ${margin.top})`}
            />
          )}

          {/* Chart Line */}
          <motion.path
            d={generatePath()}
            fill="none"
            stroke="url(#lineGradient)"
            strokeWidth={2}
            filter="url(#glow)"
            variants={showAnimation ? pathVariants : {}}
            initial={showAnimation ? "hidden" : undefined}
            animate={showAnimation ? "visible" : undefined}
            transform={`translate(${margin.left}, ${margin.top})`}
          />

          {/* Data Points */}
          <AnimatePresence>
            {data.map((point, index) => (
              <motion.circle
                key={`point-${index}`}
                cx={margin.left + xScale(index)}
                cy={margin.top + yScale(point.y)}
                r={3}
                fill={colors.line}
                variants={showAnimation ? pointVariants : {}}
                custom={index}
                initial={showAnimation ? "hidden" : undefined}
                animate={showAnimation ? "visible" : undefined}
                whileHover="hover"
                className="cursor-pointer"
                filter="url(#glow)"
                onHoverStart={() => {
                  setHoveredPoint(point);
                  onDataPointHover?.(point);
                }}
                onHoverEnd={() => {
                  setHoveredPoint(null);
                  onDataPointHover?.(null);
                }}
                onClick={() => onChartClick?.(point)}
              />
            ))}
          </AnimatePresence>

          {/* Y-Axis Labels */}
          {Array.from({ length: 6 }).map((_, i) => {
            const value = yMax - (i * yRange) / 5;
            return (
              <motion.text
                key={`y-label-${i}`}
                x={margin.left - 10}
                y={margin.top + (i * chartHeight) / 5 + 4}
                textAnchor="end"
                className="text-xs fill-slate-400 font-mono"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{
                  delay: 0.5 + i * 0.1,
                  duration: 0.3,
                }}
              >
                {value.toFixed(2)}
              </motion.text>
            );
          })}
        </svg>

        {/* Tooltip */}
        <AnimatePresence>
          {showTooltip && hoveredPoint && (
            <motion.div
              className="absolute glass-surface-elevated backdrop-blur-lg rounded-lg p-3 shadow-glass-lg pointer-events-none z-10"
              variants={motionLibrary.micro.tooltip}
              initial="initial"
              animate="animate"
              exit="exit"
              style={{
                left: `${((data.findIndex(d => d === hoveredPoint) / (data.length - 1)) * 100)}%`,
                top: '10px',
                transform: 'translateX(-50%)',
              }}
            >
              <div className="text-sm font-mono space-y-1">
                <div className="text-cyan-300 font-semibold">
                  Value: {hoveredPoint.y.toFixed(4)}
                </div>
                {hoveredPoint.change && (
                  <div className={`text-xs ${hoveredPoint.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    Change: {hoveredPoint.change >= 0 ? '+' : ''}{hoveredPoint.change.toFixed(2)}%
                  </div>
                )}
                {hoveredPoint.volume && (
                  <div className="text-xs text-slate-400">
                    Volume: {hoveredPoint.volume.toLocaleString()}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Real-time indicator */}
        {realTime && (
          <motion.div
            className="absolute top-4 right-4 flex items-center gap-2 glass-surface-secondary px-3 py-1 rounded-lg backdrop-blur-sm"
            variants={motionLibrary.micro.statusPulse}
            animate="animate"
          >
            <motion.div
              className="w-2 h-2 bg-emerald-400 rounded-full"
              animate={{
                scale: [1, 1.3, 1],
                opacity: [1, 0.7, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
            <span className="text-xs font-mono text-emerald-400">LIVE</span>
          </motion.div>
        )}
      </motion.div>
    </Card>
  );
}

/**
 * AnimatedPriceChart - Specialized component for price data with candlesticks
 */
interface PriceData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface AnimatedPriceChartProps {
  data: PriceData[];
  title?: string;
  timeframe?: string;
  loading?: boolean;
  className?: string;
}

export function AnimatedPriceChart({
  data,
  title = 'Price Chart',
  timeframe = '1H',
  loading = false,
  className = '',
}: AnimatedPriceChartProps) {
  if (loading) {
    return (
      <Card variant="trading" className={`p-6 ${className}`}>
        <div className="space-y-4">
          <div className="h-6 bg-slate-700/50 rounded w-48 loading-shimmer-glass" />
          <div className="h-64 bg-slate-800/30 rounded-xl relative overflow-hidden">
            {Array.from({ length: 10 }).map((_, i) => (
              <motion.div
                key={i}
                className="absolute bg-emerald-400/20"
                style={{
                  left: `${(i / 9) * 90 + 5}%`,
                  bottom: `${Math.random() * 60 + 20}%`,
                  width: '6px',
                  height: `${Math.random() * 40 + 20}px`,
                }}
                animate={{
                  opacity: [0.2, 0.6, 0.2],
                  scaleY: [0.8, 1.1, 0.8],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: i * 0.2,
                  ease: 'easeInOut',
                }}
              />
            ))}
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card variant="trading" className={`p-6 ${className}`}>
      <motion.div
        className="mb-4 flex items-center justify-between"
        variants={motionLibrary.utils.createSlideAnimation('up', 10)}
        initial="initial"
        animate="animate"
      >
        <h3 className="text-xl font-bold text-emerald-400 text-glow">
          {title}
        </h3>
        <div className="glass-surface-secondary px-3 py-1 rounded-lg backdrop-blur-sm">
          <span className="text-xs font-mono text-emerald-400">{timeframe}</span>
        </div>
      </motion.div>

      <motion.div
        className="h-64 relative"
        variants={motionLibrary.charts.container}
        initial="initial"
        animate="animate"
      >
        {/* Simplified candlestick visualization */}
        <div className="flex items-end justify-between h-full px-4">
          {data.slice(-20).map((candle, index) => {
            const isPositive = candle.close >= candle.open;
            const bodyHeight = Math.abs(candle.close - candle.open) / Math.max(...data.map(d => d.high)) * 200;
            const wickHeight = (candle.high - candle.low) / Math.max(...data.map(d => d.high)) * 200;
            
            return (
              <motion.div
                key={candle.timestamp}
                className="flex flex-col items-center relative"
                variants={motionLibrary.charts.dataPoint}
                initial="initial"
                animate="animate"
                transition={{ delay: index * 0.05 }}
              >
                {/* Wick */}
                <motion.div
                  className="w-0.5 bg-slate-500"
                  style={{ height: `${wickHeight}px` }}
                  initial={{ scaleY: 0 }}
                  animate={{ scaleY: 1 }}
                  transition={{ delay: index * 0.05 + 0.2, duration: 0.3 }}
                />
                
                {/* Body */}
                <motion.div
                  className={`w-3 ${isPositive ? 'bg-emerald-400' : 'bg-red-400'} rounded-sm`}
                  style={{ height: `${Math.max(bodyHeight, 2)}px` }}
                  initial={{ scaleY: 0, opacity: 0 }}
                  animate={{ scaleY: 1, opacity: 1 }}
                  transition={{ 
                    delay: index * 0.05 + 0.3, 
                    duration: 0.4,
                    ease: motionLibrary.presets.easing.bounce,
                  }}
                  whileHover={{
                    scale: 1.2,
                    boxShadow: `0 0 10px ${isPositive ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'}`,
                  }}
                />
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </Card>
  );
}

export default AnimatedChart;