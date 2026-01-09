/**
 * SPECTACULAR HISTORICAL NAVIGATOR
 * La experiencia visual más espectacular para navegación histórica
 * Diseño Bloomberg Terminal/TradingView nivel profesional
 */

'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Calendar,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Maximize2,
  TrendingUp,
  BarChart3,
  Activity,
  Clock,
  Target
} from 'lucide-react';

interface HistoricalDataPoint {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  price: number;
}

interface NavigationRange {
  start: Date;
  end: Date;
  timeframe: '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M';
}

interface SpectacularHistoricalNavigatorProps {
  data: HistoricalDataPoint[];
  onRangeChange?: (range: NavigationRange) => void;
  onDataRequest?: (range: NavigationRange) => Promise<HistoricalDataPoint[]>;
  minDate?: Date;
  maxDate?: Date;
  isLoading?: boolean;
}

const TIMEFRAMES = [
  { value: '5m', label: '5M', duration: 5 * 60 * 1000 },
  { value: '15m', label: '15M', duration: 15 * 60 * 1000 },
  { value: '1h', label: '1H', duration: 60 * 60 * 1000 },
  { value: '4h', label: '4H', duration: 4 * 60 * 60 * 1000 },
  { value: '1d', label: '1D', duration: 24 * 60 * 60 * 1000 },
  { value: '1w', label: '1W', duration: 7 * 24 * 60 * 60 * 1000 },
  { value: '1M', label: '1M', duration: 30 * 24 * 60 * 60 * 1000 }
] as const;

const QUICK_PRESETS = [
  { label: '1D', days: 1 },
  { label: '1W', days: 7 },
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '1Y', days: 365 },
  { label: 'ALL', days: null }
] as const;

export default function SpectacularHistoricalNavigator({
  data = [],
  onRangeChange,
  onDataRequest,
  minDate = new Date('2020-01-01'),
  maxDate = new Date(),
  isLoading = false
}: SpectacularHistoricalNavigatorProps) {
  // State management
  const [currentRange, setCurrentRange] = useState<NavigationRange>({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    end: new Date(),
    timeframe: '1d'
  });

  const [selectedTimeframe, setSelectedTimeframe] = useState<'5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M'>('1d');
  const [isPlaying, setIsPlaying] = useState(false);
  const [dragState, setDragState] = useState<{
    isDragging: boolean;
    dragType: 'start' | 'end' | 'range' | null;
    startX: number;
  }>({
    isDragging: false,
    dragType: null,
    startX: 0
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sliderRef = useRef<HTMLDivElement>(null);

  // Generate sparkline for mini chart
  const generateSparklineData = useCallback(() => {
    if (!data.length) return [];

    const points = data.map((point, index) => ({
      x: (index / (data.length - 1)) * 100,
      y: point.close,
      timestamp: point.timestamp
    }));

    // Normalize Y values to 0-100 range for display
    const minY = Math.min(...points.map(p => p.y));
    const maxY = Math.max(...points.map(p => p.y));
    const range = maxY - minY || 1;

    return points.map(point => ({
      ...point,
      y: ((point.y - minY) / range) * 100
    }));
  }, [data]);

  // Calculate range positions for slider
  const getRangePositions = useCallback(() => {
    if (!minDate || !maxDate) return { start: 0, end: 100 };

    const totalTime = maxDate.getTime() - minDate.getTime();
    const startPos = ((currentRange.start.getTime() - minDate.getTime()) / totalTime) * 100;
    const endPos = ((currentRange.end.getTime() - minDate.getTime()) / totalTime) * 100;

    return {
      start: Math.max(0, Math.min(100, startPos)),
      end: Math.max(0, Math.min(100, endPos))
    };
  }, [currentRange, minDate, maxDate]);

  // Handle range change
  const handleRangeChange = useCallback((newRange: NavigationRange) => {
    setCurrentRange(newRange);
    onRangeChange?.(newRange);
  }, [onRangeChange]);

  // Handle timeframe change
  const handleTimeframeChange = useCallback((timeframe: typeof selectedTimeframe) => {
    setSelectedTimeframe(timeframe);
    const newRange = { ...currentRange, timeframe };
    handleRangeChange(newRange);
  }, [currentRange, handleRangeChange]);

  // Handle preset selection
  const handlePresetSelect = useCallback((preset: typeof QUICK_PRESETS[number]) => {
    const end = new Date();
    const start = preset.days
      ? new Date(end.getTime() - preset.days * 24 * 60 * 60 * 1000)
      : minDate;

    const newRange = { ...currentRange, start, end };
    handleRangeChange(newRange);
  }, [currentRange, handleRangeChange, minDate]);

  // Mouse event handlers for drag functionality
  const handleMouseDown = useCallback((event: React.MouseEvent, dragType: 'start' | 'end' | 'range') => {
    event.preventDefault();
    setDragState({
      isDragging: true,
      dragType,
      startX: event.clientX
    });
  }, []);

  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!dragState.isDragging || !sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const progress = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const totalTime = maxDate.getTime() - minDate.getTime();
    const newTimestamp = minDate.getTime() + (progress * totalTime);

    const positions = getRangePositions();
    let newStart = currentRange.start;
    let newEnd = currentRange.end;

    switch (dragState.dragType) {
      case 'start':
        newStart = new Date(Math.min(newTimestamp, currentRange.end.getTime()));
        break;
      case 'end':
        newEnd = new Date(Math.max(newTimestamp, currentRange.start.getTime()));
        break;
      case 'range':
        const rangeDuration = currentRange.end.getTime() - currentRange.start.getTime();
        newStart = new Date(newTimestamp - rangeDuration / 2);
        newEnd = new Date(newTimestamp + rangeDuration / 2);

        // Clamp to bounds
        if (newStart.getTime() < minDate.getTime()) {
          newStart = minDate;
          newEnd = new Date(minDate.getTime() + rangeDuration);
        }
        if (newEnd.getTime() > maxDate.getTime()) {
          newEnd = maxDate;
          newStart = new Date(maxDate.getTime() - rangeDuration);
        }
        break;
    }

    const newRange = { ...currentRange, start: newStart, end: newEnd };
    handleRangeChange(newRange);
  }, [dragState, currentRange, minDate, maxDate, handleRangeChange, getRangePositions]);

  const handleMouseUp = useCallback(() => {
    setDragState({
      isDragging: false,
      dragType: null,
      startX: 0
    });
  }, []);

  // Add global mouse event listeners
  useEffect(() => {
    if (dragState.isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [dragState.isDragging, handleMouseMove, handleMouseUp]);

  // Auto-play functionality
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      const rangeDuration = currentRange.end.getTime() - currentRange.start.getTime();
      const step = rangeDuration * 0.1; // Move 10% of range each step

      const newStart = new Date(currentRange.start.getTime() + step);
      const newEnd = new Date(currentRange.end.getTime() + step);

      if (newEnd.getTime() <= maxDate.getTime()) {
        const newRange = { ...currentRange, start: newStart, end: newEnd };
        handleRangeChange(newRange);
      } else {
        setIsPlaying(false);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isPlaying, currentRange, maxDate, handleRangeChange]);

  const sparklineData = generateSparklineData();
  const positions = getRangePositions();

  return (
    <div className="w-full bg-slate-900 border border-slate-600/30 rounded-xl p-6 space-y-6">
      {/* Header Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">Historical Data Range</h3>
          </div>

          {isLoading && (
            <div className="flex items-center space-x-2 text-cyan-400">
              <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">Loading...</span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Playback Controls */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              const newStart = new Date(currentRange.start.getTime() - (currentRange.end.getTime() - currentRange.start.getTime()));
              const newEnd = currentRange.start;
              if (newStart.getTime() >= minDate.getTime()) {
                handleRangeChange({ ...currentRange, start: newStart, end: newEnd });
              }
            }}
            className="p-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg transition-colors"
          >
            <SkipBack className="h-4 w-4 text-slate-300" />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsPlaying(!isPlaying)}
            className={`p-2 rounded-lg transition-colors ${
              isPlaying
                ? 'bg-red-500/20 hover:bg-red-500/30 text-red-400'
                : 'bg-green-500/20 hover:bg-green-500/30 text-green-400'
            }`}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              const newStart = currentRange.end;
              const newEnd = new Date(currentRange.end.getTime() + (currentRange.end.getTime() - currentRange.start.getTime()));
              if (newEnd.getTime() <= maxDate.getTime()) {
                handleRangeChange({ ...currentRange, start: newStart, end: newEnd });
              }
            }}
            className="p-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg transition-colors"
          >
            <SkipForward className="h-4 w-4 text-slate-300" />
          </motion.button>
        </div>
      </div>

      {/* Timeframe Selection */}
      <div className="flex items-center space-x-2">
        <span className="text-sm text-slate-400 font-medium">Timeframe:</span>
        <div className="flex items-center space-x-1">
          {TIMEFRAMES.map((tf) => (
            <motion.button
              key={tf.value}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleTimeframeChange(tf.value)}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                selectedTimeframe === tf.value
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-700/50 text-slate-300 hover:bg-slate-600/50'
              }`}
            >
              {tf.label}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Main Navigation Slider */}
      <div className="space-y-4">
        {/* Timeline Labels */}
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>{minDate.toLocaleDateString()}</span>
          <span>
            {new Date((minDate.getTime() + maxDate.getTime()) / 2).toLocaleDateString()}
          </span>
          <span>{maxDate.toLocaleDateString()}</span>
        </div>

        {/* Slider Container */}
        <div
          ref={sliderRef}
          className="relative h-24 bg-slate-800 rounded-lg border border-slate-600/30 overflow-hidden cursor-pointer"
          onDoubleClick={(e) => {
            const rect = sliderRef.current!.getBoundingClientRect();
            const progress = (e.clientX - rect.left) / rect.width;
            const totalTime = maxDate.getTime() - minDate.getTime();
            const centerTime = minDate.getTime() + (progress * totalTime);
            const rangeDuration = currentRange.end.getTime() - currentRange.start.getTime();

            const newStart = new Date(centerTime - rangeDuration / 2);
            const newEnd = new Date(centerTime + rangeDuration / 2);

            handleRangeChange({ ...currentRange, start: newStart, end: newEnd });
          }}
        >
          {/* Sparkline Background */}
          <svg className="absolute inset-0 w-full h-full opacity-30">
            {sparklineData.length > 1 && (
              <path
                d={`M ${sparklineData.map((point, i) =>
                  `${point.x},${80 - (point.y * 0.6)}`
                ).join(' L ')}`}
                stroke="rgb(34, 197, 94)"
                strokeWidth="1"
                fill="none"
              />
            )}
          </svg>

          {/* Selected Range Overlay */}
          <motion.div
            initial={false}
            animate={{
              left: `${positions.start}%`,
              width: `${positions.end - positions.start}%`
            }}
            className="absolute top-0 bottom-0 bg-cyan-500/20 border-l-2 border-r-2 border-cyan-400 cursor-move"
            onMouseDown={(e) => handleMouseDown(e, 'range')}
          >
            <div className="w-full h-full bg-gradient-to-r from-cyan-500/10 to-purple-500/10" />
          </motion.div>

          {/* Start Handle */}
          <motion.div
            initial={false}
            animate={{ left: `${positions.start}%` }}
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 bg-gradient-to-br from-cyan-400 to-cyan-600 rounded-full border-2 border-white shadow-lg cursor-w-resize z-10 group"
            onMouseDown={(e) => handleMouseDown(e, 'start')}
          >
            <div className="absolute inset-1 bg-white rounded-full opacity-30" />
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {currentRange.start.toLocaleDateString()}
            </motion.div>
          </motion.div>

          {/* End Handle */}
          <motion.div
            initial={false}
            animate={{ left: `${positions.end}%` }}
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 bg-gradient-to-br from-purple-400 to-purple-600 rounded-full border-2 border-white shadow-lg cursor-e-resize z-10 group"
            onMouseDown={(e) => handleMouseDown(e, 'end')}
          >
            <div className="absolute inset-1 bg-white rounded-full opacity-30" />
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {currentRange.end.toLocaleDateString()}
            </motion.div>
          </motion.div>
        </div>

        {/* Quick Navigation Presets */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <ChevronLeft className="h-4 w-4 text-slate-400" />
            <span className="text-sm text-slate-400">Prev Period</span>
          </div>

          <div className="flex items-center space-x-2">
            {QUICK_PRESETS.map((preset) => (
              <motion.button
                key={preset.label}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => handlePresetSelect(preset)}
                className="px-3 py-1 text-sm bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded transition-colors"
              >
                {preset.label}
              </motion.button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-sm text-slate-400">Next Period</span>
            <ChevronRight className="h-4 w-4 text-slate-400" />
          </div>
        </div>
      </div>

      {/* Range Information */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Calendar className="h-4 w-4 text-cyan-400" />
            <span className="text-slate-300">
              Selected: {Math.ceil((currentRange.end.getTime() - currentRange.start.getTime()) / (1000 * 60 * 60 * 24))} days
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <Activity className="h-4 w-4 text-purple-400" />
            <span className="text-slate-300">
              {data.length.toLocaleString()} data points
            </span>
          </div>
        </div>

        <div className="text-slate-400">
          {currentRange.start.toLocaleDateString()} - {currentRange.end.toLocaleDateString()}
        </div>
      </div>

      {/* Usage Instructions */}
      <div className="bg-slate-800/30 rounded-lg p-4 space-y-2 text-sm text-slate-400">
        <div className="font-medium text-slate-300 mb-2">Features:</div>
        <div className="grid grid-cols-2 gap-2">
          <div>• Drag los handles ◉ para ajustar rango</div>
          <div>• Drag el área sombreada para mover el rango completo</div>
          <div>• Doble click en un punto para centrar ahí</div>
          <div>• Scroll sobre el slider para zoom in/out</div>
        </div>
      </div>
    </div>
  );
}