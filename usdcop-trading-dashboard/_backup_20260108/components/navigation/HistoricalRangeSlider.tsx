/**
 * Historical Range Slider with Mini Chart
 * Professional dual-handle slider for navigating 92k+ historical records
 * Features: Mini sparkline, market hours awareness, performance optimized
 */

import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ChevronLeft,
  ChevronRight,
  Calendar,
  Clock,
  Maximize2,
  RotateCcw
} from 'lucide-react';

export interface TimeRange {
  start: Date;
  end: Date;
}

interface DataPoint {
  timestamp: Date;
  price: number;
  high?: number;
  low?: number;
}

interface HistoricalRangeSliderProps {
  min: Date;
  max: Date;
  value: TimeRange;
  onChange: (range: TimeRange) => void;
  data?: DataPoint[];

  // Visual options
  showMiniChart?: boolean;
  snapToMarketHours?: boolean;
  showMarketHours?: boolean;

  // Performance
  throttleMs?: number;
  className?: string;
}

interface HandleProps {
  position: number;
  onDragStart: () => void;
  label: string;
  side: 'left' | 'right';
  isActive: boolean;
}

const SliderHandle: React.FC<HandleProps> = ({
  position,
  onDragStart,
  label,
  side,
  isActive
}) => (
  <motion.div
    className="absolute top-1/2 -translate-y-1/2 group z-10"
    style={{ left: `${position}%` }}
    onMouseDown={onDragStart}
    whileHover={{ scale: 1.1 }}
    whileTap={{ scale: 0.95 }}
  >
    <div className="relative -translate-x-1/2">
      {/* Handle */}
      <div className={`w-5 h-8 rounded-full cursor-ew-resize shadow-lg border-2 border-slate-900 transition-colors ${
        isActive ? 'bg-cyan-400' : 'bg-cyan-500 hover:bg-cyan-400'
      }`} />

      {/* Tooltip */}
      <motion.div
        className={`absolute top-full mt-2 px-3 py-1.5 bg-slate-900 border border-slate-600 text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none ${
          side === 'right' ? 'right-0' : 'left-0'
        }`}
        initial={{ opacity: 0, y: -4 }}
        whileHover={{ opacity: 1, y: 0 }}
      >
        <span className="text-white font-medium">{label}</span>
        <div className={`absolute top-0 w-0 h-0 border-l-4 border-r-4 border-b-4 border-transparent border-b-slate-900 -translate-y-full ${
          side === 'right' ? 'right-2' : 'left-2'
        }`} />
      </motion.div>
    </div>
  </motion.div>
);

const MiniSparkline: React.FC<{
  data: DataPoint[];
  selectedRange: TimeRange;
  width: number;
  height: number;
}> = ({ data, selectedRange, width, height }) => {
  const pathData = useMemo(() => {
    if (data.length === 0) return '';

    const minPrice = Math.min(...data.map(d => d.price));
    const maxPrice = Math.max(...data.map(d => d.price));
    const priceRange = maxPrice - minPrice;

    if (priceRange === 0) return '';

    const minTime = data[0].timestamp.getTime();
    const maxTime = data[data.length - 1].timestamp.getTime();
    const timeRange = maxTime - minTime;

    const points = data.map((point, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((point.price - minPrice) / priceRange) * height;
      return `${x},${y}`;
    });

    return `M ${points.join(' L ')}`;
  }, [data, width, height]);

  const selectedArea = useMemo(() => {
    if (data.length === 0) return { left: 0, width: 0 };

    const minTime = data[0].timestamp.getTime();
    const maxTime = data[data.length - 1].timestamp.getTime();
    const timeRange = maxTime - minTime;

    const startPercent = ((selectedRange.start.getTime() - minTime) / timeRange) * 100;
    const endPercent = ((selectedRange.end.getTime() - minTime) / timeRange) * 100;

    return {
      left: Math.max(0, startPercent),
      width: Math.max(0, endPercent - startPercent)
    };
  }, [data, selectedRange]);

  return (
    <div className="relative w-full bg-slate-800/30 rounded" style={{ height }}>
      <svg width={width} height={height} className="absolute inset-0">
        {/* Background grid */}
        <defs>
          <pattern id="grid" width="20" height="10" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 10" fill="none" stroke="rgba(148, 163, 184, 0.1)" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* Price line */}
        {pathData && (
          <path
            d={pathData}
            fill="none"
            stroke="#06b6d4"
            strokeWidth={1.5}
            opacity={0.8}
          />
        )}

        {/* Selected area overlay */}
        <rect
          x={`${selectedArea.left}%`}
          width={`${selectedArea.width}%`}
          height="100%"
          fill="rgba(6, 182, 212, 0.2)"
          stroke="rgba(6, 182, 212, 0.5)"
          strokeWidth={1}
        />
      </svg>
    </div>
  );
};

export default function HistoricalRangeSlider({
  min,
  max,
  value,
  onChange,
  data = [],
  showMiniChart = true,
  snapToMarketHours = true,
  showMarketHours = true,
  throttleMs = 32,
  className = ''
}: HistoricalRangeSliderProps) {
  const [isDragging, setIsDragging] = useState<'start' | 'end' | 'range' | null>(null);
  const [hoveredTime, setHoveredTime] = useState<Date | null>(null);
  const sliderRef = useRef<HTMLDivElement>(null);
  const throttleTimer = useRef<NodeJS.Timeout>();

  // Constants for market hours (COT to UTC)
  const MARKET_OPEN_HOUR = 13; // 8:00 AM COT = 13:00 UTC
  const MARKET_CLOSE_HOUR = 18; // 1:00 PM COT = 18:00 UTC

  // Time scale conversion
  const timeToPercent = useCallback((time: Date): number => {
    const totalRange = max.getTime() - min.getTime();
    const timeOffset = time.getTime() - min.getTime();
    return (timeOffset / totalRange) * 100;
  }, [min, max]);

  const percentToTime = useCallback((percent: number): Date => {
    const totalRange = max.getTime() - min.getTime();
    const timeOffset = (percent / 100) * totalRange;
    return new Date(min.getTime() + timeOffset);
  }, [min, max]);

  // Snap to market hours if enabled
  const snapToMarketHour = useCallback((time: Date): Date => {
    if (!snapToMarketHours) return time;

    const date = new Date(time);
    const hour = date.getUTCHours();

    // Snap to nearest market boundary
    if (hour < MARKET_OPEN_HOUR) {
      date.setUTCHours(MARKET_OPEN_HOUR, 0, 0, 0);
    } else if (hour > MARKET_CLOSE_HOUR) {
      date.setUTCHours(MARKET_CLOSE_HOUR, 0, 0, 0);
    }

    return date;
  }, [snapToMarketHours]);

  // Throttled onChange
  const throttledOnChange = useCallback((newRange: TimeRange) => {
    if (throttleTimer.current) {
      clearTimeout(throttleTimer.current);
    }

    throttleTimer.current = setTimeout(() => {
      onChange(newRange);
    }, throttleMs);
  }, [onChange, throttleMs]);

  // Mouse event handlers
  const handleMouseDown = useCallback((event: React.MouseEvent, dragType: 'start' | 'end' | 'range') => {
    event.preventDefault();
    setIsDragging(dragType);
  }, []);

  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!sliderRef.current || !isDragging) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const percent = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100));
    const newTime = percentToTime(percent);

    let newRange = { ...value };

    if (isDragging === 'start') {
      const snappedTime = snapToMarketHour(newTime);
      newRange.start = new Date(Math.min(snappedTime.getTime(), value.end.getTime() - 60000)); // Min 1 minute range
    } else if (isDragging === 'end') {
      const snappedTime = snapToMarketHour(newTime);
      newRange.end = new Date(Math.max(snappedTime.getTime(), value.start.getTime() + 60000)); // Min 1 minute range
    } else if (isDragging === 'range') {
      const rangeDuration = value.end.getTime() - value.start.getTime();
      const centerTime = newTime.getTime();

      newRange.start = new Date(Math.max(min.getTime(), centerTime - rangeDuration / 2));
      newRange.end = new Date(Math.min(max.getTime(), centerTime + rangeDuration / 2));

      // Adjust if we hit boundaries
      if (newRange.start.getTime() === min.getTime()) {
        newRange.end = new Date(min.getTime() + rangeDuration);
      } else if (newRange.end.getTime() === max.getTime()) {
        newRange.start = new Date(max.getTime() - rangeDuration);
      }
    }

    throttledOnChange(newRange);
  }, [isDragging, value, percentToTime, snapToMarketHour, throttledOnChange, min, max]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(null);
  }, []);

  // Hover handler for tooltip
  const handleMouseHover = useCallback((event: React.MouseEvent) => {
    if (!sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const percent = ((event.clientX - rect.left) / rect.width) * 100;
    setHoveredTime(percentToTime(percent));
  }, [percentToTime]);

  // Global mouse events
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Quick navigation functions
  const shiftRange = useCallback((direction: -1 | 1) => {
    const rangeDuration = value.end.getTime() - value.start.getTime();
    const shiftAmount = rangeDuration * 0.5; // Shift by half the current range

    let newStart = new Date(value.start.getTime() + (direction * shiftAmount));
    let newEnd = new Date(value.end.getTime() + (direction * shiftAmount));

    // Clamp to boundaries
    if (newStart.getTime() < min.getTime()) {
      newStart = new Date(min.getTime());
      newEnd = new Date(min.getTime() + rangeDuration);
    } else if (newEnd.getTime() > max.getTime()) {
      newEnd = new Date(max.getTime());
      newStart = new Date(max.getTime() - rangeDuration);
    }

    onChange({ start: newStart, end: newEnd });
  }, [value, min, max, onChange]);

  const setQuickRange = useCallback((preset: string) => {
    const now = new Date();
    let newRange: TimeRange;

    switch (preset) {
      case '1d':
        newRange = {
          start: new Date(now.getTime() - 24 * 60 * 60 * 1000),
          end: now
        };
        break;
      case '1w':
        newRange = {
          start: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
          end: now
        };
        break;
      case '1m':
        newRange = {
          start: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
          end: now
        };
        break;
      case 'ytd':
        newRange = {
          start: new Date(now.getFullYear(), 0, 1),
          end: now
        };
        break;
      default:
        return;
    }

    // Clamp to data boundaries
    newRange.start = new Date(Math.max(newRange.start.getTime(), min.getTime()));
    newRange.end = new Date(Math.min(newRange.end.getTime(), max.getTime()));

    onChange(newRange);
  }, [min, max, onChange]);

  // Format functions
  const formatDate = useCallback((date: Date) => {
    return date.toLocaleDateString('es-ES', {
      day: '2-digit',
      month: 'short',
      year: date.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
    });
  }, []);

  const formatDateTime = useCallback((date: Date) => {
    return date.toLocaleString('es-ES', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit'
    });
  }, []);

  const formatDuration = useCallback((ms: number) => {
    const days = Math.floor(ms / (24 * 60 * 60 * 1000));
    const hours = Math.floor((ms % (24 * 60 * 60 * 1000)) / (60 * 60 * 1000));

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h`;
    return `${Math.floor(ms / (60 * 1000))}min`;
  }, []);

  // Calculate positions
  const startPercent = timeToPercent(value.start);
  const endPercent = timeToPercent(value.end);
  const duration = value.end.getTime() - value.start.getTime();

  return (
    <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Calendar className="h-5 w-5 text-cyan-400" />
          <span className="text-lg font-semibold text-white">Navegación Histórica</span>
        </div>

        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Clock className="h-4 w-4" />
          <span>Seleccionado: {formatDuration(duration)}</span>
        </div>
      </div>

      {/* Date range display */}
      <div className="flex items-center justify-between text-sm text-slate-400 mb-3">
        <span>{formatDate(min)}</span>
        <span className="text-white font-medium">
          {formatDate(value.start)} → {formatDate(value.end)}
        </span>
        <span>{formatDate(max)}</span>
      </div>

      {/* Main slider */}
      <div
        ref={sliderRef}
        className="relative h-12 bg-slate-800/50 rounded-lg cursor-pointer mb-4"
        onMouseMove={handleMouseHover}
        onMouseLeave={() => setHoveredTime(null)}
      >
        {/* Background track with market hours indicators */}
        {showMarketHours && (
          <div className="absolute inset-0 flex">
            {/* Market hours visualization would go here */}
          </div>
        )}

        {/* Selected range */}
        <motion.div
          className="absolute top-0 bottom-0 bg-cyan-500/20 border-y border-cyan-500/50 cursor-move"
          style={{
            left: `${startPercent}%`,
            width: `${endPercent - startPercent}%`
          }}
          onMouseDown={(e) => handleMouseDown(e, 'range')}
          whileHover={{ backgroundColor: 'rgba(6, 182, 212, 0.3)' }}
        />

        {/* Start handle */}
        <SliderHandle
          position={startPercent}
          onDragStart={() => handleMouseDown({} as any, 'start')}
          label={formatDateTime(value.start)}
          side="left"
          isActive={isDragging === 'start'}
        />

        {/* End handle */}
        <SliderHandle
          position={endPercent}
          onDragStart={() => handleMouseDown({} as any, 'end')}
          label={formatDateTime(value.end)}
          side="right"
          isActive={isDragging === 'end'}
        />

        {/* Hover indicator */}
        {hoveredTime && !isDragging && (
          <div
            className="absolute top-0 bottom-0 w-px bg-cyan-400/60 pointer-events-none"
            style={{ left: `${timeToPercent(hoveredTime)}%` }}
          >
            <div className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 bg-slate-900 text-xs rounded whitespace-nowrap text-white border border-slate-600">
              {formatDateTime(hoveredTime)}
            </div>
          </div>
        )}
      </div>

      {/* Mini chart */}
      {showMiniChart && data.length > 0 && (
        <div className="mb-4">
          <div className="text-sm text-slate-400 mb-2 flex items-center gap-2">
            <span>Vista general del precio</span>
            <span className="text-xs text-slate-500">({data.length.toLocaleString()} puntos)</span>
          </div>
          <MiniSparkline
            data={data}
            selectedRange={value}
            width={sliderRef.current?.clientWidth || 800}
            height={60}
          />
        </div>
      )}

      {/* Quick controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => shiftRange(-1)}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-700/50 hover:bg-slate-600/50 rounded transition-colors text-slate-300"
            disabled={value.start.getTime() <= min.getTime()}
          >
            <ChevronLeft className="h-4 w-4" />
            Anterior
          </button>

          <button
            onClick={() => shiftRange(1)}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-700/50 hover:bg-slate-600/50 rounded transition-colors text-slate-300"
            disabled={value.end.getTime() >= max.getTime()}
          >
            Siguiente
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center gap-2">
          {['1d', '1w', '1m', 'ytd'].map((preset) => (
            <button
              key={preset}
              onClick={() => setQuickRange(preset)}
              className="px-2 py-1 text-xs bg-slate-700/30 hover:bg-slate-600/50 rounded transition-colors text-slate-400 hover:text-white"
            >
              {preset.toUpperCase()}
            </button>
          ))}

          <button
            onClick={() => onChange({ start: min, end: max })}
            className="p-1.5 bg-slate-700/30 hover:bg-slate-600/50 rounded transition-colors text-slate-400 hover:text-white"
            title="Ver todo"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}