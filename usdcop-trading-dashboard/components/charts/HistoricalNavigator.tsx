/**
 * Historical Navigator Component
 * Advanced navigation for 6 years of USDCOP data (2020-2025)
 * Includes zoom levels, time period selection, and smooth scrolling
 */

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ChevronLeft,
  ChevronRight,
  Calendar,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Navigation,
  Clock
} from 'lucide-react';

interface HistoricalNavigatorProps {
  currentDate: Date;
  minDate: Date;
  maxDate: Date;
  timeframe: '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M';
  onDateChange: (date: Date) => void;
  onTimeframeChange: (timeframe: string) => void;
  onZoomChange?: (level: number) => void;
  isLoading?: boolean;
  autoPlay?: boolean;
  onAutoPlayChange?: (enabled: boolean) => void;
}

const timeframes = [
  { value: '5m', label: '5m', description: '5 minutos' },
  { value: '15m', label: '15m', description: '15 minutos' },
  { value: '1h', label: '1h', description: '1 hora' },
  { value: '4h', label: '4h', description: '4 horas' },
  { value: '1d', label: '1D', description: '1 día' },
  { value: '1w', label: '1W', description: '1 semana' },
  { value: '1M', label: '1M', description: '1 mes' }
];

const quickPeriods = [
  { label: 'Último Mes', days: 30 },
  { label: 'Últimos 3M', days: 90 },
  { label: 'Último Año', days: 365 },
  { label: 'Todo', days: null }
];

export default function HistoricalNavigator({
  currentDate,
  minDate,
  maxDate,
  timeframe,
  onDateChange,
  onTimeframeChange,
  onZoomChange,
  isLoading = false,
  autoPlay = false,
  onAutoPlayChange
}: HistoricalNavigatorProps) {
  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedPeriod, setSelectedPeriod] = useState<number | null>(null);
  const [showDatePicker, setShowDatePicker] = useState(false);

  // Calculate navigation steps based on timeframe
  const getNavigationStep = useCallback(() => {
    switch (timeframe) {
      case '5m': return 5 * 60 * 1000; // 5 minutes
      case '15m': return 15 * 60 * 1000; // 15 minutes
      case '1h': return 60 * 60 * 1000; // 1 hour
      case '4h': return 4 * 60 * 60 * 1000; // 4 hours
      case '1d': return 24 * 60 * 60 * 1000; // 1 day
      case '1w': return 7 * 24 * 60 * 60 * 1000; // 1 week
      case '1M': return 30 * 24 * 60 * 60 * 1000; // 1 month
      default: return 24 * 60 * 60 * 1000;
    }
  }, [timeframe]);

  // Navigation functions
  const navigateBackward = useCallback(() => {
    const step = getNavigationStep();
    const newDate = new Date(currentDate.getTime() - step);
    if (newDate >= minDate) {
      onDateChange(newDate);
    }
  }, [currentDate, getNavigationStep, minDate, onDateChange]);

  const navigateForward = useCallback(() => {
    const step = getNavigationStep();
    const newDate = new Date(currentDate.getTime() + step);
    if (newDate <= maxDate) {
      onDateChange(newDate);
    }
  }, [currentDate, getNavigationStep, maxDate, onDateChange]);

  // Jump to specific periods
  const jumpToPeriod = useCallback((days: number | null) => {
    if (days === null) {
      // Show all data
      onDateChange(minDate);
      setSelectedPeriod(null);
    } else {
      const newDate = new Date(maxDate.getTime() - (days * 24 * 60 * 60 * 1000));
      onDateChange(newDate >= minDate ? newDate : minDate);
      setSelectedPeriod(days);
    }
  }, [minDate, maxDate, onDateChange]);

  // Auto-play functionality
  useEffect(() => {
    if (!autoPlay) return;

    const interval = setInterval(() => {
      const step = getNavigationStep();
      const newDate = new Date(currentDate.getTime() + step);

      if (newDate <= maxDate) {
        onDateChange(newDate);
      } else {
        onAutoPlayChange?.(false);
      }
    }, 1000); // Update every second

    return () => clearInterval(interval);
  }, [autoPlay, currentDate, getNavigationStep, maxDate, onDateChange, onAutoPlayChange]);

  // Calculate progress percentage
  const progressPercentage = useMemo(() => {
    const total = maxDate.getTime() - minDate.getTime();
    const current = currentDate.getTime() - minDate.getTime();
    return Math.max(0, Math.min(100, (current / total) * 100));
  }, [currentDate, minDate, maxDate]);

  // Format current date display
  const formatCurrentDate = useMemo(() => {
    const day = currentDate.getDate().toString().padStart(2, '0');
    const month = (currentDate.getMonth() + 1).toString().padStart(2, '0');
    const year = currentDate.getFullYear();
    const hours = currentDate.getHours().toString().padStart(2, '0');
    const minutes = currentDate.getMinutes().toString().padStart(2, '0');

    return `${day}/${month}/${year} ${hours}:${minutes}`;
  }, [currentDate]);

  return (
    <div className="bg-slate-900/95 backdrop-blur-xl border border-slate-600/50 rounded-xl p-4 space-y-4">
      {/* Header with date and controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Clock className="h-4 w-4 text-cyan-400" />
            <div className="text-lg font-mono text-white font-bold">
              {formatCurrentDate}
            </div>
          </div>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowDatePicker(!showDatePicker)}
            className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          >
            <Calendar className="h-4 w-4 text-slate-300" />
          </motion.button>
        </div>

        {/* Auto-play controls */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onDateChange(minDate)}
            className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            title="Ir al inicio"
          >
            <SkipBack className="h-4 w-4 text-slate-300" />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onAutoPlayChange?.(!autoPlay)}
            className={`p-2 rounded-lg transition-colors ${
              autoPlay
                ? 'bg-cyan-600 hover:bg-cyan-700 text-white'
                : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
            }`}
            title={autoPlay ? "Pausar reproducción" : "Reproducir automáticamente"}
          >
            {autoPlay ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onDateChange(maxDate)}
            className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
            title="Ir al final"
          >
            <SkipForward className="h-4 w-4 text-slate-300" />
          </motion.button>
        </div>
      </div>

      {/* Timeline slider */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <span>2020</span>
          <span className="text-cyan-400">{progressPercentage.toFixed(1)}%</span>
          <span>2025</span>
        </div>

        <div className="relative">
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 to-purple-500"
              style={{ width: `${progressPercentage}%` }}
              initial={{ width: 0 }}
              animate={{ width: `${progressPercentage}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>

          {/* Interactive slider */}
          <input
            type="range"
            min={minDate.getTime()}
            max={maxDate.getTime()}
            value={currentDate.getTime()}
            onChange={(e) => onDateChange(new Date(parseInt(e.target.value)))}
            className="absolute inset-0 w-full h-2 opacity-0 cursor-pointer"
          />
        </div>
      </div>

      {/* Navigation controls */}
      <div className="flex items-center justify-between">
        {/* Left/Right navigation */}
        <div className="flex items-center space-x-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={navigateBackward}
            disabled={currentDate <= minDate}
            className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
          >
            <ChevronLeft className="h-4 w-4 text-slate-300" />
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={navigateForward}
            disabled={currentDate >= maxDate}
            className="p-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
          >
            <ChevronRight className="h-4 w-4 text-slate-300" />
          </motion.button>
        </div>

        {/* Timeframe selector */}
        <div className="flex items-center space-x-1">
          {timeframes.map((tf) => (
            <motion.button
              key={tf.value}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onTimeframeChange(tf.value)}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                timeframe === tf.value
                  ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-500/25'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
              title={tf.description}
            >
              {tf.label}
            </motion.button>
          ))}
        </div>
      </div>

      {/* Quick period selection */}
      <div className="flex items-center justify-center space-x-2">
        {quickPeriods.map((period) => (
          <motion.button
            key={period.label}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => jumpToPeriod(period.days)}
            className={`px-3 py-1 text-xs rounded-md transition-all ${
              selectedPeriod === period.days
                ? 'bg-purple-600 text-white'
                : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-slate-300'
            }`}
          >
            {period.label}
          </motion.button>
        ))}
      </div>

      {/* Date picker modal */}
      <AnimatePresence>
        {showDatePicker && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 right-0 mt-2 p-4 bg-slate-800 border border-slate-600 rounded-xl shadow-xl z-50"
          >
            <div className="space-y-2">
              <label className="block text-sm text-slate-400">Fecha específica:</label>
              <input
                type="datetime-local"
                value={currentDate.toISOString().slice(0, 16)}
                min={minDate.toISOString().slice(0, 16)}
                max={maxDate.toISOString().slice(0, 16)}
                onChange={(e) => {
                  onDateChange(new Date(e.target.value));
                  setShowDatePicker(false);
                }}
                className="w-full p-2 bg-slate-700 border border-slate-600 rounded-lg text-white text-sm"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex items-center justify-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="h-4 w-4 border-2 border-cyan-500 border-t-transparent rounded-full"
          />
          <span className="ml-2 text-xs text-slate-400">Cargando datos...</span>
        </div>
      )}
    </div>
  );
}