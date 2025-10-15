/**
 * Enhanced Time Range Selector
 * Professional-grade time navigation with presets and custom ranges
 * Based on actual data: 2020-01-02 to 2025-10-10 (92,936 records)
 */

import React, { useState, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  Calendar,
  Clock,
  ChevronDown,
  RotateCcw,
  TrendingUp,
  Activity
} from 'lucide-react';

export interface TimeRange {
  start: Date;
  end: Date;
  timeframe: '5m' | '15m' | '1h' | '4h' | '1d';
  preset?: string;
}

interface TimeRangePreset {
  label: string;
  value: string;
  range: () => TimeRange;
  icon?: React.ReactNode;
  description?: string;
}

interface EnhancedTimeRangeSelectorProps {
  value: TimeRange;
  onChange: (range: TimeRange) => void;
  minDate?: Date;
  maxDate?: Date;
  showTimeframe?: boolean;
  showCustomRange?: boolean;
  className?: string;
}

// Constantes basadas en datos reales
const DATA_START = new Date('2020-01-02T07:30:00Z');
const DATA_END = new Date('2025-10-10T18:55:00Z');
const MARKET_OPEN_HOUR_UTC = 13; // 8:00 AM COT = 13:00 UTC
const MARKET_CLOSE_HOUR_UTC = 18; // 1:00 PM COT = 18:00 UTC

const TIMEFRAMES = [
  { value: '5m', label: '5 Min', duration: 5 * 60 * 1000 },
  { value: '15m', label: '15 Min', duration: 15 * 60 * 1000 },
  { value: '1h', label: '1 Hora', duration: 60 * 60 * 1000 },
  { value: '4h', label: '4 Horas', duration: 4 * 60 * 60 * 1000 },
  { value: '1d', label: '1 Día', duration: 24 * 60 * 60 * 1000 }
] as const;

export default function EnhancedTimeRangeSelector({
  value,
  onChange,
  minDate = DATA_START,
  maxDate = DATA_END,
  showTimeframe = true,
  showCustomRange = true,
  className = ''
}: EnhancedTimeRangeSelectorProps) {
  const [isCustomOpen, setIsCustomOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  // Generar presets dinámicos basados en el rango de datos real
  const presets: TimeRangePreset[] = useMemo(() => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);

    return [
      // Intraday
      {
        label: 'Hoy',
        value: 'today',
        icon: <Activity className="h-4 w-4" />,
        description: 'Sesión actual de trading',
        range: () => ({
          start: new Date(today.getTime() + MARKET_OPEN_HOUR_UTC * 60 * 60 * 1000),
          end: new Date(today.getTime() + MARKET_CLOSE_HOUR_UTC * 60 * 60 * 1000),
          timeframe: '5m' as const,
          preset: 'today'
        })
      },
      {
        label: 'Ayer',
        value: 'yesterday',
        icon: <Clock className="h-4 w-4" />,
        description: 'Sesión de trading anterior',
        range: () => ({
          start: new Date(yesterday.getTime() + MARKET_OPEN_HOUR_UTC * 60 * 60 * 1000),
          end: new Date(yesterday.getTime() + MARKET_CLOSE_HOUR_UTC * 60 * 60 * 1000),
          timeframe: '5m' as const,
          preset: 'yesterday'
        })
      },
      {
        label: 'Esta Semana',
        value: 'this_week',
        icon: <Calendar className="h-4 w-4" />,
        description: 'Últimos 7 días de trading',
        range: () => ({
          start: new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000),
          end: now,
          timeframe: '1h' as const,
          preset: 'this_week'
        })
      },

      // Historical ranges
      {
        label: 'Último Mes',
        value: '1m',
        icon: <TrendingUp className="h-4 w-4" />,
        description: '30 días de datos históricos',
        range: () => ({
          start: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
          end: now,
          timeframe: '4h' as const,
          preset: '1m'
        })
      },
      {
        label: 'Últimos 3 Meses',
        value: '3m',
        icon: <TrendingUp className="h-4 w-4" />,
        description: '90 días de histórico',
        range: () => ({
          start: new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000),
          end: now,
          timeframe: '1d' as const,
          preset: '3m'
        })
      },
      {
        label: 'Último Año',
        value: '1y',
        icon: <TrendingUp className="h-4 w-4" />,
        description: '365 días de datos',
        range: () => ({
          start: new Date(now.getTime() - 365 * 24 * 60 * 60 * 1000),
          end: now,
          timeframe: '1d' as const,
          preset: '1y'
        })
      },
      {
        label: 'Todo el Histórico',
        value: 'all',
        icon: <TrendingUp className="h-4 w-4" />,
        description: '2020-2025 (92k registros)',
        range: () => ({
          start: DATA_START,
          end: DATA_END,
          timeframe: '1d' as const,
          preset: 'all'
        })
      }
    ];
  }, []);

  // Manejar selección de preset
  const handlePresetSelect = useCallback((preset: TimeRangePreset) => {
    const newRange = preset.range();
    setSelectedPreset(preset.value);
    onChange(newRange);
  }, [onChange]);

  // Manejar cambio de timeframe
  const handleTimeframeChange = useCallback((newTimeframe: string) => {
    onChange({
      ...value,
      timeframe: newTimeframe as any
    });
  }, [value, onChange]);

  // Reset a rango por defecto
  const handleReset = useCallback(() => {
    const defaultRange = presets.find(p => p.value === '1m')?.range();
    if (defaultRange) {
      setSelectedPreset('1m');
      onChange(defaultRange);
    }
  }, [presets, onChange]);

  // Formatear fecha para display
  const formatDate = useCallback((date: Date) => {
    return date.toLocaleDateString('es-ES', {
      day: '2-digit',
      month: 'short',
      year: date.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined
    });
  }, []);

  // Calcular duración del rango actual
  const rangeDuration = useMemo(() => {
    const diff = value.end.getTime() - value.start.getTime();
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));

    if (days < 1) {
      const hours = Math.floor(diff / (60 * 60 * 1000));
      return `${hours}h`;
    } else if (days < 30) {
      return `${days}d`;
    } else if (days < 365) {
      const months = Math.floor(days / 30);
      return `${months}m`;
    } else {
      const years = Math.floor(days / 365);
      return `${years}a`;
    }
  }, [value.start, value.end]);

  return (
    <div className={`bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Calendar className="h-5 w-5" />
          Rango Temporal
        </h3>

        <div className="flex items-center gap-2 text-sm text-slate-400">
          <span>Duración: {rangeDuration}</span>
          <button
            onClick={handleReset}
            className="p-1 hover:bg-slate-700 rounded transition-colors"
            title="Reset a rango por defecto"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Presets Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        {presets.map((preset) => (
          <motion.button
            key={preset.value}
            onClick={() => handlePresetSelect(preset)}
            className={`p-3 rounded-lg border transition-all duration-200 ${
              selectedPreset === preset.value
                ? 'bg-cyan-600 border-cyan-500 text-white'
                : 'bg-slate-800/50 border-slate-600/50 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500/50'
            }`}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center gap-2 mb-1">
              {preset.icon}
              <span className="font-medium text-sm">{preset.label}</span>
            </div>
            {preset.description && (
              <div className="text-xs opacity-75">{preset.description}</div>
            )}
          </motion.button>
        ))}
      </div>

      {/* Timeframe Selector */}
      {showTimeframe && (
        <div className="flex items-center gap-2 mb-4">
          <span className="text-sm text-slate-400 font-medium">Timeframe:</span>
          <div className="flex items-center gap-1">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf.value}
                onClick={() => handleTimeframeChange(tf.value)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                  value.timeframe === tf.value
                    ? 'bg-cyan-600 text-white'
                    : 'bg-slate-700/50 text-slate-300 hover:bg-slate-600/50'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Current Range Display */}
      <div className="flex items-center justify-between bg-slate-800/30 rounded-lg p-3">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-xs text-slate-400">Desde</div>
            <div className="text-sm font-mono text-white">
              {formatDate(value.start)}
            </div>
          </div>
          <div className="text-slate-500">→</div>
          <div>
            <div className="text-xs text-slate-400">Hasta</div>
            <div className="text-sm font-mono text-white">
              {formatDate(value.end)}
            </div>
          </div>
        </div>

        {/* Custom Range Button */}
        {showCustomRange && (
          <button
            onClick={() => setIsCustomOpen(!isCustomOpen)}
            className="flex items-center gap-2 px-3 py-1.5 bg-slate-700/50 hover:bg-slate-600/50 rounded text-sm text-slate-300 transition-colors"
          >
            Personalizar
            <ChevronDown className={`h-4 w-4 transition-transform ${isCustomOpen ? 'rotate-180' : ''}`} />
          </button>
        )}
      </div>

      {/* Custom Range Picker (Expandible) */}
      {isCustomOpen && showCustomRange && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-4 p-4 bg-slate-800/50 rounded-lg border border-slate-600/30"
        >
          <div className="text-sm text-slate-400 mb-2">
            Rango de datos disponible: {formatDate(minDate)} - {formatDate(maxDate)}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Fecha inicial</label>
              <input
                type="datetime-local"
                value={value.start.toISOString().slice(0, 16)}
                onChange={(e) => {
                  const newStart = new Date(e.target.value);
                  onChange({ ...value, start: newStart });
                }}
                min={minDate.toISOString().slice(0, 16)}
                max={value.end.toISOString().slice(0, 16)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-400 mb-1">Fecha final</label>
              <input
                type="datetime-local"
                value={value.end.toISOString().slice(0, 16)}
                onChange={(e) => {
                  const newEnd = new Date(e.target.value);
                  onChange({ ...value, end: newEnd });
                }}
                min={value.start.toISOString().slice(0, 16)}
                max={maxDate.toISOString().slice(0, 16)}
                className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm"
              />
            </div>
          </div>
        </motion.div>
      )}

      {/* Data Quality Indicator */}
      <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
        <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
        <span>Datos reales disponibles • Bid/Ask • 92,936 registros</span>
      </div>
    </div>
  );
}