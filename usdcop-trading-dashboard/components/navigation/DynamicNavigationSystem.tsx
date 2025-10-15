/**
 * Dynamic Navigation System
 * Integrates all navigation components into a cohesive professional system
 * Based on real data: 92k+ USDCOP records from 2020-2025
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Navigation,
  Settings,
  ChevronDown,
  ChevronUp,
  Maximize2,
  Minimize2,
  Activity,
  BarChart3
} from 'lucide-react';

import EnhancedTimeRangeSelector, { TimeRange as TimeRangeType } from './EnhancedTimeRangeSelector';
import HistoricalRangeSlider from './HistoricalRangeSlider';
import RealDataMetricsPanel from '../metrics/RealDataMetricsPanel';
import { OHLCData } from '../../lib/services/real-market-metrics';

export interface NavigationState {
  timeRange: TimeRangeType;
  isExpanded: boolean;
  showMetrics: boolean;
  showMiniChart: boolean;
  snapToMarketHours: boolean;
  autoUpdate: boolean;
}

interface DynamicNavigationSystemProps {
  // Data
  data: OHLCData[];
  currentTick?: {
    price: number;
    bid: number;
    ask: number;
    timestamp: Date;
  };

  // Callbacks
  onTimeRangeChange: (range: TimeRangeType) => void;
  onDataRequest?: (range: TimeRangeType) => Promise<OHLCData[]>;

  // Configuration
  minDate?: Date;
  maxDate?: Date;
  className?: string;

  // Layout options
  layout?: 'compact' | 'full' | 'sidebar';
  initialExpanded?: boolean;
}

// Default date range based on our actual data
const DEFAULT_MIN_DATE = new Date('2020-01-02T07:30:00Z');
const DEFAULT_MAX_DATE = new Date('2025-10-10T18:55:00Z');

export default function DynamicNavigationSystem({
  data,
  currentTick,
  onTimeRangeChange,
  onDataRequest,
  minDate = DEFAULT_MIN_DATE,
  maxDate = DEFAULT_MAX_DATE,
  className = '',
  layout = 'full',
  initialExpanded = true
}: DynamicNavigationSystemProps) {
  // Navigation state
  const [navState, setNavState] = useState<NavigationState>({
    timeRange: {
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // Default: last 30 days
      end: new Date(),
      timeframe: '1h',
      preset: '1m'
    },
    isExpanded: initialExpanded,
    showMetrics: true,
    showMiniChart: true,
    snapToMarketHours: true,
    autoUpdate: true
  });

  const [isLoading, setIsLoading] = useState(false);

  // Prepare mini chart data for the slider
  const miniChartData = useMemo(() => {
    if (data.length === 0) return [];

    // Sample data for mini chart (max 500 points for performance)
    const sampleSize = Math.min(500, data.length);
    const step = Math.floor(data.length / sampleSize);

    return data
      .filter((_, index) => index % step === 0)
      .map(d => ({
        timestamp: d.timestamp,
        price: d.price,
        high: d.high,
        low: d.low
      }));
  }, [data]);

  // Handle time range changes
  const handleTimeRangeChange = useCallback(async (newRange: TimeRangeType) => {
    setNavState(prev => ({ ...prev, timeRange: newRange }));

    // Trigger data loading if callback provided
    if (onDataRequest) {
      setIsLoading(true);
      try {
        await onDataRequest(newRange);
      } catch (error) {
        console.error('Error loading data for range:', error);
      } finally {
        setIsLoading(false);
      }
    }

    onTimeRangeChange(newRange);
  }, [onTimeRangeChange, onDataRequest]);

  // Handle slider-specific range change
  const handleSliderRangeChange = useCallback((range: { start: Date; end: Date }) => {
    const newTimeRange: TimeRangeType = {
      ...navState.timeRange,
      start: range.start,
      end: range.end
    };
    handleTimeRangeChange(newTimeRange);
  }, [navState.timeRange, handleTimeRangeChange]);

  // Toggle navigation panel
  const toggleExpanded = useCallback(() => {
    setNavState(prev => ({ ...prev, isExpanded: !prev.isExpanded }));
  }, []);

  // Toggle metrics panel
  const toggleMetrics = useCallback(() => {
    setNavState(prev => ({ ...prev, showMetrics: !prev.showMetrics }));
  }, []);

  // Update settings
  const updateSettings = useCallback((updates: Partial<NavigationState>) => {
    setNavState(prev => ({ ...prev, ...updates }));
  }, []);

  // Auto-update effect (for real-time data)
  useEffect(() => {
    if (!navState.autoUpdate) return;

    const interval = setInterval(() => {
      // Auto-extend range to include new data if we're viewing "recent" data
      const now = new Date();
      const isViewingRecent = navState.timeRange.end.getTime() > (now.getTime() - 60 * 60 * 1000); // Within 1 hour

      if (isViewingRecent) {
        const newRange = {
          ...navState.timeRange,
          end: now
        };
        handleTimeRangeChange(newRange);
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, [navState.autoUpdate, navState.timeRange, handleTimeRangeChange]);

  // Layout-specific rendering
  const renderCompactLayout = () => (
    <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={toggleExpanded}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg transition-colors text-white"
          >
            <Navigation className="h-4 w-4" />
            <span className="text-sm font-medium">Navegación</span>
            {navState.isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>

          <div className="text-sm text-slate-400">
            {navState.timeRange.start.toLocaleDateString()} - {navState.timeRange.end.toLocaleDateString()}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleMetrics}
            className={`p-2 rounded-lg transition-colors ${
              navState.showMetrics
                ? 'bg-cyan-600 text-white'
                : 'bg-slate-700/50 text-slate-400 hover:bg-slate-600/50'
            }`}
            title="Toggle metrics"
          >
            <BarChart3 className="h-4 w-4" />
          </button>

          <div className="w-px h-6 bg-slate-600" />

          <div className="flex items-center gap-1 text-xs text-slate-500">
            <Activity className="h-3 w-3" />
            {data.length.toLocaleString()} puntos
          </div>
        </div>
      </div>
    </div>
  );

  const renderFullLayout = () => (
    <div className="space-y-6">
      {/* Time Range Selector */}
      <EnhancedTimeRangeSelector
        value={navState.timeRange}
        onChange={handleTimeRangeChange}
        minDate={minDate}
        maxDate={maxDate}
        showTimeframe={true}
        showCustomRange={true}
      />

      {/* Historical Range Slider */}
      <HistoricalRangeSlider
        min={minDate}
        max={maxDate}
        value={{ start: navState.timeRange.start, end: navState.timeRange.end }}
        onChange={handleSliderRangeChange}
        data={miniChartData}
        showMiniChart={navState.showMiniChart}
        snapToMarketHours={navState.snapToMarketHours}
        showMarketHours={true}
      />

      {/* Metrics Panel */}
      {navState.showMetrics && (
        <RealDataMetricsPanel
          data={data}
          currentTick={currentTick}
        />
      )}
    </div>
  );

  const renderSidebarLayout = () => (
    <div className="w-80 space-y-4">
      <div className="bg-slate-900/50 backdrop-blur border border-slate-600/30 rounded-xl p-4">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Navigation className="h-5 w-5" />
          Navegación
        </h3>

        <div className="space-y-4">
          <EnhancedTimeRangeSelector
            value={navState.timeRange}
            onChange={handleTimeRangeChange}
            minDate={minDate}
            maxDate={maxDate}
            showTimeframe={false}
            showCustomRange={false}
          />

          {navState.showMiniChart && miniChartData.length > 0 && (
            <div>
              <div className="text-sm text-slate-400 mb-2">Vista General</div>
              <div className="h-20 bg-slate-800/30 rounded">
                {/* Mini chart would be rendered here */}
              </div>
            </div>
          )}
        </div>
      </div>

      {navState.showMetrics && (
        <RealDataMetricsPanel
          data={data}
          currentTick={currentTick}
        />
      )}
    </div>
  );

  return (
    <div className={className}>
      {/* Settings Panel (Collapsible) */}
      <AnimatePresence>
        {navState.isExpanded && layout !== 'sidebar' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-6"
          >
            <div className="bg-slate-900/30 border border-slate-600/20 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-white font-medium flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Configuración de Navegación
                </h4>

                <button
                  onClick={toggleExpanded}
                  className="p-1 hover:bg-slate-700 rounded transition-colors text-slate-400"
                >
                  <ChevronUp className="h-4 w-4" />
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={navState.showMiniChart}
                    onChange={(e) => updateSettings({ showMiniChart: e.target.checked })}
                    className="rounded border-slate-600 bg-slate-700 text-cyan-600"
                  />
                  Mini Chart
                </label>

                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={navState.snapToMarketHours}
                    onChange={(e) => updateSettings({ snapToMarketHours: e.target.checked })}
                    className="rounded border-slate-600 bg-slate-700 text-cyan-600"
                  />
                  Snap a Horarios
                </label>

                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={navState.autoUpdate}
                    onChange={(e) => updateSettings({ autoUpdate: e.target.checked })}
                    className="rounded border-slate-600 bg-slate-700 text-cyan-600"
                  />
                  Auto Update
                </label>

                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input
                    type="checkbox"
                    checked={navState.showMetrics}
                    onChange={(e) => updateSettings({ showMetrics: e.target.checked })}
                    className="rounded border-slate-600 bg-slate-700 text-cyan-600"
                  />
                  Mostrar Métricas
                </label>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Indicator */}
      {isLoading && (
        <div className="mb-4 flex items-center gap-2 text-cyan-400 text-sm">
          <div className="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
          Cargando datos...
        </div>
      )}

      {/* Main Content */}
      {layout === 'compact' && renderCompactLayout()}
      {layout === 'full' && renderFullLayout()}
      {layout === 'sidebar' && renderSidebarLayout()}

      {/* Expanded Panel for Compact Layout */}
      <AnimatePresence>
        {layout === 'compact' && navState.isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 space-y-4"
          >
            {renderFullLayout()}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Status Bar */}
      <div className="mt-4 flex items-center justify-between text-xs text-slate-500">
        <div className="flex items-center gap-4">
          <span>
            Rango: {navState.timeRange.start.toLocaleDateString()} - {navState.timeRange.end.toLocaleDateString()}
          </span>
          <span>
            Timeframe: {navState.timeRange.timeframe}
          </span>
          <span>
            Datos: {data.length.toLocaleString()} registros
          </span>
        </div>

        <div className="flex items-center gap-2">
          {currentTick && (
            <span>
              Último: {currentTick.timestamp.toLocaleTimeString()}
            </span>
          )}
          <span>
            Calidad: {data.length > 0 ? '✓' : '⚠️'}
          </span>
        </div>
      </div>
    </div>
  );
}