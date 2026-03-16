'use client';

import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { LineChart, ToggleLeft, ToggleRight } from 'lucide-react';
import type { MacroChartData, MacroChartPoint } from '@/lib/contracts/weekly-analysis.contract';

interface UnifiedMacroChartProps {
  charts: Record<string, MacroChartData>;
  onVariableClick?: (key: string) => void;
}

const VARIABLE_CONFIG: Record<string, { label: string; color: string }> = {
  dxy:      { label: 'DXY',   color: '#3B82F6' },
  vix:      { label: 'VIX',   color: '#EF4444' },
  wti:      { label: 'WTI',   color: '#F59E0B' },
  embi_col: { label: 'EMBI',  color: '#8B5CF6' },
  ust10y:   { label: '10Y',   color: '#10B981' },
  ibr:      { label: 'IBR',   color: '#EC4899' },
  gold:     { label: 'Oro',   color: '#D4AF37' },
  brent:    { label: 'Brent', color: '#6366F1' },
};

const DEFAULT_SELECTED = ['dxy', 'vix', 'wti'];

export function UnifiedMacroChart({ charts, onVariableClick }: UnifiedMacroChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof import('lightweight-charts').createChart> | null>(null);
  const seriesRef = useRef<Map<string, unknown>>(new Map());

  const [selected, setSelected] = useState<Set<string>>(new Set(DEFAULT_SELECTED));
  const [normalized, setNormalized] = useState(false);

  const availableVars = useMemo(() =>
    Object.keys(charts).filter(k => k in VARIABLE_CONFIG && charts[k]?.data?.length > 0),
    [charts]
  );

  const toggleVariable = useCallback((key: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  // Normalize data to z-scores if enabled
  const processData = useCallback((data: MacroChartPoint[], shouldNormalize: boolean) => {
    const values = data.map(d => d.value).filter((v): v is number => v !== null);
    if (values.length === 0) return [];

    if (!shouldNormalize) {
      return data
        .filter(d => d.value !== null)
        .map(d => ({ time: d.date, value: d.value as number }));
    }

    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length) || 1;

    return data
      .filter(d => d.value !== null)
      .map(d => ({
        time: d.date,
        value: ((d.value as number) - mean) / std,
      }));
  }, []);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    let isMounted = true;

    const initChart = async () => {
      try {
        const lc = await import('lightweight-charts');
        const { createChart, ColorType, LineSeries } = lc;

        if (!isMounted || !chartContainerRef.current) return;

        // Dispose existing
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
          seriesRef.current.clear();
        }

        const chart = createChart(chartContainerRef.current, {
          layout: {
            background: { type: ColorType.Solid, color: 'transparent' },
            textColor: '#9CA3AF',
            fontSize: 11,
          },
          grid: {
            vertLines: { color: 'rgba(75, 85, 99, 0.15)' },
            horzLines: { color: 'rgba(75, 85, 99, 0.15)' },
          },
          crosshair: {
            mode: 0, // Normal
          },
          rightPriceScale: {
            borderColor: 'rgba(75, 85, 99, 0.3)',
          },
          timeScale: {
            borderColor: 'rgba(75, 85, 99, 0.3)',
            timeVisible: false,
          },
          width: chartContainerRef.current.clientWidth,
          height: 320,
        });

        chartRef.current = chart;

        // Add series for each selected variable (v5 API: addSeries)
        for (const varKey of selected) {
          const chartData = charts[varKey];
          if (!chartData?.data?.length) continue;

          const config = VARIABLE_CONFIG[varKey];
          if (!config) continue;

          const series = chart.addSeries(LineSeries, {
            color: config.color,
            lineWidth: 2,
            title: config.label,
            priceLineVisible: false,
            lastValueVisible: true,
          });

          const processedData = processData(chartData.data, normalized);
          if (processedData.length > 0) {
            series.setData(processedData as { time: string; value: number }[]);
          }

          seriesRef.current.set(varKey, series);
        }

        chart.timeScale().fitContent();

        // Resize observer
        const observer = new ResizeObserver(entries => {
          for (const entry of entries) {
            chart.applyOptions({ width: entry.contentRect.width });
          }
        });
        observer.observe(chartContainerRef.current);

        return () => {
          observer.disconnect();
        };
      } catch (err) {
        console.warn('Failed to initialize lightweight-charts:', err);
      }
    };

    initChart();

    return () => {
      isMounted = false;
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        seriesRef.current.clear();
      }
    };
  }, [charts, selected, normalized, processData]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-5"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <LineChart className="w-4 h-4 text-cyan-400" />
          Indicadores Macro
        </h3>

        {/* Normalize toggle */}
        <button
          onClick={() => setNormalized(!normalized)}
          className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-300 transition-colors"
        >
          {normalized ? (
            <ToggleRight className="w-4 h-4 text-cyan-400" />
          ) : (
            <ToggleLeft className="w-4 h-4" />
          )}
          {normalized ? 'Normalizado' : 'Absoluto'}
        </button>
      </div>

      {/* Variable toggle pills */}
      <div className="flex flex-wrap gap-2 mb-4">
        {availableVars.map(varKey => {
          const config = VARIABLE_CONFIG[varKey];
          if (!config) return null;
          const isSelected = selected.has(varKey);

          return (
            <button
              key={varKey}
              onClick={() => toggleVariable(varKey)}
              className={`
                px-2.5 py-1 rounded-full text-xs font-medium transition-all
                ${isSelected
                  ? 'border-2 text-white'
                  : 'border border-gray-700 text-gray-500 hover:text-gray-400 hover:border-gray-600'
                }
              `}
              style={isSelected ? {
                borderColor: config.color,
                backgroundColor: `${config.color}20`,
                color: config.color,
              } : undefined}
            >
              {config.label}
            </button>
          );
        })}
      </div>

      {/* Chart container */}
      <div
        ref={chartContainerRef}
        className="w-full rounded-lg overflow-hidden"
        style={{ minHeight: 320 }}
        onClick={() => {
          // Click on chart → open detail for first selected variable
          const firstSelected = [...selected][0];
          if (firstSelected && onVariableClick) {
            onVariableClick(firstSelected);
          }
        }}
      />
    </motion.div>
  );
}
