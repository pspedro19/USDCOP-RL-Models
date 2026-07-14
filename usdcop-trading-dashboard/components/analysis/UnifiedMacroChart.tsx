'use client';

import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion } from 'framer-motion';
import { LineChart, ToggleLeft, ToggleRight } from 'lucide-react';
import type { MacroChartData, MacroChartPoint } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT, CHART } from './gm-analysis';

interface UnifiedMacroChartProps {
  charts: Record<string, MacroChartData>;
  onVariableClick?: (key: string) => void;
}

/** Per-series colors are chart-library props (lightweight-charts) — CHART palette, not Tailwind. */
const VARIABLE_CONFIG: Record<string, { label: string; color: string }> = {
  dxy:      { label: 'DXY',   color: CHART.info },
  vix:      { label: 'VIX',   color: CHART.neg },
  wti:      { label: 'WTI',   color: CHART.warn },
  embi_col: { label: 'EMBI',  color: CHART.violet },
  ust10y:   { label: '10Y',   color: CHART.pos },
  ibr:      { label: 'IBR',   color: CHART.pink },
  gold:     { label: 'Oro',   color: CHART.gold },
  brent:    { label: 'Brent', color: CHART.indigo },
};

const DEFAULT_SELECTED = ['dxy', 'vix', 'wti'];

export function UnifiedMacroChart({ charts, onVariableClick }: UnifiedMacroChartProps) {
  const t = useGmT(ANALYSIS_DICT);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof import('lightweight-charts').createChart> | null>(null);
  const seriesRef = useRef<Map<string, unknown>>(new Map());
  const observerRef = useRef<ResizeObserver | null>(null);

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
            textColor: CHART.text,
            fontSize: 11,
          },
          grid: {
            vertLines: { color: CHART.grid },
            horzLines: { color: CHART.grid },
          },
          crosshair: {
            mode: 0, // Normal
          },
          rightPriceScale: {
            borderColor: CHART.border,
          },
          timeScale: {
            borderColor: CHART.border,
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

        // Resize observer — tracked in a ref so the effect cleanup can disconnect it.
        // (Returning a cleanup from this async fn would be a lost Promise, so the
        // observer would otherwise leak and call applyOptions on a disposed chart →
        // "Object is disposed" on every asset switch.) Guard the callback too.
        observerRef.current?.disconnect();
        const observer = new ResizeObserver(entries => {
          if (chartRef.current !== chart) return; // chart was disposed/replaced
          for (const entry of entries) {
            try {
              chart.applyOptions({ width: entry.contentRect.width });
            } catch {
              observer.disconnect();
            }
          }
        });
        observer.observe(chartContainerRef.current);
        observerRef.current = observer;
      } catch (err) {
        console.warn('Failed to initialize lightweight-charts:', err);
      }
    };

    initChart();

    return () => {
      isMounted = false;
      observerRef.current?.disconnect();
      observerRef.current = null;
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
      className={`${GM.panel} gm-contain p-5`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2`}>
          <LineChart className={`w-4 h-4 ${GM.accent}`} />
          {t('macroTitle')}
        </h3>

        {/* Normalize toggle */}
        <button
          onClick={() => setNormalized(!normalized)}
          className={`flex items-center gap-1.5 ${GMT.meta} ${GM.textSec} hover:text-[var(--gm-text)] transition-colors duration-[var(--gm-dur-fast)] ${GM.focus} rounded`}
        >
          {normalized ? (
            <ToggleRight className={`w-4 h-4 ${GM.accent}`} />
          ) : (
            <ToggleLeft className="w-4 h-4" />
          )}
          {normalized ? t('normalized') : t('absolute')}
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
              aria-pressed={isSelected}
              className={`
                px-2.5 py-1 rounded-full ${GMT.meta} font-semibold transition-colors duration-[var(--gm-dur-fast)] ${GM.focus}
                ${isSelected
                  ? 'border-2'
                  : `border border-[rgba(148,163,184,.16)] ${GM.textMuted} hover:text-[var(--gm-text-sec)] hover:border-[rgba(148,163,184,.3)]`
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
