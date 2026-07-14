'use client';

import { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import type { MacroChartPoint } from '@/lib/contracts/weekly-analysis.contract';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT, CHART } from './gm-analysis';

interface MacroVariableChartProps {
  data: MacroChartPoint[];
  variableName: string;
  className?: string;
}

export function MacroVariableChart({ data, variableName, className = '' }: MacroVariableChartProps) {
  const t = useGmT(ANALYSIS_DICT);
  const { chartData, yDomain } = useMemo(() => {
    // Get valid values
    const validValues = data
      .map(d => d.value)
      .filter((v): v is number => v !== null && v !== undefined && !isNaN(v));

    if (validValues.length === 0) return { chartData: [], yDomain: [0, 1] as [number, number] };

    // IQR-based outlier detection for price-like series
    const sorted = [...validValues].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const median = sorted[Math.floor(sorted.length * 0.5)];
    const isPriceSeries = Math.abs(median) > 1;

    // For price series, filter using IQR fences (generous: 2.5x IQR)
    const lowerFence = isPriceSeries ? q1 - 2.5 * iqr : -Infinity;
    const upperFence = isPriceSeries ? q3 + 2.5 * iqr : Infinity;

    const isOutlier = (v: number | null | undefined): boolean => {
      if (v === null || v === undefined || isNaN(v as number)) return true;
      if (isPriceSeries && ((v as number) < lowerFence || (v as number) > upperFence)) return true;
      return false;
    };

    const cleaned = data
      .filter(d => !isOutlier(d.value))
      .map(d => ({
        ...d,
        bb_upper: isOutlier(d.bb_upper) ? null : d.bb_upper,
        bb_lower: isOutlier(d.bb_lower) ? null : d.bb_lower,
        sma20: isOutlier(d.sma20) ? null : d.sma20,
        dateStr: d.date.slice(5), // "MM-DD"
      }));

    // Compute Y-axis domain from value + BB bands (only clean data)
    const allCleanValues = cleaned.flatMap(d =>
      [d.value, d.bb_upper, d.bb_lower, d.sma20].filter(
        (v): v is number => v !== null && v !== undefined
      )
    );
    const minVal = Math.min(...allCleanValues);
    const maxVal = Math.max(...allCleanValues);
    const padding = (maxVal - minVal) * 0.05 || 1;

    return {
      chartData: cleaned,
      yDomain: [minVal - padding, maxVal + padding] as [number, number],
    };
  }, [data]);

  if (!chartData.length) {
    return (
      <div className={`flex items-center justify-center h-48 ${GM.textMuted} ${GMT.body} ${className}`}>
        {t('noChartData')}
      </div>
    );
  }

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
          <XAxis
            dataKey="dateStr"
            tick={{ fill: CHART.tick, fontSize: 10 }}
            interval="preserveStartEnd"
            tickLine={false}
          />
          <YAxis
            tick={{ fill: CHART.tick, fontSize: 10 }}
            tickLine={false}
            domain={yDomain}
            width={55}
            allowDataOverflow
            tickFormatter={(v: number) => {
              if (Math.abs(v) >= 1000) return v.toFixed(0);
              if (Math.abs(v) >= 10) return v.toFixed(1);
              return v.toFixed(2);
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: CHART.tooltipBg,
              border: `1px solid ${CHART.border}`,
              borderRadius: '10px',
              fontSize: '12px',
            }}
            labelStyle={{ color: CHART.text }}
          />
          <Legend
            wrapperStyle={{ fontSize: '11px' }}
            iconSize={8}
          />

          {/* Bollinger bands fill */}
          <Area
            dataKey="bb_upper"
            stroke="none"
            fill={CHART.tick}
            fillOpacity={0.08}
            name="BB Upper"
            legendType="none"
          />
          <Area
            dataKey="bb_lower"
            stroke="none"
            fill={CHART.maskFill}
            fillOpacity={1}
            name="BB Lower"
            legendType="none"
          />

          {/* SMA-20 */}
          <Line
            dataKey="sma20"
            stroke={CHART.accent}
            strokeWidth={1}
            strokeDasharray="4 4"
            dot={false}
            name="SMA 20"
            connectNulls
          />

          {/* Main price line */}
          <Line
            dataKey="value"
            stroke={CHART.line}
            strokeWidth={1.5}
            dot={false}
            name={variableName}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
