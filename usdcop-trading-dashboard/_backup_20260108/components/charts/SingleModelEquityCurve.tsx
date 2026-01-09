'use client';

/**
 * SingleModelEquityCurve Component
 * =================================
 * Displays the equity curve for a SINGLE selected model.
 * Simplified version that doesn't show multiple models overlaid.
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { useSelectedModel } from '@/contexts/ModelContext';
import { useEquityCurveStream } from '@/hooks/useEquityCurveStream';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, Loader2, AlertCircle } from 'lucide-react';

interface SingleModelEquityCurveProps {
  className?: string;
  height?: number;
}

const INITIAL_CAPITAL = 10000;

export function SingleModelEquityCurve({
  className = '',
  height = 300,
}: SingleModelEquityCurveProps) {
  const { model, modelId } = useSelectedModel();
  const { data, isConnected, connectionType, error } = useEquityCurveStream({
    strategies: modelId ? [modelId] : [],
    enabled: true,
  });

  // Get data for the selected model
  const chartData = useMemo(() => {
    if (!modelId || !data) return [];

    // Map model ID to data key
    const dataKey = modelId.toUpperCase().replace(/_/g, '_');
    const modelData = data[dataKey] || data[modelId] || [];

    // If no specific model data, try to find any matching key
    if (modelData.length === 0) {
      const keys = Object.keys(data);
      const matchingKey = keys.find((k) =>
        k.toLowerCase().includes(modelId.toLowerCase().replace('_', ''))
      );
      if (matchingKey && data[matchingKey]) {
        return data[matchingKey].map((point) => ({
          timestamp: point.timestamp,
          equity: point.equity_value,
          return_pct: point.return_pct,
        }));
      }

      // If still no data, use first available dataset
      if (keys.length > 0 && data[keys[0]]) {
        return data[keys[0]].map((point) => ({
          timestamp: point.timestamp,
          equity: point.equity_value,
          return_pct: point.return_pct,
        }));
      }
    }

    return modelData.map((point) => ({
      timestamp: point.timestamp,
      equity: point.equity_value,
      return_pct: point.return_pct,
    }));
  }, [data, modelId]);

  // Calculate stats
  const stats = useMemo(() => {
    if (chartData.length === 0) {
      return { current: INITIAL_CAPITAL, profit: 0, profitPct: 0, max: INITIAL_CAPITAL, min: INITIAL_CAPITAL };
    }

    const current = chartData[chartData.length - 1]?.equity || INITIAL_CAPITAL;
    const profit = current - INITIAL_CAPITAL;
    const profitPct = (profit / INITIAL_CAPITAL) * 100;
    const values = chartData.map((d) => d.equity);
    const max = Math.max(...values);
    const min = Math.min(...values);

    return { current, profit, profitPct, max, min };
  }, [chartData]);

  // Format timestamp for X axis
  const formatXAxis = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    const date = new Date(label);

    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="text-xs text-slate-400 mb-2">
          {date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
        <p className="text-sm font-mono text-white">
          Equity: ${data.equity?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </p>
        <p
          className={`text-xs font-mono ${
            data.return_pct >= 0 ? 'text-green-400' : 'text-red-400'
          }`}
        >
          Return: {data.return_pct >= 0 ? '+' : ''}{data.return_pct?.toFixed(2)}%
        </p>
      </div>
    );
  };

  // Loading state
  if (!isConnected && chartData.length === 0) {
    return (
      <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm font-medium text-slate-400">
            Equity Curve
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4 flex items-center justify-center" style={{ height }}>
          <div className="flex flex-col items-center gap-2 text-slate-500">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span className="text-sm">Loading equity data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm font-medium text-slate-400">
            Equity Curve
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4 flex items-center justify-center" style={{ height }}>
          <div className="flex flex-col items-center gap-2 text-red-400">
            <AlertCircle className="w-6 h-6" />
            <span className="text-sm">{error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No data state
  if (chartData.length === 0) {
    return (
      <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm font-medium text-slate-400">
            Equity Curve
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4 flex items-center justify-center" style={{ height }}>
          <div className="flex flex-col items-center gap-2 text-slate-500">
            <TrendingUp className="w-6 h-6" />
            <span className="text-sm">No equity data available</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const modelColor = model?.color || '#10B981';
  const gradientId = `equity-gradient-${modelId || 'default'}`;

  return (
    <Card className={`bg-slate-900/50 border-slate-800 ${className}`}>
      <CardHeader className="py-3 px-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium text-slate-400 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" style={{ color: modelColor }} />
            Equity Curve
            {model && (
              <span
                className="text-xs px-2 py-0.5 rounded"
                style={{ backgroundColor: `${modelColor}20`, color: modelColor }}
              >
                {model.name}
              </span>
            )}
          </CardTitle>

          {/* Current equity and return */}
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs text-slate-500">Current</div>
              <div className="font-mono text-sm text-white">
                ${stats.current.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500">Return</div>
              <div
                className={`font-mono text-sm font-bold ${
                  stats.profitPct >= 0 ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {stats.profitPct >= 0 ? '+' : ''}
                {stats.profitPct.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="px-2 pb-2">
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={modelColor} stopOpacity={0.3} />
                <stop offset="95%" stopColor={modelColor} stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />

            <XAxis
              dataKey="timestamp"
              tickFormatter={formatXAxis}
              stroke="#475569"
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />

            <YAxis
              domain={[
                (dataMin: number) => Math.min(dataMin, INITIAL_CAPITAL) - 100,
                (dataMax: number) => Math.max(dataMax, INITIAL_CAPITAL) + 100,
              ]}
              stroke="#475569"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Reference line at initial capital */}
            <ReferenceLine
              y={INITIAL_CAPITAL}
              stroke="#64748b"
              strokeDasharray="5 5"
              strokeWidth={1}
              label={{
                value: 'Initial',
                position: 'right',
                fill: '#64748b',
                fontSize: 10,
              }}
            />

            <Area
              type="monotone"
              dataKey="equity"
              stroke={modelColor}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              dot={false}
              activeDot={{ r: 4, fill: modelColor }}
            />
          </AreaChart>
        </ResponsiveContainer>

        {/* Connection status indicator */}
        <div className="flex items-center justify-end gap-2 px-2 pt-1">
          <span
            className={`w-2 h-2 rounded-full ${
              connectionType === 'sse'
                ? 'bg-green-500'
                : connectionType === 'polling'
                ? 'bg-yellow-500'
                : 'bg-red-500'
            }`}
          />
          <span className="text-xs text-slate-500">
            {connectionType === 'sse' ? 'Live' : connectionType === 'polling' ? 'Polling' : 'Disconnected'}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default SingleModelEquityCurve;
