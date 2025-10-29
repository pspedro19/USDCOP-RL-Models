'use client';

import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Download, Eye, EyeOff } from 'lucide-react';
import { motion } from 'framer-motion';

interface EquityCurveChartProps {
  data: Array<{
    timestamp: string;
    RL_PPO: number;
    ML_LGBM: number;
    ML_XGB: number | null;
    LLM_CLAUDE: number | null;
    PORTFOLIO: number;
    CAPITAL_INICIAL: number;
  }>;
}

type TimeRange = 'ALL' | '72H' | '24H' | 'TODAY';

export default function EquityCurveChart({ data }: EquityCurveChartProps) {
  // ========== STATE ==========
  const [timeRange, setTimeRange] = useState<TimeRange>('24H');
  const [visibleLines, setVisibleLines] = useState({
    RL_PPO: true,
    ML_LGBM: true,
    ML_XGB: true,
    LLM_CLAUDE: true,
    PORTFOLIO: true,
    BASELINE: true
  });

  // ========== FILTERED DATA BY TIME RANGE ==========
  const filteredData = useMemo(() => {
    if (data.length === 0) return [];

    const now = new Date();
    let cutoffTime: Date;

    switch (timeRange) {
      case 'TODAY':
        cutoffTime = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0);
        break;
      case '24H':
        cutoffTime = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        break;
      case '72H':
        cutoffTime = new Date(now.getTime() - 72 * 60 * 60 * 1000);
        break;
      case 'ALL':
      default:
        return data;
    }

    return data.filter(d => new Date(d.timestamp) >= cutoffTime);
  }, [data, timeRange]);

  // Flag for data availability
  const hasRealData = data.length > 0;

  // ========== LATEST VALUES & METRICS ==========
  const latestValues = useMemo(() => {
    if (filteredData.length === 0) return null;

    const latest = filteredData[filteredData.length - 1];
    const initial = filteredData[0];

    const calculateReturn = (current: number, initial: number) => {
      return ((current - initial) / initial) * 100;
    };

    return {
      RL_PPO: {
        value: latest.RL_PPO || 10000,
        return: calculateReturn(latest.RL_PPO || 10000, initial.RL_PPO || 10000)
      },
      ML_LGBM: {
        value: latest.ML_LGBM || 10000,
        return: calculateReturn(latest.ML_LGBM || 10000, initial.ML_LGBM || 10000)
      },
      ML_XGB: {
        value: latest.ML_XGB || 10000,
        return: calculateReturn(latest.ML_XGB || 10000, initial.ML_XGB || 10000)
      },
      LLM_CLAUDE: {
        value: latest.LLM_CLAUDE || 10000,
        return: calculateReturn(latest.LLM_CLAUDE || 10000, initial.LLM_CLAUDE || 10000)
      },
      PORTFOLIO: {
        value: latest.PORTFOLIO || 40000,
        return: calculateReturn(latest.PORTFOLIO || 40000, initial.PORTFOLIO || 40000)
      }
    };
  }, [filteredData]);

  // ========== FORMATTERS ==========
  const formatCurrency = (value: number) => {
    if (value >= 1000) {
      return `$${(value / 1000).toFixed(1)}k`;
    }
    return `$${value.toFixed(0)}`;
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // ========== CUSTOM TOOLTIP ==========
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900/95 backdrop-blur border border-cyan-500/30 rounded-lg p-3 shadow-xl">
          <p className="text-slate-400 text-xs font-mono mb-2 border-b border-slate-700 pb-2">
            {formatDate(label)}
          </p>
          {payload.map((entry: any, index: number) => {
            if (!entry.value || entry.dataKey === 'CAPITAL_INICIAL') return null;

            const initialValue = payload[0]?.payload?.CAPITAL_INICIAL || 10000;
            const returnPct = ((entry.value - initialValue) / initialValue) * 100;

            return (
              <div key={index} className="flex items-center justify-between gap-6 text-sm py-1">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                  <span className="font-mono text-xs" style={{ color: entry.color }}>
                    {entry.name}:
                  </span>
                </div>
                <div className="flex flex-col items-end">
                  <span className="font-mono font-bold text-white text-sm">
                    ${entry.value.toFixed(2)}
                  </span>
                  <span className={`font-mono text-xs ${returnPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {returnPct >= 0 ? '+' : ''}{returnPct.toFixed(2)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      );
    }
    return null;
  };

  // ========== TOGGLE LINE VISIBILITY ==========
  const toggleLine = (line: keyof typeof visibleLines) => {
    setVisibleLines(prev => ({ ...prev, [line]: !prev[line] }));
  };

  // ========== EXPORT TO CSV ==========
  const exportToCSV = () => {
    const headers = ['Timestamp', 'RL_PPO', 'ML_LGBM', 'ML_XGB', 'LLM_CLAUDE', 'PORTFOLIO', 'BASELINE'];
    const rows = filteredData.map(d => [
      d.timestamp,
      d.RL_PPO,
      d.ML_LGBM,
      d.ML_XGB || '',
      d.LLM_CLAUDE || '',
      d.PORTFOLIO,
      d.CAPITAL_INICIAL
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `equity_curves_${timeRange}_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  // ========== HANDLE EMPTY DATA ==========
  // Create placeholder data to show axes even when no real data
  const hasData = data.length > 0;
  const displayData = hasData ? filteredData : [
    {
      timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      RL_PPO: 10000,
      ML_LGBM: 10000,
      ML_XGB: 10000,
      LLM_CLAUDE: 10000,
      PORTFOLIO: 40000,
      CAPITAL_INICIAL: 10000
    },
    {
      timestamp: new Date().toISOString(),
      RL_PPO: 10000,
      ML_LGBM: 10000,
      ML_XGB: 10000,
      LLM_CLAUDE: 10000,
      PORTFOLIO: 40000,
      CAPITAL_INICIAL: 10000
    }
  ];

  // ========== MAIN RENDER ==========
  return (
    <Card className="bg-slate-900 border-cyan-500/20 shadow-xl">
      <CardHeader className="space-y-4">

        {/* Title Row */}
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-cyan-500 font-mono flex items-center gap-2 text-xl">
              <TrendingUp className="h-6 w-6 text-green-400" />
              EQUITY CURVES - ALPHA ARENA
            </CardTitle>
            <p className="text-slate-400 text-sm mt-1">
              {hasRealData
                ? `Real-time performance comparison • ${filteredData.length} data points`
                : 'No data available - Showing empty chart structure'}
            </p>
          </div>

          <button
            onClick={exportToCSV}
            className="flex items-center gap-2 px-3 py-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-600/50 rounded-lg text-slate-300 text-sm font-mono transition-all"
            title="Export to CSV"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
        </div>

        {/* Toolbar: Time Range + Line Toggles */}
        <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-800/30 rounded-lg p-4 border border-slate-700/50">

          {/* Time Range Buttons */}
          <div className="flex items-center gap-2">
            {(['ALL', '72H', '24H', 'TODAY'] as TimeRange[]).map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-4 py-1.5 rounded-md text-xs font-mono font-bold transition-all ${
                  timeRange === range
                    ? 'bg-cyan-500 text-slate-950'
                    : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-slate-200'
                }`}
              >
                {range}
              </button>
            ))}
          </div>

          {/* Line Toggle Checkboxes */}
          <div className="flex flex-wrap items-center gap-3">
            {[
              { key: 'RL_PPO', label: 'RL', color: 'bg-blue-500' },
              { key: 'ML_LGBM', label: 'ML-LGBM', color: 'bg-purple-500' },
              { key: 'ML_XGB', label: 'ML-XGB', color: 'bg-orange-500' },
              { key: 'LLM_CLAUDE', label: 'LLM', color: 'bg-amber-600' },
              { key: 'PORTFOLIO', label: 'Portfolio', color: 'bg-white' },
              { key: 'BASELINE', label: 'Baseline', color: 'bg-gray-400' }
            ].map(({ key, label, color }) => (
              <button
                key={key}
                onClick={() => toggleLine(key as keyof typeof visibleLines)}
                className={`flex items-center gap-2 px-2 py-1 rounded text-xs font-mono transition-all ${
                  visibleLines[key as keyof typeof visibleLines]
                    ? 'bg-slate-700/50 text-white'
                    : 'bg-slate-800/30 text-slate-500'
                }`}
              >
                {visibleLines[key as keyof typeof visibleLines] ? (
                  <Eye className="h-3 w-3" />
                ) : (
                  <EyeOff className="h-3 w-3" />
                )}
                <div className={`w-2 h-2 rounded-full ${color}`} />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Current Values Summary */}
        {hasRealData && latestValues && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { key: 'RL_PPO', label: 'RL PPO', color: 'text-blue-400', bgColor: 'bg-blue-500/10' },
              { key: 'ML_LGBM', label: 'ML LGBM', color: 'text-purple-400', bgColor: 'bg-purple-500/10' },
              { key: 'ML_XGB', label: 'ML XGB', color: 'text-orange-400', bgColor: 'bg-orange-500/10' },
              { key: 'LLM_CLAUDE', label: 'LLM', color: 'text-amber-400', bgColor: 'bg-amber-500/10' },
              { key: 'PORTFOLIO', label: 'Portfolio', color: 'text-cyan-400', bgColor: 'bg-cyan-500/10' }
            ].map(({ key, label, color, bgColor }) => {
              const values = latestValues[key as keyof typeof latestValues];
              return (
                <div key={key} className={`${bgColor} rounded-lg p-3 border border-slate-700/50`}>
                  <p className="text-slate-500 text-xs font-mono">{label}</p>
                  <p className={`${color} text-lg font-bold font-mono mt-1`}>
                    ${values.value.toFixed(0)}
                  </p>
                  <Badge className={`mt-1 text-xs ${values.return >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {values.return >= 0 ? '+' : ''}{values.return.toFixed(2)}%
                  </Badge>
                </div>
              );
            })}
          </div>
        )}
      </CardHeader>

      <CardContent>
        {/* Warning Banner for No Data */}
        {!hasRealData && (
          <div className="mb-4 bg-yellow-950/20 border border-yellow-500/30 rounded p-3">
            <p className="text-yellow-400 text-sm font-mono">
              ⚠️ No data • Execute DAGs: L5-Serving, L6-Backtest
            </p>
          </div>
        )}

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="relative"
        >
          <ResponsiveContainer width="100%" height={500}>
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />

              <XAxis
                dataKey="timestamp"
                stroke="#94a3b8"
                fontSize={11}
                tickFormatter={formatDate}
                tick={{ fill: '#94a3b8' }}
                angle={-20}
                textAnchor="end"
                height={60}
              />

              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                tickFormatter={formatCurrency}
                tick={{ fill: '#94a3b8' }}
                domain={['dataMin - 500', 'dataMax + 500']}
              />

              <Tooltip content={<CustomTooltip />} />

              <Legend
                wrapperStyle={{
                  paddingTop: '20px'
                }}
                iconType="line"
              />

              {/* Baseline - Línea punteada gris */}
              {visibleLines.BASELINE && (
                <Line
                  type="monotone"
                  dataKey="CAPITAL_INICIAL"
                  stroke="#9CA3AF"
                  strokeWidth={1.5}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Baseline"
                  animationDuration={300}
                />
              )}

              {/* RL_PPO - Línea azul */}
              {visibleLines.RL_PPO && (
                <Line
                  type="monotone"
                  dataKey="RL_PPO"
                  stroke="#3B82F6"
                  strokeWidth={2.5}
                  dot={false}
                  name="🤖 RL PPO"
                  animationDuration={1000}
                />
              )}

              {/* ML_LGBM - Línea púrpura */}
              {visibleLines.ML_LGBM && (
                <Line
                  type="monotone"
                  dataKey="ML_LGBM"
                  stroke="#A855F7"
                  strokeWidth={2.5}
                  dot={false}
                  name="📊 ML LightGBM"
                  animationDuration={1000}
                />
              )}

              {/* ML_XGB - Línea naranja */}
              {visibleLines.ML_XGB && (
                <Line
                  type="monotone"
                  dataKey="ML_XGB"
                  stroke="#F97316"
                  strokeWidth={2.5}
                  dot={false}
                  name="📊 ML XGBoost"
                  animationDuration={1000}
                />
              )}

              {/* LLM_CLAUDE - Línea verde */}
              {visibleLines.LLM_CLAUDE && (
                <Line
                  type="monotone"
                  dataKey="LLM_CLAUDE"
                  stroke="#22C55E"
                  strokeWidth={2.5}
                  dot={false}
                  name="🧠 LLM Claude"
                  animationDuration={1000}
                />
              )}

              {/* PORTFOLIO - Línea blanca gruesa */}
              {visibleLines.PORTFOLIO && (
                <Line
                  type="monotone"
                  dataKey="PORTFOLIO"
                  stroke="#FFFFFF"
                  strokeWidth={3}
                  dot={false}
                  name="Portfolio Total"
                  animationDuration={1000}
                />
              )}
            </LineChart>
          </ResponsiveContainer>

          {/* Overlay when no real data */}
          {!hasRealData && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-900/70 backdrop-blur-sm rounded-lg pointer-events-none">
              <p className="text-slate-400 font-mono text-sm">
                Awaiting data from L5-L6 pipeline
              </p>
            </div>
          )}
        </motion.div>
      </CardContent>
    </Card>
  );
}
