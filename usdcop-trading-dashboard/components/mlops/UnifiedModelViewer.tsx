'use client';

/**
 * Unified Model Viewer
 * ====================
 * Single component for both:
 * - Backtest Replay: Dynamic visualization passing data through L1+L5 inference
 * - Production NRT: Real-time visualization during market hours
 *
 * MLOps Features:
 * - Dynamic equity curve building bar-by-bar
 * - Price chart with signals overlay
 * - Floating approval panel for experiments
 * - Seamless transition between backtest and production modes
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Pause,
  RotateCcw,
  FastForward,
  Activity,
  TrendingUp,
  TrendingDown,
  Loader2,
  Calendar,
  Clock,
  Zap,
  BarChart3,
  Target,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid,
  ComposedChart,
  Line,
  Scatter,
} from 'recharts';
import { cn } from '@/lib/utils';
import { BacktestTradeEvent, BacktestResult } from '@/lib/contracts/backtest.contract';
import { PIPELINE_DATE_RANGES, getTestEndDate } from '@/lib/contracts/ssot.contract';
import { FloatingApprovalPanel, ApprovalMetrics } from './FloatingApprovalPanel';

// ============================================================================
// Types
// ============================================================================

export type ViewerMode = 'backtest' | 'production';

export interface EquityDataPoint {
  timestamp: string;
  time: string;
  equity: number;
  drawdown: number;
  position: 'LONG' | 'SHORT' | 'NEUTRAL';
  signal?: 'BUY' | 'SELL';
  price?: number;
}

export interface TradeMarker {
  timestamp: string;
  price: number;
  side: 'BUY' | 'SELL';
  pnl?: number;
}

export interface UnifiedModelViewerProps {
  /** Viewer mode */
  mode: ViewerMode;
  /** Model ID being visualized */
  modelId: string;
  /** Experiment name (for display) */
  experimentName?: string;
  /** Whether backtest is currently running */
  isRunning: boolean;
  /** Progress percentage (0-100) */
  progress: number;
  /** Equity curve data points */
  equityData: EquityDataPoint[];
  /** Trade markers for chart */
  trades: TradeMarker[];
  /** Current metrics summary */
  metrics?: ApprovalMetrics;
  /** Start date of backtest/session */
  startDate?: string;
  /** End date of backtest/session */
  endDate?: string;
  /** Playback speed (1x, 2x, 4x, 8x) */
  playbackSpeed?: number;
  /** Callback when playback speed changes */
  onSpeedChange?: (speed: number) => void;
  /** Callback when play/pause toggled */
  onPlayPause?: () => void;
  /** Callback when reset requested */
  onReset?: () => void;
  /** Whether to show approval panel */
  showApprovalPanel?: boolean;
  /** L4 recommendation */
  recommendation?: 'PROMOTE' | 'REJECT' | 'REVIEW';
  /** L4 confidence */
  confidence?: number;
  /** Proposal ID (if from experiment) */
  proposalId?: string;
  /** Callback when approved */
  onApprove?: (notes: string) => Promise<void>;
  /** Callback when rejected */
  onReject?: (reason: string) => Promise<void>;
  /** Is market currently open (for production mode) */
  isMarketOpen?: boolean;
}

// ============================================================================
// Speed Options
// ============================================================================

const SPEED_OPTIONS = [
  { value: 1, label: '1x' },
  { value: 2, label: '2x' },
  { value: 4, label: '4x' },
  { value: 8, label: '8x' },
  { value: 16, label: '16x' },
];

// ============================================================================
// Component
// ============================================================================

export function UnifiedModelViewer({
  mode,
  modelId,
  experimentName,
  isRunning,
  progress,
  equityData,
  trades,
  metrics,
  startDate,
  endDate,
  playbackSpeed = 1,
  onSpeedChange,
  onPlayPause,
  onReset,
  showApprovalPanel = false,
  recommendation = 'REVIEW',
  confidence = 0,
  proposalId,
  onApprove,
  onReject,
  isMarketOpen = false,
}: UnifiedModelViewerProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const currentEquity = useMemo(() => {
    if (equityData.length === 0) return 100000;
    return equityData[equityData.length - 1].equity;
  }, [equityData]);

  const initialEquity = useMemo(() => {
    if (equityData.length === 0) return 100000;
    return equityData[0].equity;
  }, [equityData]);

  const totalReturn = useMemo(() => {
    return (currentEquity - initialEquity) / initialEquity;
  }, [currentEquity, initialEquity]);

  const maxEquity = useMemo(() => {
    if (equityData.length === 0) return 100000;
    return Math.max(...equityData.map(d => d.equity));
  }, [equityData]);

  const currentDrawdown = useMemo(() => {
    return (maxEquity - currentEquity) / maxEquity;
  }, [maxEquity, currentEquity]);

  const maxDrawdown = useMemo(() => {
    if (equityData.length === 0) return 0;
    return Math.max(...equityData.map(d => d.drawdown));
  }, [equityData]);

  const tradesCount = trades.length;

  const winRate = useMemo(() => {
    const closedTrades = trades.filter(t => t.pnl !== undefined);
    if (closedTrades.length === 0) return 0;
    const wins = closedTrades.filter(t => (t.pnl || 0) > 0).length;
    return wins / closedTrades.length;
  }, [trades]);

  // Current position from latest data point
  const currentPosition = useMemo(() => {
    if (equityData.length === 0) return 'NEUTRAL';
    return equityData[equityData.length - 1].position;
  }, [equityData]);

  // ============================================================================
  // Chart Data Formatting
  // ============================================================================

  const chartData = useMemo(() => {
    return equityData.map((point, index) => ({
      ...point,
      index,
      formattedTime: point.time,
      equityFormatted: point.equity.toFixed(0),
      returnPct: ((point.equity - initialEquity) / initialEquity * 100).toFixed(2),
    }));
  }, [equityData, initialEquity]);

  // Buy/Sell markers for scatter plot
  const buyMarkers = useMemo(() => {
    return chartData.filter(d => d.signal === 'BUY');
  }, [chartData]);

  const sellMarkers = useMemo(() => {
    return chartData.filter(d => d.signal === 'SELL');
  }, [chartData]);

  // ============================================================================
  // Date Range Display
  // ============================================================================

  const dateRangeDisplay = useMemo(() => {
    const start = startDate || PIPELINE_DATE_RANGES.VALIDATION_START;
    const end = endDate || getTestEndDate();
    return `${start} → ${end}`;
  }, [startDate, endDate]);

  // ============================================================================
  // Handle Approval
  // ============================================================================

  const handleApprove = async (notes: string) => {
    if (onApprove) {
      await onApprove(notes);
    }
  };

  const handleReject = async (reason: string) => {
    if (onReject) {
      await onReject(reason);
    }
  };

  return (
    <div className="relative">
      {/* Main Viewer Container */}
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Mode Badge */}
            <div className={cn(
              'px-3 py-1.5 rounded-lg border text-sm font-bold flex items-center gap-2',
              mode === 'production'
                ? 'bg-green-500/20 text-green-400 border-green-500/50'
                : 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50'
            )}>
              {mode === 'production' ? (
                <>
                  <Activity className={cn('w-4 h-4', isMarketOpen && 'animate-pulse')} />
                  {isMarketOpen ? 'LIVE' : 'PRODUCCION'}
                </>
              ) : (
                <>
                  <BarChart3 className="w-4 h-4" />
                  BACKTEST REPLAY
                </>
              )}
            </div>

            {/* Date Range */}
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <Calendar className="w-4 h-4" />
              <span>{dateRangeDisplay}</span>
            </div>

            {/* Progress (when running) */}
            {isRunning && (
              <div className="flex items-center gap-2 text-cyan-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm font-mono">{progress.toFixed(1)}%</span>
              </div>
            )}
          </div>

          {/* Playback Controls (for backtest mode) */}
          {mode === 'backtest' && (
            <div className="flex items-center gap-3">
              {/* Speed Selector */}
              <div className="flex items-center gap-1 bg-gray-800/50 rounded-lg p-1">
                {SPEED_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => onSpeedChange?.(opt.value)}
                    className={cn(
                      'px-2 py-1 rounded text-xs font-bold transition-colors',
                      playbackSpeed === opt.value
                        ? 'bg-cyan-500 text-white'
                        : 'text-gray-400 hover:text-white'
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>

              {/* Play/Pause */}
              <button
                onClick={onPlayPause}
                className={cn(
                  'p-2 rounded-lg transition-colors',
                  isRunning
                    ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
                    : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                )}
              >
                {isRunning ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>

              {/* Reset */}
              <button
                onClick={onReset}
                className="p-2 rounded-lg bg-gray-800/50 text-gray-400 hover:text-white hover:bg-gray-700/50 transition-colors"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          )}
        </div>

        {/* Metrics Row */}
        <div className="px-6 py-3 border-b border-gray-800 grid grid-cols-2 md:grid-cols-6 gap-4">
          {/* Equity */}
          <div className="text-center">
            <p className="text-gray-500 text-xs">Equity</p>
            <p className="text-white text-lg font-bold font-mono">
              ${currentEquity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </p>
          </div>

          {/* Return */}
          <div className="text-center">
            <p className="text-gray-500 text-xs">Return</p>
            <p className={cn(
              'text-lg font-bold',
              totalReturn >= 0 ? 'text-green-400' : 'text-red-400'
            )}>
              {totalReturn >= 0 ? '+' : ''}{(totalReturn * 100).toFixed(2)}%
            </p>
          </div>

          {/* Drawdown */}
          <div className="text-center">
            <p className="text-gray-500 text-xs">Drawdown</p>
            <p className="text-red-400 text-lg font-bold">
              {(currentDrawdown * 100).toFixed(2)}%
            </p>
          </div>

          {/* Max DD */}
          <div className="text-center">
            <p className="text-gray-500 text-xs">Max DD</p>
            <p className="text-red-400 text-lg font-bold">
              {(maxDrawdown * 100).toFixed(2)}%
            </p>
          </div>

          {/* Trades */}
          <div className="text-center">
            <p className="text-gray-500 text-xs">Trades</p>
            <p className="text-amber-400 text-lg font-bold">{tradesCount}</p>
          </div>

          {/* Position */}
          <div className="text-center">
            <p className="text-gray-500 text-xs">Posicion</p>
            <p className={cn(
              'text-lg font-bold',
              currentPosition === 'LONG' ? 'text-green-400' :
              currentPosition === 'SHORT' ? 'text-red-400' : 'text-gray-400'
            )}>
              {currentPosition}
            </p>
          </div>
        </div>

        {/* Equity Chart */}
        <div ref={chartRef} className="h-80 p-4">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                  </linearGradient>
                </defs>

                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

                <XAxis
                  dataKey="formattedTime"
                  stroke="#6B7280"
                  fontSize={10}
                  tickLine={false}
                  interval="preserveStartEnd"
                />

                <YAxis
                  stroke="#6B7280"
                  fontSize={10}
                  tickLine={false}
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                  domain={['dataMin - 1000', 'dataMax + 1000']}
                />

                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  labelStyle={{ color: '#9CA3AF' }}
                  formatter={(value: number, name: string) => {
                    if (name === 'equity') return [`$${value.toLocaleString()}`, 'Equity'];
                    return [value, name];
                  }}
                />

                {/* Equity Area */}
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#10B981"
                  strokeWidth={2}
                  fill="url(#equityGradient)"
                  animationDuration={300}
                />

                {/* Initial equity reference line */}
                <ReferenceLine
                  y={initialEquity}
                  stroke="#6B7280"
                  strokeDasharray="3 3"
                  label={{
                    value: 'Inicio',
                    fill: '#6B7280',
                    fontSize: 10,
                    position: 'right',
                  }}
                />

                {/* Buy markers */}
                <Scatter
                  data={buyMarkers}
                  dataKey="equity"
                  fill="#10B981"
                  shape={(props: any) => (
                    <circle cx={props.cx} cy={props.cy} r={4} fill="#10B981" stroke="#fff" strokeWidth={1} />
                  )}
                />

                {/* Sell markers */}
                <Scatter
                  data={sellMarkers}
                  dataKey="equity"
                  fill="#EF4444"
                  shape={(props: any) => (
                    <circle cx={props.cx} cy={props.cy} r={4} fill="#EF4444" stroke="#fff" strokeWidth={1} />
                  )}
                />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center text-gray-500">
                <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>{isRunning ? 'Cargando datos...' : 'Inicia el replay para ver la curva de equity'}</p>
              </div>
            </div>
          )}
        </div>

        {/* Recent Trades Mini-Table */}
        {trades.length > 0 && (
          <div className="px-6 py-4 border-t border-gray-800">
            <h4 className="text-sm font-semibold text-gray-400 mb-3">Trades Recientes</h4>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {trades.slice(-5).reverse().map((trade, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between text-sm bg-gray-800/30 rounded-lg px-3 py-2"
                >
                  <div className="flex items-center gap-3">
                    <span className={cn(
                      'px-2 py-0.5 rounded text-xs font-bold',
                      trade.side === 'BUY'
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-red-500/20 text-red-400'
                    )}>
                      {trade.side}
                    </span>
                    <span className="text-gray-400">{trade.timestamp}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-white font-mono">${trade.price.toFixed(2)}</span>
                    {trade.pnl !== undefined && (
                      <span className={cn(
                        'font-mono',
                        trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      )}>
                        {trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Floating Approval Panel */}
      {showApprovalPanel && onApprove && onReject && (
        <FloatingApprovalPanel
          modelId={modelId}
          experimentName={experimentName}
          recommendation={recommendation}
          confidence={confidence}
          metrics={metrics}
          isBacktestRunning={isRunning}
          backestProgress={progress}
          onApprove={handleApprove}
          onReject={handleReject}
          visible={true}
          position="bottom"
          isExperimentProposal={!!proposalId}
          proposalId={proposalId}
        />
      )}
    </div>
  );
}

export default UnifiedModelViewer;
