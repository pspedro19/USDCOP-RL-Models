'use client';

/**
 * ForecastingBacktestSection — Self-contained 2025 Backtest Viewer + Approval
 * ============================================================================
 * Renders inside /dashboard BEFORE the RL sections.
 * Fetches file-based JSON on mount (no ModelContext dependency).
 *
 * Sections:
 * 1. Header + StatusBadge
 * 2. KPI cards (5)
 * 3. Statistical significance badge
 * 4. Candlestick chart with trade markers
 * 5. Interactive Approval Panel (Vote 2)
 * 6. Gates Panel
 * 7. Exit reasons + Trade table
 *
 * Spec: .claude/rules/sdd-dashboard-integration.md
 * Contract: lib/contracts/strategy.contract.ts
 */

import { useSession } from 'next-auth/react';
import { MetricBadge } from '@/components/ui/MetricBadge';
import { calmarRatio } from '@/lib/contracts/ui.contract';
import { useState, useEffect, useCallback, useRef, useMemo, lazy, Suspense } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp, TrendingDown, Activity, Target, BarChart3,
  Shield, CheckCircle2, XCircle, AlertTriangle, ChevronDown,
  ChevronUp, Loader2, ArrowUpDown, DollarSign,
  Percent, Zap, RefreshCw, LineChart, Play, Pause,
  RotateCcw, CalendarDays, Calendar, Gauge, Rocket,
  ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  AreaChart, Area, XAxis, YAxis,
  Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { getExitReasonColor } from '@/lib/contracts/strategy.contract';
import type { StrategyStats, StrategyTrade } from '@/lib/contracts/strategy.contract';
import type {
  ApprovalState,
  ProductionSummary,
  ProductionTradeFile,
  GateResult,
  ProductionStatus,
  DeployStatus,
  DeployResponse,
} from '@/lib/contracts/production-approval.contract';

// ============================================================================
// Strategy Registry Types
// ============================================================================
interface StrategyRegistryEntry {
  strategy_id: string;
  strategy_name: string;
  pipeline: string;
  status: string;
  backtest_year: number;
  return_pct: number;
  sharpe: number;
  p_value: number;
}

interface StrategyRegistry {
  strategies: StrategyRegistryEntry[];
  default_strategy: string;
}

const TradingChartWithSignals = lazy(() => import('@/components/charts/TradingChartWithSignals'));

// ============================================================================
// KPI Card (matches production/page.tsx KPICard exactly)
// ============================================================================
function KPICard({ title, value, subtitle, icon, trend, color = '#10B981', isLoading }: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  color?: string;
  isLoading?: boolean;
}) {
  if (isLoading) {
    return (
      <Card className="bg-slate-900/50 border-slate-800">
        <CardContent className="p-4 sm:p-6">
          <div className="animate-pulse space-y-2">
            <div className="h-3 w-16 bg-slate-700 rounded mx-auto" />
            <div className="h-8 w-24 bg-slate-700 rounded mx-auto" />
            <div className="h-2 w-20 bg-slate-700 rounded mx-auto" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-slate-900/50 border-slate-800 hover:border-slate-600 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/5">
      <CardContent className="p-5 sm:p-6 flex flex-col items-center text-center">
        <div className="p-2.5 sm:p-3 rounded-xl mb-4" style={{ backgroundColor: `${color}15` }}>
          <div style={{ color }}>{icon}</div>
        </div>
        <span className="text-xs sm:text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
          {title}
        </span>
        <div className="text-2xl sm:text-3xl font-bold tracking-tight mb-1" style={{ color }}>
          {value}
        </div>
        {subtitle && (
          <div className="flex items-center justify-center gap-1.5 text-xs sm:text-sm text-slate-400">
            {trend === 'up' && <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4 text-emerald-400" />}
            {trend === 'down' && <TrendingDown className="w-3 h-3 sm:w-4 sm:h-4 text-red-400" />}
            <span>{subtitle}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Strategy Selector Dropdown
// ============================================================================
function StrategySelector({
  strategies,
  selectedId,
  onSelect,
}: {
  strategies: StrategyRegistryEntry[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const selected = strategies.find(s => s.strategy_id === selectedId);

  return (
    <div className="relative">
      <button
        onClick={() => strategies.length > 1 ? setIsOpen(!isOpen) : undefined}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-xl",
          "bg-slate-800/80 border border-slate-700",
          strategies.length > 1 && "hover:border-cyan-500/50 cursor-pointer",
          "transition-all duration-200 text-sm"
        )}
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <div className="text-left">
            <div className="font-semibold text-white text-xs">
              {selected?.strategy_name || 'Select Strategy'}
            </div>
            <div className="text-[10px] text-slate-400">
              {selected?.pipeline} | +{selected?.return_pct.toFixed(1)}% | Sharpe {selected?.sharpe.toFixed(2)}
            </div>
          </div>
        </div>
        {selected?.status && (
          <Badge
            variant="outline"
            className={cn(
              "text-[9px] font-bold ml-2 shrink-0",
              selected.status === 'APPROVED'
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : "bg-amber-500/10 text-amber-400 border-amber-500/30"
            )}
          >
            {selected.status}
          </Badge>
        )}
        {strategies.length > 1 && (
          <ChevronDown className={cn(
            "w-3.5 h-3.5 text-slate-400 transition-transform",
            isOpen && "rotate-180"
          )} />
        )}
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 z-50 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-2xl min-w-[260px]">
          {strategies.map((s) => (
            <button
              key={s.strategy_id}
              className={cn(
                "w-full flex items-center justify-between px-4 py-3",
                "hover:bg-slate-700/50 transition-colors text-left",
                s.strategy_id === selectedId && "bg-slate-700/30"
              )}
              onClick={() => {
                onSelect(s.strategy_id);
                setIsOpen(false);
              }}
            >
              <div>
                <div className="text-white font-medium text-sm">{s.strategy_name}</div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {s.pipeline} | +{s.return_pct.toFixed(1)}% | Sharpe {s.sharpe.toFixed(2)} | p={s.p_value.toFixed(4)}
                </div>
              </div>
              <Badge
                variant="outline"
                className={cn(
                  "text-[9px] font-bold ml-3 shrink-0",
                  s.status === 'APPROVED'
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                )}
              >
                {s.status}
              </Badge>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Status Badge
// ============================================================================
function StatusBadge({ status }: { status: ProductionStatus }) {
  const config: Record<ProductionStatus, { bg: string; text: string; border: string; label: string }> = {
    PENDING_APPROVAL: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30', label: 'Pendiente' },
    APPROVED: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30', label: 'Aprobado' },
    REJECTED: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/30', label: 'Rechazado' },
    LIVE: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', border: 'border-cyan-500/30', label: 'En Vivo' },
  };
  const c = config[status];

  return (
    <Badge variant="outline" className={cn(c.bg, c.text, c.border, 'text-[10px] font-bold')}>
      {status === 'PENDING_APPROVAL' && <AlertTriangle className="w-3 h-3 mr-1" />}
      {status === 'APPROVED' && <CheckCircle2 className="w-3 h-3 mr-1" />}
      {status === 'REJECTED' && <XCircle className="w-3 h-3 mr-1" />}
      {status === 'LIVE' && <Activity className="w-3 h-3 mr-1 animate-pulse" />}
      {c.label}
    </Badge>
  );
}

// ============================================================================
// Interactive Approval Panel (Vote 2/2)
// ============================================================================
function ApprovalPanel({
  approval,
  onApprove,
  onReject,
  isSubmitting,
}: {
  approval: ApprovalState;
  onApprove: (notes: string) => void;
  onReject: (reason: string) => void;
  isSubmitting: boolean;
}) {
  const [notes, setNotes] = useState('');
  const [showConfirm, setShowConfirm] = useState<'approve' | 'reject' | null>(null);

  if (approval.status !== 'PENDING_APPROVAL') return null;

  const recColor = approval.backtest_recommendation === 'PROMOTE'
    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
    : approval.backtest_recommendation === 'REJECT'
    ? 'bg-red-500/10 text-red-400 border-red-500/30'
    : 'bg-amber-500/10 text-amber-400 border-amber-500/30';

  return (
    <Card variant="glow" className="border-purple-500/30">
      <CardHeader variant="minimal">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Shield className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <CardTitle variant="default" gradient={false} className="text-base text-white">
                Aprobacion de Estrategia
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5">Revision humana requerida (Vote 2/2)</p>
            </div>
          </div>
          <Badge variant="outline" className={cn('text-[10px] font-bold', recColor)}>
            L4: {approval.backtest_recommendation} ({Math.round(approval.backtest_confidence * 100)}%)
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="pt-0 p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            placeholder="Notas del revisor (opcional)..."
            value={notes}
            onChange={e => setNotes(e.target.value)}
            className="flex-1 bg-slate-800/60 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50 transition-colors"
          />

          {showConfirm === 'approve' ? (
            <div className="flex gap-2">
              <button
                onClick={() => { onApprove(notes); setShowConfirm(null); }}
                disabled={isSubmitting}
                className="px-4 py-2.5 bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50 flex items-center gap-1.5"
              >
                {isSubmitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
                Confirmar
              </button>
              <button onClick={() => setShowConfirm(null)} className="px-3 py-2.5 text-slate-400 hover:text-white text-sm transition-colors">
                Cancelar
              </button>
            </div>
          ) : showConfirm === 'reject' ? (
            <div className="flex gap-2">
              <button
                onClick={() => { onReject(notes); setShowConfirm(null); }}
                disabled={isSubmitting}
                className="px-4 py-2.5 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-red-500/20 disabled:opacity-50 flex items-center gap-1.5"
              >
                {isSubmitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <XCircle className="w-3.5 h-3.5" />}
                Confirmar Rechazo
              </button>
              <button onClick={() => setShowConfirm(null)} className="px-3 py-2.5 text-slate-400 hover:text-white text-sm transition-colors">
                Cancelar
              </button>
            </div>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={() => setShowConfirm('approve')}
                className="px-4 py-2.5 bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-emerald-500/20 flex items-center gap-1.5"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                Aprobar y Promover
              </button>
              <button
                onClick={() => setShowConfirm('reject')}
                className="px-4 py-2.5 bg-slate-800 border border-slate-700 hover:bg-red-600/20 hover:border-red-500/50 text-slate-300 hover:text-red-400 text-sm font-semibold rounded-lg transition-all flex items-center gap-1.5"
              >
                <XCircle className="w-3.5 h-3.5" />
                Rechazar
              </button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Gates Panel
// ============================================================================
function GatesPanel({ gates }: { gates: GateResult[] }) {
  const passed = gates.filter(g => g.passed).length;
  const total = gates.length;

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal" className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Shield className="w-4 h-4 text-purple-400" />
            </div>
            <CardTitle variant="default" gradient={false} className="text-sm text-white uppercase tracking-wider">
              Gates de Validacion
            </CardTitle>
          </div>
          <Badge
            variant="outline"
            className={cn(
              'text-[10px] font-bold',
              passed === total
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
            )}
          >
            {passed}/{total} pasaron
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-2">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {gates.map((g) => (
            <div
              key={g.gate}
              className={cn(
                'rounded-xl p-3 border text-center transition-all duration-200',
                g.passed
                  ? 'bg-emerald-500/5 border-emerald-500/20 hover:border-emerald-500/40'
                  : 'bg-red-500/5 border-red-500/20 hover:border-red-500/40'
              )}
            >
              <div className="flex items-center justify-center gap-1 mb-1">
                {g.passed
                  ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  : <XCircle className="w-4 h-4 text-red-400" />}
                <span className="text-[10px] font-medium text-slate-400 uppercase">{g.label}</span>
              </div>
              <div className={cn(
                'text-base font-bold font-mono',
                g.passed ? 'text-emerald-400' : 'text-red-400'
              )}>
                {typeof g.value === 'number'
                  ? g.gate === 'min_trades'
                    ? g.value
                    : `${g.value.toFixed(g.gate === 'statistical_significance' ? 4 : 1)}${g.gate !== 'min_trades' ? (g.gate === 'statistical_significance' ? '' : '%') : ''}`
                  : g.value}
              </div>
              <div className="text-[9px] text-slate-500 mt-0.5">
                umbral: {g.gate === 'min_trades' ? g.threshold : g.gate === 'statistical_significance' ? g.threshold : `${g.threshold}%`}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Exit Reasons Summary
// ============================================================================
function ExitReasonsSummary({ reasons }: { reasons: Record<string, number> }) {
  const total = Object.values(reasons).reduce((s, v) => s + v, 0);

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal" className="pb-2">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-purple-500/10">
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <CardTitle variant="default" gradient={false} className="text-sm text-white uppercase tracking-wider">
            Razones de Salida
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="pt-2">
        <div className="space-y-3">
          {Object.entries(reasons).map(([reason, count]) => {
            const color = getExitReasonColor(reason);
            const pct = total > 0 ? (count / total) * 100 : 0;
            return (
              <div key={reason} className="flex items-center gap-3">
                <span className={cn('text-xs font-medium w-28 px-2 py-0.5 rounded', color.bg, color.text)}>{reason}</span>
                <div className="flex-1 h-2.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={cn('h-full rounded-full transition-all duration-500', color.bar)}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-slate-400 w-16 text-right">{count} ({Math.round(pct)}%)</span>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Equity Curve Chart (built from trades data)
// ============================================================================
function EquityCurveChart({ trades, initialCapital, seriesPoints }: {
  trades: StrategyTrade[];
  initialCapital: number;
  /** Daily equity series (window-sliced) — the SSOT for continuous strategies; when
   *  present it replaces the trade-step curve (trades can't represent partial windows). */
  seriesPoints?: Array<{ d: string; eq: number }>;
}) {
  const months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'];
  const formatTradeDate = (iso: string | undefined): string => {
    if (!iso) return '?';
    const d = new Date(String(iso));
    if (isNaN(d.getTime())) return String(iso).slice(5, 10);
    return `${d.getDate()} ${months[d.getMonth()]}`;
  };

  const dataPoints = seriesPoints && seriesPoints.length > 1
    ? seriesPoints.map((r, i) => ({ label: formatTradeDate(r.d), equity: r.eq, idx: i }))
    : [
      { label: trades.length > 0 ? formatTradeDate(trades[0].timestamp as string) : 'Inicio', equity: initialCapital, idx: 0 },
      ...trades.map((t, i) => ({
        label: formatTradeDate((t.exit_timestamp || t.timestamp) as string),
        equity: Number(t.equity_at_exit),
        idx: i + 1,
      }))
    ];

  const equities = dataPoints.map(d => d.equity);
  const minEq = Math.min(...equities);
  const maxEq = Math.max(...equities);
  const padding = (maxEq - minEq) * 0.1 || 100;

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal" className="pb-2">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-cyan-500/10">
            <LineChart className="w-4 h-4 text-cyan-400" />
          </div>
          <CardTitle variant="default" gradient={false} className="text-sm text-white uppercase tracking-wider">
            Curva de Equity
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="pt-2">
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={dataPoints}>
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                domain={[minEq - padding, maxEq + padding]}
                tick={{ fontSize: 10, fill: '#64748b' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
              />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px', fontSize: '12px' }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(value: number) => [`$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`, 'Equity']}
              />
              <ReferenceLine y={initialCapital} stroke="#475569" strokeDasharray="3 3" />
              <Area
                type="monotone"
                dataKey="equity"
                stroke="#06b6d4"
                strokeWidth={2}
                fill="url(#equityGradient)"
                dot={{ fill: '#06b6d4', r: 3 }}
                activeDot={{ r: 5, stroke: '#06b6d4', strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Trading Summary (Backtest)
// ============================================================================
function TradingSummary({
  stats,
  trades,
  initialCapital = 10000,
  strategyName,
  year = 2025,
}: {
  stats: StrategyStats | undefined;
  trades: StrategyTrade[];
  initialCapital?: number;
  strategyName?: string;
  year?: number;
}) {
  const totalTrades = trades.length;
  const wins = trades.filter(t => Number(t.pnl_usd) > 0).length;
  const losses = totalTrades - wins;
  const currentEquity = trades.length > 0 ? Number(trades[trades.length - 1].equity_at_exit) : initialCapital;
  const profit = currentEquity - initialCapital;
  const profitPct = stats?.total_return_pct ?? (profit / initialCapital * 100);
  const winRate = stats?.win_rate_pct ?? (totalTrades > 0 ? (wins / totalTrades) * 100 : 0);
  const tradingDays = stats?.trading_days ?? 0;

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm font-medium text-slate-400 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Trading Summary
          {strategyName && (
            <span className="ml-auto text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400">
              {strategyName}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-4 pb-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <DollarSign className="w-3.5 h-3.5" />
            <span className="text-sm">Capital Inicial</span>
          </div>
          <span className="font-mono text-sm text-slate-300">${initialCapital.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <TrendingUp className="w-3.5 h-3.5" />
            <span className="text-sm">Equity Final</span>
          </div>
          <span className="font-mono text-sm font-bold text-white">
            ${currentEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <Activity className="w-3.5 h-3.5" />
            <span className="text-sm">Operaciones</span>
          </div>
          <span className="font-mono text-sm text-slate-300">
            {totalTrades} <span className="text-slate-500">({wins}W / {losses}L)</span>
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <Target className="w-3.5 h-3.5" />
            <span className="text-sm">Win Rate</span>
          </div>
          <span className={cn('font-mono text-sm', winRate >= 50 ? 'text-emerald-400' : 'text-amber-400')}>
            {winRate.toFixed(1)}%
          </span>
        </div>
        {tradingDays > 0 && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-slate-500">
              <CalendarDays className="w-3.5 h-3.5" />
              <span className="text-sm">Dias de Trading</span>
            </div>
            <span className="font-mono text-sm text-slate-300">{tradingDays}</span>
          </div>
        )}
        <div className="border-t border-slate-700 pt-3 mt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-400">Profit/Loss</span>
            <div className="text-right">
              <div className={cn('font-mono text-lg font-bold', profit >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {profit >= 0 ? '+' : ''}${profit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className={cn('text-xs font-mono', profitPct >= 0 ? 'text-emerald-400/70' : 'text-red-400/70')}>
                {profitPct >= 0 ? '+' : ''}{profitPct.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
        <div className="mt-2 px-2 py-1.5 rounded bg-blue-500/10 border border-blue-500/20">
          <span className="text-xs text-blue-400">
            Backtest {year} — {totalTrades} trades OOS
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Trade Table
// ============================================================================
function formatTsCOT(iso: string | undefined | null): string {
  if (!iso) return '-';
  const months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'];
  const d = new Date(String(iso));
  if (isNaN(d.getTime())) return String(iso).split('T')[0];
  const day = d.getDate().toString().padStart(2, '0');
  const mon = months[d.getMonth()];
  const hh = d.getHours().toString().padStart(2, '0');
  const mm = d.getMinutes().toString().padStart(2, '0');
  return `${day} ${mon} ${hh}:${mm}`;
}

function TradeTable({ trades, showAll, onToggle }: {
  trades: StrategyTrade[];
  showAll: boolean;
  onToggle: () => void;
}) {
  const [sortField, setSortField] = useState<string>('trade_id');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const sorted = [...trades].sort((a, b) => {
    const av = a[sortField as keyof StrategyTrade];
    const bv = b[sortField as keyof StrategyTrade];
    if (typeof av === 'number' && typeof bv === 'number') {
      return sortDir === 'asc' ? av - bv : bv - av;
    }
    return sortDir === 'asc'
      ? String(av).localeCompare(String(bv))
      : String(bv).localeCompare(String(av));
  });

  const visible = showAll ? sorted : sorted.slice(0, 10);

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  const SortHeader = ({ field, label }: { field: string; label: string }) => (
    <th
      onClick={() => handleSort(field)}
      className="px-3 py-3 text-left text-[10px] font-semibold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-cyan-400 transition-colors select-none"
    >
      <span className="flex items-center gap-1">
        {label}
        {sortField === field && (
          sortDir === 'asc' ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
        )}
      </span>
    </th>
  );

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal" className="pb-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-cyan-500/10">
              <ArrowUpDown className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <CardTitle variant="default" gradient={false} className="text-sm text-white">
                Historial de Trades (Backtest 2025)
              </CardTitle>
              <p className="text-[10px] text-slate-500 mt-0.5">{trades.length} operaciones</p>
            </div>
          </div>
          {trades.length > 10 && (
            <button
              onClick={onToggle}
              className="text-xs text-cyan-400 hover:text-cyan-300 font-medium transition-colors flex items-center gap-1"
            >
              {showAll ? (
                <><ChevronUp className="w-3.5 h-3.5" /> Menos</>
              ) : (
                <><ChevronDown className="w-3.5 h-3.5" /> Todos ({trades.length})</>
              )}
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50">
                <SortHeader field="trade_id" label="#" />
                <SortHeader field="side" label="Lado" />
                <SortHeader field="timestamp" label="Entrada" />
                <SortHeader field="exit_timestamp" label="Salida" />
                <SortHeader field="entry_price" label="Precio E." />
                <SortHeader field="exit_price" label="Precio S." />
                <SortHeader field="leverage" label="Lev" />
                <th className="px-3 py-2.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider text-center">Confianza</th>
                <th className="px-3 py-2.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider text-right">HS%</th>
                <th className="px-3 py-2.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider text-right">TP%</th>
                <SortHeader field="pnl_usd" label="PnL $" />
                <SortHeader field="pnl_pct" label="PnL %" />
                <SortHeader field="exit_reason" label="Razon" />
              </tr>
            </thead>
            <tbody>
              {visible.map((t) => {
                const exitColor = getExitReasonColor(t.exit_reason);
                const pnlPositive = Number(t.pnl_usd) >= 0;
                return (
                  <tr key={t.trade_id} className="border-b border-slate-800/30 hover:bg-slate-800/30 transition-colors">
                    <td className="px-3 py-2.5 text-slate-500 text-xs font-mono">{t.trade_id}</td>
                    <td className="px-3 py-2.5">
                      <span className={cn(
                        'px-2 py-0.5 rounded text-[10px] font-bold',
                        t.side === 'LONG' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-red-500/15 text-red-400'
                      )}>
                        {t.side}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono whitespace-nowrap">
                      {formatTsCOT(t.timestamp as string)}
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono whitespace-nowrap">
                      {formatTsCOT(t.exit_timestamp as string | undefined)}
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono text-right">
                      {t.entry_price != null ? `$${Number(t.entry_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}` : '—'}
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono text-right">
                      {t.exit_price != null ? `$${Number(t.exit_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}` : '—'}
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono text-right">{Number(t.leverage).toFixed(2)}x</td>
                    <td className="px-3 py-2.5 text-center">
                      <span className={cn(
                        'px-1.5 py-0.5 rounded text-[10px] font-bold',
                        (t as Record<string, unknown>).confidence_tier === 'HIGH' ? 'bg-emerald-500/15 text-emerald-400' :
                        (t as Record<string, unknown>).confidence_tier === 'MEDIUM' ? 'bg-amber-500/15 text-amber-400' :
                        'bg-slate-500/15 text-slate-400'
                      )}>
                        {String((t as Record<string, unknown>).confidence_tier ?? '-')}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-red-400/70 text-xs font-mono text-right">
                      {(t as Record<string, unknown>).hard_stop_pct != null ? `${Number((t as Record<string, unknown>).hard_stop_pct).toFixed(1)}%` : '-'}
                    </td>
                    <td className="px-3 py-2.5 text-emerald-400/70 text-xs font-mono text-right">
                      {(t as Record<string, unknown>).take_profit_pct != null ? `${Number((t as Record<string, unknown>).take_profit_pct).toFixed(1)}%` : '-'}
                    </td>
                    <td className={cn('px-3 py-2.5 text-xs font-mono font-semibold text-right', pnlPositive ? 'text-emerald-400' : 'text-red-400')}>
                      {t.pnl_usd != null ? `${pnlPositive ? '+' : ''}${Number(t.pnl_usd).toFixed(2)}` : '—'}
                    </td>
                    <td className={cn('px-3 py-2.5 text-xs font-mono text-right', pnlPositive ? 'text-emerald-400' : 'text-red-400')}>
                      {t.pnl_pct != null ? `${pnlPositive ? '+' : ''}${Number(t.pnl_pct).toFixed(2)}%` : '—'}
                    </td>
                    <td className="px-3 py-2.5">
                      <span className={cn('px-2 py-0.5 rounded text-[10px] font-medium', exitColor.bg, exitColor.text)}>
                        {t.exit_reason}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Section Header
// ============================================================================
function SectionHeader({ title, subtitle, icon }: {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="mb-8 sm:mb-10 lg:mb-12 text-center flex flex-col items-center">
      <h2 className="text-lg sm:text-xl lg:text-2xl font-bold text-white flex items-center justify-center gap-3">
        {icon && <span className="text-cyan-400">{icon}</span>}
        {title}
      </h2>
      {subtitle && (
        <p className="mt-3 text-sm sm:text-base text-slate-400 max-w-2xl">
          {subtitle}
        </p>
      )}
      <div className="mt-4 flex items-center justify-center gap-1">
        <div className="h-0.5 w-8 rounded-full bg-gradient-to-r from-transparent to-cyan-500/50" />
        <div className="h-0.5 w-16 rounded-full bg-gradient-to-r from-cyan-500/50 to-blue-500/50" />
        <div className="h-0.5 w-8 rounded-full bg-gradient-to-r from-blue-500/50 to-transparent" />
      </div>
    </div>
  );
}

// ============================================================================
// Chart Loading Fallback
// ============================================================================
function ChartLoadingFallback() {
  return (
    <div className="rounded-2xl overflow-hidden border border-slate-700/50 bg-slate-900/40">
      <div className="px-6 py-4 border-b border-slate-700/50">
        <span className="text-sm font-medium text-slate-300">USD/COP Price Chart</span>
      </div>
      <div className="h-[400px] flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-10 h-10 text-cyan-400 animate-spin mx-auto mb-3" />
          <p className="text-slate-400 text-sm">Loading chart...</p>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================
function formatProfitFactor(pf: number | null | undefined): string {
  if (pf == null) return 'N/A';
  if (pf > 100) return '>100';
  return pf.toFixed(2);
}

// ============================================================================
// Deploy Panel — One-Click Production Deploy
// ============================================================================
const DEPLOY_PHASE_LABELS: Record<string, string> = {
  initializing: 'Inicializando...',
  retraining: 'Reentrenando modelos (2020-2025)...',
  exporting: 'Exportando datos de produccion...',
  done: 'Completado',
};

const DEPLOY_PHASE_PROGRESS: Record<string, number> = {
  initializing: 10,
  retraining: 50,
  exporting: 80,
  done: 100,
};

function DeployPanel({ approval }: { approval: ApprovalState }) {
  const [deployStatus, setDeployStatus] = useState<DeployStatus>({ status: 'idle' });
  const [isDeploying, setIsDeploying] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Check deploy status on mount (resume polling if deploy was already running)
  useEffect(() => {
    const checkInitial = async () => {
      try {
        const res = await fetch('/api/production/deploy/status');
        if (res.ok) {
          const data: DeployStatus = await res.json();
          setDeployStatus(data);
          if (data.status === 'running') {
            startPolling();
          }
        }
      } catch {
        // Ignore
      }
    };
    checkInitial();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startPolling = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch('/api/production/deploy/status');
        if (res.ok) {
          const data: DeployStatus = await res.json();
          setDeployStatus(data);
          if (data.status !== 'running') {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      } catch {
        // Continue polling
      }
    }, 3000);
  };

  const handleDeploy = async () => {
    setIsDeploying(true);
    try {
      const res = await fetch('/api/production/deploy', { method: 'POST' });
      const data: DeployResponse = await res.json();
      if (data.success) {
        setDeployStatus({ status: 'running', phase: 'initializing', started_at: new Date().toISOString() });
        startPolling();
      } else {
        setDeployStatus({ status: 'failed', error: data.message });
      }
    } catch (err) {
      setDeployStatus({ status: 'failed', error: err instanceof Error ? err.message : 'Error de red' });
    } finally {
      setIsDeploying(false);
    }
  };

  const handleRetry = () => {
    setDeployStatus({ status: 'idle' });
  };

  // Only show after approval
  if (approval.status !== 'APPROVED') return null;

  const phase = deployStatus.phase || 'initializing';
  const progress = DEPLOY_PHASE_PROGRESS[phase] || 0;

  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardContent className="p-4 sm:p-6">
        {/* Idle — Show deploy button */}
        {deployStatus.status === 'idle' && (
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="flex items-center gap-3 flex-1">
              <div className="p-2.5 rounded-xl bg-cyan-500/10">
                <Rocket className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-white">Desplegar a Produccion</p>
                <p className="text-xs text-slate-400 mt-0.5">
                  Reentrenar con datos completos (2020-2025) y generar trades 2026
                </p>
              </div>
            </div>
            <button
              onClick={handleDeploy}
              disabled={isDeploying}
              className="px-5 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-50 flex items-center gap-2 whitespace-nowrap"
            >
              {isDeploying ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Rocket className="w-4 h-4" />
              )}
              Desplegar a Produccion
            </button>
          </div>
        )}

        {/* Running — Progress bar */}
        {deployStatus.status === 'running' && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-cyan-500/10">
                <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-white">Desplegando...</p>
                <p className="text-xs text-cyan-400 mt-0.5">
                  {DEPLOY_PHASE_LABELS[phase] || 'Procesando...'}
                </p>
              </div>
              {deployStatus.started_at && (
                <span className="text-[10px] text-slate-500 font-mono">
                  {Math.round((Date.now() - new Date(deployStatus.started_at).getTime()) / 1000)}s
                </span>
              )}
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-1000 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>Inicializar</span>
              <span>Reentrenar</span>
              <span>Exportar</span>
              <span>Listo</span>
            </div>
          </div>
        )}

        {/* Completed — Success banner */}
        {deployStatus.status === 'completed' && (
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="flex items-center gap-3 flex-1">
              <div className="p-2.5 rounded-xl bg-emerald-500/10">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-emerald-300">Deploy completado!</p>
                <p className="text-xs text-slate-400 mt-0.5">
                  Estrategia desplegada exitosamente. Los datos de produccion 2026 estan listos.
                </p>
              </div>
            </div>
            <a
              href="/production"
              className="px-4 py-2.5 bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-emerald-500/20 flex items-center gap-2 whitespace-nowrap"
            >
              <ExternalLink className="w-4 h-4" />
              Ver Produccion
            </a>
          </div>
        )}

        {/* Failed — Error message */}
        {deployStatus.status === 'failed' && (
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="flex items-center gap-3 flex-1">
              <div className="p-2.5 rounded-xl bg-red-500/10">
                <XCircle className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <p className="text-sm font-medium text-red-300">Error en deploy</p>
                <p className="text-xs text-slate-400 mt-0.5 max-w-md break-words">
                  {deployStatus.error || 'Error desconocido durante el despliegue.'}
                </p>
              </div>
            </div>
            <button
              onClick={handleRetry}
              className="px-4 py-2.5 bg-slate-800 border border-slate-700 hover:border-cyan-500/50 text-slate-300 hover:text-white text-sm font-semibold rounded-lg transition-all flex items-center gap-2 whitespace-nowrap"
            >
              <RefreshCw className="w-4 h-4" />
              Reintentar
            </button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Compute dynamic metrics from a list of trades (for replay)
// ============================================================================
function computeTradeMetrics(trades: StrategyTrade[], initialCapital: number) {
  if (trades.length === 0) return null;

  const wins = trades.filter(t => Number(t.pnl_usd) > 0).length;
  const losses = trades.length - wins;
  const winRate = (wins / trades.length) * 100;

  // Use equity_at_entry of the first trade as the starting equity for this subset
  // This gives correct returns when filtering by date range
  const startEquity = Number(trades[0].equity_at_entry) || initialCapital;
  const finalEquity = Number(trades[trades.length - 1].equity_at_exit);
  const totalReturnPct = ((finalEquity - startEquity) / startEquity) * 100;

  // Sharpe from trade returns (annualized for weekly trades)
  const returns = trades.map(t => Number(t.pnl_pct) / 100);
  const meanReturn = returns.reduce((s, r) => s + r, 0) / returns.length;
  const stdReturn = returns.length > 1
    ? Math.sqrt(returns.reduce((s, r) => s + (r - meanReturn) ** 2, 0) / (returns.length - 1))
    : 0;
  const sharpe = stdReturn > 0 ? (meanReturn / stdReturn) * Math.sqrt(52) : 0;

  // Profit factor
  const grossProfit = trades.filter(t => Number(t.pnl_usd) > 0).reduce((s, t) => s + Number(t.pnl_usd), 0);
  const grossLoss = Math.abs(trades.filter(t => Number(t.pnl_usd) < 0).reduce((s, t) => s + Number(t.pnl_usd), 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : null;

  // Max drawdown (relative to subset start equity)
  let peak = startEquity;
  let maxDD = 0;
  for (const t of trades) {
    const eq = Number(t.equity_at_exit);
    if (eq > peak) peak = eq;
    const dd = ((peak - eq) / peak) * 100;
    if (dd > maxDD) maxDD = dd;
  }

  // Exit reasons
  const exitReasons: Record<string, number> = {};
  for (const t of trades) {
    const reason = t.exit_reason || 'unknown';
    exitReasons[reason] = (exitReasons[reason] || 0) + 1;
  }

  // Direction counts
  const nLong = trades.filter(t => t.side === 'LONG').length;
  const nShort = trades.filter(t => t.side === 'SHORT').length;

  // t-test for statistical significance
  const n = returns.length;
  const tStat = n > 1 && stdReturn > 0 ? meanReturn / (stdReturn / Math.sqrt(n)) : 0;
  // Approximate two-sided p-value
  const pValue = n > 2 ? Math.min(1, Math.exp(-0.717 * Math.abs(tStat) - 0.416 * tStat * tStat)) : 1;

  // Direction accuracy (predicted SHORT when price actually went down)
  const da = winRate; // simplified: win rate ≈ DA for directional strategy

  return {
    total_return_pct: totalReturnPct,
    sharpe,
    profit_factor: profitFactor,
    win_rate_pct: winRate,
    max_dd_pct: maxDD,
    final_equity: finalEquity,
    trading_days: trades.length,
    exit_reasons: exitReasons,
    n_long: nLong,
    n_short: nShort,
    p_value: Math.max(0.0001, pValue),
    t_stat: tStat,
    significant: pValue < 0.05,
    wins,
    losses,
    direction_accuracy_pct: da,
  };
}

// ============================================================================
// Main Component Export
// ============================================================================
// ── Dynamic strategy registry (CTR-STRAT-REGISTRY-001) — lightweight manifest types.
// Served by /api/strategies/[id]/manifest. Purely additive: if the route/manifest is
// absent (e.g. legacy build) the version UI silently no-ops and the page behaves as before.
interface ManifestBacktest {
  model_version: string;
  year: number;
  immutable_id: string;
  summary: string; // path relative to public/data
  trades: string; // path relative to public/data
  signals: string | null;
  replayable: boolean;
  gates?: Record<string, unknown>;
  headline?: Record<string, unknown>;
}
interface ManifestModelVersion {
  version: string;
  active: boolean;
  trained_at?: string | null;
}
interface StrategyManifestLite {
  strategy_id: string;
  chart_symbol?: string;
  status?: string;
  backtests: ManifestBacktest[];
  model_versions: ManifestModelVersion[];
}

export function ForecastingBacktestSection({
  controlledStrategyId,
  onStrategyChange,
}: {
  /** Controlled selection from a parent (e.g. the header Backtest selector). Optional —
   *  when omitted the section manages selection entirely on its own (backward compatible).
   *  Named `controlledStrategyId` to avoid colliding with the local `strategyId` below. */
  controlledStrategyId?: string;
  /** Notifies the parent when the in-card selector changes, keeping both in sync. */
  onStrategyChange?: (id: string) => void;
} = {}) {
  const [registry, setRegistry] = useState<StrategyRegistry | null>(null);
  const [selectedStrategyId, setSelectedStrategyId] = useState<string>('');
  const [summary, setSummary] = useState<ProductionSummary | null>(null);
  const [approval, setApproval] = useState<ApprovalState | null>(null);
  const [trades, setTrades] = useState<StrategyTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAllTrades, setShowAllTrades] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Replay state
  // Daily equity series from the bundle (signals_<year>.json) — SSOT for window-dynamic
  // metrics/equity AND the replay clock for continuous strategies (Gold B1 = 1 trade).
  const [dailySeries, setDailySeries] = useState<Array<{ d: string; eq: number }> | null>(null);
  const [replayIndex, setReplayIndex] = useState<number>(-1); // -1 = show all
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(1);
  const replayRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [replayStartDate, setReplayStartDate] = useState('2025-01-01');
  const [replayEndDate, setReplayEndDate] = useState('2025-12-31');
  // The auto-initialized window for the loaded trades (set by the effect below). Comparing the
  // current range against THIS (not against summary.year) tells us whether the USER filtered.
  // Audit I-4 fix: for gold/btc (full-history bundles, year=2026) the old summary.year comparison
  // was spuriously true forever, silently replacing bundle metrics with recomputed ones.
  const autoRangeRef = useRef<{ start: string; end: string }>({ start: '2025-01-01', end: '2025-12-31' });

  // Version registry state (additive; graceful no-op if manifest/route absent)
  const [manifest, setManifest] = useState<StrategyManifestLite | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<string>('');
  const [isPromoting, setIsPromoting] = useState(false);
  const [promoteMsg, setPromoteMsg] = useState<string | null>(null);

  // Filter trades by replay date range, then apply replay index
  // A trade belongs to the window when its SPAN overlaps it (entry ≤ end AND exit ≥ start).
  // Entry-only filtering broke always-in strategies: Gold B1's single 2004→ position never
  // "entered" in 2025, so the 2025 replay had zero trades and looked dead (operator report).
  // AND the trade must be CLIPPED to the window for display/replay/chart: raw out-of-window
  // timestamps dragged the chart axis to the trade's 2026 exit (view jumped to Dec-2025→Jul-2026,
  // ignoring the selected 2025 range) and the table showed nonsense dates (operator report #2).
  const filteredTrades = trades
    .filter(t => {
      const entry = String(t.timestamp).slice(0, 10);
      const exit = t.exit_timestamp ? String(t.exit_timestamp).slice(0, 10) : entry;
      return entry <= replayEndDate && exit >= replayStartDate;
    })
    .map(t => {
      const entry = String(t.timestamp).slice(0, 10);
      const exit = t.exit_timestamp ? String(t.exit_timestamp).slice(0, 10) : entry;
      const clipStart = entry < replayStartDate;
      const clipEnd = exit > replayEndDate;
      if (!clipStart && !clipEnd) return t;
      // Continuous position crossing the window: clamp the DISPLAY span to the window and
      // tag it — the row's PnL is the full trade's (metrics cards carry the OOS-2025 truth).
      return {
        ...t,
        timestamp: clipStart ? `${replayStartDate}T00:00:00Z` : t.timestamp,
        exit_timestamp: clipEnd ? `${replayEndDate}T23:59:59Z` : t.exit_timestamp,
        exit_reason: 'posicion_continua',
        // Full-trade prices/P&L must NOT masquerade as the window's — blank them; the
        // window truth is the bundle's OOS-2025 slice shown in the metric cards.
        entry_price: clipStart ? (null as unknown as number) : t.entry_price,
        exit_price: clipEnd ? (null as unknown as number) : t.exit_price,
        pnl_usd: (null as unknown as number),
        pnl_pct: (null as unknown as number),
      } as typeof t;
    });
  // Series replay domain: when the bundle carries the daily series, the REPLAY CLOCK is
  // TIME (series days, strided to ~60 steps), not trade count — an always-in strategy has
  // 1 trade and a trade-clock replay is a single dead step (operator: "gold sigue sin
  // funcionar"). Trades become visible as the clock passes their exit.
  const seriesInWindow = useMemo(() => {
    if (!dailySeries) return null;
    const rows = dailySeries.filter(r => r.d >= replayStartDate && r.d <= replayEndDate);
    return rows.length >= 3 ? rows : null;
  }, [dailySeries, replayStartDate, replayEndDate]);
  const seriesStride = seriesInWindow ? Math.max(1, Math.ceil(seriesInWindow.length / 60)) : 1;
  const replaySteps = seriesInWindow
    ? Math.ceil(seriesInWindow.length / seriesStride)
    : filteredTrades.length;
  const seriesClockDate = seriesInWindow && replayIndex >= 0
    ? seriesInWindow[Math.min(replayIndex * seriesStride, seriesInWindow.length - 1)].d
    : null;
  const visibleTrades = replayIndex < 0
    ? filteredTrades
    : seriesClockDate
      ? filteredTrades.filter(tr =>
          String(tr.exit_timestamp ?? tr.timestamp).slice(0, 10) <= seriesClockDate)
      : filteredTrades.slice(0, replayIndex);

  // Default the replay window to the 2025 BACKTEST year for every pair (methodology: trained
  // ≤ Dec-2024, backtest = 2025, production = 2026). COP trades are all 2025 already; a multi-year
  // backtest (Gold/BTC 2018-2026) is CLAMPED to its 2025 slice by default so the operator lands on
  // the OOS backtest year — they can widen the date inputs to see full history. Falls back to the
  // full span only when the data doesn't cover 2025. Uses BOTH entry+exit timestamps so buy-and-hold
  // strategies (a single trade held for years) don't collapse to one candle.
  useEffect(() => {
    if (!trades.length) return;
    const dates = trades
      .flatMap(t => [String(t.timestamp), t.exit_timestamp ? String(t.exit_timestamp) : ''])
      .map(d => d.slice(0, 10))
      .filter(d => /^\d{4}-\d{2}-\d{2}$/.test(d))
      .sort();
    if (!dates.length) return;
    const minD = dates[0];
    const maxD = dates[dates.length - 1];
    const BT_START = '2025-01-01';
    const BT_END = '2025-12-31';
    const covers2025 = minD <= BT_END && maxD >= BT_START;
    if (covers2025) {
      const s = minD > BT_START ? minD : BT_START;
      const e = maxD < BT_END ? maxD : BT_END;
      setReplayStartDate(s);
      setReplayEndDate(e);
      autoRangeRef.current = { start: s, end: e };
    } else {
      setReplayStartDate(minD);
      setReplayEndDate(maxD);
      autoRangeRef.current = { start: minD, end: maxD };
    }
    setReplayIndex(-1);
  }, [trades]);

  // During replay, chart endDate follows the latest visible trade (progressive reveal)
  const chartEndDate = (() => {
    if (replayIndex < 0) {
      return new Date(`${replayEndDate}T23:59:59`);
    }
    if (visibleTrades.length === 0) {
      // Replay started but no trades yet — show first month for context
      const d = new Date(replayStartDate);
      d.setDate(d.getDate() + 30);
      return d;
    }
    const lastTrade = visibleTrades[visibleTrades.length - 1];
    const lastTs = lastTrade.exit_timestamp || lastTrade.timestamp;
    if (!lastTs) return new Date(`${replayEndDate}T23:59:59`);
    const d = new Date(String(lastTs));
    d.setDate(d.getDate() + 14); // 2-week buffer after last trade
    const maxDate = new Date(`${replayEndDate}T23:59:59`);
    return d > maxDate ? maxDate : d;
  })();

  // Load-race guard (audit A5-07): loadStrategyData and loadVersionData both write
  // summary/trades with no cancellation — a slow older fetch could render version-A
  // trades over version-B metrics. Each load takes a monotonically-increasing token;
  // state writes only apply while the token is still current.
  const loadSeqRef = useRef(0);
  // Approval is STRATEGY-level, only loadStrategyData writes it — it gets its own
  // sequence. Guarding it with the shared seq made the on-mount loadVersionData
  // (manifest default) invalidate the approval write → ApprovalPanel never rendered
  // (admin could not promote; found by promotion-e2e regression).
  const stratSeqRef = useRef(0);

  // Load strategy data for a given strategy_id
  const loadStrategyData = useCallback(async (sid: string) => {
    const seq = ++loadSeqRef.current;
    const sseq = ++stratSeqRef.current;
    try {
      setLoading(true);
      setError(null);
      setDailySeries(null); // registry bundles repopulate it via loadVersionData

      // Try per-strategy summary first, fall back to generic
      let summaryData: ProductionSummary | null = null;
      const summaryPaths = [
        '/data/production/summary_2025.json',
      ];
      for (const path of summaryPaths) {
        const res = await fetch(path);
        if (res.ok) {
          const data = await res.json();
          // Only accept the legacy summary if it is THIS strategy. Registry-backed strategies
          // (e.g. Gold, whose legacy file doesn't exist) defer to loadVersionData — don't clobber.
          if (data.strategy_id === sid) {
            summaryData = data;
            break;
          }
        }
      }
      // A5-07: guard the WRITE only — never early-return before the approval fetch below
      // (an on-mount version load would otherwise skip approval → no ApprovalPanel).
      if (summaryData && seq === loadSeqRef.current) setSummary(summaryData);

      // Try per-strategy approval first, fall back to generic
      let approvalData: ApprovalState | null = null;
      const approvalPaths = [
        // singleton (ACTIVE strategy) first — its `strategy` field must match this sid;
        // else fall through to the per-strategy file (multi-strategy production, fs-backed).
        '/api/production/status',
        `/api/data/production/approval_state_${sid}.json`,
      ];
      for (const path of approvalPaths) {
        const res = await fetch(path);
        if (res.ok) {
          const data = await res.json();
          if (data.strategy === sid || path === approvalPaths[approvalPaths.length - 1]) {
            approvalData = data;
            break;
          }
        }
      }
      // Approval uses its own strategy-level sequence (version loads must not void it).
      if (sseq === stratSeqRef.current) setApproval(approvalData);
      if (seq !== loadSeqRef.current) return; // superseded (A5-07) — summary/trades only

      // Load trades
      const tradesRes = await fetch(`/data/production/trades/${sid}_2025.json`);
      if (tradesRes.ok && seq === loadSeqRef.current) {
        const tradeData: ProductionTradeFile = await tradesRes.json();
        if (seq === loadSeqRef.current) setTrades(tradeData.trades || []);
      }
      // If the legacy trades file is absent (registry-backed strategy like Gold), do NOT clear —
      // loadVersionData is authoritative and provides the versioned trades.
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error loading backtest data');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load strategy registry on mount. Prefer the dynamic /api/registry (includes new
  // strategies/branches with active-version metrics); fall back to the legacy strategies.json
  // so older builds keep working. Additive — no behavior change when only one strategy exists.
  useEffect(() => {
    const loadRegistry = async () => {
      try {
        let data: StrategyRegistry | null = null;
        const regRes = await fetch('/api/registry');
        if (regRes.ok) {
          const idx = await regRes.json();
          data = {
            strategies: (idx.strategies ?? []).map((s: {
              strategy_id: string; display_name?: string; pipeline_type?: string; status?: string;
              backtest_years?: number[]; return_pct?: number; sharpe?: number; p_value?: number;
            }) => ({
              strategy_id: s.strategy_id,
              strategy_name: s.display_name ?? s.strategy_id,
              pipeline: s.pipeline_type ?? 'ml_forecasting',
              status: s.status === 'production' ? 'APPROVED' : (s.status ?? 'PAPER'),
              backtest_year: s.backtest_years?.[0] ?? 2025,
              return_pct: s.return_pct ?? 0,
              sharpe: s.sharpe ?? 0,
              p_value: s.p_value ?? 0,
            })),
            default_strategy: idx.default?.strategy_id ?? '',
          };
        }
        if (!data) {
          const res = await fetch('/data/production/strategies.json');
          if (res.ok) data = await res.json();
        }
        if (data && data.strategies?.length) {
          setRegistry(data);
          const defaultId = data.default_strategy || data.strategies[0]?.strategy_id || 'smart_simple_v11';
          setSelectedStrategyId(defaultId);
          loadStrategyData(defaultId);
        } else {
          setSelectedStrategyId('smart_simple_v11');
          loadStrategyData('smart_simple_v11');
        }
      } catch {
        setSelectedStrategyId('smart_simple_v11');
        loadStrategyData('smart_simple_v11');
      }
    };
    loadRegistry();
  }, [loadStrategyData]);

  // When strategy changes, reload data + reset replay
  const handleStrategyChange = useCallback((sid: string) => {
    setSelectedStrategyId(sid);
    setReplayIndex(-1);
    setIsPlaying(false);
    setShowAllTrades(false);
    loadStrategyData(sid);
    onStrategyChange?.(sid);
  }, [loadStrategyData, onStrategyChange]);

  // Sync a controlled selection coming from a parent selector (additive; no-op when
  // uncontrolled or already in sync). Keeps the header Backtest selector and the in-card
  // selector pointing at the same strategy.
  useEffect(() => {
    if (controlledStrategyId && controlledStrategyId !== selectedStrategyId) {
      handleStrategyChange(controlledStrategyId);
    }
  }, [controlledStrategyId, selectedStrategyId, handleStrategyChange]);

  // ── Version registry: load a per-version immutable bundle (CTR-STRAT-REGISTRY-001) ──
  // Overrides summary + trades from the versioned artifacts; all downstream derived
  // state (KPIs, chart replay, trade table) reacts automatically. Approval is left
  // untouched — it is strategy-level, not version-level.
  const loadVersionData = useCallback(async (bt: ManifestBacktest) => {
    const seq = ++loadSeqRef.current; // A5-07: this load supersedes any in-flight one
    try {
      setLoading(true);
      setError(null);
      // Fetch via the fs-backed API route (not the static /data/ path): Next standalone only serves
      // public/ files that existed at image-build time, so a freshly published version's bundle 404s
      // through static serving even though it's on the mount. /api/data reads from disk at request time.
      const [sRes, tRes] = await Promise.all([
        fetch(`/api/data/${bt.summary}`),
        fetch(`/api/data/${bt.trades}`),
      ]);
      if (seq !== loadSeqRef.current) return; // superseded (A5-07)
      if (sRes.ok) setSummary(await sRes.json());
      // Defensive: only replace trades when the fetch succeeds — never wipe to [] on a transient
      // failure/404 (that is what silently emptied the replay for newly-published versions).
      if (tRes.ok && seq === loadSeqRef.current) {
        const td = await tRes.json();
        if (seq === loadSeqRef.current) setTrades(td?.trades ?? []);
      }
      // Daily equity series (window-dynamic metrics SSOT) — best-effort, additive.
      if (bt.signals) {
        try {
          const sr = await fetch(`/api/data/${bt.signals}`);
          if (sr.ok && seq === loadSeqRef.current) {
            const sd = await sr.json();
            if (seq === loadSeqRef.current && Array.isArray(sd?.rows)) setDailySeries(sd.rows);
          }
        } catch { /* series optional */ }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error loading version bundle');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch the strategy manifest; default the version selection to the entry matching the
  // already-shown legacy data (production/*), else the active version — so the dropdown is
  // consistent with what loadStrategyData rendered without changing default behavior.
  const loadManifest = useCallback(async (sid: string) => {
    try {
      const res = await fetch(`/api/strategies/${sid}/manifest`);
      if (!res.ok) { setManifest(null); setSelectedVersion(''); return; }
      const m: StrategyManifestLite = await res.json();
      setManifest(m);
      // Default to the active version; make the displayed bundle match it so any strategy
      // (including new branches) renders its own data. For smart_simple_v11 the active 2.0.0
      // bundle is byte-equivalent to the legacy view, so the default render is unchanged.
      const active = (m.model_versions ?? []).find(v => v.active)?.version;
      const first = (m.backtests ?? [])[0]?.model_version;
      const version = active ?? first ?? '';
      setSelectedVersion(version);
      const bt = (m.backtests ?? []).find(b => b.model_version === version);
      if (bt) loadVersionData(bt);
    } catch {
      setManifest(null);
      setSelectedVersion('');
    }
  }, [loadVersionData]);

  // Load the manifest whenever the strategy changes (separate from the legacy loaders
  // so the default view stays byte-identical to prior behavior).
  useEffect(() => {
    if (selectedStrategyId) loadManifest(selectedStrategyId);
  }, [selectedStrategyId, loadManifest]);

  const handleVersionChange = useCallback((version: string) => {
    setSelectedVersion(version);
    setReplayIndex(-1);
    setIsPlaying(false);
    setShowAllTrades(false);
    const bt = manifest?.backtests?.find(b => b.model_version === version);
    if (bt) loadVersionData(bt);
  }, [manifest, loadVersionData]);

  const handlePromote = useCallback(async () => {
    if (!manifest || !selectedVersion) return;
    setIsPromoting(true);
    setPromoteMsg(null);
    try {
      const res = await fetch('/api/registry/promote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy_id: manifest.strategy_id, version: selectedVersion, status: 'production' }),
      });
      const data = await res.json();
      if (res.ok) {
        setPromoteMsg(`v${selectedVersion} promovida`);
        await loadManifest(manifest.strategy_id);
      } else {
        setPromoteMsg(data.error ?? 'promote falló');
      }
    } catch (e) {
      setPromoteMsg(e instanceof Error ? e.message : 'error');
    } finally {
      setIsPromoting(false);
    }
  }, [manifest, selectedVersion, loadManifest]);

  // Replay animation effect
  useEffect(() => {
    if (!isPlaying) {
      if (replayRef.current) clearInterval(replayRef.current);
      return;
    }
    replayRef.current = setInterval(() => {
      setReplayIndex(prev => {
        const next = prev + 1;
        if (next >= replaySteps) {
          setIsPlaying(false);
          return replaySteps;
        }
        return next;
      });
    }, 500 / playSpeed);
    return () => { if (replayRef.current) clearInterval(replayRef.current); };
  }, [isPlaying, playSpeed, replaySteps]);

  const handleApprove = async (notes: string) => {
    setIsSubmitting(true);
    try {
      const res = await fetch('/api/production/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'APPROVE', notes, reviewer: 'dashboard_user', strategy_id: strategyId }),
      });
      if (res.ok) {
        const approvalRes = await fetch('/api/production/status');
        if (approvalRes.ok) setApproval(await approvalRes.json());
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReject = async (reason: string) => {
    setIsSubmitting(true);
    try {
      const res = await fetch('/api/production/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'REJECT', notes: reason, reviewer: 'dashboard_user', strategy_id: strategyId }),
      });
      if (res.ok) {
        const approvalRes = await fetch('/api/production/status');
        if (approvalRes.ok) setApproval(await approvalRes.json());
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // Role gating (ux-navigation §3.4): developer sees everything EXCEPT promote (admin-only).
  const { data: session } = useSession();
  const userRole = (session?.user as { role?: string } | undefined)?.role ?? 'free';
  const canPromote = userRole === 'admin';

  // Dynamic strategy lookup
  const strategyId = summary?.strategy_id || 'smart_simple_v11';
  // METHODOLOGY DEFAULT (operator convention): trained ≤ Dec-2024, BACKTEST = full 2025.
  // Registry bundles (Gold/BTC) are labeled with the FULL-history year but carry a true
  // `oos` (2025) slice — the DEFAULT display metrics are that OOS-2025 slice, so what the
  // operator sees matches "Backtest 2025" (full-history stays in the bundle for auditing).
  const oosSlice = (summary as unknown as {
    oos?: { year?: number; n_trading_days?: number;
      metrics?: { total_return_pct?: number; sharpe?: number; max_dd?: number; calmar?: number };
      statistical_tests?: { p_value?: number; significant?: boolean } };
  } | null)?.oos;
  const fullStats: StrategyStats | undefined = summary?.strategies[strategyId];
  const best: StrategyStats | undefined = oosSlice?.metrics
    ? ({
        ...fullStats,
        total_return_pct: oosSlice.metrics.total_return_pct ?? fullStats?.total_return_pct,
        sharpe: oosSlice.metrics.sharpe ?? fullStats?.sharpe,
        max_dd_pct: oosSlice.metrics.max_dd != null ? Math.abs(oosSlice.metrics.max_dd) : fullStats?.max_dd_pct,
        trading_days: oosSlice.n_trading_days ?? fullStats?.trading_days,
      } as StrategyStats)
    : fullStats;
  const initialCap = summary?.initial_capital ?? 10000;

  // Compute dynamic metrics from currently visible trades — REPLAY PREVIEW ONLY (audit I-4).
  // Active during replay (replayIndex >= 0) or when the USER narrowed the date range (compared
  // against the auto-initialized window, not summary.year — see autoRangeRef). These numbers are
  // labeled as preview and NEVER feed the approval decision (gates/Vote 2 read the bundle).
  const isDateFiltered =
    replayStartDate !== autoRangeRef.current.start || replayEndDate !== autoRangeRef.current.end;
  const dynamicMetrics = useMemo(() => {
    const tradesForMetrics = replayIndex < 0 ? filteredTrades : visibleTrades;
    if (tradesForMetrics.length === 0) return null;
    // Only compute dynamic when replaying or date-filtered
    if (replayIndex < 0 && !isDateFiltered) return null;
    return computeTradeMetrics(tradesForMetrics, initialCap);
  }, [replayIndex, filteredTrades, visibleTrades, initialCap, isDateFiltered]);

  // WINDOW metrics from the DAILY SERIES (SSOT for continuous strategies): computed over
  // exactly [replayStartDate, effEnd] — during replay effEnd tracks the last visible trade,
  // so the cards animate with the replay AND respond to any custom date range.
  const effEnd = (replayIndex >= 0 && seriesClockDate)
    ? seriesClockDate
    : (replayIndex >= 0 && visibleTrades.length > 0)
      ? String(visibleTrades[visibleTrades.length - 1].exit_timestamp
          ?? visibleTrades[visibleTrades.length - 1].timestamp).slice(0, 10)
      : replayEndDate;
  const windowMetrics = useMemo(() => {
    if (!dailySeries || dailySeries.length < 3) return null;
    const inWin = dailySeries.filter(r => r.d >= replayStartDate && r.d <= effEnd);
    if (inWin.length < 2) return null;
    // Base = the row BEFORE the window (position carried in), else the first in-window row.
    const firstIdx = dailySeries.findIndex(r => r.d === inWin[0].d);
    const baseEq = firstIdx > 0 ? dailySeries[firstIdx - 1].eq : inWin[0].eq;
    const rets: number[] = [];
    let prev = baseEq;
    for (const r of inWin) { if (prev > 0) rets.push(r.eq / prev - 1); prev = r.eq; }
    const totalRet = (inWin[inWin.length - 1].eq / baseEq - 1) * 100;
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(rets.length - 1, 1));
    // Periods/year inferred from calendar density (BTC ~365, Gold/COP ~252)
    const spanDays = (new Date(inWin[inWin.length - 1].d).getTime() - new Date(inWin[0].d).getTime()) / 86_400_000;
    const ann = spanDays > 0 && inWin.length / spanDays > 0.9 ? 365 : 252;
    const sharpe = sd > 0 ? (mean / sd) * Math.sqrt(ann) : null;
    let peak = baseEq, maxDd = 0;
    for (const r of inWin) { peak = Math.max(peak, r.eq); maxDd = Math.min(maxDd, r.eq / peak - 1); }
    const maxDdPct = Math.abs(maxDd) * 100;
    const annRet = (Math.pow(1 + totalRet / 100, ann / Math.max(inWin.length, 1)) - 1) * 100;
    return {
      total_return_pct: Math.round(totalRet * 100) / 100,
      sharpe: sharpe == null ? null : Math.round(sharpe * 100) / 100,
      max_dd_pct: Math.round(maxDdPct * 100) / 100,
      trading_days: inWin.length,
      calmar: maxDdPct > 1e-9 ? Math.round((annRet / maxDdPct) * 100) / 100 : null,
      final_equity: Math.round((10000 * (1 + totalRet / 100)) * 100) / 100,
      series: inWin,
    };
  }, [dailySeries, replayStartDate, effEnd]);

  // Display stats: dynamic during replay, static otherwise
  const displayStats: StrategyStats | undefined = windowMetrics ? ({
    ...(best ?? {}),
    final_equity: windowMetrics.final_equity,
    total_return_pct: windowMetrics.total_return_pct,
    sharpe: windowMetrics.sharpe,
    max_dd_pct: windowMetrics.max_dd_pct,
    trading_days: windowMetrics.trading_days,
    // trade-derived counters still come from the visible window trades
    win_rate_pct: dynamicMetrics?.win_rate_pct ?? best?.win_rate_pct,
    profit_factor: dynamicMetrics?.profit_factor ?? best?.profit_factor,
  } as StrategyStats) : dynamicMetrics ? {
    final_equity: dynamicMetrics.final_equity,
    total_return_pct: dynamicMetrics.total_return_pct,
    sharpe: dynamicMetrics.sharpe,
    profit_factor: dynamicMetrics.profit_factor,
    win_rate_pct: dynamicMetrics.win_rate_pct,
    max_dd_pct: dynamicMetrics.max_dd_pct,
    trading_days: dynamicMetrics.trading_days,
    exit_reasons: dynamicMetrics.exit_reasons,
    n_long: dynamicMetrics.n_long,
    n_short: dynamicMetrics.n_short,
  } as StrategyStats : best;

  // Dynamic p-value and significance
  const displayPValue = dynamicMetrics ? dynamicMetrics.p_value
    : (oosSlice?.statistical_tests?.p_value ?? summary?.statistical_tests?.p_value);
  const displayIsSignificant = dynamicMetrics ? dynamicMetrics.significant
    : (oosSlice?.statistical_tests?.significant ?? summary?.statistical_tests?.significant);
  const displayDA = dynamicMetrics ? dynamicMetrics.direction_accuracy_pct : summary?.direction_accuracy_pct;

  // Decision gates — ALWAYS the published bundle's Vote-1 results (audit I-4 / CTR-QUANT-
  // CONSTITUTION-001 §7). The human Vote 2 must be cast on the bundle's numbers; gates are
  // never recomputed from replay/filtered trades (that was silently re-evaluating gates on
  // an unofficial slice right next to the Approve button).
  const displayGates = approval?.gates ?? [];

  // Loading state
  if (loading) {
    return (
      <section className="w-full py-8 sm:py-10 lg:py-12 flex flex-col items-center">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeader
            title="Backtest OOS 2025"
            subtitle="Cargando datos..."
            icon={<BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" />}
          />
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6 lg:gap-8">
            {[1,2,3,4,5].map(i => <KPICard key={i} title="" value="" icon={<div />} isLoading />)}
          </div>
        </div>
      </section>
    );
  }

  // Error or no data — hide silently (backtest data may not exist)
  if (error || !summary) return null;

  return (
    <>
      {/* ════════════════════════════════════════════════════════════════
          Backtest OOS 2025 — Header + Status
      ════════════════════════════════════════════════════════════════ */}
      <section className="w-full py-8 sm:py-10 lg:py-12 flex flex-col items-center">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="mb-8 sm:mb-10 lg:mb-12 text-center flex flex-col items-center">
            <div className="flex items-center gap-3 mb-3">
              <h2 className="text-lg sm:text-xl lg:text-2xl font-bold text-white flex items-center justify-center gap-3">
                <span className="text-cyan-400"><BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" /></span>
                Backtest OOS 2025
              </h2>
              {approval && <StatusBadge status={approval.status} />}
            </div>
            {/* Strategy selector */}
            {registry && registry.strategies.length >= 1 && (
              <div className="mb-4">
                <StrategySelector
                  strategies={registry.strategies}
                  selectedId={selectedStrategyId}
                  onSelect={handleStrategyChange}
                />
              </div>
            )}
            {/* Version selector + promote (CTR-STRAT-REGISTRY-001) — additive, renders only
                when a manifest with versioned backtests is available */}
            {manifest && (manifest.backtests?.length ?? 0) >= 1 && (
              <div className="mb-4 flex items-center justify-center gap-2 flex-wrap">
                <span className="text-[11px] uppercase tracking-wide text-slate-500">Versión</span>
                <select
                  value={selectedVersion}
                  onChange={(e) => handleVersionChange(e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-cyan-500"
                >
                  {manifest.backtests.map((b) => {
                    const isActive = manifest.model_versions?.find((v) => v.version === b.model_version)?.active;
                    return (
                      <option key={b.immutable_id} value={b.model_version}>
                        {`v${b.model_version}${isActive ? ' ★' : ''}${b.replayable ? '' : ' (sin replay)'} · ${b.year}`}
                      </option>
                    );
                  })}
                </select>
                {/* "Promover a activa" button REMOVED (operator request 2026-07-07 + audit A4-04:
                    promote-from-TS races the Python registry rebuild). The version dropdown stays
                    for REVIEW/replay; the single promotion path is the two-vote flow
                    (Aprobar y Promover → Airflow L4b deploy). API endpoint kept for tooling. */}
              </div>
            )}
            <p className="text-sm sm:text-base text-slate-400 max-w-2xl">
              <MetricBadge phase="backtest" provenance={{ strategyId, version: selectedVersion || undefined }} className="mr-2 align-middle" />
              {summary.strategy_name || strategyId} | {dynamicMetrics ? visibleTrades.length : (best ? (best.n_long ?? 0) + (best.n_short ?? 0) : trades.length)} trades | DA {displayDA?.toFixed(1) ?? '-'}%
            </p>
            {/* p-value badge */}
            {displayPValue != null && (
              <div className="mt-3">
                <Badge
                  variant="outline"
                  className={cn(
                    'text-xs font-bold',
                    displayIsSignificant
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                      : 'bg-red-500/10 text-red-400 border-red-500/30'
                  )}
                >
                  p={displayPValue.toFixed(4)} {displayIsSignificant ? '(Significativo)' : '(No Significativo)'}
                </Badge>
              </div>
            )}
            <div className="mt-4 flex items-center justify-center gap-1">
              <div className="h-0.5 w-8 rounded-full bg-gradient-to-r from-transparent to-cyan-500/50" />
              <div className="h-0.5 w-16 rounded-full bg-gradient-to-r from-cyan-500/50 to-blue-500/50" />
              <div className="h-0.5 w-8 rounded-full bg-gradient-to-r from-blue-500/50 to-transparent" />
            </div>
          </div>

          {/* Replay-preview disclaimer (audit I-4): when metrics are recomputed from visible/
              filtered trades they are a PREVIEW — the official decision numbers are the bundle's. */}
          {dynamicMetrics && (
            <div className="mb-4 flex justify-center">
              <Badge
                variant="outline"
                className="text-[11px] font-semibold bg-amber-500/10 text-amber-300 border-amber-500/40"
              >
                PREVIEW DEL REPLAY — cifras recomputadas de los trades visibles; NO son las métricas
                oficiales del bundle (el Vote 2 y los gates usan el bundle publicado)
              </Badge>
            </div>
          )}

          {/* KPI Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6 lg:gap-8">
            <KPICard
              title="Retorno Total"
              value={`${(displayStats?.total_return_pct ?? 0) >= 0 ? '+' : ''}${displayStats?.total_return_pct?.toFixed(1) ?? 'N/A'}%`}
              subtitle={`$${(summary.initial_capital + (summary.initial_capital * (displayStats?.total_return_pct ?? 0) / 100)).toFixed(0)} final`}
              icon={<TrendingUp className="w-4 h-4 sm:w-5 sm:h-5" />}
              color={(displayStats?.total_return_pct ?? 0) >= 0 ? '#10b981' : '#ef4444'}
              trend={(displayStats?.total_return_pct ?? 0) >= 0 ? 'up' : 'down'}
            />
            <KPICard
              title="Sharpe Ratio"
              value={displayStats?.sharpe?.toFixed(2) ?? 'N/A'}
              subtitle="Risk-adjusted"
              icon={<Activity className="w-4 h-4 sm:w-5 sm:h-5" />}
              color={(displayStats?.sharpe ?? 0) >= 1.5 ? '#10b981' : (displayStats?.sharpe ?? 0) >= 1 ? '#f59e0b' : '#ef4444'}
              trend={(displayStats?.sharpe ?? 0) >= 1.5 ? 'up' : 'neutral'}
            />
            <KPICard
              title="Profit Factor"
              value={formatProfitFactor(displayStats?.profit_factor)}
              subtitle="Ganancia/Perdida"
              icon={<Target className="w-4 h-4 sm:w-5 sm:h-5" />}
              color={(displayStats?.profit_factor ?? 0) >= 1.5 ? '#10b981' : (displayStats?.profit_factor ?? 0) >= 1 ? '#f59e0b' : '#ef4444'}
              trend={(displayStats?.profit_factor ?? 0) >= 1.5 ? 'up' : 'neutral'}
            />
            <KPICard
              title="Calmar Ratio"
              value={(() => { const c = calmarRatio(displayStats?.total_return_pct, displayStats?.max_dd_pct, displayStats?.trading_days); return c == null ? 'N/A' : c.toFixed(2); })()}
              subtitle="Retorno anualizado / |DD| (métrica primaria)"
              icon={<Activity className="w-4 h-4 sm:w-5 sm:h-5" />}
              color={(calmarRatio(displayStats?.total_return_pct, displayStats?.max_dd_pct, displayStats?.trading_days) ?? 0) >= 1 ? '#10b981' : '#f59e0b'}
              trend={(calmarRatio(displayStats?.total_return_pct, displayStats?.max_dd_pct, displayStats?.trading_days) ?? 0) >= 1 ? 'up' : 'neutral'}
            />
            <KPICard
              title="Win Rate"
              value={`${displayStats?.win_rate_pct?.toFixed(0) ?? 'N/A'}%`}
              subtitle="Profitable"
              icon={<Percent className="w-4 h-4 sm:w-5 sm:h-5" />}
              color={(displayStats?.win_rate_pct ?? 0) >= 50 ? '#10b981' : '#f59e0b'}
              trend={(displayStats?.win_rate_pct ?? 0) >= 50 ? 'up' : 'down'}
            />
            <KPICard
              title="Max Drawdown"
              value={`-${displayStats?.max_dd_pct?.toFixed(1) ?? 'N/A'}%`}
              subtitle="Peak-to-trough"
              icon={<TrendingDown className="w-4 h-4 sm:w-5 sm:h-5" />}
              color={(displayStats?.max_dd_pct ?? 100) < 10 ? '#10b981' : (displayStats?.max_dd_pct ?? 100) < 15 ? '#f59e0b' : '#ef4444'}
              trend="down"
            />
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          Backtest Chart — 2025 Candles with Trade Markers
      ════════════════════════════════════════════════════════════════ */}
      <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <SectionHeader
            title="Senales de Trading (2025)"
            subtitle="Grafico de velas con senales de entrada y salida del backtest"
            icon={<BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" />}
          />
          <Suspense fallback={<ChartLoadingFallback />}>
            <div className="rounded-2xl overflow-hidden border border-slate-700/50 shadow-2xl">
              <TradingChartWithSignals
                key={`backtest-chart-${strategyId}-${selectedVersion || '2025'}`}
                symbol={manifest?.chart_symbol || 'USDCOP'}
                timeframe="5m"
                height={400}
                showSignals={true}
                showPositions={false}
                showStopLossTakeProfit={false}
                enableRealTime={false}
                isReplayMode={true}
                startDate={replayStartDate}
                endDate={chartEndDate}
                replayTrades={visibleTrades.flatMap(t => {
                  const entry = {
                    trade_id: t.trade_id,
                    timestamp: String(t.timestamp),
                    side: t.side,
                    entry_price: Number(t.entry_price),
                    pnl: 0,
                    status: 'closed' as const,
                  };
                  const exitTs = t.exit_timestamp ? String(t.exit_timestamp) : null;
                  if (!exitTs) return [entry];
                  const exitSide = t.side === 'SHORT' ? 'LONG' : 'SHORT';
                  const exit = {
                    trade_id: t.trade_id + 10000,
                    timestamp: exitTs,
                    side: exitSide,
                    entry_price: Number(t.exit_price),
                    pnl: Number(t.pnl_usd),
                    status: 'closed' as const,
                  };
                  return [entry, exit];
                })}
              />
            </div>
          </Suspense>

          {/* Replay Controls — Date-Based */}
          {trades.length > 0 && (
            <div className="mt-6 p-4 bg-slate-800/40 rounded-xl border border-slate-700/30">
              <div className="flex items-center gap-2 mb-3">
                <Calendar className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-medium text-white">Replay de Trades</span>
                <span className="text-[10px] text-slate-500 ml-auto">
                  {filteredTrades.length} trades en rango
                </span>
              </div>

              <div className="flex flex-wrap items-end gap-4">
                {/* Date range */}
                <div className="flex items-center gap-3">
                  <div>
                    <label className="block text-[10px] text-slate-500 uppercase tracking-wider mb-1">Desde</label>
                    <input
                      type="date"
                      value={replayStartDate}
                      onChange={e => { setReplayStartDate(e.target.value); setIsPlaying(false); setReplayIndex(-1); }}
                      className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500/50 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-slate-500 uppercase tracking-wider mb-1">Hasta</label>
                    <input
                      type="date"
                      value={replayEndDate}
                      onChange={e => { setReplayEndDate(e.target.value); setIsPlaying(false); setReplayIndex(-1); }}
                      className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500/50 transition-colors"
                    />
                  </div>
                </div>

                {/* Action buttons */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      if (isPlaying) {
                        setIsPlaying(false);
                      } else {
                        if (replayIndex >= replaySteps || replayIndex < 0) {
                          setReplayIndex(0);
                        }
                        setIsPlaying(true);
                      }
                    }}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all',
                      isPlaying
                        ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 border border-amber-500/30'
                        : 'bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white shadow-lg shadow-cyan-500/20'
                    )}
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    {isPlaying ? 'Pausar' : 'Replay'}
                  </button>
                  <button
                    onClick={() => { setIsPlaying(false); setReplayIndex(-1); }}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-700/50 hover:bg-slate-700 text-slate-400 hover:text-white text-sm transition-colors"
                    title="Mostrar todos los trades"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    Todos
                  </button>
                </div>

                {/* Speed controls */}
                <div className="flex items-center gap-1.5">
                  <Gauge className="w-3.5 h-3.5 text-slate-500" />
                  {[1, 2, 4].map(s => (
                    <button
                      key={s}
                      onClick={() => setPlaySpeed(s)}
                      className={cn(
                        'px-2.5 py-1.5 rounded text-xs font-bold transition-colors',
                        playSpeed === s
                          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                          : 'text-slate-500 hover:text-slate-300 border border-transparent'
                      )}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Progress bar (visible during replay) */}
              {replayIndex >= 0 && (
                <div className="mt-3 flex items-center gap-3">
                  <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-200"
                      style={{ width: `${replaySteps > 0 ? (Math.min(replayIndex, replaySteps) / replaySteps) * 100 : 0}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-slate-400">
                    {seriesClockDate
                      ? `${seriesClockDate} · ${visibleTrades.length}/${filteredTrades.length} trades`
                      : `${replayIndex}/${filteredTrades.length} trades`}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          Equity Curve + Trading Summary
      ════════════════════════════════════════════════════════════════ */}
      {trades.length > 0 && (
        <section className="w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Curva de Equity"
              subtitle={`${summary.strategy_name || strategyId} — Backtest 2025`}
              icon={<LineChart className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
              <div className="lg:col-span-2">
                <EquityCurveChart
                  seriesPoints={windowMetrics?.series}
                  trades={visibleTrades}
                  initialCapital={summary.initial_capital ?? 10000}
                />
              </div>
              <div>
                <TradingSummary
                  stats={displayStats}
                  trades={visibleTrades}
                  initialCapital={summary.initial_capital ?? 10000}
                  strategyName={summary.strategy_name || strategyId}
                />
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════════════════════════
          Approval + Gates
      ════════════════════════════════════════════════════════════════ */}
      {approval && (
        <section className="w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Aprobacion y Gates"
              subtitle="Validacion automatica y revision humana"
              icon={<Shield className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <div className="space-y-6 sm:space-y-8">
              {/* Approved/Rejected status display */}
              {approval.status === 'APPROVED' && (
                <Card className="bg-slate-900/50 border-emerald-500/30">
                  <CardContent className="p-4 sm:p-6">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0" />
                      <div>
                        <p className="text-sm text-emerald-300 font-medium">
                          Estrategia aprobada{approval.approved_by ? ` por ${approval.approved_by}` : ''}
                        </p>
                        {approval.approved_at && (
                          <p className="text-xs text-slate-400 mt-0.5">
                            {new Date(approval.approved_at).toLocaleDateString('es-CO', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                          </p>
                        )}
                        {approval.reviewer_notes && (
                          <p className="text-xs text-slate-500 mt-1 italic">&quot;{approval.reviewer_notes}&quot;</p>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
              {approval.status === 'REJECTED' && (
                <Card className="bg-slate-900/50 border-red-500/30">
                  <CardContent className="p-4 sm:p-6">
                    <div className="flex items-center gap-3">
                      <XCircle className="w-6 h-6 text-red-400 shrink-0" />
                      <div>
                        <p className="text-sm text-red-300 font-medium">
                          Estrategia rechazada{approval.rejected_by ? ` por ${approval.rejected_by}` : ''}
                        </p>
                        {approval.rejection_reason && (
                          <p className="text-xs text-slate-400 mt-0.5">{approval.rejection_reason}</p>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Interactive approval + deploy — ADMIN ONLY (Vote 2 / promote / deploy).
                  Developers propose and review but never approve or promote (RBAC §4). */}
              {canPromote && (
                <>
                  <ApprovalPanel
                    approval={approval}
                    onApprove={handleApprove}
                    onReject={handleReject}
                    isSubmitting={isSubmitting}
                  />
                  <DeployPanel approval={approval} />
                </>
              )}

              {/* Gates — always the published bundle's Vote-1 results (never replay-recomputed) */}
              {displayGates.length > 0 && (
                <GatesPanel gates={displayGates} />
              )}
            </div>
          </div>
        </section>
      )}

      {/* ════════════════════════════════════════════════════════════════
          Exit Reasons + Trade Table
      ════════════════════════════════════════════════════════════════ */}
      <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          {displayStats?.exit_reasons && Object.keys(displayStats.exit_reasons).length > 0 && (
            <div className="mb-8">
              <ExitReasonsSummary reasons={displayStats.exit_reasons} />
            </div>
          )}
          <TradeTable
            trades={replayIndex < 0 ? filteredTrades : visibleTrades}
            showAll={showAllTrades}
            onToggle={() => setShowAllTrades(!showAllTrades)}
          />
        </div>
      </section>
    </>
  );
}
