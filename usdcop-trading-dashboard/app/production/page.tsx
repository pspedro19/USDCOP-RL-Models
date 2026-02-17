'use client';

/**
 * Production — Live Real-Time Monitoring Dashboard (SDD)
 * =======================================================
 * Renders live state from H5 DB tables with file-based fallback.
 * Uses useLiveProduction() hook for adaptive polling (30s market open / 5m closed).
 *
 * Spec: .claude/rules/sdd-dashboard-integration.md
 * Contract: lib/contracts/strategy.contract.ts
 */

import { useState, lazy, Suspense } from 'react';
import { GlobalNavbar } from '@/components/navigation/GlobalNavbar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp, TrendingDown, Activity, Target, BarChart3,
  Shield, CheckCircle2, XCircle, AlertTriangle, ChevronDown,
  ChevronUp, ArrowUpDown, DollarSign,
  Percent, LineChart, Zap, RefreshCw, Clock, X, Database, FileText, Crosshair,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { getExitReasonColor } from '@/lib/contracts/strategy.contract';
import type { StrategyStats, StrategyTrade } from '@/lib/contracts/strategy.contract';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import type {
  GateResult,
  ProductionStatus,
  ApprovalState,
} from '@/lib/contracts/production-approval.contract';
import type { LiveDataSource } from '@/lib/contracts/production-monitor.contract';
import { useLiveProduction } from '@/hooks/useLiveProduction';
import { LivePositionCard } from '@/components/production/LivePositionCard';
import { GuardrailsCard } from '@/components/production/GuardrailsCard';

const TradingChartWithSignals = lazy(() => import('@/components/charts/TradingChartWithSignals'));

// ============================================================================
// LivePriceDisplay — polls /api/market/realtime-price every 60s
// ============================================================================
function LivePriceDisplay() {
  const [price, setPrice] = useState<number | null>(null);
  const [change, setChange] = useState<number | null>(null);
  const [changePct, setChangePct] = useState<number | null>(null);
  const [isMarketOpen, setIsMarketOpen] = useState(false);

  useState(() => {
    const fetchPrice = async () => {
      try {
        const res = await fetch('/api/market/realtime-price');
        if (res.ok) {
          const data = await res.json();
          setPrice(data.price);
          setChange(data.change);
          setChangePct(data.changePct);
          setIsMarketOpen(data.isMarketOpen);
        }
      } catch { /* price is optional */ }
    };
    fetchPrice();
    const interval = setInterval(fetchPrice, 60000);
    return () => clearInterval(interval);
  });

  if (price === null) return null;

  const isPositive = (change ?? 0) >= 0;

  return (
    <div className="flex items-center gap-2.5">
      <div className="flex items-center gap-1.5">
        <DollarSign className="w-3.5 h-3.5 text-slate-400" />
        <span className="font-mono text-sm font-bold text-white">
          {price.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
        </span>
      </div>
      {change !== null && (
        <span className={cn(
          'font-mono text-xs font-semibold',
          isPositive ? 'text-emerald-400' : 'text-red-400'
        )}>
          {isPositive ? '+' : ''}{change.toFixed(1)} ({isPositive ? '+' : ''}{changePct?.toFixed(2)}%)
        </span>
      )}
      <Badge
        variant="outline"
        className={cn(
          "text-[9px] font-semibold px-1.5 py-0.5",
          isMarketOpen
            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
            : "bg-slate-500/10 text-slate-400 border-slate-500/30"
        )}
      >
        <span className={cn(
          "w-1.5 h-1.5 rounded-full mr-1 inline-block",
          isMarketOpen ? "bg-emerald-500 animate-pulse" : "bg-slate-500"
        )} />
        {isMarketOpen ? 'Open' : 'Closed'}
      </Badge>
    </div>
  );
}

// ============================================================================
// KPI Card — matches dashboard/page.tsx KPICard exactly
// ============================================================================
interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  color?: string;
  isLoading?: boolean;
}

function KPICard({ title, value, subtitle, icon, trend, color = '#10B981', isLoading }: KPICardProps) {
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
        <div
          className="p-2.5 sm:p-3 rounded-xl mb-4"
          style={{ backgroundColor: `${color}15` }}
        >
          <div style={{ color }}>{icon}</div>
        </div>
        <span className="text-xs sm:text-sm font-medium text-slate-400 uppercase tracking-wider mb-2">
          {title}
        </span>
        <div
          className="text-2xl sm:text-3xl font-bold tracking-tight mb-1"
          style={{ color }}
        >
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
// Data Source Badge
// ============================================================================
function DataSourceBadge({ source }: { source: LiveDataSource }) {
  const configs: Record<LiveDataSource, { icon: React.ReactNode; label: string; cls: string }> = {
    db: {
      icon: <Database className="w-3 h-3" />,
      label: 'DB Live',
      cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    },
    file: {
      icon: <FileText className="w-3 h-3" />,
      label: 'Archivo',
      cls: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    },
    unavailable: {
      icon: <AlertTriangle className="w-3 h-3" />,
      label: 'Sin DB',
      cls: 'bg-red-500/10 text-red-400 border-red-500/30',
    },
  };
  const c = configs[source];
  return (
    <Badge variant="outline" className={cn('text-[9px] font-bold', c.cls)}>
      {c.icon}
      <span className="ml-1">{c.label}</span>
    </Badge>
  );
}

// ============================================================================
// Read-Only Approval Status Card
// ============================================================================
function ApprovalStatusCard({ approval }: { approval: ApprovalState }) {
  return (
    <Card className="bg-slate-900/50 border-slate-800">
      <CardHeader variant="minimal">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Shield className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <CardTitle variant="default" gradient={false} className="text-base text-white">
                Estado de Aprobacion
              </CardTitle>
              <p className="text-xs text-slate-400 mt-0.5">
                {approval.strategy_name || approval.strategy}
              </p>
            </div>
          </div>
          <StatusBadge status={approval.status} />
        </div>
      </CardHeader>
      <CardContent className="pt-0 p-4 sm:p-6">
        {approval.status === 'PENDING_APPROVAL' && (
          <div className="flex items-center gap-3 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
            <div>
              <p className="text-sm text-amber-300 font-medium">Pendiente de revision</p>
              <p className="text-xs text-slate-400 mt-0.5">
                Revisar y aprobar en <span className="text-cyan-400 font-medium">Dashboard</span>
              </p>
            </div>
          </div>
        )}
        {approval.status === 'APPROVED' && (
          <div className="flex items-center gap-3 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <div>
              <p className="text-sm text-emerald-300 font-medium">
                Aprobado{approval.approved_by ? ` por ${approval.approved_by}` : ''}
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
        )}
        {approval.status === 'REJECTED' && (
          <div className="flex items-center gap-3 p-3 rounded-lg bg-red-500/5 border border-red-500/20">
            <XCircle className="w-5 h-5 text-red-400 shrink-0" />
            <div>
              <p className="text-sm text-red-300 font-medium">
                Rechazado{approval.rejected_by ? ` por ${approval.rejected_by}` : ''}
              </p>
              {approval.rejection_reason && (
                <p className="text-xs text-slate-400 mt-0.5">{approval.rejection_reason}</p>
              )}
            </div>
          </div>
        )}
        {approval.status === 'LIVE' && (
          <div className="flex items-center gap-3 p-3 rounded-lg bg-cyan-500/5 border border-cyan-500/20">
            <Activity className="w-5 h-5 text-cyan-400 shrink-0 animate-pulse" />
            <p className="text-sm text-cyan-300 font-medium">Estrategia en produccion activa</p>
          </div>
        )}
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
// Helper: Format ISO timestamp to short COT display
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

// ============================================================================
// Trade Table
// ============================================================================
function TradeTable({ trades, showAll, onToggle, highlightIds }: {
  trades: StrategyTrade[];
  showAll: boolean;
  onToggle: () => void;
  highlightIds?: Set<number>;
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
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('asc'); }
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
                Historial de Trades
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
                <SortHeader field="equity_at_entry" label="Equity" />
                <SortHeader field="leverage" label="Lev" />
                <th className="px-3 py-2.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider text-right">Nocional</th>
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
                const isHighlighted = highlightIds?.has(t.trade_id);
                return (
                  <tr key={t.trade_id} className={cn(
                    "border-b border-slate-800/30 hover:bg-slate-800/30 transition-colors",
                    isHighlighted && "bg-cyan-500/5 animate-pulse"
                  )}>
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
                      ${Number(t.entry_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono text-right">
                      ${Number(t.exit_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}
                    </td>
                    <td className="px-3 py-2.5 text-slate-400 text-xs font-mono text-right">
                      ${Number(t.equity_at_entry ?? 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
                    <td className="px-3 py-2.5 text-slate-300 text-xs font-mono text-right">{Number(t.leverage).toFixed(2)}x</td>
                    <td className="px-3 py-2.5 text-cyan-300 text-xs font-mono font-medium text-right">
                      ${(Number(t.equity_at_entry ?? 0) * Number(t.leverage)).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </td>
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
                      {pnlPositive ? '+' : ''}{Number(t.pnl_usd).toFixed(2)}
                    </td>
                    <td className={cn('px-3 py-2.5 text-xs font-mono text-right', pnlPositive ? 'text-emerald-400' : 'text-red-400')}>
                      {pnlPositive ? '+' : ''}{Number(t.pnl_pct).toFixed(2)}%
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
// Exit Reasons
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
// Chart Card — graceful PNG fallback
// ============================================================================
function ChartCard({ src, alt, eager = false }: { src: string; alt: string; eager?: boolean }) {
  return (
    <Card className="bg-slate-900/50 border-slate-800 overflow-hidden">
      <img
        src={src}
        alt={alt}
        className="w-full h-auto"
        loading={eager ? 'eager' : 'lazy'}
        onError={(e) => { (e.target as HTMLImageElement).closest('[class*="Card"]')!.style.display = 'none'; }}
      />
    </Card>
  );
}

// ============================================================================
// Equity Curve (Recharts)
// ============================================================================
function EquityCurveChart({ points, initialCapital }: {
  points: { date: string; equity: number; pnl_pct: number }[];
  initialCapital: number;
}) {
  if (points.length === 0) return null;

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { date: string; equity: number; pnl_pct: number } }> }) => {
    if (!active || !payload?.[0]) return null;
    const d = payload[0].payload;
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs">
        <p className="text-slate-400 mb-1">{d.date}</p>
        <p className="text-white font-mono font-bold">${d.equity.toLocaleString('en-US', { maximumFractionDigits: 0 })}</p>
        <p className={cn('font-mono', d.pnl_pct >= 0 ? 'text-emerald-400' : 'text-red-400')}>
          {d.pnl_pct >= 0 ? '+' : ''}{d.pnl_pct.toFixed(2)}%
        </p>
      </div>
    );
  };

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
      <CardContent className="pt-0 pb-4">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={points} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
              domain={['auto', 'auto']}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={initialCapital}
              stroke="#475569"
              strokeDasharray="3 3"
              label={{ value: `$${(initialCapital / 1000).toFixed(0)}k`, fill: '#64748b', fontSize: 10, position: 'left' }}
            />
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#equityGradient)"
              dot={false}
              activeDot={{ r: 4, fill: '#10b981', stroke: '#0f172a', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Helper: format profit_factor safely
// ============================================================================
function formatProfitFactor(pf: number | null | undefined): string {
  if (pf == null) return 'N/A';
  if (pf > 100) return '>100';
  return pf.toFixed(2);
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
// Trading Summary Card
// ============================================================================
function ProductionTradingSummary({
  stats,
  trades,
  initialCapital = 10000,
  strategyName,
}: {
  stats: StrategyStats | undefined;
  trades: StrategyTrade[];
  initialCapital?: number;
  strategyName?: string;
}) {
  const totalTrades = (stats?.n_long ?? 0) + (stats?.n_short ?? 0);
  const profit = stats?.final_equity != null ? stats.final_equity - initialCapital : 0;
  const profitPct = stats?.total_return_pct ?? 0;
  const currentEquity = stats?.final_equity ?? initialCapital;
  const winRate = stats?.win_rate_pct ?? 0;

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
            <span className="text-sm">Equity Actual</span>
          </div>
          <span className="font-mono text-sm font-bold text-white">
            ${currentEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <Activity className="w-3.5 h-3.5" />
            <span className="text-sm">Total Operaciones</span>
          </div>
          <span className="font-mono text-sm text-slate-300">{totalTrades}</span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-500">
            <TrendingUp className="w-3.5 h-3.5" />
            <span className="text-sm">Win Rate</span>
          </div>
          <span className={cn('font-mono text-sm', winRate >= 50 ? 'text-green-400' : 'text-amber-400')}>
            {winRate.toFixed(1)}%
          </span>
        </div>
        <div className="border-t border-slate-700 pt-3 mt-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-400">Profit/Loss</span>
            <div className="text-right">
              <div className={cn('font-mono text-lg font-bold', profit >= 0 ? 'text-green-400' : 'text-red-400')}>
                {profit >= 0 ? '+' : ''}${profit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <div className={cn('text-xs font-mono', profitPct >= 0 ? 'text-green-400/70' : 'text-red-400/70')}>
                {profitPct >= 0 ? '+' : ''}{profitPct.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Data Freshness Indicator
// ============================================================================
function DataFreshnessIndicator({
  lastFetchTime,
  nextRefreshIn,
  dataSource,
}: {
  lastFetchTime: Date | null;
  nextRefreshIn: number;
  dataSource: LiveDataSource;
}) {
  if (!lastFetchTime) return null;

  const isStale = nextRefreshIn <= 0 || (Date.now() - lastFetchTime.getTime()) > 600_000;
  const minutes = Math.floor(Math.max(0, nextRefreshIn) / 60);
  const seconds = Math.max(0, nextRefreshIn) % 60;
  const lastTimeStr = lastFetchTime.toLocaleTimeString('es-CO', { hour: '2-digit', minute: '2-digit' });

  return (
    <div className="flex items-center gap-2 text-xs">
      <DataSourceBadge source={dataSource} />
      <span className="text-slate-500">|</span>
      <span className={cn(
        "w-2 h-2 rounded-full shrink-0",
        isStale ? "bg-amber-500" : "bg-emerald-500 animate-pulse"
      )} />
      <Clock className="w-3 h-3 text-slate-500" />
      <span className="text-slate-400 font-mono">{lastTimeStr}</span>
      <span className="text-slate-500">|</span>
      <span className="text-slate-500 font-mono">
        {minutes}m {seconds.toString().padStart(2, '0')}s
      </span>
    </div>
  );
}

// ============================================================================
// Loading Skeleton
// ============================================================================
function LoadingSkeleton() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-[#030712] via-[#0f172a] to-[#030712]">
      <GlobalNavbar currentPage="production" />
      <header className="sticky top-16 z-40 bg-[#030712]/95 backdrop-blur-xl border-b border-slate-800/50">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3 py-3 sm:py-4 animate-pulse">
            <div className="h-6 w-36 bg-slate-700 rounded" />
            <div className="h-5 w-20 bg-slate-700 rounded-full" />
          </div>
        </div>
      </header>
      <div className="h-16" aria-hidden="true" />
      <main className="w-full overflow-x-hidden">
        <section className="w-full py-8 sm:py-10 lg:py-12 flex flex-col items-center">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6 lg:gap-8">
              {[1,2,3,4,5].map(i => (
                <div key={i} className="rounded-xl border border-slate-700/50 bg-slate-900/80 p-6 animate-pulse">
                  <div className="flex flex-col items-center">
                    <div className="h-10 w-10 bg-slate-700 rounded-xl mb-4" />
                    <div className="h-3 w-16 bg-slate-700/60 rounded mb-2" />
                    <div className="h-8 w-20 bg-slate-700 rounded" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

// ============================================================================
// New Trade Toast
// ============================================================================
function NewTradeToast({ count, onDismiss }: { count: number; onDismiss: () => void }) {
  useState(() => {
    const timer = setTimeout(onDismiss, 8000);
    return () => clearTimeout(timer);
  });

  if (count <= 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 animate-in slide-in-from-bottom-4 fade-in duration-300">
      <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-cyan-500/10 border border-cyan-500/30 backdrop-blur-sm shadow-lg shadow-cyan-500/10">
        <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
        <span className="text-sm font-medium text-cyan-300">
          {count} {count === 1 ? 'nuevo trade detectado' : 'nuevos trades detectados'}
        </span>
        <button onClick={onDismiss} className="p-1 text-slate-400 hover:text-white transition-colors">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Main Page
// ============================================================================
export default function ProductionPage() {
  const {
    currentSignal,
    activePosition,
    equityCurve,
    guardrails,
    market,
    trades,
    stats,
    summary,
    approval,
    isLive,
    dataSource,
    loading,
    error,
    lastFetchTime,
    nextRefreshIn,
    strategyId,
    strategyName,
    refresh,
    newTradeCount,
    newTradeIds,
    dismissNewTrades,
  } = useLiveProduction();

  const [showAllTrades, setShowAllTrades] = useState(false);

  const best: StrategyStats | undefined = isLive
    ? stats
    : summary?.strategies[summary?.strategy_id || strategyId];

  const year = summary?.year ?? 2026;
  const initialCapital = summary?.initial_capital ?? 10000;
  const cacheBust = summary?.generated_at || '';
  const marketIsOpen = market?.is_open ?? false;

  if (loading) return <LoadingSkeleton />;

  if (error || (!summary && !isLive)) {
    const isPending = approval?.status === 'PENDING_APPROVAL';
    return (
      <div className="min-h-screen bg-gradient-to-b from-[#030712] via-[#0f172a] to-[#030712]">
        <GlobalNavbar currentPage="production" />
        <div className="flex flex-col items-center justify-center h-[70vh] gap-6 px-4">
          <div className="relative">
            <div className="absolute inset-0 rounded-full bg-cyan-500/10 blur-xl animate-pulse" />
            <div className="relative p-6 rounded-full bg-slate-800/50 border border-slate-700/50">
              <Shield className="w-12 h-12 text-cyan-400" />
            </div>
          </div>
          <div className="text-center max-w-lg">
            <h2 className="text-xl sm:text-2xl font-bold text-white mb-3">
              Esperando Primera Estrategia
            </h2>
            <p className="text-sm text-slate-400 leading-relaxed mb-4">
              No hay estrategias desplegadas en produccion todavia.
              {isPending
                ? ' Hay una estrategia pendiente de aprobacion en el Dashboard.'
                : ' Aprueba una estrategia desde el Dashboard para activar el monitoreo en tiempo real.'}
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              {isPending && (
                <a
                  href="/dashboard"
                  className="px-5 py-2.5 bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-cyan-500/20 flex items-center gap-2"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  Revisar en Dashboard
                </a>
              )}
              <button
                onClick={refresh}
                className="px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-slate-300 hover:text-white hover:border-cyan-500/50 transition-all flex items-center gap-2"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Reintentar
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#030712] via-[#0f172a] to-[#030712] overflow-x-hidden">
      <GlobalNavbar currentPage="production" />

      {/* Sticky Header */}
      <header className="sticky top-16 z-40 bg-[#030712]/95 backdrop-blur-xl border-b border-slate-800/50">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-3 py-3 sm:py-4">
            <div className="flex items-center gap-3 sm:gap-4 min-w-0">
              <h1 className="text-lg sm:text-xl lg:text-2xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent whitespace-nowrap">
                Produccion
              </h1>
              <div className="hidden sm:flex items-center gap-2">
                {approval && <StatusBadge status={approval.status} />}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden lg:block">
                <DataFreshnessIndicator
                  lastFetchTime={lastFetchTime}
                  nextRefreshIn={nextRefreshIn}
                  dataSource={dataSource}
                />
              </div>
              <div className="hidden sm:block">
                <LivePriceDisplay />
              </div>
              <Badge variant="outline" className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-[10px] font-bold">
                {strategyName}
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <div className="h-16" aria-hidden="true" />

      <main className="w-full overflow-x-hidden">

        {/* ═══════════════════════════════════════════════════════════════
            Section 1: KPI Cards
        ═══════════════════════════════════════════════════════════════ */}
        <section className="w-full py-8 sm:py-10 lg:py-12 flex flex-col items-center">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Metricas Clave"
              subtitle={`Rendimiento ${year} | ${summary?.n_trading_days ?? trades.length} dias de trading${summary?.direction_accuracy_pct != null ? ` | DA ${summary.direction_accuracy_pct}%` : ''}`}
              icon={<BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 sm:gap-6 lg:gap-8">
              <KPICard
                title="Retorno Total"
                value={`${(best?.total_return_pct ?? 0) >= 0 ? '+' : ''}${best?.total_return_pct?.toFixed(1) ?? 'N/A'}%`}
                subtitle={`$${((best?.total_return_pct ?? 0) * 100 + 10000).toFixed(0)} final`}
                icon={<TrendingUp className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={(best?.total_return_pct ?? 0) >= 0 ? '#10b981' : '#ef4444'}
                trend={(best?.total_return_pct ?? 0) >= 0 ? 'up' : 'down'}
              />
              <KPICard
                title="Sharpe Ratio"
                value={best?.sharpe?.toFixed(2) ?? 'N/A'}
                subtitle="Risk-adjusted"
                icon={<Activity className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={(best?.sharpe ?? 0) >= 1.5 ? '#10b981' : (best?.sharpe ?? 0) >= 1 ? '#f59e0b' : '#ef4444'}
                trend={(best?.sharpe ?? 0) >= 1.5 ? 'up' : 'neutral'}
              />
              <KPICard
                title="Profit Factor"
                value={formatProfitFactor(best?.profit_factor)}
                subtitle="Ganancia/Perdida"
                icon={<Target className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={(best?.profit_factor ?? 0) >= 1.5 ? '#10b981' : (best?.profit_factor ?? 0) >= 1 ? '#f59e0b' : '#ef4444'}
                trend={(best?.profit_factor ?? 0) >= 1.5 ? 'up' : 'neutral'}
              />
              <KPICard
                title="Win Rate"
                value={`${best?.win_rate_pct?.toFixed(0) ?? 'N/A'}%`}
                subtitle="Profitable"
                icon={<Percent className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={(best?.win_rate_pct ?? 0) >= 50 ? '#10b981' : '#f59e0b'}
                trend={(best?.win_rate_pct ?? 0) >= 50 ? 'up' : 'down'}
              />
              <KPICard
                title="Max Drawdown"
                value={`-${best?.max_dd_pct?.toFixed(1) ?? 'N/A'}%`}
                subtitle="Peak-to-trough"
                icon={<TrendingDown className="w-4 h-4 sm:w-5 sm:h-5" />}
                color={(best?.max_dd_pct ?? 100) < 10 ? '#10b981' : (best?.max_dd_pct ?? 100) < 15 ? '#f59e0b' : '#ef4444'}
                trend="down"
              />
            </div>
          </div>
        </section>

        {/* ═══════════════════════════════════════════════════════════════
            Section 2: This Week (Live Position Card) — NEW
        ═══════════════════════════════════════════════════════════════ */}
        {isLive && (
          <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
            <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <SectionHeader
                title="Esta Semana"
                subtitle="Senal activa y posicion en tiempo real"
                icon={<Crosshair className="w-5 h-5 sm:w-6 sm:h-6" />}
              />
              <LivePositionCard
                signal={currentSignal}
                position={activePosition}
                marketOpen={market?.is_open ?? false}
              />
            </div>
          </section>
        )}

        {/* ═══════════════════════════════════════════════════════════════
            Section 3: Candlestick Chart with Trading Signals
            - Market open + live DB: enableRealTime=true, refreshes 5m bars every 30s
            - Market closed / file fallback: replay mode with trade entry/exit markers
        ═══════════════════════════════════════════════════════════════ */}
        <section className={cn(
          "w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center",
          !isLive && "bg-slate-900/30"
        )}>
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Senales de Trading"
              subtitle={marketIsOpen
                ? "Velas 5m en tiempo real (refresco cada 30s)"
                : "Grafico de velas con senales de entrada y salida"
              }
              icon={<BarChart3 className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <Suspense fallback={<ChartLoadingFallback />}>
              <div className="rounded-2xl overflow-hidden border border-slate-700/50 shadow-2xl">
                <TradingChartWithSignals
                  key={`prod-chart-${strategyId}-${marketIsOpen ? 'live' : 'replay'}`}
                  symbol="USDCOP"
                  timeframe="5m"
                  height={400}
                  showSignals={true}
                  showPositions={false}
                  showStopLossTakeProfit={false}
                  enableRealTime={marketIsOpen}
                  isReplayMode={!marketIsOpen}
                  startDate={`${year}-01-01`}
                  endDate={undefined}
                  replayTrades={!marketIsOpen ? trades.flatMap(t => {
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
                  }) : []}
                />
              </div>
            </Suspense>
          </div>
        </section>

        {/* ═══════════════════════════════════════════════════════════════
            Section 4: Approval + Gates
        ═══════════════════════════════════════════════════════════════ */}
        {approval && (
          <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
            <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <SectionHeader
                title="Estado y Gates"
                subtitle="Estado de aprobacion (revision en Dashboard)"
                icon={<Shield className="w-5 h-5 sm:w-6 sm:h-6" />}
              />
              <div className="space-y-6 sm:space-y-8">
                <ApprovalStatusCard approval={approval} />
                {approval.gates.length > 0 && (
                  <GatesPanel gates={approval.gates} />
                )}
              </div>
            </div>
          </section>
        )}

        {/* ═══════════════════════════════════════════════════════════════
            Section 5: Equity Curve (Recharts when live, PNG fallback)
        ═══════════════════════════════════════════════════════════════ */}
        <section className="w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Curva de Equity"
              subtitle={`${strategyName} vs Buy & Hold`}
              icon={<LineChart className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
              <div className="lg:col-span-2">
                {equityCurve && equityCurve.points.length > 1 ? (
                  <EquityCurveChart
                    points={equityCurve.points}
                    initialCapital={equityCurve.initial_capital}
                  />
                ) : (
                  <ChartCard
                    src={`/data/production/equity_curve_${year}.png?t=${cacheBust}`}
                    alt={`Equity Curve ${year}`}
                    eager
                  />
                )}
              </div>
              <div>
                <ProductionTradingSummary
                  stats={best}
                  trades={trades}
                  initialCapital={initialCapital}
                  strategyName={strategyName}
                />
              </div>
            </div>
          </div>
        </section>

        {/* ═══════════════════════════════════════════════════════════════
            Section 6: Guardrails — NEW
        ═══════════════════════════════════════════════════════════════ */}
        {guardrails && (
          <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
            <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <SectionHeader
                title="Guardrails"
                subtitle="Estado del circuit breaker y metricas de proteccion"
                icon={<Shield className="w-5 h-5 sm:w-6 sm:h-6" />}
              />
              <div className="max-w-xl mx-auto">
                <GuardrailsCard guardrails={guardrails} />
              </div>
            </div>
          </section>
        )}

        {/* ═══════════════════════════════════════════════════════════════
            Section 7: Monthly PnL + Exit Reasons
        ═══════════════════════════════════════════════════════════════ */}
        <section className={cn(
          "w-full py-12 sm:py-16 lg:py-20 flex flex-col items-center",
          !guardrails && "bg-slate-900/30"
        )}>
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Rendimiento y Distribucion"
              subtitle="PnL mensual, distribucion de trades y razones de salida"
              icon={<DollarSign className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8 lg:gap-10">
              <div>
                <ChartCard
                  src={`/data/production/monthly_pnl_${year}.png?t=${cacheBust}`}
                  alt={`Monthly PnL ${year}`}
                />
              </div>
              <div>
                <ChartCard
                  src={`/data/production/trade_distribution_${year}.png?t=${cacheBust}`}
                  alt={`Trade Distribution ${year}`}
                />
              </div>
              <div>
                {best?.exit_reasons && Object.keys(best.exit_reasons).length > 0 && (
                  <ExitReasonsSummary reasons={best.exit_reasons} />
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ═══════════════════════════════════════════════════════════════
            Section 8: Trade History
        ═══════════════════════════════════════════════════════════════ */}
        <section className="w-full py-12 sm:py-16 lg:py-20 bg-slate-900/30 flex flex-col items-center">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <SectionHeader
              title="Historial de Trades"
              subtitle="Actividad de trading en horario Colombia (COT)"
              icon={<ArrowUpDown className="w-5 h-5 sm:w-6 sm:h-6" />}
            />
            <TradeTable
              trades={trades}
              showAll={showAllTrades}
              onToggle={() => setShowAllTrades(!showAllTrades)}
              highlightIds={newTradeIds}
            />
          </div>
        </section>

        {/* Footer */}
        <footer className="w-full py-12 sm:py-16 border-t border-slate-800/50 flex flex-col items-center pb-32">
          <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <p className="text-sm sm:text-base text-slate-400 font-medium">
              USDCOP Trading System
            </p>
            <p className="mt-2 text-xs sm:text-sm text-slate-500">
              {strategyName} | {year}
            </p>
            <div className="mt-6 flex justify-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700/50">
                <span className={cn(
                  "w-2 h-2 rounded-full",
                  isLive ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                )} />
                <span className="text-xs text-slate-400">
                  {isLive ? 'DB Live' : 'File-based'} | Auto-refresh {market?.is_open ? '30s' : '5m'}
                </span>
              </div>
            </div>
          </div>
        </footer>
      </main>

      {/* New trade toast notification */}
      {newTradeCount > 0 && (
        <NewTradeToast count={newTradeCount} onDismiss={dismissNewTrades} />
      )}
    </div>
  );
}
