'use client';

/**
 * ForecastingBacktestSection — Self-contained 2025 Backtest Viewer + Approval
 * ============================================================================
 * GM re-skin (CTR-GM-UI-001, prototipo Var B líneas 363-484 + view-model
 * 2304-2398). SOLO presentación: todos los fetches, endpoints, estados y
 * handlers de aprobación (Vote 2/2, approval-gates.md) están INTACTOS.
 * Los números SIEMPRE vienen del bundle publicado (audit I-4 / CTR-QUANT-
 * CONSTITUTION-001 §7); el replay solo produce PREVIEWS etiquetados.
 *
 * Sections:
 * 1. Meta (MetricBadge + status + p-value) + version selector
 * 2. KPI row (builder SSOT strategy-kpis, mismo que Producción)
 * 3. Candle panel + replay bar (gradiente + Desde/Hasta + progreso accent)
 * 4. Equity panel
 * 5. Grid 1.5fr/1fr: Resumen del periodo · Razones de salida · Tabla de trades
 *    | Gates (Voto 1) · Aprobación humana (Voto 2/2, SOLO admin) · Deploy
 *
 * Spec: .claude/rules/approval-gates.md · .claude/specs/platform/dashboard-integration.md
 * Contract: lib/contracts/strategy.contract.ts
 */

import { useSession } from 'next-auth/react';
import { MetricBadge } from '@/components/ui/MetricBadge';
import { useState, useEffect, useCallback, useRef, useMemo, lazy, Suspense } from 'react';
import {
  CheckCircle2, XCircle, AlertTriangle, Activity, ChevronDown,
  ChevronUp, Loader2, RefreshCw, Play, Pause,
  RotateCcw, Calendar, Gauge, Rocket, ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  AreaChart, Area, XAxis, YAxis,
  Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { GmBadge, GmKpi, GmPanel, GmSkeleton } from '@/components/gm';
import { buildStrategyKpis } from '@/components/gm/views/strategy-kpis';
import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, GM_HEX, type GmTone } from '@/lib/ui/gm-tokens';
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

// ── Colores de recharts (props SVG no aceptan clases Tailwind) desde GM_HEX —
//    único lugar con hex (CTR-GM-UI-001). Mismo patrón que ProductionView.tsx.
const HEX = {
  accent: GM_HEX.accent,
  tick: GM_HEX.tick,
  ref: GM_HEX.ref,
} as const;

// ── Diccionario local ES/EN (gm-core) — cero strings hardcodeados en la vista.
const DICT = defineGmDict({
  es: {
    loading: 'Cargando backtest…',
    trades: 'trades',
    significant: 'Significativo',
    notSignificant: 'No significativo',
    previewBanner: 'PREVIEW DEL REPLAY — cifras recomputadas de los trades visibles; NO son las métricas oficiales del bundle (el Vote 2 y los gates usan el bundle publicado)',
    ratiosNote: 'N<20 trades — solo conteo y P&L (quant-constitution §6)',
    versionLabel: 'Versión',
    noReplay: '(sin replay)',
    chartTitle: 'Precio y operaciones',
    lwNote: 'Lightweight Charts',
    replayTitle: 'Replay del backtest',
    inRange: 'trades en rango',
    from: 'Desde',
    to: 'Hasta',
    play: 'Reproducir replay',
    pause: 'Pausar',
    showAll: 'Todos',
    showAllTitle: 'Mostrar todos los trades',
    equityTitle: 'Curva de equity',
    equityLabel: 'Equity',
    summaryTitle: 'Resumen del periodo',
    sCapIni: 'Capital inicial',
    sEqFin: 'Equity final',
    sEntryStart: 'Precio entrada (inicio)',
    sExitEnd: 'Precio salida (fin)',
    sOps: 'Operaciones',
    sWr: 'Win rate',
    sDays: 'Días de trading',
    sRet: 'Retorno del periodo',
    sPnl: 'Profit / Loss',
    backtestNote: 'Backtest',
    tradesOos: 'trades OOS',
    exitTitle: 'Razones de salida',
    tradesTitle: 'Historial de operaciones',
    ops: 'operaciones',
    less: 'Menos',
    thDate: 'Fecha',
    thSide: 'Lado',
    thEntry: 'Entrada',
    thExit: 'Salida',
    thPnl: 'P&L',
    thReason: 'Motivo',
    gatesTitle: 'Gates automáticos (Voto 1/2)',
    passed: 'pasaron',
    threshold: 'umbral',
    approvalTitle: 'Aprobación humana',
    approvalSub: 'Revisión humana requerida (Voto 2/2)',
    approvalDesc: 'Los gates automáticos ya emitieron el Voto 1. Tu voto (2/2) decide sobre los números del bundle publicado y promueve la estrategia a producción.',
    notesPh: 'Notas del revisor (opcional)…',
    approve: 'Aprobar (Voto 2/2)',
    reject: 'Rechazar',
    confirm: 'Confirmar',
    confirmReject: 'Confirmar rechazo',
    cancel: 'Cancelar',
    approvedMsg: 'Estrategia aprobada',
    rejectedMsg: 'Estrategia rechazada',
    by: 'por',
    statusPending: 'Pendiente',
    statusApproved: 'Aprobado',
    statusRejected: 'Rechazado',
    statusLive: 'En vivo',
    deployTitle: 'Desplegar a producción',
    deployDesc: 'Reentrenar con datos completos (2020-2025) y generar trades 2026',
    deploying: 'Desplegando…',
    deployDone: '¡Deploy completado!',
    deployDoneDesc: 'Estrategia desplegada exitosamente. Los datos de producción 2026 están listos.',
    seeProd: 'Ver producción',
    deployErr: 'Error en deploy',
    unknownErr: 'Error desconocido durante el despliegue.',
    retry: 'Reintentar',
    phInit: 'Inicializando…',
    phRetrain: 'Reentrenando modelos (2020-2025)…',
    phExport: 'Exportando datos de producción…',
    phDone: 'Completado',
    phGeneric: 'Procesando…',
    stepInit: 'Inicializar',
    stepRetrain: 'Reentrenar',
    stepExport: 'Exportar',
    stepReady: 'Listo',
  },
  en: {
    loading: 'Loading backtest…',
    trades: 'trades',
    significant: 'Significant',
    notSignificant: 'Not significant',
    previewBanner: 'REPLAY PREVIEW — figures recomputed from visible trades; NOT the official bundle metrics (Vote 2 and the gates use the published bundle)',
    ratiosNote: 'N<20 trades — count and P&L only (quant-constitution §6)',
    versionLabel: 'Version',
    noReplay: '(no replay)',
    chartTitle: 'Price & trades',
    lwNote: 'Lightweight Charts',
    replayTitle: 'Backtest replay',
    inRange: 'trades in range',
    from: 'From',
    to: 'To',
    play: 'Play replay',
    pause: 'Pause',
    showAll: 'All',
    showAllTitle: 'Show all trades',
    equityTitle: 'Equity curve',
    equityLabel: 'Equity',
    summaryTitle: 'Period summary',
    sCapIni: 'Initial capital',
    sEqFin: 'Final equity',
    sEntryStart: 'Entry price (start)',
    sExitEnd: 'Exit price (end)',
    sOps: 'Trades',
    sWr: 'Win rate',
    sDays: 'Trading days',
    sRet: 'Period return',
    sPnl: 'Profit / Loss',
    backtestNote: 'Backtest',
    tradesOos: 'OOS trades',
    exitTitle: 'Exit reasons',
    tradesTitle: 'Trade history',
    ops: 'trades',
    less: 'Less',
    thDate: 'Date',
    thSide: 'Side',
    thEntry: 'Entry',
    thExit: 'Exit',
    thPnl: 'P&L',
    thReason: 'Reason',
    gatesTitle: 'Automatic gates (Vote 1/2)',
    passed: 'passed',
    threshold: 'threshold',
    approvalTitle: 'Human approval',
    approvalSub: 'Human review required (Vote 2/2)',
    approvalDesc: 'The automatic gates already cast Vote 1. Your vote (2/2) decides on the published bundle numbers and promotes the strategy to production.',
    notesPh: 'Reviewer notes (optional)…',
    approve: 'Approve (Vote 2/2)',
    reject: 'Reject',
    confirm: 'Confirm',
    confirmReject: 'Confirm rejection',
    cancel: 'Cancel',
    approvedMsg: 'Strategy approved',
    rejectedMsg: 'Strategy rejected',
    by: 'by',
    statusPending: 'Pending',
    statusApproved: 'Approved',
    statusRejected: 'Rejected',
    statusLive: 'Live',
    deployTitle: 'Deploy to production',
    deployDesc: 'Retrain on full data (2020-2025) and generate 2026 trades',
    deploying: 'Deploying…',
    deployDone: 'Deploy completed!',
    deployDoneDesc: 'Strategy deployed successfully. 2026 production data is ready.',
    seeProd: 'View production',
    deployErr: 'Deploy error',
    unknownErr: 'Unknown error during deployment.',
    retry: 'Retry',
    phInit: 'Initializing…',
    phRetrain: 'Retraining models (2020-2025)…',
    phExport: 'Exporting production data…',
    phDone: 'Done',
    phGeneric: 'Processing…',
    stepInit: 'Initialize',
    stepRetrain: 'Retrain',
    stepExport: 'Export',
    stepReady: 'Ready',
  },
});

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
// Strategy Selector Dropdown (GM re-skin — misma lógica de selección)
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
        className={`${GM.ctaGhost} ${GM.focus} flex items-center gap-2.5 min-h-[44px] px-3.5 py-1.5`}
        aria-expanded={isOpen}
      >
        <span className="w-2 h-2 rounded-full bg-[var(--gm-pos)] motion-safe:animate-pulse shrink-0" aria-hidden />
        <span className="text-left">
          <span className={`block text-[0.8125rem] font-bold leading-tight ${GM.text}`}>
            {selected?.strategy_name || '—'}
          </span>
          <span className={`block ${GMT.micro} ${GM.textMuted} font-mono leading-tight mt-0.5`}>
            {selected?.pipeline} · +{selected?.return_pct.toFixed(1)}% · Sharpe {selected?.sharpe.toFixed(2)}
          </span>
        </span>
        {selected?.status && (
          <GmBadge tone={selected.status === 'APPROVED' ? 'pos' : 'warn'}>{selected.status}</GmBadge>
        )}
        {strategies.length > 1 && (
          <ChevronDown className={cn('w-4 h-4 transition-transform', isOpen && 'rotate-180')} aria-hidden />
        )}
      </button>

      {isOpen && (
        <div className={`absolute top-full left-0 mt-1.5 z-40 min-w-[300px] ${GM.popover} p-1.5 flex flex-col gap-0.5`}>
          {strategies.map((s) => (
            <button
              key={s.strategy_id}
              className={cn(
                `${GM.rowHover} ${GM.focus} w-full flex items-center justify-between gap-3 px-3 py-2.5 rounded-[9px] text-left`,
                s.strategy_id === selectedId && 'bg-[rgba(34,211,238,.07)]'
              )}
              onClick={() => {
                onSelect(s.strategy_id);
                setIsOpen(false);
              }}
            >
              <span className="min-w-0">
                <span className={`block text-[0.8125rem] font-bold ${GM.text}`}>{s.strategy_name}</span>
                <span className={`block ${GMT.micro} ${GM.textMuted} font-mono mt-0.5`}>
                  {s.pipeline} · +{s.return_pct.toFixed(1)}% · Sharpe {s.sharpe.toFixed(2)} · p={s.p_value.toFixed(4)}
                </span>
              </span>
              <GmBadge tone={s.status === 'APPROVED' ? 'pos' : 'warn'} className="shrink-0">
                {s.status}
              </GmBadge>
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
  const t = useGmT(DICT);
  const config: Record<ProductionStatus, { tone: GmTone; label: string }> = {
    PENDING_APPROVAL: { tone: 'warn', label: t('statusPending') },
    APPROVED: { tone: 'pos', label: t('statusApproved') },
    REJECTED: { tone: 'neg', label: t('statusRejected') },
    LIVE: { tone: 'accent', label: t('statusLive') },
  };
  const c = config[status];

  return (
    <GmBadge tone={c.tone}>
      {status === 'PENDING_APPROVAL' && <AlertTriangle className="w-3 h-3" aria-hidden />}
      {status === 'APPROVED' && <CheckCircle2 className="w-3 h-3" aria-hidden />}
      {status === 'REJECTED' && <XCircle className="w-3 h-3" aria-hidden />}
      {status === 'LIVE' && <Activity className="w-3 h-3 motion-safe:animate-pulse" aria-hidden />}
      {c.label}
    </GmBadge>
  );
}

// ============================================================================
// Interactive Approval Panel (Vote 2/2 — SOLO admin; lógica intacta)
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
  const t = useGmT(DICT);
  const [notes, setNotes] = useState('');
  const [showConfirm, setShowConfirm] = useState<'approve' | 'reject' | null>(null);

  if (approval.status !== 'PENDING_APPROVAL') return null;

  const recTone: GmTone = approval.backtest_recommendation === 'PROMOTE'
    ? 'pos'
    : approval.backtest_recommendation === 'REJECT'
    ? 'neg'
    : 'warn';

  return (
    <section className="gm-contain rounded-2xl border border-[rgba(34,211,238,.28)] bg-[rgba(34,211,238,.05)] p-[18px]">
      <div className="flex items-start justify-between gap-3 mb-1.5">
        <div className={`${GMT.panelTitle} ${GM.accent}`}>{t('approvalTitle')}</div>
        <GmBadge tone={recTone}>
          L4: {approval.backtest_recommendation} ({Math.round(approval.backtest_confidence * 100)}%)
        </GmBadge>
      </div>
      <p className={`${GMT.meta} ${GM.textSec} leading-relaxed m-0 mb-4`}>
        {t('approvalSub')}. {t('approvalDesc')}
      </p>

      <input
        type="text"
        placeholder={t('notesPh')}
        value={notes}
        onChange={e => setNotes(e.target.value)}
        className={`${GM.input} ${GM.focus} w-full min-h-[44px] mb-2.5`}
      />

      {showConfirm === 'approve' ? (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => { onApprove(notes); setShowConfirm(null); }}
            disabled={isSubmitting}
            className={`${GM.ctaPrimary} ${GM.focus} w-full h-11 text-[0.8125rem] disabled:opacity-50 inline-flex items-center justify-center gap-1.5`}
          >
            {isSubmitting ? <Loader2 className="w-3.5 h-3.5 motion-safe:animate-spin" aria-hidden /> : <CheckCircle2 className="w-3.5 h-3.5" aria-hidden />}
            {t('confirm')}
          </button>
          <button
            onClick={() => setShowConfirm(null)}
            className={`${GM.ctaGhost} ${GM.focus} w-full h-11 text-[0.8125rem] font-semibold`}
          >
            {t('cancel')}
          </button>
        </div>
      ) : showConfirm === 'reject' ? (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => { onReject(notes); setShowConfirm(null); }}
            disabled={isSubmitting}
            className={`${GM.ctaDanger} ${GM.focus} w-full h-11 text-[0.8125rem] font-bold disabled:opacity-50 inline-flex items-center justify-center gap-1.5`}
          >
            {isSubmitting ? <Loader2 className="w-3.5 h-3.5 motion-safe:animate-spin" aria-hidden /> : <XCircle className="w-3.5 h-3.5" aria-hidden />}
            {t('confirmReject')}
          </button>
          <button
            onClick={() => setShowConfirm(null)}
            className={`${GM.ctaGhost} ${GM.focus} w-full h-11 text-[0.8125rem] font-semibold`}
          >
            {t('cancel')}
          </button>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setShowConfirm('approve')}
            className={`${GM.ctaPrimary} ${GM.focus} w-full h-11 text-[0.8125rem] inline-flex items-center justify-center gap-1.5`}
          >
            <CheckCircle2 className="w-3.5 h-3.5" aria-hidden />
            {t('approve')}
          </button>
          <button
            onClick={() => setShowConfirm('reject')}
            className={`${GM.ctaDanger} ${GM.focus} w-full h-11 text-[0.8125rem] font-bold inline-flex items-center justify-center gap-1.5`}
          >
            <XCircle className="w-3.5 h-3.5" aria-hidden />
            {t('reject')}
          </button>
        </div>
      )}
    </section>
  );
}

// ============================================================================
// Gates Panel — SIEMPRE los resultados Vote-1 del bundle publicado
// ============================================================================
function GatesPanel({ gates }: { gates: GateResult[] }) {
  const t = useGmT(DICT);
  const passed = gates.filter(g => g.passed).length;
  const total = gates.length;

  return (
    <GmPanel
      title={t('gatesTitle')}
      actions={
        <GmBadge tone={passed === total ? 'pos' : 'warn'}>
          {passed}/{total} {t('passed')}
        </GmBadge>
      }
    >
      <div className="flex flex-col gap-2.5">
        {gates.map((g) => (
          <div key={g.gate} className="flex items-center gap-2.5">
            {g.passed
              ? <CheckCircle2 className={`w-[17px] h-[17px] shrink-0 ${GM.pos}`} aria-hidden />
              : <XCircle className={`w-[17px] h-[17px] shrink-0 ${GM.neg}`} aria-hidden />}
            <span className={`${GMT.meta} ${GM.textStrong} font-mono min-w-0 flex-1`}>{g.label}</span>
            <span className={`${GMT.micro} ${GMT.mono} ${g.passed ? GM.pos : GM.neg} text-right shrink-0`}>
              {typeof g.value === 'number'
                ? g.gate === 'min_trades'
                  ? g.value
                  : `${g.value.toFixed(g.gate === 'statistical_significance' ? 4 : 1)}${g.gate !== 'min_trades' ? (g.gate === 'statistical_significance' ? '' : '%') : ''}`
                : g.value}
              <span className={`block ${GM.textMuted} font-normal`}>
                {t('threshold')}: {g.gate === 'min_trades' ? g.threshold : g.gate === 'statistical_significance' ? g.threshold : `${g.threshold}%`}
              </span>
            </span>
          </div>
        ))}
      </div>
    </GmPanel>
  );
}

// ============================================================================
// Exit Reasons Summary — barras con colores del EXIT_REASON_COLORS (contrato)
// ============================================================================
function ExitReasonsSummary({ reasons }: { reasons: Record<string, number> }) {
  const t = useGmT(DICT);
  const total = Object.values(reasons).reduce((s, v) => s + v, 0);

  return (
    <GmPanel title={t('exitTitle')}>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {Object.entries(reasons).map(([reason, count]) => {
          const color = getExitReasonColor(reason);
          const pct = total > 0 ? (count / total) * 100 : 0;
          return (
            <div key={reason}>
              <div className={`flex items-center justify-between gap-2 ${GMT.meta} mb-1.5`}>
                <span className={`${GM.textStrong} font-mono truncate`}>{reason}</span>
                <span className={`${GM.textMuted} ${GMT.mono} shrink-0`}>{count} · {Math.round(pct)}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-[rgba(148,163,184,.12)] overflow-hidden">
                <div
                  className={cn('h-full rounded-full transition-all duration-[var(--gm-dur-slow)]', color.bar)}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </GmPanel>
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
  const t = useGmT(DICT);
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
      ...trades.map((tr, i) => ({
        label: formatTradeDate((tr.exit_timestamp || tr.timestamp) as string),
        equity: Number(tr.equity_at_exit),
        idx: i + 1,
      }))
    ];

  const equities = dataPoints.map(d => d.equity);
  const minEq = Math.min(...equities);
  const maxEq = Math.max(...equities);
  const padding = (maxEq - minEq) * 0.1 || 100;

  return (
    <GmPanel title={t('equityTitle')}>
      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={dataPoints}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={HEX.accent} stopOpacity={0.3} />
                <stop offset="95%" stopColor={HEX.accent} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: HEX.tick }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[minEq - padding, maxEq + padding]}
              tick={{ fontSize: 10, fill: HEX.tick }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
            />
            <RechartsTooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload[0]) return null;
                const d = payload[0].payload as { label: string; equity: number };
                return (
                  <div className={`${GM.popover} px-3 py-2`}>
                    <p className={`${GMT.micro} ${GM.textSec} m-0 mb-1`}>{d.label}</p>
                    <p className={`${GMT.mono} text-[0.8125rem] font-bold ${GM.text} m-0`}>
                      ${d.equity.toLocaleString(undefined, { minimumFractionDigits: 2 })} <span className={`${GMT.micro} ${GM.textMuted} font-normal`}>{t('equityLabel')}</span>
                    </p>
                  </div>
                );
              }}
            />
            <ReferenceLine y={initialCapital} stroke={HEX.ref} strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="equity"
              stroke={HEX.accent}
              strokeWidth={2}
              fill="url(#equityGradient)"
              dot={{ fill: HEX.accent, r: 3 }}
              activeDot={{ r: 5, stroke: HEX.accent, strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </GmPanel>
  );
}

// ============================================================================
// Resumen del periodo (Backtest) — grid 4 col (prototipo btSummary)
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
  const t = useGmT(DICT);
  const totalTrades = trades.length;
  const wins = trades.filter(tr => Number(tr.pnl_usd) > 0).length;
  const losses = totalTrades - wins;
  const currentEquity = trades.length > 0 ? Number(trades[trades.length - 1].equity_at_exit) : initialCapital;
  const profit = currentEquity - initialCapital;
  const profitPct = stats?.total_return_pct ?? (profit / initialCapital * 100);
  const winRate = stats?.win_rate_pct ?? (totalTrades > 0 ? (wins / totalTrades) * 100 : 0);
  const tradingDays = stats?.trading_days ?? 0;
  const entryStart = trades.length > 0 && trades[0].entry_price != null
    ? `$${Number(trades[0].entry_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}` : '—';
  const exitEnd = trades.length > 0 && trades[trades.length - 1].exit_price != null
    ? `$${Number(trades[trades.length - 1].exit_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}` : '—';

  const items: Array<{ label: string; value: string; cls: string }> = [
    { label: t('sCapIni'), value: `$${initialCapital.toLocaleString()}`, cls: GM.text },
    { label: t('sEqFin'), value: `$${currentEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, cls: GM.textStrong },
    { label: t('sEntryStart'), value: entryStart, cls: GM.textSec },
    { label: t('sExitEnd'), value: exitEnd, cls: GM.textSec },
    { label: t('sOps'), value: `${totalTrades} (${wins}W / ${losses}L)`, cls: GM.text },
    { label: t('sWr'), value: `${winRate.toFixed(1)}%`, cls: winRate >= 50 ? GM.pos : GM.warn },
    { label: t('sDays'), value: tradingDays > 0 ? String(tradingDays) : '—', cls: GM.text },
    { label: t('sRet'), value: `${profitPct >= 0 ? '+' : ''}${profitPct.toFixed(2)}%`, cls: profitPct >= 0 ? GM.pos : GM.neg },
  ];

  return (
    <GmPanel
      title={t('summaryTitle')}
      actions={strategyName ? <GmBadge tone="accent">{strategyName}</GmBadge> : undefined}
    >
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
        {items.map((it) => (
          <div key={it.label}>
            <div className={`${GMT.label} ${GM.textMuted} mb-1.5`}>{it.label}</div>
            <div className={`text-[0.9375rem] font-extrabold ${GMT.mono} ${it.cls}`}>{it.value}</div>
          </div>
        ))}
      </div>
      <div className="border-t border-[rgba(148,163,184,.10)] pt-3 mt-4 flex items-center justify-between gap-3">
        <span className={`${GMT.meta} ${GM.textSec}`}>{t('sPnl')}</span>
        <div className="text-right">
          <div className={`${GMT.mono} text-lg font-extrabold ${profit >= 0 ? GM.pos : GM.neg}`}>
            {profit >= 0 ? '+' : ''}${profit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className={`${GMT.micro} ${GMT.mono} ${profitPct >= 0 ? GM.pos : GM.neg}`}>
            {profitPct >= 0 ? '+' : ''}{profitPct.toFixed(2)}%
          </div>
        </div>
      </div>
      <div className={`mt-3 rounded-[9px] px-2.5 py-1.5 ${GM.infoBadge}`}>
        <span className={GMT.micro}>
          {t('backtestNote')} {year} — {totalTrades} {t('tradesOos')}
        </span>
      </div>
    </GmPanel>
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
  const t = useGmT(DICT);
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

  const SortHeader = ({ field, label, align = 'left' }: { field: string; label: string; align?: 'left' | 'right' }) => (
    <th
      onClick={() => handleSort(field)}
      className={cn(
        `px-3 py-3 ${GMT.label} ${GM.textMuted} cursor-pointer hover:text-[var(--gm-accent)] transition-colors duration-[var(--gm-dur-fast)] select-none`,
        align === 'right' ? 'text-right' : 'text-left'
      )}
    >
      <span className={cn('inline-flex items-center gap-1', align === 'right' && 'flex-row-reverse')}>
        {label}
        {sortField === field && (
          sortDir === 'asc' ? <ChevronUp className="w-3 h-3" aria-hidden /> : <ChevronDown className="w-3 h-3" aria-hidden />
        )}
      </span>
    </th>
  );

  return (
    <GmPanel
      title={t('tradesTitle')}
      meta={`${trades.length} ${t('ops')}`}
      actions={trades.length > 10 ? (
        <button
          onClick={onToggle}
          className={`${GM.focus} ${GMT.meta} ${GM.accent} font-semibold inline-flex items-center gap-1 min-h-[44px] px-2 transition-colors duration-[var(--gm-dur-fast)] hover:opacity-80`}
        >
          {showAll ? (
            <><ChevronUp className="w-3.5 h-3.5" aria-hidden /> {t('less')}</>
          ) : (
            <><ChevronDown className="w-3.5 h-3.5" aria-hidden /> {t('showAll')} ({trades.length})</>
          )}
        </button>
      ) : undefined}
      className="!p-0 [&>header]:px-[18px] [&>header]:pt-4 [&>header]:mb-2"
    >
      <div className="overflow-x-auto">
        <table className={`w-full ${GMT.body}`}>
          <thead>
            <tr className="border-b border-[rgba(148,163,184,.10)]">
              <SortHeader field="timestamp" label={t('thDate')} />
              <SortHeader field="side" label={t('thSide')} />
              <SortHeader field="entry_price" label={t('thEntry')} align="right" />
              <SortHeader field="exit_price" label={t('thExit')} align="right" />
              <SortHeader field="pnl_usd" label={t('thPnl')} align="right" />
              <SortHeader field="exit_reason" label={t('thReason')} align="right" />
            </tr>
          </thead>
          <tbody>
            {visible.map((tr) => {
              const exitColor = getExitReasonColor(tr.exit_reason);
              const pnlPositive = Number(tr.pnl_usd) >= 0;
              return (
                <tr key={tr.trade_id} className={`border-b border-[rgba(148,163,184,.07)] ${GM.rowHover} transition-colors duration-[var(--gm-dur-fast)]`}>
                  <td className={`px-3 py-2.5 ${GMT.meta} ${GMT.mono} whitespace-nowrap`}>
                    <span className={GM.textSec}>{formatTsCOT(tr.timestamp as string)}</span>
                    <span className={`block ${GMT.micro} ${GM.textMuted}`}>
                      → {formatTsCOT(tr.exit_timestamp as string | undefined)}
                    </span>
                  </td>
                  <td className="px-3 py-2.5">
                    <span className={cn(
                      'inline-flex px-2 py-0.5 rounded-[7px] text-[10px] font-bold',
                      tr.side === 'LONG' ? GM.posBadge : GM.negBadge
                    )}>
                      {tr.side}
                    </span>
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.meta} ${GMT.mono} ${GM.textStrong} text-right whitespace-nowrap`}>
                    {tr.entry_price != null ? `$${Number(tr.entry_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}` : '—'}
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.meta} ${GMT.mono} ${GM.textStrong} text-right whitespace-nowrap`}>
                    {tr.exit_price != null ? `$${Number(tr.exit_price).toLocaleString('en-US', { minimumFractionDigits: 0 })}` : '—'}
                  </td>
                  <td className={cn(`px-3 py-2.5 ${GMT.meta} ${GMT.mono} font-bold text-right whitespace-nowrap`, pnlPositive ? GM.pos : GM.neg)}>
                    {tr.pnl_usd != null ? `${pnlPositive ? '+' : ''}${Number(tr.pnl_usd).toFixed(2)}` : '—'}
                    <span className="block text-[10px] font-normal opacity-80">
                      {tr.pnl_pct != null ? `${pnlPositive ? '+' : ''}${Number(tr.pnl_pct).toFixed(2)}%` : ''}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <span className={cn('inline-flex px-2 py-0.5 rounded-[7px] text-[10px] font-medium', exitColor.bg, exitColor.text)}>
                      {tr.exit_reason}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </GmPanel>
  );
}

// ============================================================================
// Chart Loading Fallback
// ============================================================================
function ChartLoadingFallback() {
  const t = useGmT(DICT);
  return (
    <div className={`${GM.panelInner} h-[400px] flex items-center justify-center gap-3`}>
      <RefreshCw className={`w-6 h-6 ${GM.accent} motion-safe:animate-spin`} aria-hidden />
      <p className={`${GMT.body} ${GM.textMuted} m-0`}>{t('loading')}</p>
    </div>
  );
}

// ============================================================================
// Deploy Panel — One-Click Production Deploy (lógica intacta)
// ============================================================================
const DEPLOY_PHASE_PROGRESS: Record<string, number> = {
  initializing: 10,
  retraining: 50,
  exporting: 80,
  done: 100,
};

function DeployPanel({ approval }: { approval: ApprovalState }) {
  const t = useGmT(DICT);
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
  const phaseLabels: Record<string, string> = {
    initializing: t('phInit'),
    retraining: t('phRetrain'),
    exporting: t('phExport'),
    done: t('phDone'),
  };

  return (
    <GmPanel>
      {/* Idle — Show deploy button */}
      {deployStatus.status === 'idle' && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="p-2.5 rounded-xl bg-[rgba(34,211,238,.10)] shrink-0">
              <Rocket className={`w-5 h-5 ${GM.accent}`} aria-hidden />
            </span>
            <span>
              <span className={`block ${GMT.body} font-semibold ${GM.textStrong}`}>{t('deployTitle')}</span>
              <span className={`block ${GMT.micro} ${GM.textMuted} mt-0.5`}>{t('deployDesc')}</span>
            </span>
          </div>
          <button
            onClick={handleDeploy}
            disabled={isDeploying}
            className={`${GM.ctaPrimary} ${GM.focus} w-full h-11 text-[0.8125rem] disabled:opacity-50 inline-flex items-center justify-center gap-2`}
          >
            {isDeploying ? (
              <Loader2 className="w-4 h-4 motion-safe:animate-spin" aria-hidden />
            ) : (
              <Rocket className="w-4 h-4" aria-hidden />
            )}
            {t('deployTitle')}
          </button>
        </div>
      )}

      {/* Running — Progress bar */}
      {deployStatus.status === 'running' && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <span className="p-2.5 rounded-xl bg-[rgba(34,211,238,.10)] shrink-0">
              <Loader2 className={`w-5 h-5 ${GM.accent} motion-safe:animate-spin`} aria-hidden />
            </span>
            <span className="flex-1 min-w-0">
              <span className={`block ${GMT.body} font-semibold ${GM.textStrong}`}>{t('deploying')}</span>
              <span className={`block ${GMT.micro} ${GM.accent} mt-0.5`}>
                {phaseLabels[phase] || t('phGeneric')}
              </span>
            </span>
            {deployStatus.started_at && (
              <span className={`${GMT.micro} ${GM.textMuted} font-mono shrink-0`}>
                {Math.round((Date.now() - new Date(deployStatus.started_at).getTime()) / 1000)}s
              </span>
            )}
          </div>
          <div className="h-1.5 rounded-full bg-[rgba(148,163,184,.14)] overflow-hidden">
            <div
              className="h-full bg-[var(--gm-accent)] rounded-full transition-all duration-[var(--gm-dur-slow)] ease-[var(--gm-ease-out)]"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className={`flex justify-between ${GMT.micro} ${GM.textMuted}`}>
            <span>{t('stepInit')}</span>
            <span>{t('stepRetrain')}</span>
            <span>{t('stepExport')}</span>
            <span>{t('stepReady')}</span>
          </div>
        </div>
      )}

      {/* Completed — Success banner */}
      {deployStatus.status === 'completed' && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="p-2.5 rounded-xl bg-[rgba(52,211,153,.12)] shrink-0">
              <CheckCircle2 className={`w-5 h-5 ${GM.pos}`} aria-hidden />
            </span>
            <span>
              <span className={`block ${GMT.body} font-semibold ${GM.pos}`}>{t('deployDone')}</span>
              <span className={`block ${GMT.micro} ${GM.textMuted} mt-0.5`}>{t('deployDoneDesc')}</span>
            </span>
          </div>
          <a
            href="/production"
            className={`${GM.ctaSoft} ${GM.focus} w-full h-11 text-[0.8125rem] inline-flex items-center justify-center gap-2`}
          >
            <ExternalLink className="w-4 h-4" aria-hidden />
            {t('seeProd')}
          </a>
        </div>
      )}

      {/* Failed — Error message */}
      {deployStatus.status === 'failed' && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="p-2.5 rounded-xl bg-[rgba(251,113,133,.10)] shrink-0">
              <XCircle className={`w-5 h-5 ${GM.neg}`} aria-hidden />
            </span>
            <span className="min-w-0">
              <span className={`block ${GMT.body} font-semibold ${GM.neg}`}>{t('deployErr')}</span>
              <span className={`block ${GMT.micro} ${GM.textMuted} mt-0.5 break-words`}>
                {deployStatus.error || t('unknownErr')}
              </span>
            </span>
          </div>
          <button
            onClick={handleRetry}
            className={`${GM.ctaGhost} ${GM.focus} w-full h-11 text-[0.8125rem] font-semibold inline-flex items-center justify-center gap-2`}
          >
            <RefreshCw className="w-4 h-4" aria-hidden />
            {t('retry')}
          </button>
        </div>
      )}
    </GmPanel>
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
  const t = useGmT(DICT);
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

  // Presentational only: the parent (page header) owns the selector when controlled —
  // header sin duplicados (prototipo). Uncontrolled/standalone keeps the in-card selector.
  const isControlled = controlledStrategyId !== undefined || onStrategyChange !== undefined;

  // Trades shown in meta line + KPI builder (misma expresión que la versión previa)
  const nTradesShown = dynamicMetrics
    ? visibleTrades.length
    : (best ? (best.n_long ?? 0) + (best.n_short ?? 0) : trades.length);
  const { kpis, ratiosHidden } = buildStrategyKpis(displayStats, {
    initialCapital: initialCap,
    nTrades: nTradesShown,
  });

  // Loading state
  if (loading) {
    return <GmSkeleton label={t('loading')} />;
  }

  // Error or no data — hide silently (backtest data may not exist)
  if (error || !summary) return null;

  return (
    <div className="flex flex-col gap-4">
      {/* ════════════════════════════════════════════════════════════════
          Meta: procedencia (MetricBadge), estado, p-value, selector, versión
      ════════════════════════════════════════════════════════════════ */}
      <div className="flex flex-wrap items-center gap-2.5">
        <MetricBadge phase="backtest" provenance={{ strategyId, version: selectedVersion || undefined }} />
        <span className={`${GMT.body} font-bold ${GM.textStrong}`}>{summary.strategy_name || strategyId}</span>
        <span className={`${GMT.meta} ${GMT.mono} ${GM.textMuted}`}>
          {nTradesShown} {t('trades')} · DA {displayDA?.toFixed(1) ?? '-'}%
        </span>
        {approval && <StatusBadge status={approval.status} />}
        {displayPValue != null && (
          <GmBadge tone={displayIsSignificant ? 'pos' : 'neg'}>
            p={displayPValue.toFixed(4)} · {displayIsSignificant ? t('significant') : t('notSignificant')}
          </GmBadge>
        )}
        <span className="flex-1" />
        {!isControlled && registry && registry.strategies.length >= 1 && (
          <StrategySelector
            strategies={registry.strategies}
            selectedId={selectedStrategyId}
            onSelect={handleStrategyChange}
          />
        )}
        {/* Version selector (CTR-STRAT-REGISTRY-001) — additive, renders only
            when a manifest with versioned backtests is available */}
        {manifest && (manifest.backtests?.length ?? 0) >= 1 && (
          <label className="flex items-center gap-2">
            <span className={`${GMT.label} ${GM.textMuted}`}>{t('versionLabel')}</span>
            <select
              value={selectedVersion}
              onChange={(e) => handleVersionChange(e.target.value)}
              className={`${GM.input} ${GM.focus} min-h-[44px]`}
            >
              {manifest.backtests.map((b) => {
                const isActive = manifest.model_versions?.find((v) => v.version === b.model_version)?.active;
                return (
                  <option key={b.immutable_id} value={b.model_version}>
                    {`v${b.model_version}${isActive ? ' ★' : ''}${b.replayable ? '' : ` ${t('noReplay')}`} · ${b.year}`}
                  </option>
                );
              })}
            </select>
            {/* "Promover a activa" button REMOVED (operator request 2026-07-07 + audit A4-04:
                promote-from-TS races the Python registry rebuild). The version dropdown stays
                for REVIEW/replay; the single promotion path is the two-vote flow
                (Aprobar y Promover → Airflow L4b deploy). API endpoint kept for tooling. */}
          </label>
        )}
      </div>

      {/* Replay-preview disclaimer (audit I-4): when metrics are recomputed from visible/
          filtered trades they are a PREVIEW — the official decision numbers are the bundle's. */}
      {dynamicMetrics && (
        <div className={`${GM.warnBadge} rounded-xl px-3.5 py-2.5 ${GMT.micro} font-semibold leading-relaxed`}>
          {t('previewBanner')}
        </div>
      )}
      {ratiosHidden && (
        <div className={`${GM.neutralBadge} rounded-xl px-3.5 py-2 ${GMT.micro}`}>
          {t('ratiosNote')}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════
          KPI row — builder SSOT (mismas métricas del bundle que Producción)
      ════════════════════════════════════════════════════════════════ */}
      <div className="grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(118px,1fr))]">
        {kpis.map((k) => (
          <GmKpi key={k.label} label={k.label} value={k.value} tone={k.tone} sub={k.sub} />
        ))}
      </div>

      {/* ════════════════════════════════════════════════════════════════
          Candle panel + replay bar (prototipo: título + nota Lightweight Charts)
      ════════════════════════════════════════════════════════════════ */}
      <GmPanel title={t('chartTitle')} meta={t('lwNote')}>
        <Suspense fallback={<ChartLoadingFallback />}>
          <div className="rounded-xl overflow-hidden border border-[rgba(148,163,184,.10)]">
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
              replayTrades={visibleTrades.flatMap(tr => {
                const entry = {
                  trade_id: tr.trade_id,
                  timestamp: String(tr.timestamp),
                  side: tr.side,
                  entry_price: Number(tr.entry_price),
                  pnl: 0,
                  status: 'closed' as const,
                };
                const exitTs = tr.exit_timestamp ? String(tr.exit_timestamp) : null;
                if (!exitTs) return [entry];
                const exitSide = tr.side === 'SHORT' ? 'LONG' : 'SHORT';
                const exit = {
                  trade_id: tr.trade_id + 10000,
                  timestamp: exitTs,
                  side: exitSide,
                  entry_price: Number(tr.exit_price),
                  pnl: Number(tr.pnl_usd),
                  status: 'closed' as const,
                };
                return [entry, exit];
              })}
            />
          </div>
        </Suspense>

        {/* Replay Controls — Date-Based (barra del prototipo) */}
        {trades.length > 0 && (
          <div className="mt-3.5 pt-3.5 border-t border-[rgba(148,163,184,.10)] flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Calendar className={`w-4 h-4 ${GM.accent}`} aria-hidden />
              <span className={`${GMT.panelTitle} ${GM.textStrong}`}>{t('replayTitle')}</span>
              <span className={`${GMT.micro} ${GMT.mono} ${GM.textMuted} ml-auto`}>
                {filteredTrades.length} {t('inRange')}
              </span>
            </div>

            <div className="flex flex-wrap items-end gap-3.5">
              {/* Action button — gradiente del prototipo */}
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
                  `${GM.focus} inline-flex items-center gap-2 h-11 px-4 text-[0.78125rem] font-bold shrink-0`,
                  isPlaying ? `${GM.ctaSoft}` : `${GM.ctaPrimary}`
                )}
              >
                {isPlaying ? <Pause className="w-4 h-4" aria-hidden /> : <Play className="w-4 h-4" aria-hidden />}
                {isPlaying ? t('pause') : t('play')}
              </button>

              {/* Date range Desde/Hasta */}
              <label className="flex flex-col gap-1">
                <span className={`${GMT.label} ${GM.textMuted}`}>{t('from')}</span>
                <input
                  type="date"
                  value={replayStartDate}
                  onChange={e => { setReplayStartDate(e.target.value); setIsPlaying(false); setReplayIndex(-1); }}
                  className={`${GM.input} ${GM.focus} ${GMT.mono} min-h-[44px]`}
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className={`${GMT.label} ${GM.textMuted}`}>{t('to')}</span>
                <input
                  type="date"
                  value={replayEndDate}
                  onChange={e => { setReplayEndDate(e.target.value); setIsPlaying(false); setReplayIndex(-1); }}
                  className={`${GM.input} ${GM.focus} ${GMT.mono} min-h-[44px]`}
                />
              </label>

              <button
                onClick={() => { setIsPlaying(false); setReplayIndex(-1); }}
                className={`${GM.ctaGhost} ${GM.focus} inline-flex items-center gap-1.5 h-11 px-3.5 text-[0.78125rem] font-semibold`}
                title={t('showAllTitle')}
              >
                <RotateCcw className="w-3.5 h-3.5" aria-hidden />
                {t('showAll')}
              </button>

              {/* Speed controls */}
              <div className="flex items-center gap-1.5">
                <Gauge className={`w-3.5 h-3.5 ${GM.textMuted}`} aria-hidden />
                {[1, 2, 4].map(s => (
                  <button
                    key={s}
                    onClick={() => setPlaySpeed(s)}
                    className={cn(
                      `${GM.focus} h-11 min-w-[44px] px-2 rounded-[10px] text-xs font-bold transition-colors duration-[var(--gm-dur-fast)]`,
                      playSpeed === s
                        ? 'bg-[rgba(34,211,238,.10)] text-[var(--gm-accent)] border border-[rgba(34,211,238,.28)]'
                        : `${GM.textMuted} hover:text-[var(--gm-text)] border border-transparent`
                    )}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>

            {/* Progress bar (visible during replay) — accent del prototipo */}
            {replayIndex >= 0 && (
              <div className="flex items-center gap-3">
                <div className="flex-1 h-1.5 rounded-full bg-[rgba(148,163,184,.14)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--gm-accent)] rounded-full transition-all duration-[var(--gm-dur-base)]"
                    style={{ width: `${replaySteps > 0 ? (Math.min(replayIndex, replaySteps) / replaySteps) * 100 : 0}%` }}
                  />
                </div>
                <span className={`${GMT.micro} ${GMT.mono} ${GM.textMuted} shrink-0`}>
                  {seriesClockDate
                    ? `${seriesClockDate} · ${visibleTrades.length}/${filteredTrades.length} ${t('trades')}`
                    : `${replayIndex}/${filteredTrades.length} ${t('trades')}`}
                </span>
              </div>
            )}
          </div>
        )}
      </GmPanel>

      {/* ════════════════════════════════════════════════════════════════
          Equity panel
      ════════════════════════════════════════════════════════════════ */}
      {trades.length > 0 && (
        <EquityCurveChart
          seriesPoints={windowMetrics?.series}
          trades={visibleTrades}
          initialCapital={summary.initial_capital ?? 10000}
        />
      )}

      {/* ════════════════════════════════════════════════════════════════
          Grid 1.5fr/1fr — Resumen · Razones · Tabla | Gates (Voto 1) · Voto 2
      ════════════════════════════════════════════════════════════════ */}
      <div className="grid lg:grid-cols-[1.5fr_1fr] gap-4 items-start">
        <div className="flex flex-col gap-4 min-w-0">
          {trades.length > 0 && (
            <TradingSummary
              stats={displayStats}
              trades={visibleTrades}
              initialCapital={summary.initial_capital ?? 10000}
              strategyName={summary.strategy_name || strategyId}
            />
          )}
          {displayStats?.exit_reasons && Object.keys(displayStats.exit_reasons).length > 0 && (
            <ExitReasonsSummary reasons={displayStats.exit_reasons} />
          )}
          <TradeTable
            trades={replayIndex < 0 ? filteredTrades : visibleTrades}
            showAll={showAllTrades}
            onToggle={() => setShowAllTrades(!showAllTrades)}
          />
        </div>

        <div className="flex flex-col gap-4 min-w-0">
          {approval && (
            <>
              {/* Gates — always the published bundle's Vote-1 results (never replay-recomputed) */}
              {displayGates.length > 0 && (
                <GatesPanel gates={displayGates} />
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

              {/* Approved/Rejected status display */}
              {approval.status === 'APPROVED' && (
                <section className={`${GM.panel} gm-contain p-[18px] !border-[rgba(52,211,153,.3)]`}>
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className={`w-6 h-6 ${GM.pos} shrink-0`} aria-hidden />
                    <div>
                      <p className={`${GMT.body} font-semibold ${GM.pos} m-0`}>
                        {t('approvedMsg')}{approval.approved_by ? ` ${t('by')} ${approval.approved_by}` : ''}
                      </p>
                      {approval.approved_at && (
                        <p className={`${GMT.micro} ${GMT.mono} ${GM.textMuted} m-0 mt-1`}>
                          {new Date(approval.approved_at).toLocaleDateString('es-CO', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
                        </p>
                      )}
                      {approval.reviewer_notes && (
                        <p className={`${GMT.micro} ${GM.textMuted} italic m-0 mt-1`}>&quot;{approval.reviewer_notes}&quot;</p>
                      )}
                    </div>
                  </div>
                </section>
              )}
              {approval.status === 'REJECTED' && (
                <section className={`${GM.panel} gm-contain p-[18px] !border-[rgba(251,113,133,.24)]`}>
                  <div className="flex items-start gap-3">
                    <XCircle className={`w-6 h-6 ${GM.neg} shrink-0`} aria-hidden />
                    <div>
                      <p className={`${GMT.body} font-semibold ${GM.neg} m-0`}>
                        {t('rejectedMsg')}{approval.rejected_by ? ` ${t('by')} ${approval.rejected_by}` : ''}
                      </p>
                      {approval.rejection_reason && (
                        <p className={`${GMT.micro} ${GM.textMuted} m-0 mt-1`}>{approval.rejection_reason}</p>
                      )}
                    </div>
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
