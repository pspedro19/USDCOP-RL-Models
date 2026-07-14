'use client';

/**
 * ProductionView — vista PRODUCCIÓN / SEÑALES del GlobalMarkets Terminal
 * (CTR-GM-UI-001, hi-fi Var B líneas 484-615).
 *
 * Fuentes de datos (las mismas que la página legacy /legacy/production):
 *   - /api/production/strategies          → selector dinámico (strategy_id NUNCA hardcodeado)
 *   - /api/production/live                → señal de la semana, posición, guardrails, equity (DB)
 *   - /data/production/summary.json       → KPIs del bundle publicado (regla Vote-2/P1)
 *   - /api/data/production/summary_<sid>.json (estrategias no-default, fs-backed)
 *   - /api/production/status | approval_state_<sid>.json → estado de aprobación (read-only)
 *   - /data/production/trades/<sid>.json  → historial de trades (fallback archivo)
 *
 * KPIs vía el builder SSOT `strategy-kpis.ts` (compartido con Backtest);
 * quant-constitution §6: con N<20 trades solo conteo y P&L (canShowRatios).
 */
import { useMemo, useState, type ReactNode } from 'react';
import { useSession } from 'next-auth/react';
import dynamic from 'next/dynamic';
import {
  Area, AreaChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, XCircle,
} from 'lucide-react';

import {
  AsyncBoundary, GmBadge, GmEmpty, GmKpi, GmPageHeader, GmPanel, GmSkeleton,
  useGmQuery, type AsyncState,
} from '@/components/gm';
import { ClientApiError } from '@/lib/api/gm-client';
import { defineGmDict, useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT, GM_TONE_TEXT, GM_HEX, type GmTone } from '@/lib/ui/gm-tokens';
import { getExitReasonColor } from '@/lib/contracts/strategy.contract';
import type { StrategyStats, StrategyTrade } from '@/lib/contracts/strategy.contract';
import type {
  ApprovalState, ProductionStatus, ProductionSummary, ProductionTradeFile,
} from '@/lib/contracts/production-approval.contract';
import type {
  ActivePosition, CurrentSignal, Guardrails, LiveProductionResponse, LiveTrade,
} from '@/lib/contracts/production-monitor.contract';
import { buildStrategyKpis } from './strategy-kpis';

const TradingChartWithSignals = dynamic(
  () => import('@/components/charts/TradingChartWithSignals'),
  { ssr: false, loading: () => <div className={`${GM.panelInner} h-[400px] motion-safe:animate-pulse`} /> },
);

// Recharts y los dots de estado necesitan valores crudos, no clases Tailwind → se
// importan de GM_HEX (único lugar con hex permitido, CTR-GM-UI-001).
const HEX = {
  pos: GM_HEX.pos,
  neg: GM_HEX.neg,
  tick: GM_HEX.tick,
  ref: GM_HEX.ref,
  grid: GM_HEX.gridStroke,
} as const;

// ─────────────────────────────────────────────────────────── tipos locales

interface StrategyOption {
  strategy_id: string;
  strategy_name: string;
  status: string;
  year: number | null;
  return_pct: number | null;
  mode: string;
  is_active_default: boolean;
}

interface StrategyListResponse {
  strategies: StrategyOption[];
  active_strategy_id: string | null;
}

interface RealtimePriceData {
  price: number;
  change?: number | null;
  changePct?: number | null;
  isMarketOpen?: boolean;
}

// ─────────────────────────────────────────────────────────── helpers

function formatTsCOT(iso: string | undefined | null): string {
  if (!iso) return '—';
  const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
  const d = new Date(String(iso));
  if (isNaN(d.getTime())) return String(iso).split('T')[0];
  return `${d.getDate().toString().padStart(2, '0')} ${months[d.getMonth()]} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
}

function fmtPrice(n: number | null | undefined, digits = 1): string {
  if (n == null) return '—';
  return `$${n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

function unwrapPrice(d: unknown): RealtimePriceData | null {
  if (!d || typeof d !== 'object') return null;
  const o = d as Record<string, unknown>;
  const inner = (o.data && typeof o.data === 'object' ? o.data : o) as Record<string, unknown>;
  return typeof inner.price === 'number' ? (inner as unknown as RealtimePriceData) : null;
}

function liveTradesToStrategyTrades(liveTrades: LiveTrade[]): StrategyTrade[] {
  return liveTrades.map((t) => ({
    trade_id: t.trade_id,
    timestamp: t.timestamp,
    exit_timestamp: t.exit_timestamp ?? undefined,
    side: t.side,
    entry_price: t.entry_price,
    exit_price: t.exit_price,
    pnl_usd: t.pnl_usd,
    pnl_pct: t.pnl_pct,
    exit_reason: t.exit_reason,
    equity_at_entry: t.equity_at_entry,
    equity_at_exit: t.equity_at_exit,
    leverage: t.leverage,
    confidence_tier: t.confidence_tier,
    hard_stop_pct: t.hard_stop_pct,
    take_profit_pct: t.take_profit_pct,
  }));
}

// ─────────────────────────────────────────────── i18n (strings de esta vista)

/** Prototipo Var B: sigTitle (l. 2401 — "Monitor de producción" admin vs "Señales"),
 *  sigCard + labels (l. 2405-2411, 2461-2468). */
const PROD_DICT = defineGmDict({
  es: {
    titleInternal: 'Monitor de producción',
    titleClient: 'Señales',
    sigCurrentTitle: 'Señal de la semana',
    confLabel: 'Confianza',
    entryLabel: 'Zona de entrada',
    tpLabel: 'Take profit',
    slLabel: 'Stop',
    estReturnLabel: 'Retorno estimado',
    sizeLabel: 'Tamaño',
    closeNote: 'Cierre programado el viernes al cierre de sesión (COT).',
    noSignal: 'Sin señal activa. Próxima señal: lunes 08:15 COT.',
    skipTrade: 'Trade omitido',
    inPosition: 'En posición',
    waitingEntry: 'Esperando entrada',
    btMetricsTitle: 'Métricas del backtest',
    btMetricsSub: 'Muestra OOS completa (N alto) — ratios de referencia de la estrategia',
    btSignificant: 'Significativo',
    btNotSignificant: 'No significativo',
  },
  en: {
    titleInternal: 'Production monitor',
    titleClient: 'Signals',
    sigCurrentTitle: "This week's signal",
    confLabel: 'Confidence',
    entryLabel: 'Entry zone',
    tpLabel: 'Take profit',
    slLabel: 'Stop',
    estReturnLabel: 'Estimated return',
    sizeLabel: 'Size',
    closeNote: 'Scheduled close Friday at session end (COT).',
    noSignal: 'No active signal. Next signal: Monday 08:15 COT.',
    skipTrade: 'Trade skipped',
    inPosition: 'In position',
    waitingEntry: 'Awaiting entry',
    btMetricsTitle: 'Backtest metrics',
    btMetricsSub: 'Full OOS sample (large N) — the strategy’s reference ratios',
    btSignificant: 'Significant',
    btNotSignificant: 'Not significant',
  },
});

type ProdT = (key: keyof (typeof PROD_DICT)['es']) => string;

const APPROVAL_TONE: Record<ProductionStatus, { tone: GmTone; label: string }> = {
  PENDING_APPROVAL: { tone: 'warn', label: 'Pendiente' },
  APPROVED: { tone: 'pos', label: 'Aprobado' },
  REJECTED: { tone: 'neg', label: 'Rechazado' },
  LIVE: { tone: 'accent', label: 'En vivo' },
};

// ─────────────────────────────────────────────────────────── subcomponentes

function StrategySelector({ label, sub, options, currentSid, onSelect }: {
  label: string;
  sub?: string;
  options: StrategyOption[];
  currentSid: string;
  onSelect: (sid: string | null) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`${GM.ctaGhost} ${GM.focus} flex items-center gap-2.5 h-10 px-3.5`}
        aria-expanded={open}
      >
        <span className="w-2 h-2 rounded-full motion-safe:animate-pulse" style={{ background: HEX.pos }} aria-hidden />
        <span className="text-left">
          <span className="block text-[13px] font-bold leading-tight">{label}</span>
          {sub && <span className={`block ${GMT.micro} ${GM.textMuted} font-mono leading-tight`}>{sub}</span>}
        </span>
        <ChevronDown className={`w-4 h-4 ${GM.accent} transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
        <div className={`absolute top-[46px] right-0 z-40 min-w-[320px] ${GM.popover} p-1.5 flex flex-col gap-0.5`}>
          {options.map((o) => (
            <button
              key={o.strategy_id}
              onClick={() => { onSelect(o.is_active_default ? null : o.strategy_id); setOpen(false); }}
              className={`${GM.rowHover} ${GM.focus} flex items-center justify-between gap-3 w-full text-left px-3 py-2.5 rounded-[9px]
                ${o.strategy_id === currentSid ? 'bg-[rgba(148,163,184,.07)]' : ''}`}
            >
              <span className="min-w-0">
                <span className={`block text-[13px] font-bold ${GM.text}`}>{o.strategy_name}</span>
                <span className={`block ${GMT.micro} ${GM.textSec} font-mono mt-0.5`}>
                  {o.return_pct != null ? `${o.return_pct >= 0 ? '+' : ''}${o.return_pct.toFixed(2)}% YTD` : 'sin datos'}
                  {o.mode === 'paper' ? ' · PAPER' : ''}
                </span>
              </span>
              <GmBadge tone={o.status === 'APPROVED' ? 'pos' : o.status === 'PENDING_APPROVAL' ? 'warn' : 'accent'}>
                {o.status === 'PENDING_APPROVAL' ? 'PENDIENTE' : o.status}
              </GmBadge>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function EquityChart({ points, initialCapital }: {
  points: { date: string; equity: number; pnl_pct: number }[];
  initialCapital: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={points} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="gmEquityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={HEX.pos} stopOpacity={0.3} />
            <stop offset="95%" stopColor={HEX.pos} stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="date" tick={{ fill: HEX.tick, fontSize: 10 }} axisLine={{ stroke: HEX.grid }} tickLine={false} />
        <YAxis
          tick={{ fill: HEX.tick, fontSize: 10 }}
          axisLine={{ stroke: HEX.grid }}
          tickLine={false}
          domain={['auto', 'auto']}
          tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload || !payload[0]) return null;
            const d = payload[0].payload as { date: string; equity: number; pnl_pct: number };
            return (
              <div className={`${GM.popover} px-3 py-2`}>
                <p className={`${GMT.micro} ${GM.textSec} mb-1`}>{d.date}</p>
                <p className={`${GMT.mono} text-[13px] font-bold ${GM.text}`}>
                  ${d.equity.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                </p>
                <p className={`${GMT.mono} ${GMT.micro} ${d.pnl_pct >= 0 ? GM.pos : GM.neg}`}>
                  {d.pnl_pct >= 0 ? '+' : ''}{d.pnl_pct.toFixed(2)}%
                </p>
              </div>
            );
          }}
        />
        <ReferenceLine
          y={initialCapital}
          stroke={HEX.ref}
          strokeDasharray="3 3"
          label={{ value: `$${(initialCapital / 1000).toFixed(0)}k`, fill: HEX.tick, fontSize: 10, position: 'left' }}
        />
        <Area type="monotone" dataKey="equity" stroke={HEX.pos} strokeWidth={2} fill="url(#gmEquityGradient)" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function KeyVal({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[12.5px]">
      <span className={GM.textSec}>{label}</span>
      {children}
    </div>
  );
}

/** Signal-card semanal del prototipo (Var B l. 577-589): badge grande de dirección +
 *  confianza al lado, filas entrada/TP/SL y nota de tamaño bajo un divisor.
 *  Solo campos REALES de /api/production/live — lo que no exista se omite. */
function SignalPanel({ signal, position, marketOpen, t }: {
  signal: CurrentSignal | null;
  position: ActivePosition | null;
  marketOpen: boolean;
  t: ProdT;
}) {
  if (!signal) {
    return (
      <GmPanel title={t('sigCurrentTitle')}>
        <p className={`${GMT.body} ${GM.textSec} m-0`}>{t('noSignal')}</p>
      </GmPanel>
    );
  }
  const isShort = signal.direction === -1;
  const estado = signal.skip_trade ? t('skipTrade') : position ? t('inPosition') : t('waitingEntry');
  const estadoTone: GmTone = signal.skip_trade ? 'warn' : position ? 'pos' : 'accent';
  const confTone: GmTone = signal.confidence_tier === 'HIGH' ? 'pos'
    : signal.confidence_tier === 'LOW' ? 'neg'
    : signal.confidence_tier === 'MEDIUM' ? 'warn' : 'neutral';
  // Nota de tamaño (prototipo sigCard.note): solo con leverage real publicado.
  const sizeNote = signal.adjusted_leverage != null
    ? `${t('sizeLabel')} ${signal.adjusted_leverage.toFixed(2)}× (${isShort ? 'SHORT' : 'LONG'}). ${t('closeNote')}`
    : null;
  return (
    <GmPanel title={t('sigCurrentTitle')} meta={signal.signal_date}>
      {/* Badge grande de dirección + confianza (prototipo l. 579-582) */}
      <div className="flex items-center gap-3 mb-3.5">
        <span
          className={`text-[22px] font-extrabold px-4 py-1.5 rounded-[11px] ${isShort ? GM.negBadge : GM.posBadge}`}
          data-testid="prod-signal-direction"
        >
          {isShort ? 'SHORT' : 'LONG'}
        </span>
        <div>
          <div className={`${GMT.label} ${GM.textMuted}`}>{t('confLabel')}</div>
          <div className={`text-[14px] font-bold mt-0.5 ${confTone === 'neutral' ? GM.textStrong : GM_TONE_TEXT[confTone]}`}>
            {signal.confidence_tier ?? '—'}
          </div>
        </div>
        <div className="ml-auto flex flex-col items-end gap-1.5">
          <GmBadge tone={estadoTone}>{estado}</GmBadge>
          {marketOpen && <GmBadge tone="pos">LIVE</GmBadge>}
        </div>
      </div>
      <div className="flex flex-col gap-2.5">
        <KeyVal label={t('entryLabel')}>
          <span className={`${GMT.mono} ${GM.textStrong}`}>{fmtPrice(position?.entry_price)}</span>
        </KeyVal>
        <KeyVal label={t('tpLabel')}>
          <span className={`${GMT.mono} ${GM.pos}`}>{signal.take_profit_pct != null ? `${signal.take_profit_pct.toFixed(1)}%` : '—'}</span>
        </KeyVal>
        <KeyVal label={t('slLabel')}>
          <span className={`${GMT.mono} ${GM.neg}`}>{signal.hard_stop_pct != null ? `${signal.hard_stop_pct.toFixed(1)}%` : '—'}</span>
        </KeyVal>
        <KeyVal label={t('estReturnLabel')}>
          <span className={`${GMT.mono} font-bold ${signal.ensemble_return >= 0 ? GM.pos : GM.neg}`}>
            {signal.ensemble_return >= 0 ? '+' : ''}{(signal.ensemble_return * 100).toFixed(2)}%
          </span>
        </KeyVal>
      </div>
      {sizeNote && (
        <p className={`mt-3.5 pt-3 border-t border-dashed border-[rgba(148,163,184,.16)] ${GMT.micro} ${GM.textSec} m-0`}>
          {sizeNote}
        </p>
      )}
    </GmPanel>
  );
}

function PositionPanel({ position, realtimePrice }: {
  position: ActivePosition | null;
  realtimePrice: number | null;
}) {
  if (!position) {
    return (
      <GmPanel title="Posición abierta y P&L">
        <p className={`${GMT.body} ${GM.textSec} m-0`}>Sin posición abierta.</p>
      </GmPanel>
    );
  }
  const isShort = position.direction === -1;
  const currentPrice = realtimePrice ?? position.current_price;
  const unrealized = isShort
    ? ((position.entry_price - currentPrice) / position.entry_price) * position.leverage * 100
    : ((currentPrice - position.entry_price) / position.entry_price) * position.leverage * 100;
  return (
    <GmPanel title="Posición abierta y P&L" meta={`Bar #${position.bar_count}`}>
      <div className="flex items-center justify-between mb-3">
        <span className={`${GMT.mono} text-[15px] font-extrabold ${GM.text}`}>USD/COP</span>
        <span className={`text-[12.5px] font-bold ${isShort ? GM.neg : GM.pos}`}>{isShort ? 'SHORT' : 'LONG'}</span>
      </div>
      <div className="flex flex-col gap-2.5">
        <KeyVal label="Entrada">
          <span className={`${GMT.mono} ${GM.textStrong}`}>{fmtPrice(position.entry_price)}</span>
        </KeyVal>
        <KeyVal label={`Precio actual${realtimePrice == null ? ' (DB)' : ''}`}>
          <span className={`${GMT.mono} ${GM.accent}`}>{fmtPrice(currentPrice)}</span>
        </KeyVal>
        <KeyVal label="Leverage">
          <span className={`${GMT.mono} ${GM.textStrong}`}>{position.leverage.toFixed(2)}x</span>
        </KeyVal>
        <div className="flex items-center justify-between pt-2.5 border-t border-[rgba(148,163,184,.1)]">
          <span className={`text-[13px] font-semibold ${GM.textStrong}`}>P&L no realizado</span>
          <span className={`${GMT.mono} text-[15px] font-extrabold ${unrealized >= 0 ? GM.pos : GM.neg}`}>
            {unrealized >= 0 ? '+' : ''}{unrealized.toFixed(2)}%
          </span>
        </div>
      </div>
    </GmPanel>
  );
}

function GuardrailsPanel({ guardrails }: { guardrails: Guardrails }) {
  const cb = guardrails.circuit_breaker_active;
  return (
    <GmPanel
      title="Guardrails"
      actions={
        <GmBadge tone={cb ? 'neg' : guardrails.alerts.length > 0 ? 'warn' : 'pos'}>
          {cb ? 'CB activo' : guardrails.alerts.length > 0 ? 'Alerta' : 'Normal'}
        </GmBadge>
      }
    >
      <div className="flex flex-col gap-2.5">
        <KeyVal label="P&L acumulado">
          <span className={`${GMT.mono} font-semibold ${guardrails.cumulative_pnl_pct == null ? GM.textMuted : guardrails.cumulative_pnl_pct >= 0 ? GM.pos : GM.neg}`}>
            {guardrails.cumulative_pnl_pct != null
              ? `${guardrails.cumulative_pnl_pct >= 0 ? '+' : ''}${guardrails.cumulative_pnl_pct.toFixed(2)}%` : '—'}
          </span>
        </KeyVal>
        <KeyVal label="Pérdidas consecutivas">
          <span className={`${GMT.mono} font-semibold ${guardrails.consecutive_losses >= 3 ? GM.warn : GM.textStrong}`}>
            {guardrails.consecutive_losses}
          </span>
        </KeyVal>
        <KeyVal label="Sharpe 16 sem.">
          <span className={`${GMT.mono} font-semibold ${GM.textStrong}`}>
            {guardrails.rolling_sharpe_16w != null ? guardrails.rolling_sharpe_16w.toFixed(2) : '< 16 semanas'}
          </span>
        </KeyVal>
        <KeyVal label="DA SHORT 16 sem.">
          <span className={`${GMT.mono} font-semibold ${GM.textStrong}`}>
            {guardrails.rolling_da_short_16w != null ? `${guardrails.rolling_da_short_16w.toFixed(1)}%` : '< 16 semanas'}
          </span>
        </KeyVal>
      </div>
      {guardrails.alerts.length > 0 && (
        <div className="mt-3.5 flex flex-col gap-2">
          {guardrails.alerts.map((a, i) => (
            <div key={i} className={`${GM.warnBadge} rounded-[9px] px-2.5 py-2 flex items-center gap-2`}>
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" aria-hidden />
              <span className="text-[11.5px] font-medium normal-case tracking-normal">{a}</span>
            </div>
          ))}
        </div>
      )}
    </GmPanel>
  );
}

/** Re-skin GM del ApprovalStatusCard read-only de la página vieja (+ gates Vote-1 del bundle). */
function ApprovalPanel({ approval }: { approval: ApprovalState }) {
  const cfg = APPROVAL_TONE[approval.status] ?? APPROVAL_TONE.PENDING_APPROVAL;
  const gatesPassed = approval.gates.filter((g) => g.passed).length;
  return (
    <GmPanel
      title="Estado de aprobación"
      meta={approval.strategy_name || approval.strategy}
      actions={<GmBadge tone={cfg.tone}>{cfg.label}</GmBadge>}
    >
      <div className="flex flex-col gap-2.5">
        {approval.status === 'PENDING_APPROVAL' && (
          <p className={`${GMT.body} ${GM.warn} m-0 flex items-center gap-2`}>
            <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden />
            Pendiente de revisión — Vote 2 humano en Backtest (/dashboard).
          </p>
        )}
        {approval.status === 'APPROVED' && (
          <p className={`${GMT.body} ${GM.pos} m-0 flex items-center gap-2`}>
            <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden />
            Aprobado{approval.approved_by ? ` por ${approval.approved_by}` : ''}
            {approval.approved_at ? ` · ${formatTsCOT(approval.approved_at)}` : ''}
          </p>
        )}
        {approval.status === 'REJECTED' && (
          <p className={`${GMT.body} ${GM.neg} m-0 flex items-center gap-2`}>
            <XCircle className="w-4 h-4 shrink-0" aria-hidden />
            Rechazado{approval.rejected_by ? ` por ${approval.rejected_by}` : ''}
            {approval.rejection_reason ? ` — ${approval.rejection_reason}` : ''}
          </p>
        )}
        {approval.status === 'LIVE' && (
          <p className={`${GMT.body} ${GM.accent} m-0`}>Estrategia en producción activa.</p>
        )}
        {approval.reviewer_notes && (
          <p className={`${GMT.micro} ${GM.textMuted} italic m-0`}>&quot;{approval.reviewer_notes}&quot;</p>
        )}
        {approval.gates.length > 0 && (
          <div className="pt-2.5 border-t border-[rgba(148,163,184,.1)]">
            <div className={`${GMT.label} ${GM.textMuted} mb-2`}>
              Gates Vote 1 · {gatesPassed}/{approval.gates.length} pasaron
            </div>
            <div className="grid grid-cols-1 gap-1.5">
              {approval.gates.map((g) => (
                <div key={g.gate} className="flex items-center justify-between gap-2 text-[11.5px]">
                  <span className={`flex items-center gap-1.5 ${GM.textSec}`}>
                    {g.passed
                      ? <CheckCircle2 className={`w-3.5 h-3.5 ${GM.pos}`} aria-hidden />
                      : <XCircle className={`w-3.5 h-3.5 ${GM.neg}`} aria-hidden />}
                    {g.label}
                  </span>
                  <span className={`${GMT.mono} ${g.passed ? GM.pos : GM.neg}`}>
                    {g.value}{' '}<span className={GM.textFaint}>/ {g.threshold}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </GmPanel>
  );
}

function TradesTable({ trades }: { trades: StrategyTrade[] }) {
  const [showAll, setShowAll] = useState(false);
  const sorted = useMemo(
    () => [...trades].sort((a, b) => a.trade_id - b.trade_id),
    [trades],
  );
  const visible = showAll ? sorted : sorted.slice(0, 10);
  const th = `px-3 py-2.5 text-left ${GMT.label} ${GM.textMuted}`;
  return (
    <GmPanel
      title="Historial de trades"
      meta={`${trades.length} operaciones · horario COT`}
      actions={trades.length > 10 ? (
        <button
          onClick={() => setShowAll((s) => !s)}
          className={`${GM.ctaGhost} ${GM.focus} flex items-center gap-1 h-8 px-2.5 text-[12px]`}
        >
          {showAll ? <><ChevronUp className="w-3.5 h-3.5" aria-hidden /> Menos</> : <><ChevronDown className="w-3.5 h-3.5" aria-hidden /> Todos ({trades.length})</>}
        </button>
      ) : undefined}
    >
      <div className="overflow-x-auto -mx-[18px] -mb-[18px]">
        <table className="w-full text-[12.5px]">
          <thead>
            <tr className="border-b border-[rgba(148,163,184,.1)]">
              <th className={th}>#</th>
              <th className={th}>Lado</th>
              <th className={th}>Entrada</th>
              <th className={th}>Salida</th>
              <th className={`${th} text-right`}>Precio E.</th>
              <th className={`${th} text-right`}>Precio S.</th>
              <th className={`${th} text-right`}>Lev</th>
              <th className={`${th} text-right`}>PnL $</th>
              <th className={`${th} text-right`}>PnL %</th>
              <th className={th}>Razón</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((t) => {
              const isOpen = t.exit_price == null || t.exit_reason == null;
              const exitColor = getExitReasonColor(t.exit_reason ?? '');
              const pnlPos = isOpen ? true : Number(t.pnl_usd) >= 0;
              return (
                <tr key={t.trade_id} className={`border-t border-[rgba(148,163,184,.07)] ${GM.rowHover}`}>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textMuted}`}>{t.trade_id}</td>
                  <td className="px-3 py-2.5">
                    <span className={`${GMT.mono} font-bold ${t.side === 'LONG' ? GM.pos : GM.neg}`}>{t.side}</span>
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textSec} whitespace-nowrap`}>{formatTsCOT(t.timestamp)}</td>
                  <td className={`px-3 py-2.5 ${GMT.mono} whitespace-nowrap`}>
                    {isOpen
                      ? <GmBadge tone="accent">Abierto</GmBadge>
                      : <span className={GM.textSec}>{formatTsCOT(t.exit_timestamp)}</span>}
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textStrong} text-right`}>{fmtPrice(t.entry_price, 0)}</td>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textStrong} text-right`}>{isOpen ? '—' : fmtPrice(t.exit_price, 0)}</td>
                  <td className={`px-3 py-2.5 ${GMT.mono} ${GM.textSec} text-right`}>{Number(t.leverage).toFixed(2)}x</td>
                  <td className={`px-3 py-2.5 ${GMT.mono} font-bold text-right ${isOpen ? GM.accent : pnlPos ? GM.pos : GM.neg}`}>
                    {isOpen ? '—' : `${pnlPos ? '+' : ''}${Number(t.pnl_usd).toFixed(2)}`}
                  </td>
                  <td className={`px-3 py-2.5 ${GMT.mono} text-right ${isOpen ? GM.accent : pnlPos ? GM.pos : GM.neg}`}>
                    {isOpen ? '—' : `${pnlPos ? '+' : ''}${Number(t.pnl_pct).toFixed(2)}%`}
                  </td>
                  <td className="px-3 py-2.5">
                    {isOpen
                      ? <GmBadge tone="accent">ABIERTO</GmBadge>
                      : (
                        <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${exitColor.bg} ${exitColor.text}`}>
                          {t.exit_reason}
                        </span>
                      )}
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

function ExitReasonsPanel({ reasons }: { reasons: Record<string, number> }) {
  const total = Object.values(reasons).reduce((s, v) => s + v, 0);
  return (
    <GmPanel title="Razones de salida">
      <div className="flex flex-col gap-3">
        {Object.entries(reasons).map(([reason, count]) => {
          const color = getExitReasonColor(reason);
          const pct = total > 0 ? (count / total) * 100 : 0;
          return (
            <div key={reason}>
              <div className="flex justify-between text-[12px] mb-1.5">
                <span className={`${GMT.mono} ${GM.textStrong}`}>{reason}</span>
                <span className={GM.textSec}>{count} · {Math.round(pct)}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-[rgba(148,163,184,.12)] overflow-hidden">
                <div className={`h-full rounded-full ${color.bar}`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </GmPanel>
  );
}

// ─────────────────────────────────────────────────────────── vista principal

export function ProductionView() {
  // ux-navigation §3.2 / RBAC §8: subscribers/free ven la vista CLIENTE ("Señales")
  // — sin estados de aprobación ni gates (outputs, no internals).
  const t = useGmT(PROD_DICT);
  const { data: session } = useSession();
  const role = (session?.user as { role?: string } | undefined)?.role ?? 'free';
  const isClientView = role === 'subscriber' || role === 'free';

  const [selectedSid, setSelectedSid] = useState<string | null>(null);

  // ── fetches (todos vía useGmQuery, CTR-FE-BE-001) ─────────────────────────
  const strategiesQ = useGmQuery<StrategyListResponse>('/api/production/strategies');
  const options = strategiesQ.data?.strategies ?? [];
  const selected = selectedSid ? options.find((o) => o.strategy_id === selectedSid) ?? null : null;
  const isDefault = !selected || selected.is_active_default;
  const sidNonDefault = selected && !selected.is_active_default ? selected.strategy_id : null;

  // Estrategias no-default leen vía /api/data (fs-backed, visible post-build sin rebuild).
  const summaryQ = useGmQuery<ProductionSummary>(
    sidNonDefault ? `/api/data/production/summary_${sidNonDefault}.json` : '/data/production/summary.json',
    { refreshMs: 300_000 },
  );
  // Bundle de BACKTEST (año OOS) — mismas fuentes fs-backed, junto al summary de producción.
  // Da la tabla de métricas de referencia (Calmar/Sharpe/… con N alto, sin gate N<20).
  const backtestSummaryQ = useGmQuery<ProductionSummary>(
    sidNonDefault
      ? `/api/data/production/summary_${sidNonDefault}_2025.json`
      : '/data/production/summary_2025.json',
    { refreshMs: 300_000 },
  );
  const approvalQ = useGmQuery<ApprovalState>(
    sidNonDefault ? `/api/data/production/approval_state_${sidNonDefault}.json` : '/api/production/status',
  );
  const liveQ = useGmQuery<LiveProductionResponse>(
    isDefault ? '/api/production/live' : null,
    { refreshMs: 60_000 },
  );
  // strategy_id dinámico del summary publicado — NUNCA hardcodeado.
  const sid = summaryQ.data?.strategy_id ?? sidNonDefault;
  const tradesQ = useGmQuery<ProductionTradeFile>(
    sid
      ? (sidNonDefault ? `/api/data/production/trades/${sid}.json` : `/data/production/trades/${sid}.json`)
      : null,
  );
  const priceQ = useGmQuery<unknown>(isDefault ? '/api/market/realtime-price' : null, { refreshMs: 300_000 });
  const priceData = unwrapPrice(priceQ.data);

  // ── merge live/archivo (misma lógica que useLiveProduction) ───────────────
  const live = liveQ.data;
  const dbLive = !!live && live.data_source === 'db'
    && (live.trades.length > 0 || live.active_position != null);
  const marketIsOpen = (dbLive && live?.market.is_open) ?? false;

  const summary = summaryQ.data ?? null;
  const strategyId = (dbLive ? live?.strategy_id : null) ?? summary?.strategy_id ?? sid ?? '';
  const strategyName = (dbLive ? live?.strategy_name : null)
    ?? summary?.strategy_name ?? selected?.strategy_name ?? 'Estrategia activa';
  const year = summary?.year ?? new Date().getFullYear();
  const initialCapital = summary?.initial_capital ?? 10000;

  // KPIs SIEMPRE del bundle publicado (regla Vote-2/P1 — nunca recomputados en el front).
  const bundleStats: StrategyStats | undefined = summary
    ? summary.strategies[summary.strategy_id]
    : undefined;

  const displayTrades: StrategyTrade[] = useMemo(() => {
    if (dbLive && live) {
      const converted = liveTradesToStrategyTrades(live.trades);
      if (live.active_position) {
        const ap = live.active_position;
        const already = converted.some(
          (t) => t.entry_price === ap.entry_price && t.timestamp === ap.entry_timestamp,
        );
        if (!already) {
          converted.push({
            trade_id: converted.length + 1,
            timestamp: ap.entry_timestamp,
            exit_timestamp: null,
            side: ap.direction === -1 ? 'SHORT' : 'LONG',
            entry_price: ap.entry_price,
            exit_price: null,
            pnl_usd: null,
            pnl_pct: null,
            exit_reason: null,
            equity_at_entry: live.equity_curve.current_equity,
            equity_at_exit: null,
            leverage: ap.leverage,
          });
        }
      }
      return converted;
    }
    return tradesQ.data?.trades ?? [];
  }, [dbLive, live, tradesQ.data]);

  const tradesState: AsyncState<StrategyTrade[]> = dbLive
    ? { data: displayTrades, error: null, loading: false, reload: liveQ.reload }
    : {
        data: tradesQ.data ? tradesQ.data.trades ?? [] : null,
        error: tradesQ.error,
        loading: tradesQ.loading,
        reload: tradesQ.reload,
      };

  const nTrades = (bundleStats?.n_long ?? 0) + (bundleStats?.n_short ?? 0) || displayTrades.length;
  const { kpis, ratiosHidden } = useMemo(
    () => buildStrategyKpis(bundleStats, { initialCapital, nTrades }),
    [bundleStats, initialCapital, nTrades],
  );

  // ── Métricas del backtest (OOS): mismo builder SSOT pero con el N del backtest
  //    (alto) → los ratios SIEMPRE se muestran (Calmar/Sharpe/PF/…). La fila
  //    YTD-live de arriba conserva su gate N<20 (quant-constitution §6).
  const backtestSummary = backtestSummaryQ.data ?? null;
  const backtestStats: StrategyStats | undefined = backtestSummary
    ? backtestSummary.strategies[backtestSummary.strategy_id]
    : undefined;
  const backtestYear = backtestSummary?.year ?? 2025;
  const backtestN = (backtestStats?.n_long ?? 0) + (backtestStats?.n_short ?? 0);
  const backtestPValue =
    (backtestSummary as { statistical_tests?: { p_value?: number; significant?: boolean } } | null)
      ?.statistical_tests ?? null;
  const { kpis: backtestKpis } = useMemo(
    () => buildStrategyKpis(backtestStats, { initialCapital, nTrades: backtestN }),
    [backtestStats, initialCapital, backtestN],
  );

  // Curva de equity — mismo origen que la página vieja: DB live si existe;
  // si no, los puntos de equity del archivo de trades publicado.
  const equity = useMemo(() => {
    if (dbLive && live && live.equity_curve.points.length > 1) {
      return { points: live.equity_curve.points, initial: live.equity_curve.initial_capital };
    }
    const closed = displayTrades
      .filter((t) => t.equity_at_exit != null)
      .sort((a, b) => String(a.exit_timestamp ?? a.timestamp).localeCompare(String(b.exit_timestamp ?? b.timestamp)));
    if (closed.length === 0) return null;
    const points = [
      { date: 'Inicio', equity: initialCapital, pnl_pct: 0 },
      ...closed.map((t) => ({
        date: String(t.exit_timestamp ?? t.timestamp).slice(0, 10),
        equity: Number(t.equity_at_exit),
        pnl_pct: (Number(t.equity_at_exit) / initialCapital - 1) * 100,
      })),
    ];
    return { points, initial: initialCapital };
  }, [dbLive, live, displayTrades, initialCapital]);

  const approval = approvalQ.data ?? null;
  const chartSymbol = (summary as { chart_symbol?: string } | null)?.chart_symbol;
  const summaryNotFound = !summaryQ.data
    && summaryQ.error instanceof ClientApiError
    && summaryQ.error.status === 404;

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="motion-safe:animate-in motion-safe:fade-in" data-testid="production-view">
      <GmPageHeader
        kicker="GlobalMarkets · Producción"
        title={isClientView ? t('titleClient') : t('titleInternal')}
        subtitle={`${strategyName} · ${year} · USD/COP y estrategias aprobadas`}
        actions={
          <div className="flex items-center gap-2.5">
            {priceData && (
              <span className={`hidden md:flex items-center gap-2 ${GMT.meta} ${GM.textStrong}`}>
                <span className={`${GMT.mono} font-bold`}>{fmtPrice(priceData.price)}</span>
                {priceData.changePct != null && (
                  <span className={`${GMT.mono} ${GMT.micro} ${(priceData.change ?? 0) >= 0 ? GM.pos : GM.neg}`}>
                    {(priceData.change ?? 0) >= 0 ? '+' : ''}{priceData.changePct.toFixed(2)}%
                  </span>
                )}
              </span>
            )}
            <GmBadge tone={dbLive ? 'pos' : 'neutral'}>
              <span
                className={`w-1.5 h-1.5 rounded-full ${dbLive ? 'motion-safe:animate-pulse' : ''}`}
                style={{ background: dbLive ? HEX.pos : HEX.tick }}
                aria-hidden
              />
              {dbLive ? 'LIVE · DB' : 'Archivo'}
            </GmBadge>
            {options.length > 0 && (
              <StrategySelector
                label={strategyName}
                sub={bundleStats ? `${(bundleStats.total_return_pct ?? 0) >= 0 ? '+' : ''}${bundleStats.total_return_pct?.toFixed(1) ?? '—'}% · ${strategyId}` : strategyId}
                options={options}
                currentSid={strategyId}
                onSelect={setSelectedSid}
              />
            )}
          </div>
        }
      />

      {summaryNotFound ? (
        <GmEmpty
          title="Esperando primera estrategia"
          body="No hay estrategias desplegadas en producción todavía. Aprueba una estrategia desde Backtest (Vote 2) para activar el monitoreo."
          action={
            <a href="/dashboard" className={`${GM.ctaPrimary} ${GM.focus} inline-flex items-center h-[42px] px-5 text-[13.5px]`}>
              Revisar en Backtest
            </a>
          }
        />
      ) : (
        <AsyncBoundary
          state={summaryQ}
          skeleton={<GmSkeleton label="Cargando producción…" />}
          emptyProps={{
            title: 'Sin datos de producción',
            body: 'El bundle publicado aún no está disponible.',
          }}
        >
          {() => (
            <>
              {/* Banner de la semana */}
              <div className={`${GM.posBadge} rounded-xl px-4 py-3 mb-4 flex items-center gap-2.5 normal-case tracking-normal`}>
                <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden />
                <span className="text-[13px] font-medium">
                  {dbLive && live?.current_signal
                    ? `Semana del ${live.current_signal.signal_date} — señal ${live.current_signal.direction === -1 ? 'SHORT' : 'LONG'}${live.current_signal.skip_trade ? ' (trade omitido)' : live.active_position ? ' · en posición' : ' · esperando entrada'}`
                    : `Números del bundle publicado${summary?.generated_at ? ` · ${String(summary.generated_at).slice(0, 10)}` : ''} — sin datos en vivo`}
                </span>
              </div>

              {/* KPIs — builder SSOT compartido con Backtest */}
              <div
                className={`grid grid-cols-2 md:grid-cols-4 ${ratiosHidden ? 'xl:grid-cols-3' : 'xl:grid-cols-7'} gap-3 mb-3`}
                data-testid="prod-kpis"
              >
                {kpis.map((k) => (
                  <GmKpi key={k.label} label={k.label} value={k.value} tone={k.tone} sub={k.sub} />
                ))}
              </div>
              {ratiosHidden && (
                <p className={`${GMT.micro} ${GM.textMuted} mb-4 m-0`}>
                  Con menos de 20 operaciones se publica solo conteo y P&L (quant-constitution §6).
                </p>
              )}

              {/* Tabla de métricas del BACKTEST (OOS) — Calmar/Sharpe/… con N alto,
                  siempre visibles; complementa la fila YTD-live gateada arriba. */}
              {backtestStats && backtestN >= 20 && (
                <GmPanel className="mb-4">
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <div>
                      <h3 className={`${GMT.panelTitle} ${GM.textStrong} m-0`}>
                        {t('btMetricsTitle')} · {backtestYear}
                      </h3>
                      <p className={`${GMT.micro} ${GM.textMuted} m-0 mt-0.5`}>{t('btMetricsSub')}</p>
                    </div>
                    {backtestPValue?.p_value != null && (
                      <GmBadge tone={backtestPValue.significant ? 'pos' : 'neutral'}>
                        {backtestPValue.significant ? t('btSignificant') : t('btNotSignificant')} · p={backtestPValue.p_value.toFixed(4)}
                      </GmBadge>
                    )}
                  </div>
                  <div
                    className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3"
                    data-testid="prod-backtest-kpis"
                  >
                    {backtestKpis.map((k) => (
                      <GmKpi key={k.label} label={k.label} value={k.value} tone={k.tone} sub={k.sub} />
                    ))}
                  </div>
                </GmPanel>
              )}

              {/* Gráfico de velas con señales (mismo framework que Backtest) */}
              {!isDefault && !chartSymbol ? (
                <GmPanel className="mb-4">
                  <p className={`${GMT.body} ${GM.textSec} m-0 text-center`}>
                    Esta estrategia opera en barras <b className={GM.textStrong}>diarias</b> (cierre UTC 00:00).
                    El gráfico de velas con señales está en <b className={GM.textStrong}>Backtest</b> — aquí se
                    rastrean KPIs, operaciones y estado de aprobación del forward paper.
                  </p>
                </GmPanel>
              ) : (
                <GmPanel
                  title="Señales de trading"
                  meta={marketIsOpen ? 'Velas 5m en tiempo real · refresco 30s' : 'Replay con señales de entrada/salida'}
                  className="mb-4"
                >
                  <div className="rounded-xl overflow-hidden">
                    <TradingChartWithSignals
                      key={`gm-prod-chart-${strategyId}-${marketIsOpen ? 'live' : 'replay'}`}
                      symbol={chartSymbol ?? 'USDCOP'}
                      timeframe="5m"
                      height={400}
                      showSignals
                      showPositions={false}
                      showStopLossTakeProfit={false}
                      enableRealTime={marketIsOpen && isDefault}
                      isReplayMode={!marketIsOpen || !isDefault}
                      startDate={`${year}-01-01`}
                      replayTrades={!marketIsOpen ? displayTrades.flatMap((t) => {
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
                        return [entry, {
                          trade_id: t.trade_id + 10000,
                          timestamp: exitTs,
                          side: t.side === 'SHORT' ? 'LONG' as const : 'SHORT' as const,
                          entry_price: Number(t.exit_price),
                          pnl: Number(t.pnl_usd),
                          status: 'closed' as const,
                        }];
                      }) : []}
                    />
                  </div>
                </GmPanel>
              )}

              {/* Curva de equity (con eje Y) */}
              <GmPanel
                title="Curva de equity"
                meta={`${strategyName} · capital inicial $${initialCapital.toLocaleString('en-US')}`}
                className="mb-4"
              >
                {equity ? (
                  <EquityChart points={equity.points} initialCapital={equity.initial} />
                ) : (
                  // Degradación elegante: PNG del pipeline si no hay puntos (onError lo oculta).
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`/data/production/equity_curve_${year}.png?t=${summary?.generated_at ?? ''}`}
                    alt={`Curva de equity ${year}`}
                    className="w-full h-auto rounded-xl"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                  />
                )}
              </GmPanel>

              {/* Grid principal: trades + distribución | señal/posición/guardrails/aprobación */}
              <div className="grid grid-cols-1 xl:grid-cols-[1.5fr_1fr] gap-4 items-start">
                <div className="flex flex-col gap-4 min-w-0">
                  {bundleStats?.exit_reasons && Object.keys(bundleStats.exit_reasons).length > 0 && (
                    <ExitReasonsPanel reasons={bundleStats.exit_reasons} />
                  )}
                  <AsyncBoundary
                    state={tradesState}
                    skeleton={<div className={`${GM.panel} h-[320px] motion-safe:animate-pulse`} />}
                    empty={(t) => t.length === 0}
                    emptyProps={{
                      title: 'Sin operaciones',
                      body: 'Aún no hay trades registrados para esta estrategia en el año en curso.',
                    }}
                  >
                    {(t) => <TradesTable trades={t} />}
                  </AsyncBoundary>
                </div>

                <div className="flex flex-col gap-4 min-w-0">
                  {isDefault ? (
                    <AsyncBoundary
                      state={liveQ}
                      skeleton={<div className={`${GM.panel} h-[220px] motion-safe:animate-pulse`} />}
                      empty={(d) => d.data_source !== 'db'}
                      emptyProps={{
                        title: 'Sin datos en vivo',
                        body: 'La base de datos en vivo no está disponible — se muestran los números del bundle publicado.',
                      }}
                    >
                      {(d) => (
                        <>
                          <SignalPanel
                            signal={d.current_signal}
                            position={d.active_position}
                            marketOpen={d.market.is_open}
                            t={t}
                          />
                          <PositionPanel
                            position={d.active_position}
                            realtimePrice={priceData?.price ?? null}
                          />
                          {d.guardrails && <GuardrailsPanel guardrails={d.guardrails} />}
                        </>
                      )}
                    </AsyncBoundary>
                  ) : (
                    <GmPanel title={t('sigCurrentTitle')}>
                      <p className={`${GMT.body} ${GM.textSec} m-0`}>
                        Estrategia en forward <b className={GM.textStrong}>paper</b> sobre barras diarias — sin
                        monitoreo intradía en vivo. Los resultados se publican en el bundle semanal.
                      </p>
                    </GmPanel>
                  )}
                  {approval && !isClientView && <ApprovalPanel approval={approval} />}
                </div>
              </div>

              {/* Pie: fuente de datos */}
              <div className="mt-6 flex justify-center">
                <span className={`inline-flex items-center gap-2 ${GM.panelSoft} px-4 py-2 ${GMT.micro} ${GM.textSec}`}>
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: dbLive ? HEX.pos : HEX.tick }}
                    aria-hidden
                  />
                  {dbLive ? 'DB live' : 'Bundle publicado (archivo)'} · {strategyName} · {year}
                  {' · '}auto-refresh {marketIsOpen ? '60s' : '5m'}
                </span>
              </div>
            </>
          )}
        </AsyncBoundary>
      )}
    </div>
  );
}
