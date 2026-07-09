/**
 * UI governance contract (spec `.claude/specs/platform/ux-navigation.md` §6).
 *
 * P1 — every performance figure declares PROVENANCE (strategy · version · bundle date).
 * P2 — every performance figure carries a phase badge: LIVE / BACKTEST / PAPER.
 * Design tokens centralize the palette so components stop hardcoding ad-hoc slate/cyan
 * variants (sobriety + AA contrast come from ONE place).
 */

export type MetricPhase = 'live' | 'backtest' | 'paper';

export interface MetricProvenance {
  strategyId: string;
  version?: string;
  /** ISO date of the published bundle the figure comes from. */
  bundleDate?: string;
}

/** Phase badge tokens — the ONLY allowed styling for performance-metric badges. */
export const PHASE_TOKENS: Record<MetricPhase, { symbol: string; label: string; className: string }> = {
  live: {
    symbol: '●', label: 'LIVE',
    className: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/40',
  },
  backtest: {
    symbol: '◆', label: 'BACKTEST',
    className: 'bg-blue-500/10 text-blue-300 border-blue-500/40',
  },
  paper: {
    symbol: '○', label: 'PAPER',
    className: 'bg-amber-500/10 text-amber-300 border-amber-500/40',
  },
};

/**
 * Sober design tokens (AA on dark). Components should consume these instead of ad-hoc
 * class soup; migrate incrementally.
 */
export const UI_TOKENS = {
  surface: 'bg-slate-950',
  card: 'bg-slate-900/40 border border-slate-700/60 rounded-2xl',
  cardHighlight: 'bg-cyan-500/5 border border-cyan-500/60 rounded-2xl',
  // slate-400 fails AA on slate-950 for small text — use these instead of slate-500/600.
  textPrimary: 'text-slate-100',
  textSecondary: 'text-slate-300',
  textMuted: 'text-slate-400',
  ctaPrimary: 'bg-cyan-600 hover:bg-cyan-500 text-white font-semibold rounded-xl',
  ctaSecondary: 'bg-slate-800 hover:bg-slate-700 text-slate-100 font-semibold rounded-xl',
  positive: 'text-emerald-400',
  negative: 'text-red-400',
  warning: 'text-amber-300',
} as const;

/** N<20 trades ⇒ only count + PnL are publishable (quant-constitution §6). */
export const MIN_TRADES_FOR_RATIOS = 20;

export function canShowRatios(nTrades: number | null | undefined): boolean {
  return (nTrades ?? 0) >= MIN_TRADES_FOR_RATIOS;
}

export function provenanceLabel(p: MetricProvenance): string {
  const parts = [p.strategyId, p.version, p.bundleDate && `bundle ${p.bundleDate}`];
  return `fuente: ${parts.filter(Boolean).join(' · ')}`;
}

/**
 * Calmar ratio = annualized return / |max drawdown| — the quant-constitution's PRIMARY
 * graduation metric (§2), so it must be a first-class KPI wherever Sharpe is shown.
 * Annualizes from trading days when available (252/yr); with no day count it falls back
 * to the raw period return over |DD| (labelled by the caller). Null when not computable.
 */
export function calmarRatio(
  totalReturnPct: number | null | undefined,
  maxDdPct: number | null | undefined,
  tradingDays?: number | null,
): number | null {
  if (totalReturnPct == null || maxDdPct == null) return null;
  const dd = Math.abs(maxDdPct);
  if (dd < 1e-9) return null; // no drawdown ⇒ ratio undefined (never Infinity)
  const r = totalReturnPct / 100;
  const annualized = tradingDays && tradingDays > 0
    ? (Math.pow(1 + r, 252 / tradingDays) - 1) * 100
    : totalReturnPct;
  return annualized / dd;
}
