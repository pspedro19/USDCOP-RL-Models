/**
 * Composition of the PUBLIC track record, shared by the page and the API route.
 *
 * It lives here rather than in the route handler because the page is a server component:
 * having it fetch its own `/api/public/track-record` over HTTP made the page prerender at
 * BUILD time, when no server is listening, so the deployed page was permanently frozen in
 * its "results unavailable" state. Reading the bundles directly removes the self-call
 * entirely — one composition, two callers, no network hop.
 *
 * WHAT IS PUBLIC (and why it is safe): results that already happened. Returns, drawdown,
 * closed trades, the p-value. The sellable asset is the signal published BEFORE the fact —
 * direction, levels, sizing, timing — and none of that appears here. A trade closed more
 * than a week ago proves the method without handing anyone a trade.
 *
 * WHAT IS DELIBERATELY ABSENT: approval gates and the Deflated Sharpe. Those are
 * `research:read` (CXD-057); a marketing surface must not become the bypass for the one
 * projection the SSOT reserves to research.
 *
 * Honesty invariants enforced here rather than hoped for:
 *  - `phase` is `paper`: execution is simulated until the SFC gate (rbac.md §9).
 *  - No Sharpe, profit factor or p-value for a series under 20 trades
 *    (quant-constitution §6) — stripped here, never trusted from the bundle.
 *  - Every figure comes from the PUBLISHED bundle, never recomputed (quant-constitution §7).
 */
import { promises as fs } from 'fs';
import path from 'path';

const PROD_DIR = path.join(process.cwd(), 'public', 'data', 'production');

/** Below this many trades only count and PnL may be published (quant-constitution §6). */
export const MIN_TRADES_FOR_RATIOS = 20;
/** Closed trades younger than this stay private — the lag is what keeps signals sellable. */
export const TRADE_LAG_DAYS = 7;

export interface PeriodView {
  year: number;
  label: string;
  return_pct: number | null;
  buy_hold_pct: number | null;
  max_dd_pct: number | null;
  trades: number | null;
  win_rate_pct: number | null;
  /** null whenever trades < 20 — never "not computed", always "not publishable". */
  sharpe: number | null;
  profit_factor: number | null;
  p_value: number | null;
  significant: boolean | null;
  ratios_withheld: boolean;
}

export interface DefenceRow {
  asset: string;
  strategy_name: string;
  year: number;
  market_pct: number | null;
  strategy_pct: number | null;
  trades: number | null;
}

export interface ClosedTrade {
  opened: string;
  closed: string;
  side: string;
  pnl_pct: number | null;
  exit_reason: string | null;
}

export interface TrackRecord {
  phase: 'paper';
  unavailable?: boolean;
  disclaimer: string;
  strategy: { id: string; name: string; bundle_date: string | null };
  periods: { forward: PeriodView | null; backtest: PeriodView | null };
  defence: DefenceRow[];
  closed_trades: { lag_days: number; forward: ClosedTrade[]; backtest: ClosedTrade[] };
  min_trades_for_ratios: number;
}

type Bundle = {
  year?: number;
  strategy_id?: string;
  strategy_name?: string;
  generated_at?: string;
  n_trades?: number;
  strategies?: Record<string, Record<string, unknown>>;
  statistical_tests?: { p_value?: number | null; significant?: boolean | null };
};

const num = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

async function readJson<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(PROD_DIR, file), 'utf-8')) as T;
  } catch {
    return null;
  }
}

/** Project one published bundle into a period view, applying the N<20 rule at the boundary. */
function toPeriod(bundle: Bundle | null, label: string): PeriodView | null {
  if (!bundle) return null;
  const sid = bundle.strategy_id ?? 'smart_simple_v11';
  const s = bundle.strategies?.[sid] ?? {};
  const bh = bundle.strategies?.buy_and_hold ?? {};

  // Prefer the bundle's own count; fall back to the long/short split, and treat a zero
  // split as "not reported" rather than as zero trades.
  const counted = (num(s.n_long) ?? 0) + (num(s.n_short) ?? 0);
  const trades = bundle.n_trades ?? (counted > 0 ? counted : null);
  const enough = (trades ?? 0) >= MIN_TRADES_FOR_RATIOS;
  const dd = num(s.max_dd_pct);

  return {
    year: bundle.year ?? 0,
    label,
    return_pct: num(s.total_return_pct),
    buy_hold_pct: num(bh.total_return_pct),
    // Exporters publish drawdown with either sign; normalize to a magnitude so a reader
    // never has to work out whether −7.84 and 7.84 mean different things.
    max_dd_pct: dd === null ? null : Math.abs(dd),
    trades,
    win_rate_pct: num(s.win_rate_pct),
    sharpe: enough ? num(s.sharpe) : null,
    profit_factor: enough ? num(s.profit_factor) : null,
    p_value: enough ? (num(bundle.statistical_tests?.p_value) ?? null) : null,
    significant: enough ? (bundle.statistical_tests?.significant ?? null) : null,
    ratios_withheld: !enough,
  };
}

/** Closed trades older than the lag window, stripped of anything actionable. */
function publishableTrades(raw: unknown): ClosedTrade[] {
  const list: unknown[] = Array.isArray(raw)
    ? raw
    : Array.isArray((raw as { trades?: unknown[] })?.trades)
      ? (raw as { trades: unknown[] }).trades
      : [];
  const cutoff = Date.now() - TRADE_LAG_DAYS * 86_400_000;

  return list
    .map((t) => t as Record<string, unknown>)
    .filter((t) => {
      const exit = typeof t.exit_timestamp === 'string' ? Date.parse(t.exit_timestamp) : NaN;
      return Number.isFinite(exit) && exit < cutoff; // still open ⇒ never published
    })
    .map((t) => ({
      opened: String(t.timestamp ?? '').slice(0, 10),
      closed: String(t.exit_timestamp ?? '').slice(0, 10),
      side: String(t.side ?? ''),
      pnl_pct: num(t.pnl_pct),
      exit_reason: typeof t.exit_reason === 'string' ? t.exit_reason : null,
      // Entry/exit levels and sizing are omitted on purpose: those are the product.
    }))
    .sort((a, b) => (a.closed < b.closed ? 1 : -1));
}

/**
 * Market-vs-strategy rows, built ONLY from bundles that publish a buy_and_hold baseline.
 * An asset without a published baseline is omitted rather than compared against a number
 * taken from somewhere else — the comparison IS the claim, so it has to come from the same
 * artifact as the result.
 */
const DEFENCE_SOURCES: Array<{ file: string; asset: string }> = [
  { file: 'summary.json', asset: 'USD/COP' },
  { file: 'summary_2025.json', asset: 'USD/COP' },
  { file: 'summary_gold_dynamic_exit.json', asset: 'XAU/USD' },
  { file: 'summary_gold_trend_simple.json', asset: 'XAU/USD' },
];

export async function buildTrackRecord(): Promise<TrackRecord> {
  const forward = await readJson<Bundle>('summary.json');
  const backtest = await readJson<Bundle>('summary_2025.json');

  const base = {
    phase: 'paper' as const,
    disclaimer:
      'Ejecución simulada (paper). No operamos capital de terceros. Rendimientos pasados ' +
      'no garantizan resultados futuros.',
    min_trades_for_ratios: MIN_TRADES_FOR_RATIOS,
  };

  if (!forward && !backtest) {
    return {
      ...base,
      unavailable: true,
      strategy: { id: '', name: '', bundle_date: null },
      periods: { forward: null, backtest: null },
      defence: [],
      closed_trades: { lag_days: TRADE_LAG_DAYS, forward: [], backtest: [] },
    };
  }

  const defence: DefenceRow[] = [];
  for (const src of DEFENCE_SOURCES) {
    const b = await readJson<Bundle>(src.file);
    if (!b) continue;
    const sid = b.strategy_id ?? '';
    const s = b.strategies?.[sid] ?? {};
    const market = num(b.strategies?.buy_and_hold?.total_return_pct);
    if (market === null) continue; // no published baseline ⇒ no comparison
    defence.push({
      asset: src.asset,
      strategy_name: b.strategy_name ?? sid,
      year: b.year ?? 0,
      market_pct: market,
      strategy_pct: num(s.total_return_pct),
      trades: b.n_trades ?? null,
    });
  }

  return {
    ...base,
    strategy: {
      id: forward?.strategy_id ?? backtest?.strategy_id ?? 'smart_simple_v11',
      name: forward?.strategy_name ?? backtest?.strategy_name ?? 'Smart Simple',
      bundle_date: (forward?.generated_at ?? backtest?.generated_at ?? '').slice(0, 10) || null,
    },
    periods: {
      forward: toPeriod(forward, 'Producción forward (papel)'),
      backtest: toPeriod(backtest, 'Backtest fuera de muestra'),
    },
    defence,
    closed_trades: {
      lag_days: TRADE_LAG_DAYS,
      forward: publishableTrades(await readJson('trades/smart_simple_v11.json')),
      backtest: publishableTrades(await readJson('trades/smart_simple_v11_2025.json')),
    },
  };
}
