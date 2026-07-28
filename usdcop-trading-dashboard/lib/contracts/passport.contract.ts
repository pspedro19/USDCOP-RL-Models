/**
 * Strategy Passport + Control Tower contract (CTR-PASSPORT-001) — TS mirror.
 * =========================================================================
 *
 * BL-32 / FABRIC §24.4-§24.5. Python SSOT twin: `src/contracts/passport.py`.
 * Spec: `.claude/specs/platform/passport-control-tower.md`.
 *
 * The Passport is a DERIVED read-only view (identity + governance + lineage +
 * performance across five environments + execution/risk). FABRIC says it must be
 * "un SELECT" over fact tables. Those fact tables do not exist yet (BL-18 metric
 * engine, BL-21/BL-22 exec + fact_pnl, BL-24 lineage), so the contract is built
 * around ONE primitive:
 *
 *     Sourced<T> = { value: T | null; source: { path, status, pending, note } }
 *
 * Every number names the published artifact it came from. A number with no
 * artifact is never rendered as 0 / "—" / an estimate: it is `value: null` with
 * `status: 'unavailable'` and a `pending` string naming what will supply it.
 * That is what makes the missing pieces a DOCUMENTED INTERFACE instead of a hole.
 *
 * Invariants encoded (asserted by `validate*`, not merely commented):
 *  - quant-constitution §6 — N<20 ⇒ only count + PnL; ratios/p/DSR suppressed.
 *  - §2 — N_MAX is a spend cap and NEVER enters the DSR; N×3 is disclosure.
 *  - strategy-contract §2 — no Infinity/NaN reaches JSON.
 *  - approval-gates §3 — DIAGNOSTIC surface: no approve/promote/deploy/kill.
 */

// ───────────────────────────────────────────────────────── contract identity

export const PASSPORT_CONTRACT_ID = 'CTR-PASSPORT-001';
export const PASSPORT_CONTRACT_VERSION = '1.0.0';

// ─────────────────────────────────────────────────────────────── vocabularies

/** Published or not. There is no "estimated" — that would be a modelling call. */
export const SOURCE_STATUSES = ['published', 'unavailable'] as const;
export type SourceStatus = (typeof SOURCE_STATUSES)[number];

/** FABRIC §24.4 — the five columns of the same metric from the same engine.
 *  The single engine is BL-18; until then each column declares its own source
 *  and `metric_engine` is reported unavailable. */
export const PASSPORT_ENVS = ['backtest', 'held_out', 'paper', 'canary', 'live'] as const;
export type PassportEnv = (typeof PASSPORT_ENVS)[number];

/** FABRIC §24.5 LIBRO — "conteo por estado". */
export const BOOK_STATES = ['CHAMPION', 'CANARY', 'PAPER', 'REDUCED', 'QUARANTINED'] as const;
export type BookState = (typeof BOOK_STATES)[number];

/** Retirement traffic light (quant-constitution §5). `unknown` is first-class:
 *  a strategy without a signed withdrawal protocol is NOT "green". */
export const RETIREMENT_SIGNALS = ['green', 'yellow', 'red', 'unknown'] as const;
export type RetirementSignal = (typeof RETIREMENT_SIGNALS)[number];

/** Three-clock monitoring (BL-25). `exec` has no producer yet. */
export const HEALTH_CLOCKS = ['data', 'model', 'exec'] as const;
export type HealthClock = (typeof HEALTH_CLOCKS)[number];

/** Trial lineage (ADR-0022): FT = predictive, AT = economic. */
export const TRIAL_KINDS = ['forecast', 'action'] as const;
export type TrialKind = (typeof TRIAL_KINDS)[number];

// ────────────────────────────────────────────────────── constitutional consts

/** quant-constitution §6 — below this only count and PnL are publishable. */
export const MIN_TRADES_FOR_RATIOS = 20;

/** FABRIC §9.7 — commitment device against p-hacking. SPEND CAP ONLY: it never
 *  enters the DSR or any statistical formula. Disclosed beside N_global. */
export const N_MAX_TRIALS = 989;

/** quant-constitution §2 — the bar for any edge claim. */
export const DSR_BAR = 0.95;

/** Inferential fields suppressed by the small-sample guard. */
export const SMALL_SAMPLE_SUPPRESSED_FIELDS = [
  'sharpe', 'sortino', 'calmar', 'p_value', 'dsr_family', 'dsr_cluster',
  'dsr_global', 'psr', 'bootstrap_ci_low', 'bootstrap_ci_high',
] as const;

/** The §6 verdict when the published source does not let us determine N.
 *  FAIL-CLOSED (S-04): the guard used to return untouched on an unknown N
 *  ("absence of N is not evidence of N<20"), which is backwards for a
 *  PUBLICATION guard — the manifests that omit the count are exactly the ones
 *  with 1-3 trades. `btc_hodl_b1` published Sharpe 0.793 and p=0.0242 off ONE
 *  trade with both guards green in both languages. Mirrors
 *  `src/contracts/passport.py::UNDETERMINABLE_N_REASON`. */
export const UNDETERMINABLE_N_REASON =
  'N no determinable desde la fuente publicada (fail-closed, quant-constitution §6: '
  + 'sin conteo de trades no se publica Sharpe/p-value/DSR)';

/** The Passport/Tower is DIAGNOSTIC. Vote 2 lives ONLY on /dashboard.
 *  Any of these keys in a payload is a contract violation, not a feature. */
export const FORBIDDEN_PASSPORT_ACTIONS = [
  'approve', 'reject', 'promote', 'deploy', 'vote', 'kill_switch', 'execute',
] as const;

// ─────────────────────────────────────────────────────── the Sourced primitive

export interface FieldSource {
  /** Repo-root-relative artifact path. `null` ⟺ status === 'unavailable'. */
  path: string | null;
  status: SourceStatus;
  /** What will supply this field (backlog id + system). Required when unavailable. */
  pending: string | null;
  /** Free-form caveat shown next to the value (e.g. "forward sin reconciliar"). */
  note?: string | null;
}

export interface Sourced<T> {
  value: T | null;
  source: FieldSource;
}

export type SourcedNumber = Sourced<number>;
export type SourcedString = Sourced<string>;

/** NaN/Infinity ⇒ null. Applied at the value level so a bad float never ships. */
export function sanitizeNumber<T>(value: T): T | null {
  if (typeof value === 'number' && !Number.isFinite(value)) return null;
  return value;
}

/** A value backed by a published artifact. `path` must be a real published file. */
export function sourced<T>(value: T | null, path: string, note?: string): Sourced<T> {
  return { value: sanitizeNumber(value), source: { path, status: 'published', pending: null, note: note ?? null } };
}

/** A value NO published artifact can supply yet — the documented acople point. */
export function unavailable<T = number>(pending: string, note?: string): Sourced<T> {
  return { value: null, source: { path: null, status: 'unavailable', pending, note: note ?? null } };
}

export function isAvailable(field: Sourced<unknown> | null | undefined): boolean {
  return !!field && field.source?.status === 'published' && field.value != null;
}

/** Mirror of `canShowRatios` (ui.contract.ts) kept local so the contract is standalone. */
export function canShowRatios(nTrades: number | null | undefined): boolean {
  return (nTrades ?? 0) >= MIN_TRADES_FOR_RATIOS;
}

// ───────────────────────────────────────────────────────── passport structures

export interface PassportIdentity {
  strategy_id: string;
  asset_id: string;
  display_name: string;
  /** 'action' (trades capital) vs 'diagnostic' (forecast only) — BL-13 surface. */
  surface: string | null;
  engine_type: string | null;      // rule_based | ml | rl | composite (strategy-engines.md)
  status: string | null;           // registry status vocabulary (production/experimental/…)
  active_version: SourcedString;
  timeframe: string | null;
}

/** Governance = trials (FT/AT, N×3), votes, gates, withdrawal protocol. */
export interface PassportGovernance {
  n_trials_total: SourcedNumber;
  n_trials_forecast: SourcedNumber;   // FT — predictive lineage
  n_trials_action: SourcedNumber;     // AT — economic lineage
  n_family: SourcedNumber;
  n_cluster: SourcedNumber;
  n_global: SourcedNumber;
  /** §2: the DSR bar. Reported ONLY when a published gate carries it. */
  dsr_family: SourcedNumber;
  dsr_bar: number;
  approval_status: SourcedString;
  gates: Sourced<PassportGate[]>;
  withdrawal_protocol: SourcedString;   // path to the signed protocol
  retirement_signal: RetirementSignal;
  retirement_reason: string | null;
}

export interface PassportGate {
  gate: string;
  label: string;
  passed: boolean;
  value: number | null;
  threshold: number | null;
}

/** Lineage = snapshots + fingerprints. BL-17/BL-24 own the real graph. */
export interface PassportLineage {
  model_versions: Sourced<PassportModelVersion[]>;
  spec_fingerprint: SourcedString;
  feature_set_hash: SourcedString;
  policy_hash: SourcedString;
  /** BL-24 nodes/edges + camino dorado. */
  lineage_graph: Sourced<unknown>;
}

export interface PassportModelVersion {
  version: string;
  active: boolean;
  trained_at: string | null;
  train_window: string | null;
  feature_hash: string | null;
  norm_stats_hash: string | null;
  artifact_uri: string | null;
}

/** Performance in ONE environment. Every figure is Sourced. */
export interface EnvPerformance {
  env: PassportEnv;
  period_label: SourcedString;
  return_pct: SourcedNumber;
  n_trades: SourcedNumber;
  max_dd_pct: SourcedNumber;
  win_rate_pct: SourcedNumber;
  profit_factor: SourcedNumber;
  sharpe: SourcedNumber;
  calmar: SourcedNumber;
  p_value: SourcedNumber;
  dsr_family: SourcedNumber;
  timing_ratio: SourcedNumber;
  /** True ⇒ ratios were suppressed by the §6 guard (N<20). */
  insufficient_trades: boolean;
}

/** §24.4 `v_strategy_passport_live` — NOT materialized; operational state. */
export interface PassportLiveState {
  open_orders: SourcedNumber;
  last_fill_at: SourcedString;
  quarantined: Sourced<boolean>;
  reconciled: Sourced<boolean>;
  kill_switch_engaged: Sourced<boolean>;
  deploy_status: SourcedString;
  last_signal_at: SourcedString;
}

/** Risk block (BL-26/BL-27 own the real numbers). */
export interface PassportRisk {
  current_exposure: SourcedNumber;
  vol_target_pct: SourcedNumber;
  vol_forecast_pct: SourcedNumber;
  m_forward: SourcedNumber;
  m_dd: SourcedNumber;
  rho_max: SourcedNumber;
  turnover: SourcedNumber;
}

/** §24.4 `v_strategy_passport` — the composition. One object, one SELECT. */
export interface StrategyPassport {
  contract: typeof PASSPORT_CONTRACT_ID;
  contract_version: string;
  strategy_id: string;
  generated_at: string;
  identity: PassportIdentity;
  governance: PassportGovernance;
  lineage: PassportLineage;
  /** All five envs are always present; missing ones are fully `unavailable`. */
  performance: Record<PassportEnv, EnvPerformance>;
  live: PassportLiveState;
  risk: PassportRisk;
  /** FABRIC §24.4: the five columns must come from ONE engine (BL-18). */
  metric_engine: SourcedString;
  pending_interfaces: PendingInterface[];
}

// ───────────────────────────────────────────────────── control tower (§24.5)

/** A capability the Tower/Passport declares but cannot yet source. This IS the
 *  hand-off contract: when `produced_by` lands, only the composer changes. */
export interface PendingInterface {
  /** Backlog id that unblocks it (e.g. 'BL-22'). */
  backlog_id: string;
  /** The field(s) it feeds, dotted. */
  field: string;
  /** The artifact/table the composer will read once it exists. */
  produced_by: string;
  note: string;
}

/** LIBRO — capital, PnL, vol, DD, gross/net, CVaR, state counts, ρ, attribution. */
export interface TowerBook {
  capital: SourcedNumber;
  pnl_d_pct: SourcedNumber;
  pnl_m_pct: SourcedNumber;
  pnl_y_pct: SourcedNumber;
  vol_forecast_pct: SourcedNumber;
  vol_target_pct: SourcedNumber;
  max_dd_pct: SourcedNumber;
  gross_exposure: SourcedNumber;
  net_exposure: SourcedNumber;
  cvar_pct: SourcedNumber;
  /** Counts per FABRIC state; null ⇒ no source publishes that state yet. */
  state_counts: Record<BookState, number | null>;
  state_counts_source: FieldSource;
  correlation_matrix: Sourced<TowerCorrelation>;
  diversification_ratio: SourcedNumber;
  /** timing vs beta vs carry (§24.5). BL-07 ran it one-off and did NOT persist. */
  pnl_attribution: Sourced<TowerAttribution>;
  /** Frozen book weights when the paper ledger publishes them. */
  weights: Sourced<Record<string, number>>;
}

export interface TowerCorrelation {
  sleeves: string[];
  matrix: number[][];
}

export interface TowerAttribution {
  timing_pct: number | null;
  beta_pct: number | null;
  carry_pct: number | null;
}

/** SLEEVES row (§24.5): sleeve | research | tier | ops | env | ρ_max | Sharpe |
 *  DSR(N×3) | timing_ratio | m_forward·m_dd | turnover | días al juez | semáforo. */
export interface TowerSleeve {
  strategy_id: string;
  asset_id: string;
  display_name: string;
  research_state: string | null;   // registry status
  tier: string | null;             // BL-27 tiering
  ops_state: string | null;        // BL-30 ops lifecycle
  /** Which of the five envs actually has published data. */
  envs_with_data: PassportEnv[];
  rho_max: SourcedNumber;
  sharpe: SourcedNumber;
  n_trades: SourcedNumber;
  return_pct: SourcedNumber;
  dsr_family: SourcedNumber;
  n_family: SourcedNumber;
  n_cluster: SourcedNumber;
  n_global: SourcedNumber;
  timing_ratio: SourcedNumber;
  m_forward: SourcedNumber;
  m_dd: SourcedNumber;
  turnover: SourcedNumber;
  /** Days elapsed in the sealed judge window (arithmetic over published dates). */
  judge_days: SourcedNumber;
  judge_starts_after: SourcedString;
  retirement_signal: RetirementSignal;
  retirement_reason: string | null;
  insufficient_trades: boolean;
}

/** The paired v11-live vs v12/v14-paper test (§24.5). NO p-value is invented:
 *  it is reported only if a published artifact carries it. */
export interface TowerPairedTest {
  baseline_id: string;
  candidate_id: string;
  baseline_return_pct: SourcedNumber;
  candidate_return_pct: SourcedNumber;
  n_paired: SourcedNumber;
  p_value: SourcedNumber;
  e_value: SourcedNumber;
  note: string;
}

/** DATOS — freshness, STALE, replay parity, trials per family, N_global vs N_MAX. */
export interface TowerData {
  clocks: Record<HealthClock, Sourced<TowerClock>>;
  promotions_frozen: Sourced<boolean>;
  withdrawal_triggered: Sourced<boolean>;
  stale_probes: SourcedNumber;
  replay_parity: SourcedString;
  trials_by_family: Sourced<TowerFamilyTrials[]>;
  n_global: SourcedNumber;
  /** Spend cap ONLY (§9.7) — never in the DSR. Always {N_MAX_TRIALS}. */
  n_max_trials: SourcedNumber;
  last_vintage_revision: SourcedString;
}

export interface TowerClock {
  clock: HealthClock;
  signal: string;
  actions: string[];
  evaluated_at: string | null;
}

export interface TowerFamilyTrials {
  family_id: string;
  kind: TrialKind;
  cluster_id: string | null;
  n_trials: number;
  closed: boolean;
}

export interface ControlTowerSnapshot {
  contract: typeof PASSPORT_CONTRACT_ID;
  contract_version: string;
  generated_at: string;
  book: TowerBook;
  sleeves: TowerSleeve[];
  data: TowerData;
  paired_tests: TowerPairedTest[];
  pending_interfaces: PendingInterface[];
}

// ──────────────────────────────────────────────────────────────── small sample

/**
 * The trade count of a performance/sleeve block, or `null` when it cannot be
 * determined WITH CERTAINTY from what was published. `null` covers all three
 * shapes the real artifacts produce: key absent, field `unavailable`, or field
 * published-with-value-null (`sourced(null, path)` — what the composer emits
 * when a manifest headline carries no trade count). All three mean "unknown".
 * Mirror of `src/contracts/passport.py::resolve_n_trades`.
 */
export function resolveNTrades(block: unknown): number | null {
  if (!block || typeof block !== 'object') return null;
  const field = (block as Record<string, unknown>).n_trades;
  const n = (field && typeof field === 'object' && 'value' in (field as object))
    ? (field as Sourced<unknown>).value
    : field;
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

/** The single sentence both runtimes attach to a suppressed field. */
export function smallSampleReason(n: number | null): string {
  if (n == null) return UNDETERMINABLE_N_REASON;
  return `N=${n} < ${MIN_TRADES_FOR_RATIOS} (quant-constitution §6: solo conteo y PnL)`;
}

/** Inferential fields published on a block whose N does not license them.
 *  A ratio is publishable ONLY when the block carries a published trade count
 *  >= 20. Unknown N is a violation, not a pass. */
export function smallSampleViolations(block: unknown, label: string): string[] {
  const n = resolveNTrades(block);
  if (n != null && n >= MIN_TRADES_FOR_RATIOS) return [];
  const detail = n != null
    ? `N=${n} < ${MIN_TRADES_FOR_RATIOS}`
    : 'N no determinable desde la fuente publicada (fail-closed)';
  const rec = (block ?? {}) as Record<string, Sourced<unknown>>;
  return SMALL_SAMPLE_SUPPRESSED_FIELDS
    .filter((key) => isAvailable(rec[key]))
    .map((key) => `${label}.${key}: published with ${detail} (quant-constitution §6)`);
}

/**
 * Null every inferential field of an env-performance block unless a published
 * trade count of at least 20 licenses it. Descriptive quantities survive;
 * ratios and p-values do not. The suppressed field keeps its Sourced shape but
 * flips to `unavailable` with the reason, so the UI says WHY instead of
 * rendering a blank.
 *
 * FAIL-CLOSED on an unknown N (S-04): publishing a Sharpe requires PROVING
 * N >= 20 from the artifact.
 */
export function suppressSmallSample(perf: EnvPerformance): EnvPerformance {
  const n = resolveNTrades(perf);
  if (n != null && n >= MIN_TRADES_FOR_RATIOS) return perf;
  const reason = smallSampleReason(n);
  const out = { ...perf, insufficient_trades: true } as EnvPerformance & Record<string, unknown>;
  for (const key of SMALL_SAMPLE_SUPPRESSED_FIELDS) {
    if (key in out) out[key] = unavailable(reason);
  }
  return out as EnvPerformance;
}

// ───────────────────────────────────────────────────────────────── validators

function walkSourced(node: unknown, prefix = '', out: Array<[string, Sourced<unknown>]> = []) {
  if (Array.isArray(node)) {
    node.forEach((v, i) => walkSourced(v, `${prefix}[${i}]`, out));
    return out;
  }
  if (node && typeof node === 'object') {
    const rec = node as Record<string, unknown>;
    if ('value' in rec && rec.source && typeof rec.source === 'object') {
      out.push([prefix || '<root>', rec as unknown as Sourced<unknown>]);
      return out;
    }
    for (const [k, v] of Object.entries(rec)) walkSourced(v, prefix ? `${prefix}.${k}` : k, out);
  }
  return out;
}

export function validateSourced(field: unknown, label: string): string[] {
  const errors: string[] = [];
  if (!field || typeof field !== 'object') return [`${label}: not a Sourced object`];
  const rec = field as Record<string, unknown>;
  if (!('value' in rec)) errors.push(`${label}: missing 'value'`);
  const src = rec.source as Record<string, unknown> | undefined;
  if (!src || typeof src !== 'object') return [...errors, `${label}: missing/invalid 'source'`];
  const status = src.status as string;
  if (!(SOURCE_STATUSES as readonly string[]).includes(status)) {
    errors.push(`${label}: bad source.status ${JSON.stringify(status)}`);
  }
  if (status === 'published' && !src.path) {
    errors.push(`${label}: published fields MUST name their artifact path`);
  }
  if (status === 'unavailable') {
    if (rec.value != null) errors.push(`${label}: unavailable fields MUST have value=null`);
    if (!src.pending) errors.push(`${label}: unavailable fields MUST declare what they are pending on`);
  }
  if (typeof rec.value === 'number' && !Number.isFinite(rec.value)) {
    errors.push(`${label}: non-finite number reached the contract`);
  }
  return errors;
}

export function validateStrategyPassport(payload: unknown): string[] {
  const errors: string[] = [];
  if (!payload || typeof payload !== 'object') return ['passport: not an object'];
  const p = payload as Record<string, unknown>;
  if (p.contract !== PASSPORT_CONTRACT_ID) errors.push(`passport.contract must be ${PASSPORT_CONTRACT_ID}`);
  for (const key of ['strategy_id', 'generated_at', 'identity', 'governance', 'lineage', 'performance', 'live', 'risk']) {
    if (!(key in p)) errors.push(`passport: missing '${key}'`);
  }
  const perf = p.performance as Record<string, EnvPerformance> | undefined;
  if (!perf || typeof perf !== 'object') {
    errors.push('passport.performance: not an object');
  } else {
    const missing = PASSPORT_ENVS.filter((e) => !(e in perf));
    if (missing.length) errors.push(`passport.performance must declare all five envs; missing ${missing.join(',')}`);
    for (const [env, block] of Object.entries(perf)) {
      if (!(PASSPORT_ENVS as readonly string[]).includes(env)) {
        errors.push(`passport.performance: unknown env ${JSON.stringify(env)}`);
      }
      errors.push(...smallSampleViolations(block, `passport.performance.${env}`));
    }
  }
  for (const key of FORBIDDEN_PASSPORT_ACTIONS) {
    if (key in p) errors.push(`passport: DIAGNOSTIC surface must not expose action ${JSON.stringify(key)}`);
  }
  for (const [label, field] of walkSourced(payload)) {
    errors.push(...validateSourced(field, `passport.${label}`));
  }
  return errors;
}

export function validateControlTower(payload: unknown): string[] {
  const errors: string[] = [];
  if (!payload || typeof payload !== 'object') return ['tower: not an object'];
  const t = payload as Record<string, unknown>;
  if (t.contract !== PASSPORT_CONTRACT_ID) errors.push(`tower.contract must be ${PASSPORT_CONTRACT_ID}`);
  for (const key of ['generated_at', 'book', 'sleeves', 'data', 'pending_interfaces']) {
    if (!(key in t)) errors.push(`tower: missing '${key}'`);
  }
  const book = t.book as Record<string, unknown> | undefined;
  const counts = book?.state_counts as Record<string, unknown> | undefined;
  if (!counts || typeof counts !== 'object') {
    errors.push('tower.book.state_counts: not an object');
  } else {
    const unknownStates = Object.keys(counts).filter((s) => !(BOOK_STATES as readonly string[]).includes(s));
    if (unknownStates.length) errors.push(`tower.book.state_counts: unknown states ${unknownStates.join(',')}`);
  }
  const sleeves = t.sleeves;
  if (!Array.isArray(sleeves)) {
    errors.push('tower.sleeves: not a list');
  } else {
    sleeves.forEach((s, i) => {
      const row = s as Record<string, unknown>;
      if (!row?.strategy_id) errors.push(`tower.sleeves[${i}]: missing strategy_id`);
      if (!(RETIREMENT_SIGNALS as readonly string[]).includes(row?.retirement_signal as string)) {
        errors.push(`tower.sleeves[${i}].retirement_signal: bad value ${JSON.stringify(row?.retirement_signal)}`);
      }
      // §6 applies to the SLEEVE row too: it is a decision surface, and it is
      // where `btc_hodl_b1: sharpe=0.793 n_trades=null` was published.
      errors.push(...smallSampleViolations(row, `tower.sleeves[${i}]`));
    });
  }
  const data = t.data as Record<string, unknown> | undefined;
  const nMax = (data?.n_max_trials as Sourced<number> | undefined)?.value;
  if (nMax !== N_MAX_TRIALS) {
    errors.push(`tower.data.n_max_trials must be ${N_MAX_TRIALS} (spend cap, never in the DSR)`);
  }
  for (const key of FORBIDDEN_PASSPORT_ACTIONS) {
    if (key in t) errors.push(`tower: DIAGNOSTIC surface must not expose action ${JSON.stringify(key)}`);
  }
  for (const [label, field] of walkSourced(payload)) {
    errors.push(...validateSourced(field, `tower.${label}`));
  }
  return errors;
}
