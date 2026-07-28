/**
 * StrategyBundleManifest / RegistryIndex — TypeScript mirror of the Python SSOT
 * `src/contracts/strategy_manifest.py` (CTR-STRAT-REGISTRY-001, audit A4-03).
 *
 * These types describe what the dynamic registry ships to the frontend:
 *   public/data/registry.json                      → RegistryIndex
 *   public/data/strategies/<sid>/manifest.json     → StrategyBundleManifest
 *
 * Keep field-for-field parity with the Python dataclasses — the Python side is
 * the producer; change there first, mirror here second.
 */

export const MANIFEST_SCHEMA_VERSION = '1.0.0';

export type PipelineType = 'ml_forecasting' | 'rl' | 'rule_based' | 'hybrid';
export type StrategyTimeframe = 'weekly' | 'daily' | 'intraday_5m';
export type LifecycleStatus = 'experimental' | 'paper' | 'production' | 'archived';
/**
 * BL-13 / C-005: "action" = tradeable strategy, "diagnostic" = look-only research
 * surface. Diagnostic may never be visible (non-archived) nor champion.
 * Python mirror: strategy_manifest.SURFACES (default "action" — additive).
 */
export type StrategySurface = 'action' | 'diagnostic';

/**
 * Runtime whitelist — mirror of Python `strategy_manifest.SURFACES` (same values,
 * same order). The closed union type above exists only at compile time; untrusted
 * JSON (registry.json, manifest.json) must be validated at RUNTIME with the same
 * fail-closed rule as the Python producer (`validate_surface`): absence/null keeps
 * legacy semantics ("action"); any other unknown value is REJECTED, never coerced.
 * K-024: bilateral semantic parity with rejection tests on both sides.
 */
export const STRATEGY_SURFACES = ['action', 'diagnostic'] as const;

/** Type guard: is this a known surface value? */
export function isStrategySurface(raw: unknown): raw is StrategySurface {
  return typeof raw === 'string' && (STRATEGY_SURFACES as readonly string[]).includes(raw);
}

/**
 * Policy-contract-style validator (cf. policy.contract.ts / forecast-output.contract.ts):
 * returns the list of violations, [] = valid. Absence (undefined/null) is legal legacy
 * (-> "action"); a PRESENT unknown value is a contract violation.
 */
export function validateStrategySurface(raw: unknown): string[] {
  if (raw === undefined || raw === null) return [];
  if (isStrategySurface(raw)) return [];
  return [
    `surface ${JSON.stringify(raw)} not in [${STRATEGY_SURFACES.join(', ')}] — ` +
      'fail-closed (BL-13/C-005): unknown surfaces are rejected, never coerced to "action"',
  ];
}

/**
 * Fail-closed accessor — functional mirror of Python `__post_init__`/`validate_surface`:
 * absence/null -> legacy "action"; unknown value -> throw.
 */
export function assertStrategySurface(raw: unknown): StrategySurface {
  const errors = validateStrategySurface(raw);
  if (errors.length > 0) throw new Error(errors[0]);
  return raw === undefined || raw === null ? 'action' : (raw as StrategySurface);
}

/** One immutable backtest, keyed by (model_version, year). NEVER overwritten (spec §5). */
export interface BacktestEntry {
  model_version: string;
  year: number;
  immutable_id: string;
  /** Paths relative to public/data/ */
  summary: string;
  trades: string;
  signals?: string | null; // present iff replayable
  replayable?: boolean;
  gates?: { passed?: number; of?: number; recommendation?: string } & Record<string, unknown>;
  headline?: {
    return_pct?: number | null;
    sharpe?: number | null;
    max_dd_pct?: number | null;
    win_rate_pct?: number | null;
    p_value?: number | null;
    trades?: number | null;
  } & Record<string, unknown>;
}

export interface ModelVersionEntry {
  version: string;
  active?: boolean;
  trained_at?: string | null;
  train_window?: string | null;
  feature_hash?: string | null;
  norm_stats_hash?: string | null;
  artifact_uri?: string | null;
}

/** Mutable live-year pointer (audit A4-01): the production files grow weekly. */
export interface ProductionPointer {
  model_version: string;
  year: number | null;
  /** Paths relative to public/data/ */
  summary: string;
  trades: string;
  signals?: string;
  updated_at?: string;
}

/** Self-describing strategy bundle — the only thing the UI needs to render a strategy. */
export interface StrategyBundleManifest {
  strategy_id: string;
  asset_id: string;
  symbol: string;
  chart_symbol: string;
  display_name: string;
  pipeline_type: PipelineType | string;
  timeframe: StrategyTimeframe | string;
  status: LifecycleStatus | string;
  /** BL-13/C-005 — producer default "action"; optional here for legacy manifests. */
  surface?: StrategySurface;
  schema_version?: string;
  capabilities?: { replay?: boolean; live?: boolean; approval?: boolean } & Record<string, boolean>;
  produced_by?: Record<string, unknown>;
  backtests: BacktestEntry[];
  production?: ProductionPointer | null;
  approval?: Record<string, unknown> | null;
  model_versions: ModelVersionEntry[];
}

export interface RegistryStrategyEntry {
  strategy_id: string;
  asset_id: string;
  status: string;
  display_name: string;
  pipeline_type: PipelineType | string;
  timeframe: StrategyTimeframe | string;
  /** Path to the manifest, relative to public/data/ */
  manifest: string;
  /** BL-13/C-005 — producer default "action"; optional here for legacy registries. */
  surface?: StrategySurface;
  backtest_years?: number[];
  has_production?: boolean;
  has_replay?: boolean;
  // Active-version headline (optional, additive — lets selectors skip N+1 fetches)
  active_version?: string | null;
  return_pct?: number | null;
  sharpe?: number | null;
  p_value?: number | null;
}

export interface RegistryAssetEntry {
  asset_id: string;
  symbol: string;
  chart_symbol: string;
  display_name: string;
  asset_class?: string;
}

/** The dynamic index the frontend fetches to build all selectors (spec §4). */
export interface RegistryIndex {
  generated_at: string;
  assets: RegistryAssetEntry[];
  strategies: RegistryStrategyEntry[];
  default: { asset_id: string; strategy_id: string };
  schema_version?: string;
}

// ── helpers (read-side mirrors of the Python @property accessors) ─────────────

export function activeVersion(m: StrategyBundleManifest): string | null {
  const active = m.model_versions?.find((v) => v.active);
  if (active) return active.version;
  return m.model_versions?.length ? m.model_versions[m.model_versions.length - 1].version : null;
}

export function activeBacktest(m: StrategyBundleManifest): BacktestEntry | null {
  const av = activeVersion(m);
  const matches = (m.backtests ?? []).filter((b) => b.model_version === av);
  return matches.find((b) => b.replayable) ?? matches[0]
    ?? (m.backtests?.length ? m.backtests[m.backtests.length - 1] : null);
}
