/**
 * Policy Contract (single evaluation engine, all strategy engines)
 * ================================================================
 * TypeScript mirror of the universal strategy-decision contract. Every
 * engine — rule_based, ml or composite — emits the SAME StrategyDecision;
 * execution, facts, BI and frontend never branch on the engine
 * (invariant 1 of `.claude/rules/strategy-engines.md`).
 *
 * The frontend ONLY renders decisions and rule traces — it NEVER
 * re-evaluates conditions (invariant 7). The whitelist below exists so
 * the UI can label/validate operators it displays; evaluation is
 * backend-only (`src/contracts/policy_dsl.py`).
 *
 * Spec: .claude/specs/planes/05-rule-based-strategies.md §4, §9, §15
 * Python mirror: src/contracts/policy.py + src/contracts/rule_trace.py
 *                (whitelist: src/contracts/policy_dsl.py)
 * Contract: CTR-POLICY-001 (BL-45 R1)
 */

// -----------------------------------------------------------------------------
// Constants (mirrored in policy.py / policy_dsl.py — change BOTH sides)
// -----------------------------------------------------------------------------

export const ENGINE_TYPES = ['rule_based', 'ml', 'composite'] as const;

export type EngineType = (typeof ENGINE_TYPES)[number];

export const VALID_DIRECTIONS = ['LONG', 'SHORT', 'FLAT'] as const;

export type Direction = (typeof VALID_DIRECTIONS)[number];

/** Evaluation modes a PolicyContext can run under. */
export const POLICY_MODES = ['DECISION', 'FREEZE', 'REVALIDATE', 'BACKFILL'] as const;

export type PolicyMode = (typeof POLICY_MODES)[number];

// -----------------------------------------------------------------------------
// DSL operator whitelist (mirrors policy_dsl.py — the ONLY operators a
// declarative spec may use; anything else is rejected server-side)
// -----------------------------------------------------------------------------

export const COMPARISON_OPERATORS = [
  'greater_than', 'less_than', 'equal', 'crosses_above', 'crosses_below',
] as const;

export const LOGICAL_OPERATORS = ['all', 'any', 'not'] as const;

export const RANGE_OPERATORS = ['between'] as const;

export const ALLOWED_OPERATORS = [
  ...COMPARISON_OPERATORS,
  ...LOGICAL_OPERATORS,
  ...RANGE_OPERATORS,
] as const;

export type AllowedOperator = (typeof ALLOWED_OPERATORS)[number];

// -----------------------------------------------------------------------------
// EngineRef — discriminated by `type` (spec §4)
// -----------------------------------------------------------------------------

/**
 * rule_based: policy_hash required, model snapshots FORBIDDEN
 * (a rules policy has no trained weights — `never` makes a record that
 * carries one fail to typecheck, mirroring the Python ValueError).
 */
export interface RuleBasedEngineRef {
  type: 'rule_based';
  policy_version_id?: string | null;
  policy_hash: string;              // required
  model_snapshot_id?: never;        // forbidden
  model_snapshot_ids?: never;       // forbidden
}

/** ml: model_snapshot_id required. */
export interface MlEngineRef {
  type: 'ml';
  policy_version_id?: string | null;
  policy_hash?: string | null;
  model_snapshot_id: string;        // required
  model_snapshot_ids?: string[];
}

/** composite: policy_hash required + model_snapshot_ids (predictor components). */
export interface CompositeEngineRef {
  type: 'composite';
  policy_version_id?: string | null;
  policy_hash: string;              // required
  model_snapshot_id?: string | null;
  model_snapshot_ids?: string[];    // predictor components
}

export type EngineRef = RuleBasedEngineRef | MlEngineRef | CompositeEngineRef;

// -----------------------------------------------------------------------------
// Rule trace (rule_trace_v1 — mirrors rule_trace.py)
// -----------------------------------------------------------------------------

export const RULE_TRACE_SCHEMA_V1 = 'rule_trace_v1' as const;

/** One evaluated condition inside a policy run. */
export interface RuleTraceEntry {
  rule_id: string;
  label: string;
  observed: Record<string, unknown>;
  result: boolean;
  reason_code: string;
}

/**
 * Full trace of one policy evaluation. The backend produces it; the
 * frontend only renders it (never re-evaluates conditions).
 */
export interface RuleTrace {
  trace_schema: typeof RULE_TRACE_SCHEMA_V1;   // literal 'rule_trace_v1'
  rules: RuleTraceEntry[];
}

// -----------------------------------------------------------------------------
// PolicyContext — evaluation context shape (decision §15.2: WITH state)
// -----------------------------------------------------------------------------

/**
 * Context handed to every Policy.evaluate call (backend-owned; mirrored
 * here so tooling/UI can type serialized contexts). `state` is the
 * per-strategy mutable store that persists between evaluations;
 * `previous_snapshot` enables crossing operators.
 */
export interface PolicyContext {
  as_of: string | null;
  mode: PolicyMode;                              // DECISION | FREEZE | REVALIDATE | BACKFILL
  state: Record<string, unknown>;
  previous_snapshot: Record<string, unknown> | null;
  extras: Record<string, unknown>;
}

// -----------------------------------------------------------------------------
// StrategyDecision — the universal output of every engine
// -----------------------------------------------------------------------------

/**
 * Common decision emitted by ALL engines (spec §4 `strategy_decision`),
 * as serialized by `StrategyDecision.to_dict()`. Deterministic:
 * `signal_id` and `decision_fingerprint` derive from the economic
 * content, so same inputs + same policy => identical decision (CI §11).
 */
export interface StrategyDecision {
  signal_id: string;
  sleeve_id: string;
  strategy_version: string;
  engine_ref: EngineRef;
  as_of: string;
  direction: Direction;                          // LONG | SHORT | FLAT
  target_exposure: number;
  reason_codes: string[];
  decision_components: Record<string, unknown>;
  rule_trace: RuleTrace | null;
  feature_snapshot_id: string | null;
  decision_fingerprint: string;                  // "sha256:..."
}
