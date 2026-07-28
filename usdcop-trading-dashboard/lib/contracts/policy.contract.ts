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

export type ComparisonOperator = (typeof COMPARISON_OPERATORS)[number];

export type LogicalOperator = (typeof LOGICAL_OPERATORS)[number];

export type RangeOperator = (typeof RANGE_OPERATORS)[number];

// -----------------------------------------------------------------------------
// DSL AST — operands (mirrors policy_dsl.py operand grammar)
// -----------------------------------------------------------------------------

/**
 * `"feature.<name>"` string form — any OTHER string (raw expressions, code,
 * SQL) is rejected server-side and by `validateOperand` below.
 */
export type FeatureRefString = `feature.${string}`;

/** `{ feature: "<name>" }` mapping form (exactly one key: `feature`). */
export interface FeatureRefObject {
  feature: string;
}

export type FeatureRef = FeatureRefString | FeatureRefObject;

/**
 * An operand is a feature reference or a FINITE numeric literal —
 * booleans, NaN/Infinity and arbitrary strings are rejected on both sides.
 */
export type Operand = FeatureRef | number;

// -----------------------------------------------------------------------------
// DSL AST — condition nodes (mirrors policy_dsl.py::validate_condition)
// -----------------------------------------------------------------------------

/** greater_than | less_than | equal | crosses_above | crosses_below */
export interface ComparisonCondition {
  operator: ComparisonOperator;
  left: Operand;
  right: Operand;
}

/** all | any — non-empty list of sub-conditions (empty list is rejected). */
export interface AllAnyCondition {
  operator: 'all' | 'any';
  conditions: ConditionNode[];
}

/** not — negates a single sub-condition. */
export interface NotCondition {
  operator: 'not';
  condition: ConditionNode;
}

/** between — lower <= value <= upper (inclusive on both ends). */
export interface BetweenCondition {
  operator: 'between';
  value: Operand;
  lower: Operand;
  upper: Operand;
}

/**
 * The COMPLETE condition AST: any node outside these four shapes (unknown
 * operator, raw string condition, eval/SQL/inline code) does not typecheck
 * here and raises ValueError in `policy_dsl.py::validate_condition`.
 */
export type ConditionNode =
  | ComparisonCondition
  | AllAnyCondition
  | NotCondition
  | BetweenCondition;

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

/**
 * Exact whitelist of supported trace schemas (mirrors
 * rule_trace.py::SUPPORTED_TRACE_SCHEMAS — fail-closed on both sides).
 */
export const SUPPORTED_TRACE_SCHEMAS = [RULE_TRACE_SCHEMA_V1] as const;

export type TraceSchema = (typeof SUPPORTED_TRACE_SCHEMAS)[number];

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

// -----------------------------------------------------------------------------
// Policy — mirror of the Python `Policy` Protocol (policy.py §4)
// -----------------------------------------------------------------------------

/** Explicit feature snapshot handed to a policy (never "the latest data"). */
export type FeatureSnapshot = Record<string, unknown>;

/**
 * Common interface every policy implements — coded_policy (class in Git) or
 * declarative (DSL spec compiled by `policy_dsl.py`). Evaluation is
 * BACKEND-ONLY (invariant 7: the frontend renders traces, never re-evaluates);
 * this mirror exists so tooling shares the exact protocol shape.
 */
export interface Policy {
  /** Feature names the snapshot must contain. */
  required_features(): string[];
  /** Human-readable errors; empty array = valid. */
  validate_inputs(snapshot: FeatureSnapshot): string[];
  /** Evaluate the policy on an explicit feature snapshot. */
  evaluate(snapshot: FeatureSnapshot, context: PolicyContext): StrategyDecision;
}

// -----------------------------------------------------------------------------
// Runtime validation (same pattern as forecast-output.contract.ts — mirrors
// the Python fail-closed constructors: unknown mode/schema/operator or a
// non-finite exposure is REJECTED, never silently accepted)
// -----------------------------------------------------------------------------

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function nonEmptyString(v: unknown): v is string {
  return typeof v === 'string' && v.length > 0;
}

/**
 * Mirrors policy_dsl.py operand rules: `"feature.<name>"` string,
 * `{ feature: "<name>" }` mapping (single key), or a FINITE numeric literal.
 * Arbitrary strings (code/SQL), booleans and NaN/Infinity are violations.
 */
export function validateOperand(raw: unknown): string[] {
  if (typeof raw === 'string') {
    return raw.startsWith('feature.')
      ? []
      : [`invalid operand ${JSON.stringify(raw)}: strings must be 'feature.<name>' references`];
  }
  if (typeof raw === 'number') {
    return Number.isFinite(raw)
      ? []
      : [`invalid operand ${String(raw)}: NaN/Infinity forbidden`];
  }
  if (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) {
    const keys = Object.keys(raw);
    if (keys.length === 1 && keys[0] === 'feature'
        && nonEmptyString((raw as Record<string, unknown>).feature)) {
      return [];
    }
    return [`invalid operand mapping ${JSON.stringify(raw)}`];
  }
  return [`invalid operand ${JSON.stringify(raw)}: feature reference or finite numeric literal only`];
}

/**
 * Mirrors policy_dsl.py::validate_condition — structural validation of a
 * condition AST. Any operator outside ALLOWED_OPERATORS is a violation.
 */
export function validateConditionNode(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return [`condition must be an object with an 'operator' key — raw string/code conditions are forbidden`];
  }
  const node = raw as Record<string, unknown>;
  const operator = node.operator;
  if (!ALLOWED_OPERATORS.includes(operator as AllowedOperator)) {
    return [`operator ${JSON.stringify(operator)} is not in the whitelist [${ALLOWED_OPERATORS.join(', ')}]`];
  }
  const errors: string[] = [];
  if ((COMPARISON_OPERATORS as readonly string[]).includes(operator as string)) {
    for (const key of ['left', 'right'] as const) {
      if (!(key in node)) errors.push(`operator '${String(operator)}' requires '${key}'`);
      else errors.push(...validateOperand(node[key]));
    }
  } else if (operator === 'not') {
    if (!('condition' in node)) errors.push(`operator 'not' requires 'condition'`);
    else errors.push(...validateConditionNode(node.condition));
  } else if (operator === 'all' || operator === 'any') {
    const conditions = node.conditions;
    if (!Array.isArray(conditions) || conditions.length === 0) {
      errors.push(`operator '${operator}' requires a non-empty 'conditions' list`);
    } else {
      for (const child of conditions) errors.push(...validateConditionNode(child));
    }
  } else if (operator === 'between') {
    for (const key of ['value', 'lower', 'upper'] as const) {
      if (!(key in node)) errors.push(`operator 'between' requires '${key}'`);
      else errors.push(...validateOperand(node[key]));
    }
  }
  return errors;
}

/** Mirrors EngineRef.__post_init__ — discriminated invariants per type. */
export function validateEngineRef(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null) return ['engine_ref must be an object'];
  const o = raw as Record<string, unknown>;
  if (!ENGINE_TYPES.includes(o.type as EngineType)) {
    return [`engine_ref.type must be one of [${ENGINE_TYPES.join(', ')}], got ${JSON.stringify(o.type)}`];
  }
  const errors: string[] = [];
  if (o.type === 'rule_based') {
    if (!nonEmptyString(o.policy_hash)) errors.push('rule_based engine_ref requires policy_hash');
    if (o.model_snapshot_id || (Array.isArray(o.model_snapshot_ids) && o.model_snapshot_ids.length > 0)) {
      errors.push('rule_based engine_ref must NOT carry model snapshots');
    }
  } else if (o.type === 'ml') {
    if (!nonEmptyString(o.model_snapshot_id)) errors.push('ml engine_ref requires model_snapshot_id');
  } else if (o.type === 'composite') {
    if (!nonEmptyString(o.policy_hash)) errors.push('composite engine_ref requires policy_hash');
  }
  return errors;
}

/** Mirrors PolicyContext.__post_init__ — mode must be in POLICY_MODES. */
export function validatePolicyContext(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null) return ['policy_context must be an object'];
  const o = raw as Record<string, unknown>;
  if (!POLICY_MODES.includes(o.mode as PolicyMode)) {
    return [`context.mode must be one of [${POLICY_MODES.join(', ')}], got ${JSON.stringify(o.mode)}`];
  }
  return [];
}

/** Mirrors RuleTrace.__post_init__ — trace_schema must be a SUPPORTED schema. */
export function validateRuleTrace(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null) return ['rule_trace must be an object'];
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  if (!SUPPORTED_TRACE_SCHEMAS.includes(o.trace_schema as TraceSchema)) {
    errors.push(`unsupported trace_schema: ${JSON.stringify(o.trace_schema)} (supported: [${SUPPORTED_TRACE_SCHEMAS.join(', ')}])`);
  }
  if (!Array.isArray(o.rules)) {
    errors.push('rule_trace.rules must be an array');
  } else {
    for (const entry of o.rules) {
      if (typeof entry !== 'object' || entry === null
          || !nonEmptyString((entry as Record<string, unknown>).rule_id)) {
        errors.push('rule_trace entry requires a non-empty rule_id');
      }
    }
  }
  return errors;
}

/**
 * Mirrors StrategyDecision.__post_init__ (fail-closed): direction in
 * VALID_DIRECTIONS, target_exposure FINITE (no NaN/Infinity — strategy-contract
 * invariant 2), engine_ref invariants, rule_trace schema pinned.
 */
export function validateStrategyDecision(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null) return ['strategy_decision must be an object'];
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  for (const name of ['signal_id', 'sleeve_id', 'strategy_version', 'as_of', 'decision_fingerprint']) {
    if (typeof o[name] !== 'string') errors.push(`${name} must be a string`);
  }
  if (!VALID_DIRECTIONS.includes(o.direction as Direction)) {
    errors.push(`direction must be one of [${VALID_DIRECTIONS.join(', ')}], got ${JSON.stringify(o.direction)}`);
  }
  if (!finite(o.target_exposure)) {
    errors.push('target_exposure must be a finite number (NaN/Infinity forbidden)');
  }
  errors.push(...validateEngineRef(o.engine_ref));
  if (o.rule_trace !== null && o.rule_trace !== undefined) {
    errors.push(...validateRuleTrace(o.rule_trace));
  }
  return errors;
}

/** Type guard for untrusted JSON (files, API responses). */
export function isStrategyDecision(raw: unknown): raw is StrategyDecision {
  return validateStrategyDecision(raw).length === 0;
}
