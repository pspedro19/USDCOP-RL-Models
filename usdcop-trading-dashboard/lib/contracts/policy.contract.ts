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

/**
 * The FOUR engines of invariant 1 (`.claude/rules/strategy-engines.md`):
 * rule_based | ml | rl | composite. `rl` was missing in the R1 cut although
 * the rule and spec §2/§12 (PPO USD/COP) list it and BL-46 R5 requires an
 * RLPolicyPanel — an engine the contract rejects can never reach the renderer.
 * Added bilaterally (mirror: src/contracts/policy.py::ENGINE_TYPES).
 */
export const ENGINE_TYPES = ['rule_based', 'ml', 'rl', 'composite'] as const;

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

/**
 * rl: model_snapshot_id required (the learned policy artifact HAS trained
 * weights, exactly like ml); policy_hash optional. Mirrors the Python branch
 * `elif self.type in ("ml", "rl")`.
 */
export interface RlEngineRef {
  type: 'rl';
  policy_version_id?: string | null;
  policy_hash?: string | null;
  model_snapshot_id: string;        // required
  model_snapshot_ids?: string[];
}

export type EngineRef =
  | RuleBasedEngineRef
  | MlEngineRef
  | RlEngineRef
  | CompositeEngineRef;

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

/**
 * One evaluated condition inside a policy run.
 *
 * `threshold` (BL-46 R5, additive/optional) carries the RIGHT-hand values the
 * condition was compared against — the "Umbral" column of §9. Without it the
 * UI would have to infer which observed value is the threshold, i.e. re-derive
 * the rule (invariant 7). Absent/empty ⇒ render "—", never a guess.
 */
export interface RuleTraceEntry {
  rule_id: string;
  label: string;
  observed: Record<string, unknown>;
  result: boolean;
  reason_code: string;
  threshold?: Record<string, unknown>;
}

/**
 * Full trace of one policy evaluation. The backend produces it; the
 * frontend only renders it (never re-evaluates conditions).
 *
 * `winning_rule_id` / `fallback_applied` (BL-46 R5, additive/optional) are the
 * RESOLUTION facts: picking "the first entry with result=true" in React would
 * re-implement the policy's resolution mode.
 */
export interface RuleTrace {
  trace_schema: typeof RULE_TRACE_SCHEMA_V1;   // literal 'rule_trace_v1'
  rules: RuleTraceEntry[];
  winning_rule_id?: string | null;
  fallback_applied?: boolean;
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

// Strict form patterns (mirror policy.py HASH_PATTERN / ID_PATTERN /
// ISO_TIMESTAMP_PATTERN — C-004 remedy-3 finding 4: policy_hash=true,
// sleeve_id=null, as_of=null are typed violations, never truthiness passes.
// Remedy-4 divergence 2: the ISO check validates the REAL calendar/clock —
// 2026-02-30, 25:00 and +25:00 are impossible values, never accepted;
// JS Date.parse is NOT used because it silently rolls Feb-30 over to Mar-2).

/**
 * `sha256:<64 lowercase hex>` — the length of the algorithm, not a range.
 *
 * This used to be `{8,64}`, which accepted 57 different lengths and made a
 * divergence between hash idioms INVISIBLE: a digest produced by a different
 * canonicalisation, or the truncated hex16 fragment embedded in `signal_id`,
 * validated as if it were the canonical fingerprint
 * (INTEGRATION-CONTRACT.md F-02). Mirrors policy.py HASH_PATTERN verbatim.
 */
export const HASH_PATTERN = /^sha256:[0-9a-f]{64}$/;

/** Identifier form for sleeve_id / snapshot ids / versions. */
export const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]*$/;

/**
 * ISO-8601 date or datetime FORM (the only accepted `as_of` shape).
 * Groups: 1=year 2=month 3=day 4=hour 5=minute 6=second 7=offset.
 * Necessary but NOT sufficient — `validIsoTimestamp` also validates the
 * real calendar/clock/offset ranges (mirrors policy.py require_iso_timestamp).
 */
export const ISO_TIMESTAMP_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})?)?$/;

export function validHash(v: unknown): v is string {
  return typeof v === 'string' && HASH_PATTERN.test(v);
}

export function validId(v: unknown): v is string {
  return typeof v === 'string' && ID_PATTERN.test(v);
}

function daysInMonth(year: number, month: number): number {
  if (month === 2) {
    const leap = (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
    return leap ? 29 : 28;
  }
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}

export function validIsoTimestamp(v: unknown): v is string {
  if (typeof v !== 'string') return false;
  const m = ISO_TIMESTAMP_PATTERN.exec(v);
  if (!m) return false;
  // Real calendar (incl. leap years) — mirrors Python datetime.date()
  const year = Number(m[1]);
  const month = Number(m[2]);
  const day = Number(m[3]);
  if (month < 1 || month > 12 || day < 1 || day > daysInMonth(year, month)) return false;
  // Real clock: hour 25 / minute 61 / second 61 are impossible
  if (m[4] !== undefined) {
    const hour = Number(m[4]);
    const minute = Number(m[5]);
    const second = m[6] === undefined ? 0 : Number(m[6]);
    if (hour > 23 || minute > 59 || second > 59) return false;
  }
  // Real UTC offset: +25:00 / +05:61 are impossible
  const offset = m[7];
  if (offset !== undefined && offset !== 'Z') {
    const offsetHour = Number(offset.slice(1, 3));
    const offsetMinute = Number(offset.slice(4, 6));
    if (offsetHour > 23 || offsetMinute > 59) return false;
  }
  return true;
}

/**
 * Recursive CLOSED-WORLD JSON sweep (mirrors rule_trace.py::ensure_json_safe,
 * C-004 remedy-4 divergence 4). The allowed types are EXACTLY: plain object /
 * array / string / finite number / boolean / null. Anything else — class
 * instances (Date, Set, Map, decoded Python-only markers), undefined, bigint,
 * function, symbol — is a violation, INCLUDING non-finite numbers whatever
 * their origin. Repo rule: JSON never carries Infinity/NaN, and non-JSON
 * types are never silently stringified.
 */
export function collectNonFinite(value: unknown, path: string, errors: string[]): void {
  if (value === null) return;
  const t = typeof value;
  if (t === 'number') {
    if (!Number.isFinite(value)) {
      errors.push(`${path} contains a non-finite number — NaN/Infinity forbidden in JSON`);
    }
    return;
  }
  if (t === 'string' || t === 'boolean') return;
  if (Array.isArray(value)) {
    value.forEach((item, i) => collectNonFinite(item, `${path}[${i}]`, errors));
    return;
  }
  if (t === 'object') {
    const proto = Object.getPrototypeOf(value);
    if (proto !== Object.prototype && proto !== null) {
      errors.push(
        `${path} has a non-JSON type (${value!.constructor?.name ?? 'unknown'}) — ` +
        'allowed types are exactly plain-object/array/string/finite-number/boolean/null',
      );
      return;
    }
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      collectNonFinite(item, `${path}.${key}`, errors);
    }
    return;
  }
  errors.push(
    `${path} has a non-JSON type (${t}) — ` +
    'allowed types are exactly plain-object/array/string/finite-number/boolean/null',
  );
}

/**
 * Mirrors DeclarativePolicy.validate_inputs strictness: every snapshot value
 * must be a REAL finite number — null/bool/string/NaN/Infinity are typed
 * violations (C-004 remedy-3 finding 3).
 */
export function validateFeatureSnapshot(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['feature_snapshot must be an object'];
  }
  const errors: string[] = [];
  for (const [name, value] of Object.entries(raw)) {
    if (typeof value !== 'number') {
      errors.push(`feature ${name} must be a real number (bool/string coercion forbidden)`);
    } else if (!Number.isFinite(value)) {
      errors.push(`feature ${name} is not finite — NaN/Infinity forbidden`);
    }
  }
  return errors;
}

/**
 * Mirrors policy_dsl.py operand rules: `"feature.<name>"` string,
 * `{ feature: "<name>" }` mapping (single key), or a FINITE numeric literal.
 * Arbitrary strings (code/SQL), booleans and NaN/Infinity are violations.
 */
export function validateOperand(raw: unknown): string[] {
  if (typeof raw === 'string') {
    if (!raw.startsWith('feature.')) {
      return [`invalid operand ${JSON.stringify(raw)}: strings must be 'feature.<name>' references`];
    }
    // "feature." with an empty name is a violation on BOTH sides
    // (C-004 remedy-3 finding 2).
    return raw.length > 'feature.'.length
      ? []
      : [`invalid operand 'feature.': the feature name must be non-empty`];
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
  // Optional fields, when present, must be WELL-FORMED strings — a bool/true
  // policy_hash is a typed violation (C-004 remedy-3 finding 4).
  if (o.policy_hash !== undefined && o.policy_hash !== null && !validHash(o.policy_hash)) {
    errors.push(`policy_hash must be a 'sha256:<hex>' string, got ${JSON.stringify(o.policy_hash)}`);
  }
  if (o.policy_version_id !== undefined && o.policy_version_id !== null && !validId(o.policy_version_id)) {
    errors.push('policy_version_id must be a non-empty identifier string');
  }
  if (o.model_snapshot_id !== undefined && o.model_snapshot_id !== null && !validId(o.model_snapshot_id)) {
    errors.push('model_snapshot_id must be a non-empty identifier string');
  }
  if (o.model_snapshot_ids !== undefined && o.model_snapshot_ids !== null) {
    if (!Array.isArray(o.model_snapshot_ids)) {
      errors.push('model_snapshot_ids must be an array of identifier strings');
    } else {
      for (const sid of o.model_snapshot_ids) {
        if (!validId(sid)) errors.push('model_snapshot_ids entries must be non-empty identifier strings');
      }
    }
  }
  if (o.type === 'rule_based') {
    if (!validHash(o.policy_hash)) errors.push('rule_based engine_ref requires policy_hash');
    if (o.model_snapshot_id || (Array.isArray(o.model_snapshot_ids) && o.model_snapshot_ids.length > 0)) {
      errors.push('rule_based engine_ref must NOT carry model snapshots');
    }
  } else if (o.type === 'ml' || o.type === 'rl') {
    if (!validId(o.model_snapshot_id)) errors.push(`${o.type} engine_ref requires model_snapshot_id`);
  } else if (o.type === 'composite') {
    if (!validHash(o.policy_hash)) errors.push('composite engine_ref requires policy_hash');
  }
  return errors;
}

/**
 * Mirrors PolicyContext.__post_init__ — mode must be in POLICY_MODES, and
 * `as_of`, when present and non-null, must be a REAL ISO-8601 timestamp
 * (C-004 remedy-4 divergence 2a: the TS side must validate context.as_of
 * exactly like Python does, impossible calendar/clock/offset rejected).
 */
export function validatePolicyContext(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null) return ['policy_context must be an object'];
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  if (!POLICY_MODES.includes(o.mode as PolicyMode)) {
    errors.push(`context.mode must be one of [${POLICY_MODES.join(', ')}], got ${JSON.stringify(o.mode)}`);
  }
  if (o.as_of !== undefined && o.as_of !== null && !validIsoTimestamp(o.as_of)) {
    errors.push(`context.as_of must be a real ISO-8601 date/datetime, got ${JSON.stringify(o.as_of)}`);
  }
  return errors;
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
        continue;
      }
      const e = entry as Record<string, unknown>;
      if (e.result !== undefined && typeof e.result !== 'boolean') {
        errors.push(`rule_trace entry ${String(e.rule_id)} result must be a boolean`);
      }
      // observed/threshold values ride into JSON exports — closed JSON only
      // (no NaN/Infinity, no non-JSON types) AND they must be OBJECTS: a bare
      // scalar would blow up `dict(...)` in Python, so both sides reject it.
      for (const key of ['observed', 'threshold'] as const) {
        const v = e[key];
        if (v === undefined) continue;
        if (typeof v !== 'object' || v === null || Array.isArray(v)) {
          errors.push(`rule_trace entry ${String(e.rule_id)} ${key} must be an object`);
          continue;
        }
        collectNonFinite(v, `rule_trace.${String(e.rule_id)}.${key}`, errors);
      }
    }
  }
  // Resolution facts (mirror RuleTrace.__post_init__): a winner must be a
  // traced rule that actually fired, and it excludes the fallback.
  if (o.fallback_applied !== undefined && typeof o.fallback_applied !== 'boolean') {
    errors.push('rule_trace fallback_applied must be a boolean');
  }
  if (o.winning_rule_id !== undefined && o.winning_rule_id !== null) {
    if (!nonEmptyString(o.winning_rule_id)) {
      errors.push('rule_trace winning_rule_id must be a non-empty string or null');
    } else if (Array.isArray(o.rules)) {
      const match = (o.rules as unknown[]).find(
        (r) => typeof r === 'object' && r !== null
          && (r as Record<string, unknown>).rule_id === o.winning_rule_id,
      ) as Record<string, unknown> | undefined;
      if (!match) {
        errors.push(
          `rule_trace winning_rule_id ${JSON.stringify(o.winning_rule_id)} is not among the traced rules`,
        );
      } else if (match.result !== true) {
        errors.push(
          `rule_trace winning_rule_id ${JSON.stringify(o.winning_rule_id)} points at a rule whose result is False`,
        );
      }
      if (o.fallback_applied === true) {
        errors.push('rule_trace cannot declare BOTH a winning_rule_id and fallback_applied=true');
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
  // Strict identifier/timestamp/hash forms (C-004 remedy-3 finding 4):
  // sleeve_id=null, as_of=null, fingerprint=true are typed violations.
  for (const name of ['sleeve_id', 'strategy_version'] as const) {
    if (!validId(o[name])) {
      errors.push(`${name} must be a non-empty identifier string, got ${JSON.stringify(o[name])}`);
    }
  }
  if (!validIsoTimestamp(o.as_of)) {
    errors.push(`as_of must be an ISO-8601 date/datetime string, got ${JSON.stringify(o.as_of)}`);
  }
  if (!validHash(o.decision_fingerprint)) {
    errors.push(`decision_fingerprint must be a 'sha256:<hex>' string, got ${JSON.stringify(o.decision_fingerprint)}`);
  }
  // Derived id (C-004 remedy-4 divergence 2d, mirrors StrategyDecision
  // __post_init__): signal_id is the composite
  // '<sleeve_id>:<as_of>:<fingerprint-hex16>' — every part is validated
  // above, and the composite must EQUAL its derivation, so an embedded
  // timestamp with an impossible offset, a foreign sleeve or a foreign
  // fingerprint prefix is rejected.
  if (typeof o.signal_id !== 'string' || o.signal_id.length === 0) {
    errors.push(`signal_id must be a non-empty string, got ${JSON.stringify(o.signal_id)}`);
  } else if (validId(o.sleeve_id) && validIsoTimestamp(o.as_of) && validHash(o.decision_fingerprint)) {
    const hex16 = (o.decision_fingerprint as string).slice('sha256:'.length).slice(0, 16);
    const derived = `${o.sleeve_id}:${o.as_of}:${hex16}`;
    if (o.signal_id !== derived) {
      errors.push(
        `signal_id must equal its derivation '<sleeve_id>:<as_of>:<fingerprint-hex16>' ` +
        `(${derived}), got ${JSON.stringify(o.signal_id)}`,
      );
    }
  }
  if (!VALID_DIRECTIONS.includes(o.direction as Direction)) {
    errors.push(`direction must be one of [${VALID_DIRECTIONS.join(', ')}], got ${JSON.stringify(o.direction)}`);
  }
  if (!finite(o.target_exposure)) {
    errors.push('target_exposure must be a finite number (NaN/Infinity forbidden)');
  }
  if (o.reason_codes !== undefined
      && (!Array.isArray(o.reason_codes) || o.reason_codes.some((c) => typeof c !== 'string'))) {
    errors.push('reason_codes must be an array of strings');
  }
  if (o.feature_snapshot_id !== undefined && o.feature_snapshot_id !== null
      && !validId(o.feature_snapshot_id)) {
    errors.push('feature_snapshot_id must be a non-empty identifier string or null');
  }
  // Closed JSON: no NaN/Infinity and no non-JSON types anywhere in the
  // serialized payload (remedy-3 finding 5 + remedy-4 divergence 4)
  if (o.decision_components !== undefined) {
    collectNonFinite(o.decision_components, 'decision_components', errors);
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
