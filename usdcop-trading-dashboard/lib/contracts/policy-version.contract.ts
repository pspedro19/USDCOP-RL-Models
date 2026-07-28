/**
 * Policy backend contract — control.policy_version + action.strategy_signal
 * =========================================================================
 * TypeScript HALF of CTR-POLICY-BACKEND-001 (BL-46 R4/R5). The Python half is
 * `src/contracts/policy_version.py`; both must ACCEPT and REJECT exactly the
 * same payloads (shared fixture: `tests/fixtures/policy_backend_cases.v1.json`,
 * executed by pytest AND vitest).
 *
 * Also carries the two NON-ECONOMIC schemas the schema-driven UI renders from:
 * `PresentationSpec` (§9.1, own `presentation_hash` — a label never changes
 * `policy_hash`) and `ConfigFieldSpec` (the tuneables of `sb_trading_configs`,
 * rendered from schema instead of fixed fields).
 *
 * Spec: .claude/specs/planes/05-rule-based-strategies.md §8, §9, §9.1
 * Rule: .claude/rules/strategy-engines.md (invariants 1, 5, 7, 9)
 */

import {
  ENGINE_TYPES,
  VALID_DIRECTIONS,
  collectNonFinite,
  validHash,
  validId,
  validIsoTimestamp,
  ISO_TIMESTAMP_PATTERN,
  type Direction,
  type EngineType,
} from './policy.contract';

// -----------------------------------------------------------------------------
// Whitelists (mirror policy_version.py — change BOTH sides)
// -----------------------------------------------------------------------------

export const IMPLEMENTATION_MODES = ['coded_policy', 'declarative'] as const;
export type ImplementationMode = (typeof IMPLEMENTATION_MODES)[number];

export const DECISION_SCHEMA_VERSIONS = ['decision_components_v1'] as const;
export const POLICY_VERSION_SCHEMAS = ['policy_version_v1'] as const;
export const STRATEGY_SIGNAL_SCHEMAS = ['strategy_signal_v1'] as const;

export const PRESENTATION_FORMATS = ['price', 'percent', 'decimal', 'integer', 'text'] as const;
export type PresentationFormat = (typeof PRESENTATION_FORMATS)[number];

export const CONFIG_FIELD_TYPES = ['number', 'boolean', 'enum'] as const;
export type ConfigFieldType = (typeof CONFIG_FIELD_TYPES)[number];

/** Declared input fallbacks (invariant 9 — no default, no freeze). */
export const FALLBACK_MODES = ['FAIL_CLOSED', 'FLAT'] as const;
export type FallbackMode = (typeof FALLBACK_MODES)[number];

export const URI_SCHEMES = ['s3', 'https', 'file'] as const;

/** Absolute URI, whitelisted scheme, no whitespace/control chars. */
export const URI_PATTERN = /^(?:s3|https|file):\/\/[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=%-]+$/;
export const URI_MAX_LENGTH = 512;

/** `package.module:ClassName` — the only accepted coded_policy reference. */
export const MODULE_REF_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$/;

/** snake_case key of a presentation component / config field. */
export const COMPONENT_KEY_PATTERN = /^[a-z][a-z0-9_]*$/;

export const LABEL_MAX_LENGTH = 80;
export const DESCRIPTION_MAX_LENGTH = 400;

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export interface PolicyVersionRecord {
  policy_version_id: string;
  sleeve_id: string;
  strategy_version: string;
  engine_type: EngineType;
  implementation_mode: ImplementationMode;
  module_reference: string | null;
  code_hash: string | null;
  params_hash: string;
  policy_hash: string;
  feature_set_hash: string;
  resample_policy_hash: string;
  frozen_at: string;
  manifest_uri: string;
  schema_version: (typeof POLICY_VERSION_SCHEMAS)[number];
}

export interface StrategySignalRecord {
  signal_id: string;
  sleeve_id: string;
  strategy_version: string;
  policy_version_id: string;
  instrument_id: string;
  as_of: string;
  valid_from: string;
  valid_until: string;
  direction: Direction;
  target_exposure: number;
  feature_snapshot_id: string | null;
  decision_fingerprint: string;
  reason_codes: string[];
  decision_components: Record<string, unknown>;
  decision_schema_version: (typeof DECISION_SCHEMA_VERSIONS)[number];
  rule_trace_uri: string | null;
  created_at: string;
  schema_version: (typeof STRATEGY_SIGNAL_SCHEMAS)[number];
}

export interface PresentationComponent {
  key: string;
  label: string;
  format: PresentationFormat;
}

export interface PresentationSpec {
  engine_label: string;
  description: string;
  components: PresentationComponent[];
}

export interface ConfigFieldSpec {
  key: string;
  label: string;
  type: ConfigFieldType;
  default?: unknown;
  minimum?: number | null;
  maximum?: number | null;
  step?: number | null;
  options?: string[];
  unit?: string | null;
}

export interface PolicyVersionIndex {
  schema: 'policy_version_index_v1';
  policy_engine_version: string;
  generated_at: string;
  versions: PolicyVersionRecord[];
}

// -----------------------------------------------------------------------------
// Scalar validators (each has a Python twin with identical verdicts)
// -----------------------------------------------------------------------------

export function validUri(v: unknown): v is string {
  return typeof v === 'string' && v.length <= URI_MAX_LENGTH && URI_PATTERN.test(v);
}

function validText(v: unknown, maxLength: number): v is string {
  return (
    typeof v === 'string' && v.length > 0 && v.length <= maxLength
    && !v.includes('\n') && !v.includes('\r')
  );
}

function finiteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * A point in TIME: full ISO datetime WITH an explicit offset. Mirrors
 * `require_instant` — a date-only or offset-less value cannot be ordered, so
 * both sides reject it.
 */
export function validInstant(v: unknown): v is string {
  if (!validIsoTimestamp(v)) return false;
  const m = ISO_TIMESTAMP_PATTERN.exec(v as string);
  return !!m && m[4] !== undefined && m[7] !== undefined;
}

/** Howard Hinnant's days-from-civil — identical integer arithmetic in Python. */
function daysFromCivil(year: number, month: number, day: number): number {
  const y = year - (month <= 2 ? 1 : 0);
  const era = Math.floor((y >= 0 ? y : y - 399) / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5) + day - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

/** Absolute seconds since epoch for a validated instant (never `Date.parse`). */
export function instantEpochSeconds(value: string): number {
  const m = ISO_TIMESTAMP_PATTERN.exec(value);
  if (!m || m[4] === undefined || m[7] === undefined) {
    throw new Error(`not a comparable instant: ${value}`);
  }
  const days = daysFromCivil(Number(m[1]), Number(m[2]), Number(m[3]));
  let seconds = days * 86400 + Number(m[4]) * 3600 + Number(m[5]) * 60
    + (m[6] === undefined ? 0 : Number(m[6]));
  const offset = m[7];
  if (offset !== 'Z') {
    const offsetSeconds = Number(offset.slice(1, 3)) * 3600 + Number(offset.slice(4, 6)) * 60;
    seconds += offset[0] === '+' ? -offsetSeconds : offsetSeconds;
  }
  return seconds;
}

// -----------------------------------------------------------------------------
// Record validators
// -----------------------------------------------------------------------------

const POLICY_VERSION_REQUIRED = [
  'policy_version_id', 'sleeve_id', 'strategy_version', 'engine_type',
  'implementation_mode', 'params_hash', 'policy_hash', 'feature_set_hash',
  'resample_policy_hash', 'frozen_at', 'manifest_uri',
] as const;

const POLICY_VERSION_OPTIONAL = ['module_reference', 'code_hash', 'schema_version'] as const;

/** Mirrors PolicyVersionRecord.__post_init__ + from_dict (closed schema). */
export function validatePolicyVersion(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['policy_version must be an object'];
  }
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  const missing = POLICY_VERSION_REQUIRED.filter((k) => !(k in o));
  if (missing.length) errors.push(`policy_version payload missing fields: ${missing.join(', ')}`);
  const known = new Set<string>([...POLICY_VERSION_REQUIRED, ...POLICY_VERSION_OPTIONAL]);
  const unknown = Object.keys(o).filter((k) => !known.has(k));
  if (unknown.length) {
    errors.push(`policy_version payload has unknown fields: ${unknown.sort().join(', ')}`);
  }
  for (const name of ['policy_version_id', 'sleeve_id', 'strategy_version'] as const) {
    if (!validId(o[name])) errors.push(`${name} must be a non-empty identifier string`);
  }
  if (!ENGINE_TYPES.includes(o.engine_type as EngineType)) {
    errors.push(`engine_type must be one of [${ENGINE_TYPES.join(', ')}]`);
  }
  if (!IMPLEMENTATION_MODES.includes(o.implementation_mode as ImplementationMode)) {
    errors.push(`implementation_mode must be one of [${IMPLEMENTATION_MODES.join(', ')}]`);
  }
  if (o.schema_version !== undefined
      && !(POLICY_VERSION_SCHEMAS as readonly string[]).includes(o.schema_version as string)) {
    errors.push(`schema_version must be one of [${POLICY_VERSION_SCHEMAS.join(', ')}]`);
  }
  for (const name of ['params_hash', 'policy_hash', 'feature_set_hash', 'resample_policy_hash'] as const) {
    if (!validHash(o[name])) errors.push(`${name} must be a 'sha256:<hex>' string`);
  }
  if (!validInstant(o.frozen_at)) {
    errors.push('frozen_at must be a full ISO datetime with an explicit UTC offset');
  }
  if (!validUri(o.manifest_uri)) {
    errors.push(`manifest_uri must be an absolute URI (schemes ${URI_SCHEMES.join('|')})`);
  }
  if (o.implementation_mode === 'coded_policy') {
    if (o.module_reference === undefined || o.module_reference === null) {
      errors.push("coded_policy requires module_reference ('package.module:ClassName')");
    } else if (typeof o.module_reference !== 'string' || !MODULE_REF_PATTERN.test(o.module_reference)) {
      errors.push('module_reference must match package.module:ClassName');
    }
    if (!validHash(o.code_hash)) errors.push("code_hash must be a 'sha256:<hex>' string");
  } else if (o.implementation_mode === 'declarative') {
    if (o.module_reference !== undefined && o.module_reference !== null) {
      errors.push('declarative policies must NOT declare a module_reference');
    }
    if (o.code_hash !== undefined && o.code_hash !== null && !validHash(o.code_hash)) {
      errors.push("code_hash must be a 'sha256:<hex>' string");
    }
  }
  return errors;
}

const SIGNAL_REQUIRED = [
  'signal_id', 'sleeve_id', 'strategy_version', 'policy_version_id', 'instrument_id',
  'as_of', 'valid_from', 'valid_until', 'direction', 'target_exposure',
  'decision_fingerprint', 'created_at',
] as const;

const SIGNAL_OPTIONAL = [
  'feature_snapshot_id', 'reason_codes', 'decision_components',
  'decision_schema_version', 'rule_trace_uri', 'schema_version',
] as const;

/** Mirrors StrategySignalRecord.__post_init__ (+ its closed kwargs). */
export function validateStrategySignal(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['strategy_signal must be an object'];
  }
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  // Closed schema — the Python constructor raises TypeError on a missing or
  // unknown kwarg, so a payload that Python cannot build must fail here too.
  const missing = SIGNAL_REQUIRED.filter((k) => !(k in o));
  if (missing.length) errors.push(`strategy_signal payload missing fields: ${missing.join(', ')}`);
  const knownKeys = new Set<string>([...SIGNAL_REQUIRED, ...SIGNAL_OPTIONAL]);
  const unknownKeys = Object.keys(o).filter((k) => !knownKeys.has(k));
  if (unknownKeys.length) {
    errors.push(`strategy_signal payload has unknown fields: ${unknownKeys.sort().join(', ')}`);
  }
  for (const name of ['sleeve_id', 'strategy_version', 'policy_version_id', 'instrument_id'] as const) {
    if (!validId(o[name])) errors.push(`${name} must be a non-empty identifier string`);
  }
  if (!validIsoTimestamp(o.as_of)) errors.push('as_of must be an ISO-8601 date/datetime string');
  for (const name of ['valid_from', 'valid_until', 'created_at'] as const) {
    if (!validInstant(o[name])) {
      errors.push(`${name} must be a full ISO datetime with an explicit UTC offset`);
    }
  }
  if (validInstant(o.valid_from) && validInstant(o.valid_until)
      && instantEpochSeconds(o.valid_from) > instantEpochSeconds(o.valid_until)) {
    errors.push('valid_from must not be after valid_until (inverted validity window)');
  }
  if (!VALID_DIRECTIONS.includes(o.direction as Direction)) {
    errors.push(`direction must be one of [${VALID_DIRECTIONS.join(', ')}]`);
  }
  if (!finiteNumber(o.target_exposure)) {
    errors.push('target_exposure must be a finite number (NaN/Infinity forbidden)');
  }
  if (!validHash(o.decision_fingerprint)) {
    errors.push("decision_fingerprint must be a 'sha256:<hex>' string");
  }
  if (validId(o.sleeve_id) && validIsoTimestamp(o.as_of) && validHash(o.decision_fingerprint)) {
    const hex16 = (o.decision_fingerprint as string).slice('sha256:'.length).slice(0, 16);
    const derived = `${o.sleeve_id}:${o.as_of}:${hex16}`;
    if (o.signal_id !== derived) {
      errors.push(`signal_id must equal its derivation (${derived})`);
    }
  } else if (typeof o.signal_id !== 'string' || o.signal_id.length === 0) {
    errors.push('signal_id must be a non-empty string');
  }
  if (o.feature_snapshot_id !== undefined && o.feature_snapshot_id !== null
      && !validId(o.feature_snapshot_id)) {
    errors.push('feature_snapshot_id must be a non-empty identifier string or null');
  }
  if (o.reason_codes !== undefined
      && (!Array.isArray(o.reason_codes) || o.reason_codes.some((c) => typeof c !== 'string'))) {
    errors.push('reason_codes must be an array of strings');
  }
  if (o.decision_components !== undefined) {
    if (typeof o.decision_components !== 'object' || o.decision_components === null
        || Array.isArray(o.decision_components)) {
      errors.push('decision_components must be an object');
    } else {
      collectNonFinite(o.decision_components, 'decision_components', errors);
    }
  }
  if (o.decision_schema_version !== undefined
      && !(DECISION_SCHEMA_VERSIONS as readonly string[]).includes(o.decision_schema_version as string)) {
    errors.push(`decision_schema_version must be one of [${DECISION_SCHEMA_VERSIONS.join(', ')}]`);
  }
  if (o.schema_version !== undefined
      && !(STRATEGY_SIGNAL_SCHEMAS as readonly string[]).includes(o.schema_version as string)) {
    errors.push(`schema_version must be one of [${STRATEGY_SIGNAL_SCHEMAS.join(', ')}]`);
  }
  if (o.rule_trace_uri !== undefined && o.rule_trace_uri !== null && !validUri(o.rule_trace_uri)) {
    errors.push(`rule_trace_uri must be an absolute URI (schemes ${URI_SCHEMES.join('|')})`);
  }
  return errors;
}

/** Mirrors PresentationSpec/PresentationComponent.__post_init__. */
export function validatePresentationSpec(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['presentation must be an object'];
  }
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  if (!validText(o.engine_label, LABEL_MAX_LENGTH)) {
    errors.push(`presentation.engine_label must be a non-empty single-line string <= ${LABEL_MAX_LENGTH} chars`);
  }
  if (!validText(o.description, DESCRIPTION_MAX_LENGTH)) {
    errors.push(`presentation.description must be a non-empty single-line string <= ${DESCRIPTION_MAX_LENGTH} chars`);
  }
  if (o.components !== undefined) {
    if (!Array.isArray(o.components)) {
      errors.push('presentation.components must be an array');
    } else {
      const keys: string[] = [];
      for (const c of o.components) {
        if (typeof c !== 'object' || c === null || Array.isArray(c)) {
          errors.push('presentation component must be an object');
          continue;
        }
        const comp = c as Record<string, unknown>;
        if (typeof comp.key !== 'string' || !COMPONENT_KEY_PATTERN.test(comp.key)) {
          errors.push('presentation component key must match ^[a-z][a-z0-9_]*$');
        } else {
          keys.push(comp.key);
        }
        if (!validText(comp.label, LABEL_MAX_LENGTH)) {
          errors.push('presentation component label must be a non-empty single-line string');
        }
        if (!(PRESENTATION_FORMATS as readonly string[]).includes(comp.format as string)) {
          errors.push(`presentation component format must be one of [${PRESENTATION_FORMATS.join(', ')}]`);
        }
      }
      if (new Set(keys).size !== keys.length) {
        errors.push('presentation.components keys must be unique');
      }
    }
  }
  return errors;
}

/** Mirrors ConfigFieldSpec.__post_init__ (schema-driven config form). */
export function validateConfigFieldSpec(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['config field must be an object'];
  }
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  // Closed schema — `ConfigFieldSpec(**payload)` raises TypeError on an
  // unknown kwarg in Python, so an unknown key must fail here too.
  const allowed = new Set([
    'key', 'label', 'type', 'default', 'minimum', 'maximum', 'step', 'options', 'unit',
  ]);
  const unknownKeys = Object.keys(o).filter((k) => !allowed.has(k));
  if (unknownKeys.length) {
    errors.push(`config field has unknown keys: ${unknownKeys.sort().join(', ')}`);
  }
  if (typeof o.key !== 'string' || !COMPONENT_KEY_PATTERN.test(o.key)) {
    errors.push('config field key must match ^[a-z][a-z0-9_]*$');
  }
  if (!validText(o.label, LABEL_MAX_LENGTH)) {
    errors.push('config field label must be a non-empty single-line string');
  }
  if (!(CONFIG_FIELD_TYPES as readonly string[]).includes(o.type as string)) {
    errors.push(`config field type must be one of [${CONFIG_FIELD_TYPES.join(', ')}]`);
    return errors;
  }
  const hasOptions = Array.isArray(o.options) && o.options.length > 0;
  const bounds = ['minimum', 'maximum', 'step'] as const;
  if (o.type === 'number') {
    for (const b of bounds) {
      if (o[b] !== undefined && o[b] !== null && !finiteNumber(o[b])) {
        errors.push(`config field ${b} must be a finite number`);
      }
    }
    if (finiteNumber(o.minimum) && finiteNumber(o.maximum) && o.minimum > o.maximum) {
      errors.push('config field minimum > maximum');
    }
    if (finiteNumber(o.step) && o.step <= 0) errors.push('config field step must be > 0');
    if (hasOptions) errors.push("options are only valid for type 'enum'");
  } else if (o.type === 'boolean') {
    if (bounds.some((b) => o[b] !== undefined && o[b] !== null) || hasOptions) {
      errors.push('boolean fields take no minimum/maximum/step/options');
    }
  } else {
    if (!Array.isArray(o.options) || o.options.length === 0) {
      errors.push('enum requires a non-empty options list');
    } else {
      if (!o.options.every((x) => typeof x === 'string' && x.length > 0)) {
        errors.push('enum options must be non-empty strings');
      }
      if (new Set(o.options as string[]).size !== (o.options as string[]).length) {
        errors.push('enum options must be unique');
      }
    }
    if (bounds.some((b) => o[b] !== undefined && o[b] !== null)) {
      errors.push('enum fields take no minimum/maximum/step');
    }
  }
  if (o.unit !== undefined && o.unit !== null && !validText(o.unit, 16)) {
    errors.push('config field unit must be a short non-empty string');
  }
  if (o.default !== undefined && o.default !== null) {
    errors.push(...validateConfigValue(o as unknown as ConfigFieldSpec, o.default));
  }
  return errors;
}

/** Mirrors ConfigFieldSpec.validate_value. */
export function validateConfigValue(fieldSpec: ConfigFieldSpec, value: unknown): string[] {
  if (fieldSpec.type === 'number') {
    if (typeof value !== 'number') return [`${fieldSpec.key} must be a number`];
    if (!Number.isFinite(value)) return [`${fieldSpec.key} must be finite (NaN/Infinity forbidden)`];
    const errors: string[] = [];
    if (finiteNumber(fieldSpec.minimum) && value < fieldSpec.minimum) {
      errors.push(`${fieldSpec.key} must be >= ${fieldSpec.minimum}`);
    }
    if (finiteNumber(fieldSpec.maximum) && value > fieldSpec.maximum) {
      errors.push(`${fieldSpec.key} must be <= ${fieldSpec.maximum}`);
    }
    return errors;
  }
  if (fieldSpec.type === 'boolean') {
    return typeof value === 'boolean' ? [] : [`${fieldSpec.key} must be a boolean`];
  }
  if (typeof value !== 'string') return [`${fieldSpec.key} must be a string`];
  return (fieldSpec.options ?? []).includes(value)
    ? []
    : [`${fieldSpec.key} must be one of ${JSON.stringify(fieldSpec.options ?? [])}`];
}

/** Mirrors ConfigSchema.validate_values — unknown keys are ERRORS, never dropped. */
export function validateConfigValues(
  fields: readonly ConfigFieldSpec[],
  values: unknown,
): string[] {
  if (typeof values !== 'object' || values === null || Array.isArray(values)) {
    return ['config values must be an object'];
  }
  const known = new Map(fields.map((f) => [f.key, f]));
  const errors: string[] = [];
  for (const key of Object.keys(values as Record<string, unknown>)) {
    if (!known.has(key)) errors.push(`unknown config key: ${key}`);
  }
  for (const [key, spec] of known) {
    if (key in (values as Record<string, unknown>)) {
      errors.push(...validateConfigValue(spec, (values as Record<string, unknown>)[key]));
    }
  }
  return errors;
}

/** Type guard for the file-based policy version index served by the API. */
export function validatePolicyVersionIndex(raw: unknown): string[] {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return ['policy_version_index must be an object'];
  }
  const o = raw as Record<string, unknown>;
  const errors: string[] = [];
  if (o.schema !== 'policy_version_index_v1') {
    errors.push("policy_version_index.schema must be 'policy_version_index_v1'");
  }
  if (typeof o.policy_engine_version !== 'string' || !o.policy_engine_version) {
    errors.push('policy_version_index.policy_engine_version must be a non-empty string');
  }
  if (!validInstant(o.generated_at)) {
    errors.push('policy_version_index.generated_at must be an instant with UTC offset');
  }
  if (!Array.isArray(o.versions)) {
    errors.push('policy_version_index.versions must be an array');
  } else {
    o.versions.forEach((v, i) => {
      validatePolicyVersion(v).forEach((e) => errors.push(`versions[${i}]: ${e}`));
    });
    const ids = o.versions
      .map((v) => (typeof v === 'object' && v !== null ? (v as Record<string, unknown>).policy_version_id : undefined))
      .filter((x): x is string => typeof x === 'string');
    if (new Set(ids).size !== ids.length) errors.push('policy_version_id must be unique in the index');
  }
  return errors;
}
