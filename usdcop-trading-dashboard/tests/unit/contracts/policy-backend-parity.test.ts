/**
 * Policy BACKEND contract parity — SHARED FIXTURE (BL-46 R4/R5).
 *
 * Vitest twin of tests/unit/test_policy_backend_contract.py. Both runners LOAD
 * the same versioned case table (tests/fixtures/policy_backend_cases.v1.json)
 * — nothing is duplicated literally — recompute its content SHA-256 before
 * running (drift ⇒ RED) and execute every case: Python against the real
 * constructors, this file against the TS validators. Same verdict, case by case.
 */

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import { validateEngineRef, validateRuleTrace } from '@/lib/contracts/policy.contract';
import {
  validateConfigFieldSpec,
  validateConfigValues,
  validatePolicyVersion,
  validatePresentationSpec,
  validateStrategySignal,
  instantEpochSeconds,
  type ConfigFieldSpec,
} from '@/lib/contracts/policy-version.contract';

const FIXTURE_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../../tests/fixtures/policy_backend_cases.v1.json',
);

type Verdict = 'valid' | 'invalid';
type Target =
  | 'policy_version' | 'strategy_signal' | 'trace' | 'engine_ref'
  | 'presentation' | 'config_field' | 'config_values';

interface ParityCase { id: string; target: Target; payload: unknown; expect: Verdict; note?: string }

/** Marker for Python-only types (Decimal, ...): a non-plain object the closed
 *  JSON check must reject, mirroring the Python raise. */
class PyOnlyValue {
  constructor(public readonly kind: string, public readonly value: unknown) {}
}

function decodeFixtureValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(decodeFixtureValue);
  if (value !== null && typeof value === 'object') {
    const o = value as Record<string, unknown>;
    const keys = Object.keys(o);
    if (keys.length === 1 && keys[0] === '$nonfinite') {
      const kind = o.$nonfinite;
      if (kind === 'NaN') return NaN;
      if (kind === 'Infinity') return Infinity;
      if (kind === '-Infinity') return -Infinity;
      throw new Error(`unknown $nonfinite ${String(kind)}`);
    }
    if (keys.length === 2 && '$pytype' in o && 'value' in o) {
      return new PyOnlyValue(String(o.$pytype), o.value);
    }
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, decodeFixtureValue(v)]));
  }
  return value;
}

function loadFixture(): { doc: Record<string, unknown>; cases: ParityCase[] } {
  const raw = readFileSync(FIXTURE_PATH, 'utf-8').replace(/\r\n/g, '\n');
  const m = raw.match(/"content_sha256":\s*"([0-9a-f]{64})"/);
  if (!m) throw new Error('fixture must declare a 64-hex content_sha256');
  const declared = m[1];
  const actual = createHash('sha256').update(raw.replace(declared, ''), 'utf-8').digest('hex');
  if (actual !== declared) {
    throw new Error(
      `FIXTURE DRIFT: declared ${declared} != recomputed ${actual} — both runners refuse to run`,
    );
  }
  const doc = JSON.parse(raw) as Record<string, unknown>;
  if (doc.fixture !== 'policy_backend_cases' || doc.version !== 'v1') {
    throw new Error('unexpected fixture identity/version');
  }
  const cases = (doc.cases as ParityCase[]).map((c) => ({
    ...c,
    payload: decodeFixtureValue(c.payload),
  }));
  if (new Set(cases.map((c) => c.id)).size !== cases.length) {
    throw new Error('duplicate case ids in fixture');
  }
  return { doc, cases };
}

const { cases: CASES } = loadFixture();

function runCase(target: Target, payload: unknown): Verdict {
  let errors: string[];
  switch (target) {
    case 'policy_version': errors = validatePolicyVersion(payload); break;
    case 'strategy_signal': errors = validateStrategySignal(payload); break;
    case 'trace': errors = validateRuleTrace(payload); break;
    case 'engine_ref': errors = validateEngineRef(payload); break;
    case 'presentation': errors = validatePresentationSpec(payload); break;
    case 'config_field': errors = validateConfigFieldSpec(payload); break;
    case 'config_values': {
      const p = payload as { fields: unknown; values: unknown };
      const fieldErrors = (p.fields as unknown[]).flatMap((f) => validateConfigFieldSpec(f));
      errors = fieldErrors.length
        ? fieldErrors
        : validateConfigValues(p.fields as ConfigFieldSpec[], p.values);
      break;
    }
  }
  return errors.length === 0 ? 'valid' : 'invalid';
}

describe('policy backend contract parity — shared fixture (LOADED + EXECUTED)', () => {
  it.each(CASES.map((c) => [c.id, c] as const))('case %s', (_id, c) => {
    expect(runCase(c.target, c.payload)).toBe(c.expect);
  });

  it('fixture SHA is verified and ids are unique', () => {
    const { cases } = loadFixture();
    expect(cases.length).toBeGreaterThanOrEqual(90);
    expect(new Set(cases.map((c) => c.id)).size).toBe(cases.length);
  });

  it('instant arithmetic resolves offsets (never string comparison)', () => {
    expect(instantEpochSeconds('2026-07-28T10:00:00-05:00'))
      .toBe(instantEpochSeconds('2026-07-28T15:00:00Z'));
    expect(instantEpochSeconds('2028-03-01T00:00:00Z') - instantEpochSeconds('2028-02-28T00:00:00Z'))
      .toBe(2 * 86400);
  });
});
