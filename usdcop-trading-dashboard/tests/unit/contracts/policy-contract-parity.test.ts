/**
 * Policy contract parity — SHARED FIXTURE (C-004 remedio-4 divergence 1).
 *
 * This is the Vitest twin of tests/unit/test_policy_contract.py::
 * TestSharedCaseTable. Both runners LOAD the same versioned case table
 * (tests/fixtures/policy_contract_cases.v1.json at the repo root) — nothing
 * is duplicated literally. Each runner recomputes the fixture's content
 * SHA-256 before running and goes RED on drift. Python executes the cases
 * against the real constructors/validators; this file executes them against
 * the TS runtime validators — same verdict, case by case.
 *
 * Sentinel encoding (see the fixture's `encoding` block):
 * - {"$nonfinite": "NaN"|"Infinity"|"-Infinity"} -> native non-finite number
 * - {"$pytype": kind, "value": v} -> Python decodes to the REAL type
 *   (numpy scalar, Decimal, datetime, set, bytes); TS decodes non-finite
 *   numeric kinds to native NaN/Infinity and every other kind to a non-plain
 *   marker object — both sides must reject them (closed JSON type set).
 */

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import {
  validateStrategyDecision,
  validateEngineRef,
  validateConditionNode,
  validateRuleTrace,
  validatePolicyContext,
  validateFeatureSnapshot,
} from '@/lib/contracts/policy.contract';

const FIXTURE_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../../tests/fixtures/policy_contract_cases.v1.json',
);

type Verdict = 'valid' | 'invalid';
type Target = 'decision' | 'engine_ref' | 'condition' | 'trace' | 'context' | 'snapshot';

interface ParityCase {
  id: string;
  target: Target;
  payload: unknown;
  expect: Verdict;
  note?: string;
}

interface ParityFixture {
  fixture: string;
  version: string;
  content_sha256: string;
  cases: ParityCase[];
}

/** Marker for Python-only types (Decimal, datetime, set, ...): a non-plain
 *  object the closed JSON check must reject, mirroring the Python raise. */
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
      throw new Error(`unknown $nonfinite ${String(kind)} in fixture`);
    }
    if (keys.length === 2 && '$pytype' in o && 'value' in o) {
      const kind = String(o.$pytype);
      if (kind === 'numpy.float32' || kind === 'numpy.float64') {
        const v = o.value;
        if (v === 'Infinity') return Infinity;
        if (v === '-Infinity') return -Infinity;
        if (v === 'NaN') return NaN;
        return new PyOnlyValue(kind, v); // finite numpy scalar: still non-JSON
      }
      return new PyOnlyValue(kind, o.value);
    }
    return Object.fromEntries(
      Object.entries(o).map(([k, v]) => [k, decodeFixtureValue(v)]),
    );
  }
  return value;
}

/** Load + SHA-verify the shared fixture. Drift => throw (RED). */
function loadParityFixture(): { doc: ParityFixture; cases: ParityCase[] } {
  const raw = readFileSync(FIXTURE_PATH, 'utf-8').replace(/\r\n/g, '\n');
  const m = raw.match(/"content_sha256":\s*"([0-9a-f]{64})"/);
  if (!m) throw new Error('fixture must declare a 64-hex content_sha256');
  const declared = m[1];
  const actual = createHash('sha256')
    .update(raw.replace(declared, ''), 'utf-8')
    .digest('hex');
  if (actual !== declared) {
    throw new Error(
      `FIXTURE DRIFT: declared content_sha256 ${declared} != recomputed ${actual} — ` +
      'the case table changed without regenerating the pin (both runners refuse to run)',
    );
  }
  const doc = JSON.parse(raw) as ParityFixture;
  if (doc.fixture !== 'policy_contract_cases' || doc.version !== 'v1') {
    throw new Error('unexpected fixture identity/version');
  }
  const cases = doc.cases.map((c) => ({ ...c, payload: decodeFixtureValue(c.payload) }));
  const ids = new Set(cases.map((c) => c.id));
  if (ids.size !== cases.length) throw new Error('duplicate case ids in fixture');
  return { doc, cases };
}

const { doc: FIXTURE_DOC, cases: PARITY_CASES } = loadParityFixture();

function runCase(target: Target, payload: unknown): Verdict {
  let errors: string[];
  switch (target) {
    case 'decision': errors = validateStrategyDecision(payload); break;
    case 'engine_ref': errors = validateEngineRef(payload); break;
    case 'condition': errors = validateConditionNode(payload); break;
    case 'trace': errors = validateRuleTrace(payload); break;
    case 'context': errors = validatePolicyContext(payload); break;
    case 'snapshot': errors = validateFeatureSnapshot(payload); break;
  }
  return errors.length === 0 ? 'valid' : 'invalid';
}

describe('policy contract parity — shared fixture (LOADED + EXECUTED, not duplicated)', () => {
  it.each(PARITY_CASES.map((c) => [c.id, c] as const))('case %s', (_id, c) => {
    expect(runCase(c.target, c.payload)).toBe(c.expect);
  });

  it('fixture content SHA is verified and case ids are unique', () => {
    // loadParityFixture() already threw on drift; re-run explicitly so the
    // guarantee is a named green test, not only a collection side effect.
    const { cases } = loadParityFixture();
    expect(cases.length).toBeGreaterThanOrEqual(35);
    expect(new Set(cases.map((c) => c.id)).size).toBe(cases.length);
    expect(FIXTURE_DOC.content_sha256).toMatch(/^[0-9a-f]{64}$/);
  });
});
