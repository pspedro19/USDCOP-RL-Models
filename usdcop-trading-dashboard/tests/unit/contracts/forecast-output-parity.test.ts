/**
 * ForecastOutput contract parity — SHARED FIXTURE (BL-15 remedio, point 2).
 *
 * This is the Vitest twin of tests/unit/test_forecast_output_contract.py::
 * TestSharedCaseTable. Both runners LOAD the same versioned case table
 * (tests/fixtures/forecast_output_cases.v1.json at the repo root) — nothing is
 * duplicated literally. Each runner recomputes the fixture's content SHA-256
 * before running and goes RED on drift. Python executes the cases against
 * validate_forecast_payload(); this file executes them against the TS runtime
 * validator — same verdict, case by case.
 *
 * This is a REAL runner (it executes TypeScript), not a regex over a Python
 * file: `validateForecastOutput` is imported and called for every case.
 *
 * Sentinel encoding (see the fixture's `encoding` block):
 * - {"$nonfinite": "NaN"|"Infinity"|"-Infinity"} -> native non-finite number.
 */

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { describe, it, expect } from 'vitest';
import {
  validateForecastOutput,
  parseForecastOutput,
  ingestForecastOutputs,
  ForecastOutputError,
  PREDICTION_TYPES,
  FORECAST_OUTPUT_FIELDS,
} from '@/lib/contracts/forecast-output.contract';

const FIXTURE_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../../tests/fixtures/forecast_output_cases.v1.json',
);

type Verdict = 'valid' | 'invalid';

interface ParityCase {
  id: string;
  payload: unknown;
  expect: Verdict;
  note?: string;
}

interface ParityFixture {
  fixture: string;
  version: string;
  content_sha256: string;
  case_count: number;
  cases: ParityCase[];
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
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) out[k] = decodeFixtureValue(v);
    return out;
  }
  return value;
}

function loadFixture(): { cases: ParityCase[]; declaredCount: number } {
  const raw = readFileSync(FIXTURE_PATH, 'utf-8').replace(/\r\n/g, '\n');
  const m = /"content_sha256":\s*"([0-9a-f]{64})"/.exec(raw);
  if (!m) throw new Error('fixture must declare a 64-hex content_sha256');
  const declared = m[1];
  const actual = createHash('sha256')
    .update(raw.replace(declared, ''), 'utf-8')
    .digest('hex');
  if (actual !== declared) {
    throw new Error(
      `FIXTURE DRIFT: declared content_sha256 ${declared} != recomputed ${actual} ` +
        '— the case table changed without regenerating the pin (both runners refuse to run)',
    );
  }
  const doc = JSON.parse(raw) as ParityFixture;
  if (doc.fixture !== 'forecast_output_cases' || doc.version !== 'v1') {
    throw new Error('unexpected fixture identity/version');
  }
  const cases = doc.cases.map((c) => ({ ...c, payload: decodeFixtureValue(c.payload) }));
  const ids = new Set(cases.map((c) => c.id));
  if (ids.size !== cases.length) throw new Error('duplicate case ids in fixture');
  if (cases.length !== doc.case_count) throw new Error('case_count pin does not match');
  return { cases, declaredCount: doc.case_count };
}

const { cases, declaredCount } = loadFixture();

describe('ForecastOutput shared case table (Py <-> TS parity)', () => {
  it('pins the case count (anti-drift)', () => {
    // case-table-v1:96 — update BOTH runners and the pin deliberately.
    expect(cases.length).toBe(96);
    expect(declaredCount).toBe(96);
  });

  for (const c of cases) {
    it(`${c.id} -> ${c.expect}${c.note ? ` (${c.note})` : ''}`, () => {
      const errors = validateForecastOutput(c.payload);
      const verdict: Verdict = errors.length > 0 ? 'invalid' : 'valid';
      expect(
        verdict,
        `${c.id}: expected ${c.expect}, got ${verdict} (errors=${JSON.stringify(errors)})`,
      ).toBe(c.expect);
    });
  }
});

describe('ForecastOutput ingest wall (TS side)', () => {
  const valid = cases.find((c) => c.id === 'valid_base_aware_z')!.payload;
  const laundered = cases.find((c) => c.id === 'wall_laundered_actionable_forecast')!.payload;

  it('parseForecastOutput accepts a contract-clean payload', () => {
    expect(parseForecastOutput(valid).diagnostic_only).toBe(true);
  });

  it('parseForecastOutput throws ForecastOutputError on an actionable payload', () => {
    expect(() => parseForecastOutput(laundered)).toThrow(ForecastOutputError);
  });

  it('ingestForecastOutputs is all-or-nothing', () => {
    expect(ingestForecastOutputs([valid, valid])).toHaveLength(2);
    expect(() => ingestForecastOutputs([valid, laundered])).toThrow(ForecastOutputError);
  });

  it('Date.parse would have accepted the impossible dates this contract rejects', () => {
    // Documents WHY the strict parser exists: the native engine rolls Feb 30 to
    // Mar 2 and accepts a space separator, so `new Date()` cannot be the wall.
    expect(Number.isNaN(Date.parse('2026-02-30T00:00:00Z'))).toBe(false);
    expect(Number.isNaN(Date.parse('2026-07-27 00:00:00Z'))).toBe(false);
    expect(validateForecastOutput({ ...(valid as object), as_of: '2026-02-30T00:00:00Z' }).length)
      .toBeGreaterThan(0);
    expect(validateForecastOutput({ ...(valid as object), as_of: '2026-07-27 00:00:00Z' }).length)
      .toBeGreaterThan(0);
  });

  it('mirrors the Python constants', () => {
    expect([...PREDICTION_TYPES]).toEqual(['return', 'log_return', 'price']);
    expect([...FORECAST_OUTPUT_FIELDS]).toEqual([
      'forecast_id', 'forecast_spec_id', 'asset', 'model_id', 'horizon',
      'as_of', 'available_at', 'target_time',
      'prediction', 'model_fingerprint', 'data_snapshot_id',
      'direction_probability', 'diagnostic_only',
    ]);
  });
});
