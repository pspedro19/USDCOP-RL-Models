/**
 * Policy contract parity — SHARED CASE TABLE (C-004 remedy-3 finding 6).
 *
 * This is the Vitest twin of tests/unit/test_policy_contract.py::
 * TestSharedCaseTable. The table below is a LITERAL duplicate of
 * PARITY_CASES (same case ids, same payloads, same expected verdicts).
 * Python EXECUTES it against the real constructors/validators; this file
 * EXECUTES it against the TS runtime validators — no text inspection.
 * If you add/change a case: update BOTH files and BOTH pins.
 */

import { describe, it, expect } from 'vitest';
import {
  validateStrategyDecision,
  validateEngineRef,
  validateConditionNode,
  validateRuleTrace,
  validatePolicyContext,
  validateFeatureSnapshot,
} from '@/lib/contracts/policy.contract';

const FULL_HASH = 'sha256:' + 'deadbeef'.repeat(8); // 64 lowercase hex chars

type Verdict = 'valid' | 'invalid';
type Target = 'decision' | 'engine_ref' | 'condition' | 'trace' | 'context' | 'snapshot';

interface ParityCase {
  id: string;
  target: Target;
  payload: unknown;
  expect: Verdict;
}

function decisionPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    signal_id: 's1:2026-07-27:deadbeefdeadbeef',
    sleeve_id: 's1',
    strategy_version: '1.0.0',
    engine_ref: { type: 'rule_based', policy_hash: FULL_HASH },
    as_of: '2026-07-27',
    direction: 'FLAT',
    target_exposure: 0.0,
    reason_codes: [],
    decision_components: { close: 1.0 },
    rule_trace: null,
    feature_snapshot_id: null,
    decision_fingerprint: FULL_HASH,
    ...overrides,
  };
}

const PARITY_CASES: ParityCase[] = [
  // --- StrategyDecision -----------------------------------------------------
  { id: 'decision_valid', target: 'decision', payload: decisionPayload(), expect: 'valid' },
  { id: 'decision_direction_drop_table', target: 'decision',
    payload: decisionPayload({ direction: 'DROP TABLE trades' }), expect: 'invalid' },
  { id: 'decision_exposure_bool', target: 'decision',
    payload: decisionPayload({ target_exposure: true }), expect: 'invalid' },
  { id: 'decision_exposure_numeric_string', target: 'decision',
    payload: decisionPayload({ target_exposure: '1.0' }), expect: 'invalid' },
  { id: 'decision_exposure_inf', target: 'decision',
    payload: decisionPayload({ target_exposure: Infinity }), expect: 'invalid' },
  { id: 'decision_exposure_nan', target: 'decision',
    payload: decisionPayload({ target_exposure: NaN }), expect: 'invalid' },
  { id: 'decision_sleeve_id_none', target: 'decision',
    payload: decisionPayload({ sleeve_id: null }), expect: 'invalid' },
  { id: 'decision_as_of_none', target: 'decision',
    payload: decisionPayload({ as_of: null }), expect: 'invalid' },
  { id: 'decision_policy_hash_true', target: 'decision',
    payload: decisionPayload({ engine_ref: { type: 'rule_based', policy_hash: true } }),
    expect: 'invalid' },
  { id: 'decision_components_infinity', target: 'decision',
    payload: decisionPayload({ decision_components: { x: Infinity } }), expect: 'invalid' },
  { id: 'decision_trace_missing_schema', target: 'decision',
    payload: decisionPayload({ rule_trace: { rules: [] } }), expect: 'invalid' },
  // --- EngineRef ------------------------------------------------------------
  { id: 'engine_ref_valid', target: 'engine_ref',
    payload: { type: 'rule_based', policy_hash: FULL_HASH }, expect: 'valid' },
  { id: 'engine_ref_hash_bool', target: 'engine_ref',
    payload: { type: 'rule_based', policy_hash: true }, expect: 'invalid' },
  { id: 'engine_ref_hash_not_hex', target: 'engine_ref',
    payload: { type: 'rule_based', policy_hash: 'sha256:NOT-HEX!' }, expect: 'invalid' },
  { id: 'engine_ref_ml_snapshot_bool', target: 'engine_ref',
    payload: { type: 'ml', model_snapshot_id: true }, expect: 'invalid' },
  // --- Condition AST ----------------------------------------------------------
  { id: 'condition_valid_gt', target: 'condition',
    payload: { operator: 'greater_than', left: 'feature.close', right: 1.5 },
    expect: 'valid' },
  { id: 'condition_op_drop_table', target: 'condition',
    payload: { operator: 'DROP TABLE trades', left: 'feature.close', right: 1.0 },
    expect: 'invalid' },
  { id: 'condition_op_eval', target: 'condition',
    payload: { operator: 'eval', code: 'close > ma_200' }, expect: 'invalid' },
  { id: 'condition_feature_true_mapping', target: 'condition',
    payload: { operator: 'greater_than', left: { feature: true }, right: 1.0 },
    expect: 'invalid' },
  { id: 'condition_feature_empty', target: 'condition',
    payload: { operator: 'greater_than', left: 'feature.', right: 1.0 },
    expect: 'invalid' },
  { id: 'condition_nan_literal', target: 'condition',
    payload: { operator: 'greater_than', left: 'feature.close', right: NaN },
    expect: 'invalid' },
  { id: 'condition_inf_literal', target: 'condition',
    payload: { operator: 'greater_than', left: 'feature.close', right: Infinity },
    expect: 'invalid' },
  { id: 'condition_bool_operand', target: 'condition',
    payload: { operator: 'greater_than', left: true, right: 1.0 }, expect: 'invalid' },
  // --- RuleTrace --------------------------------------------------------------
  { id: 'trace_valid_v1', target: 'trace',
    payload: { trace_schema: 'rule_trace_v1',
      rules: [{ rule_id: 'r1', label: 'R1', observed: { close: 1.0 },
                result: true, reason_code: 'GT' }] },
    expect: 'valid' },
  { id: 'trace_v2_rejected', target: 'trace',
    payload: { trace_schema: 'rule_trace_v2', rules: [] }, expect: 'invalid' },
  { id: 'trace_missing_schema', target: 'trace',
    payload: { rules: [] }, expect: 'invalid' },
  { id: 'trace_observed_infinity', target: 'trace',
    payload: { trace_schema: 'rule_trace_v1',
      rules: [{ rule_id: 'r1', label: 'R1', observed: { close: Infinity },
                result: true, reason_code: 'GT' }] },
    expect: 'invalid' },
  // --- PolicyContext ----------------------------------------------------------
  { id: 'context_valid', target: 'context', payload: { mode: 'DECISION' }, expect: 'valid' },
  { id: 'context_mode_drop_table', target: 'context',
    payload: { mode: 'DROP_TABLE' }, expect: 'invalid' },
  // --- FeatureSnapshot --------------------------------------------------------
  { id: 'snapshot_valid', target: 'snapshot',
    payload: { close: 2.0, ma_200: 1.0 }, expect: 'valid' },
  { id: 'snapshot_infinity', target: 'snapshot',
    payload: { close: Infinity, ma_200: 1.0 }, expect: 'invalid' },
  { id: 'snapshot_nan', target: 'snapshot',
    payload: { close: NaN, ma_200: 1.0 }, expect: 'invalid' },
  { id: 'snapshot_bool_value', target: 'snapshot',
    payload: { close: true, ma_200: 1.0 }, expect: 'invalid' },
  { id: 'snapshot_null_value', target: 'snapshot',
    payload: { close: null, ma_200: 1.0 }, expect: 'invalid' },
  { id: 'snapshot_string_value', target: 'snapshot',
    payload: { close: '5', ma_200: 1.0 }, expect: 'invalid' },
];

/** Mirrors PARITY_CASE_IDS_SHA256_PREFIX in the Python twin. */
const PARITY_CASE_TABLE_PIN = 'case-table-v1:35';

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

describe('policy contract parity — shared case table (EXECUTED, not text-inspected)', () => {
  it.each(PARITY_CASES.map((c) => [c.id, c] as const))('case %s', (_id, c) => {
    expect(runCase(c.target, c.payload)).toBe(c.expect);
  });

  it('table pin matches the Python twin (version + case count, unique ids)', () => {
    expect(PARITY_CASE_TABLE_PIN).toBe(`case-table-v1:${PARITY_CASES.length}`);
    expect(new Set(PARITY_CASES.map((c) => c.id)).size).toBe(PARITY_CASES.length);
  });
});
