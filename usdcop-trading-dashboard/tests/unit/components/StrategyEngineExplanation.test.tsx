/**
 * StrategyEngineExplanation — the renderer RENDERS, it never re-evaluates
 * (invariant 7 of `.claude/rules/strategy-engines.md`, BL-46 R5 verification).
 *
 * The decisive test is `contradictory trace`: a trace whose `result` disagrees
 * with its own observed/threshold values must render the BACKEND's verdict. If
 * anyone ever re-implements `close > ma_200` in React, that test goes red.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { StrategyDecision } from '@/lib/contracts/policy.contract';
import { StrategyEngineExplanation } from '@/components/gm/strategy/StrategyEngineExplanation';

const HASH = `sha256:${'ab'.repeat(32)}`;

function ma200Decision(overrides: Partial<StrategyDecision> = {}): StrategyDecision {
  return {
    signal_id: `spx500_daily_ma200_v1:2026-07-27:${'de'.repeat(8)}`,
    sleeve_id: 'spx500_daily_ma200_v1',
    strategy_version: '2.0.0',
    engine_ref: { type: 'rule_based', policy_hash: HASH },
    as_of: '2026-07-27',
    direction: 'LONG',
    target_exposure: 1,
    reason_codes: ['CLOSE_ABOVE_MA200'],
    decision_components: { close: 6412.8, ma_200: 5984.2 },
    rule_trace: {
      trace_schema: 'rule_trace_v1',
      rules: [
        {
          rule_id: 'trend_on',
          label: 'Precio sobre MA200',
          observed: { close: 6412.8, ma_200: 5984.2 },
          threshold: { ma_200: 5984.2 },
          result: true,
          reason_code: 'CLOSE_ABOVE_MA200',
        },
        {
          rule_id: 'trend_off',
          label: 'Precio bajo MA200',
          observed: { close: 6412.8, ma_200: 5984.2 },
          threshold: { ma_200: 5984.2 },
          result: false,
          reason_code: 'CLOSE_BELOW_MA200',
        },
      ],
      winning_rule_id: 'trend_on',
      fallback_applied: false,
    },
    feature_snapshot_id: 'fs_spx500_20260727',
    decision_fingerprint: `sha256:${'de'.repeat(32)}`,
    ...overrides,
  } as StrategyDecision;
}

describe('StrategyEngineExplanation — renders the trace, never re-evaluates', () => {
  it('renders the MA200 trace exactly as published', () => {
    render(<StrategyEngineExplanation decision={ma200Decision()} />);
    expect(screen.getByTestId('rule-observed-trend_on').textContent)
      .toBe('close: 6412.8 · ma_200: 5984.2');
    expect(screen.getByTestId('rule-threshold-trend_on').textContent).toBe('ma_200: 5984.2');
    expect(screen.getByTestId('rule-result-trend_on').textContent).toBe('PASS');
    expect(screen.getByTestId('rule-result-trend_off').textContent).toBe('FAIL');
    expect(screen.getByTestId('rule-winner').textContent).toBe('trend_on');
    expect(screen.getByTestId('rule-fallback').textContent).toBe('no');
    expect(screen.getByTestId('policy-hash').textContent).toBe(HASH);
  });

  it('a CONTRADICTORY trace still renders the backend verdict (no re-evaluation)', () => {
    const decision = ma200Decision({
      rule_trace: {
        trace_schema: 'rule_trace_v1',
        rules: [{
          rule_id: 'trend_on',
          label: 'Precio sobre MA200',
          // observed says close < ma_200, yet the backend published result=true
          observed: { close: 100, ma_200: 5984.2 },
          threshold: { ma_200: 5984.2 },
          result: true,
          reason_code: 'CLOSE_ABOVE_MA200',
        }],
        winning_rule_id: 'trend_on',
        fallback_applied: false,
      },
    });
    render(<StrategyEngineExplanation decision={decision} />);
    expect(screen.getByTestId('rule-result-trend_on').textContent).toBe('PASS');
  });

  it('shows "—" when the backend published no threshold (never guesses one)', () => {
    const decision = ma200Decision({
      rule_trace: {
        trace_schema: 'rule_trace_v1',
        rules: [{
          rule_id: 'legacy_rule',
          label: 'Regla legacy sin umbral',
          observed: { close: 6412.8 },
          result: false,
          reason_code: 'LEGACY',
        }],
        fallback_applied: true,
      },
    });
    render(<StrategyEngineExplanation decision={decision} />);
    expect(screen.getByTestId('rule-threshold-legacy_rule').textContent).toBe('—');
    expect(screen.getByTestId('rule-winner').textContent).toBe('—');
    expect(screen.getByTestId('rule-fallback').textContent).toBe('sí');
  });

  it('branches ONLY on engine_ref.type — ml/rl/composite reuse the same shell', () => {
    const ml = ma200Decision({
      engine_ref: { type: 'ml', model_snapshot_id: 'ridge_2026w30' },
      rule_trace: null,
    });
    const { unmount } = render(<StrategyEngineExplanation decision={ml} />);
    expect(screen.getByTestId('strategy-engine-explanation').dataset.engine).toBe('ml');
    expect(screen.getByTestId('ml-model-snapshot').textContent).toBe('ridge_2026w30');
    unmount();

    const rl = ma200Decision({
      engine_ref: { type: 'rl', model_snapshot_id: 'ppo_v215b_seed42' },
      rule_trace: null,
    });
    const second = render(<StrategyEngineExplanation decision={rl} />);
    expect(screen.getByTestId('rl-policy-artifact').textContent).toBe('ppo_v215b_seed42');
    expect(screen.getByTestId('rl-action').textContent).toBe('LONG');
    second.unmount();

    const composite = ma200Decision({
      engine_ref: { type: 'composite', policy_hash: HASH, model_snapshot_ids: ['ridge_2026w30'] },
    });
    render(<StrategyEngineExplanation decision={composite} />);
    expect(screen.getByTestId('composite-component-close').textContent).toContain('6412.8');
    // the composite variant reuses the SAME RuleTracePanel
    expect(screen.getByTestId('rule-result-trend_on').textContent).toBe('PASS');
  });

  it('presentation metadata only relabels — it never changes the facts', () => {
    render(
      <StrategyEngineExplanation
        decision={ma200Decision({
          engine_ref: { type: 'composite', policy_hash: HASH },
        })}
        presentation={{
          engine_label: 'Reglas MA200',
          description: 'Exposición cuando el cierre supera la media de 200 sesiones',
          components: [{ key: 'close', label: 'Cierre', format: 'price' }],
        }}
      />,
    );
    expect(screen.getByTestId('engine-label').textContent).toBe('Reglas MA200');
    expect(screen.getByTestId('composite-component-close').textContent)
      .toBe('Cierre: 6412.8');
    expect(screen.getByTestId('rule-result-trend_on').textContent).toBe('PASS');
  });
});
