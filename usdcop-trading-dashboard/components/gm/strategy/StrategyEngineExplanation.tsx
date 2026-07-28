'use client';

/**
 * StrategyEngineExplanation — ONE renderer, four engine variants (BL-46 R5).
 *
 * Invariant 7 of `.claude/rules/strategy-engines.md` is the whole point of this
 * file: **the frontend RENDERS the rule_trace, it NEVER re-evaluates
 * conditions.** Everything shown here is a FACT emitted by the backend:
 *
 * - `result` comes from `trace.rules[i].result` — never from comparing
 *   `observed` against `threshold` in React.
 * - "Regla ganadora" comes from `trace.winning_rule_id`, "Fallback" from
 *   `trace.fallback_applied` — never from "the first rule with result=true".
 * - "Umbral" comes from `trace.rules[i].threshold` — absent ⇒ "—", never a
 *   guess about which observed value plays the threshold role.
 *
 * The variant is chosen ONLY by `engine_ref.type` (invariant 1: never by
 * strategy_id), and the shared header is identical for every engine.
 *
 * Spec: .claude/specs/planes/05-rule-based-strategies.md §9
 * Contract: CTR-POLICY-BACKEND-001 (BL-46 R5)
 */

import type {
  RuleTrace,
  RuleTraceEntry,
  StrategyDecision,
} from '@/lib/contracts/policy.contract';
import type { PresentationSpec } from '@/lib/contracts/policy-version.contract';

const ENGINE_LABEL: Record<string, string> = {
  rule_based: 'Reglas',
  ml: 'Modelo ML',
  rl: 'Política RL',
  composite: 'Compuesta',
};

/** Render a value WITHOUT interpreting it (numbers get a stable locale-free form). */
function renderValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '—';
  if (typeof v === 'boolean') return v ? 'sí' : 'no';
  if (typeof v === 'string') return v;
  return JSON.stringify(v);
}

function pairs(record: Record<string, unknown> | undefined): string {
  const entries = Object.entries(record ?? {});
  if (!entries.length) return '—';
  return entries.map(([k, v]) => `${k}: ${renderValue(v)}`).join(' · ');
}

function labelFor(component: string, presentation?: PresentationSpec | null): string {
  const found = presentation?.components?.find((c) => c.key === component);
  return found ? found.label : component;
}

// ─────────────────────────────────────────────────────────── variants

/** Rule-based: the §9 table — Condición | Observado | Umbral | Resultado. */
export function RuleTracePanel({ trace }: { trace: RuleTrace | null }) {
  if (!trace || trace.rules.length === 0) {
    return (
      <p data-testid="rule-trace-empty" className="text-sm opacity-70">
        El backend no publicó un rule_trace para esta decisión.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table data-testid="rule-trace-table" className="w-full text-sm">
        <caption className="sr-only">
          Condiciones evaluadas por la política (valores publicados por el backend)
        </caption>
        <thead>
          <tr>
            <th scope="col" className="text-left">Condición</th>
            <th scope="col" className="text-right">Observado</th>
            <th scope="col" className="text-right">Umbral</th>
            <th scope="col" className="text-left">Resultado</th>
          </tr>
        </thead>
        <tbody>
          {trace.rules.map((rule: RuleTraceEntry) => (
            <tr key={rule.rule_id} data-testid={`rule-row-${rule.rule_id}`}>
              <th scope="row" className="text-left font-normal">
                {rule.label || rule.rule_id}
              </th>
              <td className="text-right" data-testid={`rule-observed-${rule.rule_id}`}>
                {pairs(rule.observed)}
              </td>
              <td className="text-right" data-testid={`rule-threshold-${rule.rule_id}`}>
                {pairs(rule.threshold)}
              </td>
              {/* FACT from the backend — never a comparison done here. */}
              <td data-testid={`rule-result-${rule.rule_id}`}>
                {rule.result ? 'PASS' : 'FAIL'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="mt-2 text-xs">
        <dt className="inline">Regla ganadora: </dt>
        <dd className="inline" data-testid="rule-winner">
          {trace.winning_rule_id ?? '—'}
        </dd>
        {' · '}
        <dt className="inline">Fallback aplicado: </dt>
        <dd className="inline" data-testid="rule-fallback">
          {trace.fallback_applied ? 'sí' : 'no'}
        </dd>
      </dl>
    </div>
  );
}

/** ML: model snapshot + published decision components (no re-scoring here). */
export function MLExplanationPanel({ decision }: { decision: StrategyDecision }) {
  const modelId = 'model_snapshot_id' in decision.engine_ref
    ? decision.engine_ref.model_snapshot_id : null;
  return (
    <dl data-testid="ml-panel" className="text-sm">
      <dt className="inline">Model snapshot: </dt>
      <dd className="inline" data-testid="ml-model-snapshot">{modelId ?? '—'}</dd>
      <div data-testid="ml-components">{pairs(decision.decision_components)}</div>
    </dl>
  );
}

/** RL: state/action/policy artifact — the published facts only. */
export function RLPolicyPanel({ decision }: { decision: StrategyDecision }) {
  const modelId = 'model_snapshot_id' in decision.engine_ref
    ? decision.engine_ref.model_snapshot_id : null;
  return (
    <dl data-testid="rl-panel" className="text-sm">
      <dt className="inline">Policy artifact: </dt>
      <dd className="inline" data-testid="rl-policy-artifact">{modelId ?? '—'}</dd>
      <dt className="inline"> · Acción: </dt>
      <dd className="inline" data-testid="rl-action">{decision.direction}</dd>
      <div data-testid="rl-state">{pairs(decision.decision_components)}</div>
    </dl>
  );
}

/** Composite: predictor components + the gate/sizing trace, same renderer. */
export function CompositeDecisionPanel({
  decision, presentation,
}: { decision: StrategyDecision; presentation?: PresentationSpec | null }) {
  return (
    <div data-testid="composite-panel">
      <ul className="text-sm">
        {Object.entries(decision.decision_components).map(([key, value]) => (
          <li key={key} data-testid={`composite-component-${key}`}>
            {labelFor(key, presentation)}: {renderValue(value)}
          </li>
        ))}
      </ul>
      <RuleTracePanel trace={decision.rule_trace} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────── single entry

export interface StrategyEngineExplanationProps {
  decision: StrategyDecision;
  /** §9.1 render-only metadata; never affects the decision or its hash. */
  presentation?: PresentationSpec | null;
}

export function StrategyEngineExplanation({
  decision, presentation = null,
}: StrategyEngineExplanationProps) {
  const engine = decision.engine_ref.type;
  const policyHash = 'policy_hash' in decision.engine_ref
    ? decision.engine_ref.policy_hash : null;
  return (
    <section data-testid="strategy-engine-explanation" data-engine={engine}>
      <header className="text-sm">
        <span data-testid="engine-label">
          {presentation?.engine_label ?? ENGINE_LABEL[engine] ?? engine}
        </span>
        {' · '}
        <span data-testid="sleeve-id">{decision.sleeve_id}</span>
        {' · v'}
        <span data-testid="strategy-version">{decision.strategy_version}</span>
        {' · exposición '}
        <span data-testid="target-exposure">{renderValue(decision.target_exposure)}</span>
        {' · '}
        <span data-testid="reason-codes">
          {decision.reason_codes.length ? decision.reason_codes.join(', ') : '—'}
        </span>
        {' · policy_hash '}
        <code data-testid="policy-hash">{policyHash ?? '—'}</code>
      </header>
      {presentation?.description ? (
        <p data-testid="engine-description" className="text-xs opacity-80">
          {presentation.description}
        </p>
      ) : null}
      {engine === 'rule_based' ? <RuleTracePanel trace={decision.rule_trace} /> : null}
      {engine === 'ml' ? <MLExplanationPanel decision={decision} /> : null}
      {engine === 'rl' ? <RLPolicyPanel decision={decision} /> : null}
      {engine === 'composite'
        ? <CompositeDecisionPanel decision={decision} presentation={presentation} />
        : null}
    </section>
  );
}

export default StrategyEngineExplanation;
