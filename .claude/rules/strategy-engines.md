---
kind: rule
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - config/assets/pipelines.yaml
  - src/contracts/signal_contract.py
  - scripts/pipeline/normalize_champions.py
---

# Rule: Motores de estrategia (rule-based, ml, rl, composite)

> **SSOT de las invariantes de motores.** Diseño completo (spec YAML, DSL, factory,
> rule_trace, plan R1-R8):
> [`05-rule-based-strategies.md`](../specs/planes/05-rule-based-strategies.md).

## Invariantes

1. Todo motor (`rule_based | ml | rl | composite`) emite el mismo contrato de decisión;
   ejecución, facts, BI y frontend no ramifican por motor.
2. `rule_based` recibe el mismo gobierno, linaje, gates, paper, retiro y Passport que ML:
   `retrain: never`, sin `model_snapshot_id`, y freeze por hashes de policy/params/features.
3. `smart_simple_v11` es `composite`; se evalúa la política completa, no sólo su predictor.
4. La misma `Policy.evaluate(snapshot, ctx)` corre en backtest, paper y live. Sólo cambia la
   fuente causal de features.
5. Toda política fija `feature_set_id`, `resample_policy_id`, cutoff y hash; nunca consulta
   “los últimos datos”.
6. El DSL permite únicamente operadores AST en whitelist. Nada de eval, SQL libre o Python
   inline; lo complejo es una feature registrada o `coded_policy` versionada.
7. El frontend renderiza `rule_trace`; jamás reevalúa condiciones ni deja que `presentation:`
   altere la decisión o `policy_hash`.
8. Cada variante mirada de una regla cobra un trial; una repetición congelada o métrica
   predeclarada no.
9. Toda política declara fallbacks de input stale/missing y prioridad de conflictos antes del
   freeze.

## DO NOT

- Un solo registry; el factory ramifica por `engine.type`, capacidades y estado, nunca por
  `strategy_id`.
- `retrain: never` no genera tarea de entrenamiento.
- Componentes nuevos van en `decision_components` JSONB versionado, no en columnas ad hoc.
