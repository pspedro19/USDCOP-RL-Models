---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - src/contracts/signal_contract.py
  - airflow/dags/asset_pipeline_factory.py
  - config/assets/pipelines.yaml
  - .claude/rules/strategy-engines.md
---

# BL-45 — Motor de políticas: contrato + registry + factory (R1-R3)

**Fuente**: planes/05-rule-based-strategies.md §2-§7, §13 R1-R3 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Las rule-based YA corren (MA200 SPX, votos SMA Oro, hodl BTC) pero cada una con su publisher ad-hoc; no existe Policy.evaluate() común, ni engine.type en specs, ni DSL declarativo, ni rule_trace. gold_dynamic_exit es stateful y vive FUERA del loop estándar (publisher propio) — prueba viva de la decisión abierta §15.2.

## Qué falta exactamente
R1: contracts/policy.py (required_features/validate_inputs/evaluate→StrategyDecision), strategy_decision con engine_ref discriminado, rule_trace_v1. R2: specs con engine.type|implementation.mode|feature_set_id|resample_policy_id|policy(params,resolution,fallbacks)|explainability. R3: factory con tareas genéricas (resolve_feature_snapshot→validate_policy_inputs→evaluate_policy→publish) que OMITE train cuando retrain=never; modos DECISION/FREEZE/REVALIDATE/BACKFILL; ramifica SOLO por engine.type, jamás por strategy_id. DSL declarativo = whitelist AST (greater_than/all/crosses_above/...), CERO eval/SQL/Python desde YAML. Resolver §15: DSL propio (recomendado) y firma de evaluate con contexto de ESTADO (dynexit lo exige). Las 17 validaciones CI de §11.

## Impacto frontend
Ninguno directo (R5 es BL-46).

## Dependencias
BL-13 (surface), BL-39 (feature_set), BL-28 (converge con factories FABRIC — decisión per-sleeve vs task-mapping se toma ahí).

## Verificación
Las 17 validaciones CI de §11 en verde; mismo input+policy ⇒ misma decisión (test de determinismo); spec MA200 declarativo evalúa idéntico al coded_policy actual.

## Notas constitución
Regla siempre-cargada nueva: .claude/rules/strategy-engines.md (invariantes 1-9). Las reglas también cobran trials — cada variante de ventana/umbral/filtro = +1 en su familia.
