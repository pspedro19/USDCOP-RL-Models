---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
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

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/unit/test_policy_contract.py -q
verde:   219 passed

muta:    src/contracts/policy_dsl.py — desactivar la whitelist de operadores del DSL
         (ALLOWED_OPERATORS deja de filtrar)
espera:  4 failed — entre ellos `eval` y `DROP TABLE` aceptados como operadores:
         test_operator_outside_whitelist_raises,
         test_eval_operator_rejected  (`pytest.raises(ValueError, match="whitelist")`
         deja de dispararse), y los casos de código-desde-YAML
         (`when: "eval(close > ma_200 and custom_python())"`,
          `left: "__import__('os').system('id')"`)

muta-2:  spec NUEVO en config/policies/ con `operator: python_eval` y
         `__import__('os').system(...)`, re-hasheado para colar el freeze
espera-2: revienta igual — scripts/validation/validate_policy_specs.py hace
         `directory.glob("*.yaml")` en vez de leer una lista a mano (K-029), así que
         añadir un spec malicioso no lo esquiva: lo mete dentro del perímetro
```

**Historial honesto**: BL-45 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`) — no hubo defecto que cerrar el 2026-07-28. Lo que lo hace fiable, y lo que se
señaló expresamente en CLD-216, es la **segunda** mutación: no basta con que el validador
rechace lo que ya está declarado; el atacante realista **añade un spec nuevo**. Que el
validador descubra los specs por glob del directorio en vez de por lista enumerada es lo que
convierte la whitelist en una garantía y no en una convención — K-029 bien aplicada.

**Aviso de CI declarado (CLD-216)**: `fabric-contracts.yml` corre `validate_policy_specs.py`,
pero **NO** `check_policy_parity.py` (el arnés de paridad de BL-47). La whitelist del DSL sí
está vigilada por CI; la paridad de los motores migrados, no.

## Notas constitución
Regla siempre-cargada nueva: .claude/rules/strategy-engines.md (invariantes 1-9). Las reglas también cobran trials — cada variante de ventana/umbral/filtro = +1 en su familia.
