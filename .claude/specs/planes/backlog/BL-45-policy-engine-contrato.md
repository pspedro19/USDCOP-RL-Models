---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-03
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


## Auditoría de las 17 validaciones §11 (CLAUDE, 2026-08-03)

La brecha declarada arriba dice *"R1: contracts/policy.py (...)"* como si R1 estuviera por
hacer. **Está en gran parte construido** y esa frase induce a error: `src/contracts/policy.py`
tiene `EngineRef`, `PolicyContext`, `StrategyDecision`, el protocolo `Policy` con
`required_features`/`validate_inputs`/`evaluate`, y `POLICY_MODES` con los cuatro modos.
Existen además `rule_trace.py`, `policy_dsl.py` y `policy_version.py`.

Lo que **no** está es R3. Y de las 17 validaciones de §11, esto es lo que se aplica de verdad
— cada fila comprobada contra el código, no contra el docstring que la anuncia:

| # | Validación §11 | Estado | Dónde |
|---|---|---|---|
| 1 | `rule_based` ⇒ `retrain=never` | ✅ | `loader.py:166` |
| 2 | `rule_based` no declara `capability=train` | ✅ | `loader.py:176` |
| 3 | `rule_based` no exige `model_snapshot_id` | ✅ | `loader.py:170` |
| 4 | toda policy referencia `feature_set` y `resample_policy` | ✅ | `loader.py:181` |
| 5 | los `required_features` existen en el snapshot | ✅ | `base.py:79-81` (`validate_inputs`) |
| 6 | **ningún input supera `decision_cutoff`** | ❌ **no implementable hoy** | ver abajo |
| 7 | todo operador declarativo pertenece al whitelist | ✅ | `loader.py:222` → AST de `policy_dsl` |
| 8 | el YAML no contiene `eval`, SQL libre ni Python | ✅ | `validate_policy_specs.py::_check_text` |
| 9 | toda política tiene default/fallback explícito | ✅ | `loader.py:44` |
| 10 | conflictos entre reglas con prioridad declarada | ✅ | `_check_outputs` |
| 11 | la salida respeta `direction` y caps de exposición | ✅ | exposición en `_check_outputs`; **`direction` en el contrato** (`policy.py:385`), no duplicado en el validador |
| 12 | determinismo (mismos inputs ⇒ misma decisión) | ✅ | `_check_determinism` |
| 13 | `rule_trace` contiene todas las condiciones relevantes | ⚠️ **schema sí, completitud NO** | `rule_trace.py` valida forma, nadie exige cobertura |
| 14 | `policy_hash` coincide con params y schema | ✅ | `loader.py` (`governance.policy_hash`) |
| 15 | cambiar ventana/threshold exige nueva versión | ⚠️ parcial | `check_policy_parity.py` cubre paridad legacy↔motor (BL-47), no "cambio ⇒ versión" |
| 16 | el frontend no recalcula condiciones | ➡️ BL-46 | fuera de R1-R3 |
| 17 | `WITHDRAWN` conserva bundles y manifiesto | ⚠️ | estado en `governance/declaration.py`; cobertura en carril CODEX |

**Recuento honesto: 12 aplicadas, 3 parciales, 1 no implementable, 1 diferida a BL-46.**

### El hueco #6 es constitucional y NO se puede tapar con un check más

`quant-constitution.md` §4 exige anti-look-ahead en tres capas. El motor tiene `as_of` en
`PolicyContext` **y lo valida como ISO**, pero el snapshot es un `Mapping[str, Any]` de
`{nombre: valor}` **sin marca temporal por feature**. No hay contra qué comparar el cutoff:
la validación #6 no es un check que falte escribir, es **una comprobación que la forma actual
del contrato no permite expresar**.

Salidas posibles, ninguna tomada aquí porque tocan contrato compartido:
1. que el snapshot lleve `available_at` por feature ⇒ **cambio de contrato, exige `C-NNN` + ACK**;
2. que lo garantice aguas arriba `resolve_feature_snapshot`, que es **precisamente la tarea de
   R3 que no existe**.

Mientras tanto, el motor **no puede demostrar** que sus inputs respeten el cutoff. Eso se
declara aquí en vez de dejar la casilla §11 como si estuviera cubierta.

### Estado real de R2 y R3

- **R2 parcial**: 4 specs en `config/policies/` compilan y pasan el validador
  (`[OK] 4 specs de política válidos`), pero los campos exigidos no están uniformemente:
  `resample_policy_id` y `explainability` aparecen en 4 ficheros y `fallbacks` en **1**.
- **R3 NO existe**: `airflow/dags/asset_pipeline_factory.py` ramifica por **`strategy_ids`**
  (`:107`, `:117`, `:176`), que es exactamente lo que `strategy-engines.md` prohíbe
  (*"el factory ramifica por `engine.type` (...), nunca por `strategy_id`"*). Las tareas
  genéricas `resolve_feature_snapshot→validate_policy_inputs→evaluate_policy→publish` no
  existen. **Verificar un cambio ahí exige Airflow vivo ⇒ STACK_OR_CI.**

**BL-45 sigue PARTIAL.** Esta auditoría no cierra alcance: sustituye una brecha mal descrita
por el mapa real, y aísla el único punto (#6) que no es trabajo sino decisión de contrato.
