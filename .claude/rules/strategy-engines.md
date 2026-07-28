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
> rule_trace, plan R1-R8): `../specs/planes/05-rule-based-strategies.md`.

## Invariantes

1. **Una estrategia es un CONTRATO, no una implementación.** Todo motor
   (`rule_based | ml | rl | composite`) emite el mismo `strategy_signal`/decision
   (dirección, target_exposure, fingerprint, reason_codes). Ejecución, facts, BI y
   frontend NO distinguen motores — ahí vive la escalabilidad.
2. **Una estrategia de reglas no es un modelo incompleto: es una política determinista
   completa.** Mismo gobierno, linaje, gates, paper, juez, retiro y Passport que ML.
   `rule_based` ⇒ `retrain: never`, sin `model_snapshot_id`; congelar la receta ES
   congelar la estrategia (`policy_hash` + `params_hash` + `feature_set_hash`).
3. **`smart_simple_v11` es `composite`, no ML puro**: predictor Ridge/BR + gate Hurst
   + sizing + TP/HS. El predictor es UN componente; la unidad evaluable es la política.
4. **Un solo motor de evaluación**: la MISMA librería `Policy.evaluate(snapshot, ctx)`
   corre en backtest, paper y live. Dos implementaciones = el backtest miente. Solo
   cambia la FUENTE de features (histórica con `available_at` vs streaming).
5. **Snapshots explícitos, jamás "los últimos datos"**: toda política referencia
   `feature_set_id` + `resample_policy_id` con cutoff y hash; prohibido
   `SELECT ... ORDER BY time DESC LIMIT n` desde una política.
6. **DSL declarativo = whitelist de operadores sobre AST** (`greater_than`, `all`,
   `crosses_above`, …). PROHIBIDO ejecutar código arbitrario desde YAML (eval, SQL
   libre, Python inline). Lo complejo vive como feature registrada
   (`code_reference + code_hash`) o como `coded_policy` en Git.
7. **El frontend renderiza el `rule_trace`, NUNCA re-evalúa condiciones.** Un solo
   renderer por spec con variantes por motor (RuleTrace/ML/RL/Composite). La sección
   `presentation:` no decide ni cambia el `policy_hash`.
8. **Las reglas también se sobreajustan**: cada variante mirada (ventana, umbral,
   filtro, salida, sizing, combinación) = +1 trial en su familia. Re-ejecutar una
   política congelada, publicar forward o recalcular métricas predeclaradas = 0.
9. **Fallbacks declarados**: toda política define `missing_input_policy` /
   `stale_input_policy` (FAIL_CLOSED / FLAT / …) y resolución de conflictos entre
   reglas (prioridad). Sin default explícito no hay freeze.

## DO NOT

- Do NOT crear un registry de reglas paralelo al de estrategias — un solo registro
  con discriminador `engine.type`.
- Do NOT emitir tarea de entrenamiento cuando `retrain: never` (el factory la omite).
- Do NOT añadir columnas a la tabla de señales por cada regla nueva — los componentes
  de decisión van en `decision_components` JSONB versionado.
- Do NOT permitir `eval`/SQL/Python arbitrario en un spec declarativo.
- Do NOT recalcular condiciones en el frontend.
- Do NOT tratar una variante de regla como "gratis" por no ser ML.
- Do NOT ramificar el factory por `strategy_id` — solo por `engine.type`,
  `capabilities` y estados.
