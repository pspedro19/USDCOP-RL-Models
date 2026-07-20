---
kind: as-built
status: IMPLEMENTED
contract: CTR-EXPERIMENT-PROC-001
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - scripts/pipeline/run_ssot_pipeline.py
  - config/experiments
---
# SDD Spec: Procedimiento de experimentos (referencia)

> **Responsibility**: los pasos concretos de un experimento y el formato de reporte.
> Las **reglas duras** (1 variable, 5 seeds, validación estadística) están en
> `../../rules/experiment-protocol.md`, que se auto-carga.

---

## Fase 1 — ANTES de empezar

1. **Revisar la cola**: ¿hay un ID que coincida en `EXPERIMENT_QUEUE.md`?
   Si no, pedir al usuario que defina la hipótesis primero.
2. **Verificar el config SSOT congelado** en `config/experiments/{experiment_id}.yaml`.
   Debe ser un `pipeline_ssot.yaml` **completo**, no un diff ni un override parcial.
3. **Declarar el protocolo en voz alta** antes de entrenar: ID del experimento, la ÚNICA
   variable que cambia, qué se mantiene constante, y la ruta del config.

## Fase 2 — DURANTE el training

4. **Correr los 5 seeds** `[42, 123, 456, 789, 1337]` vía `MultiSeedTrainer` o llamadas
   secuenciales a `run_ssot_pipeline.py --config config/experiments/{id}.yaml`.
5. **Sin scope creep.** Si aparece un bug, se corrige en un commit aparte — **no** se re-corre el
   experimento. Si el config necesita ajuste, se ABORTA y se abre un ID nuevo.

## Fase 3 — DESPUÉS de L4

6. **Reportar** en `EXPERIMENT_LOG.md`:

   Tabla por seed (obligatoria):
   ```
   | Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
   ```

   Agregados (obligatorios): media ± std · seeds positivos X/5 · CI bootstrap 95% (10.000
   muestras) · t-test vs 0 con p-value · comparación vs baseline V21.5b (+2.51%, 4/5) ·
   vs buy-and-hold (-14.66%) · vs random (-4.12%).

7. **Validación estadística** antes de declarar éxito: ≥3/5 seeds positivos · CI bootstrap
   excluye el cero · profit factor > 1.05 (1.00-1.02 es ruido) · si p > 0.05 se escribe
   explícitamente **"NO estadísticamente significativo"**.

8. **Actualizar la cola**: mover a COMPLETED, evaluar el marco de decisión para el siguiente
   experimento, y poblar `_meta.results` en el config congelado.

---

## Formatos de reporte

```
RL:  | Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars | WR_Long% | WR_Short% |
ML:  | Strategy | Return% | Sharpe | p-value | DA% | WR% | PF | MaxDD% | Trades | $10K -> |
```

Siempre acompañado de media ± std, CI bootstrap 95% y comparación contra buy-and-hold.

## Splits de datos (RL)

Train 2019-12 → 2024-12 (70K barras) · Val 2025-01→06 (7K) · Test 2025-07→12 (7K).

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Reglas duras (auto-cargadas) | `../../rules/experiment-protocol.md` |
| Versionado de configs | `../../rules/ssot-versioning.md` · `ssot-lifecycle.md` |
| Anti-selección, trials, DSR | `../../rules/quant-constitution.md` |
