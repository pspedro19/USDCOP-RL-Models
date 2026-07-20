---
kind: rule
status: IMPLEMENTED
contract: CTR-EXPERIMENT-PROC-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - scripts/pipeline/run_ssot_pipeline.py
---
# Rule: Protocolo de experimentos

> **SSOT de las reglas duras.** Pasos concretos y formatos de reporte:
> `../specs/platform/experiment-procedure.md`.

## Reglas duras (nunca se violan)

1. **UNA variable por experimento.** Variables mayores: espacio de acción, arquitectura, set de
   features, función de reward, hiperparámetros, niveles de stop, sizing.
   Si piden cambiar varias: *"Son N variables. ¿Cuál probamos primero?"* y proponer N experimentos.
   Si el usuario insiste, loguear **"Regla 1 violada"** en el log.
2. **5 seeds para RL**: `[42, 123, 456, 789, 1337]`. Sin excepciones.
3. **Un experimento = un SSOT congelado.** Config completo en `config/experiments/`, congelado al
   arrancar el training. `v215b_baseline.yaml` nunca se modifica.
4. **Reward de eval ≠ performance OOS.** JAMÁS elegir "mejor modelo" por reward de evaluación.
   Probado: seed 456 (eval=131) perdió -20.6%; seed 1337 (eval=111) ganó +9.6%.
5. **Validación estadística antes de declarar éxito**: ≥3/5 seeds positivos, CI bootstrap 95%
   excluye cero, PF > 1.05. Si p > 0.05 se escribe **"NO estadísticamente significativo"**.
6. **CPU para PPO MlpPolicy**, GPU solo para RecurrentPPO (la RTX 3050 hace throttling).
7. **Todo experimento se loguea** en `.claude/experiments/EXPERIMENT_LOG.md`.

> Las reglas de anti-selección (no grid-search sobre el test, registro de trials, DSR) son
> transversales y viven en `quant-constitution.md`.

## DO NOT

- Do NOT cambiar más de una variable por experimento.
- Do NOT entrenar RL con menos de 5 seeds.
- Do NOT declarar un modelo "rentable" sin tests estadísticos.
- Do NOT usar el reward de eval para seleccionar el modelo de producción.
- Do NOT modificar un config congelado a mitad de experimento — abre uno nuevo.
- Do NOT re-correr un experimento por un bug encontrado; corrígelo en un commit aparte.
- Do NOT usar GPU para PPO MlpPolicy.
