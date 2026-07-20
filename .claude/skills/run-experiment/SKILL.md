---
name: run-experiment
description: Run a training experiment under the repo's hard rules — one variable, 5 seeds, frozen SSOT config, statistical validation, mandatory report format. Use when the user asks to test a hypothesis, tune a parameter, or compare model variants.
---

# Run an experiment

Las reglas duras existen y **nada las hacía cumplir**. Esta skill es ese enforcement.

## Preflight (bloqueante)

1. **¿Hay un ID en `.claude/experiments/EXPERIMENT_QUEUE.md`?** Si no, pide la hipótesis primero.
2. **¿Existe el config congelado** `config/experiments/{id}.yaml`, completo (no un diff)?
3. **Declara el protocolo en voz alta**: ID · la ÚNICA variable que cambia · qué queda constante ·
   ruta del config.

**Si el usuario pide cambiar N variables**: responde *"Son N variables. ¿Cuál probamos primero?"*
y propón N experimentos. Si insiste, ejecuta pero loguea **"Regla 1 violada"**.

## Ejecutar

```bash
python scripts/pipeline/run_ssot_pipeline.py --config config/experiments/{id}.yaml
```

**5 seeds obligatorios**: `[42, 123, 456, 789, 1337]`. Sin excepciones, sin "empecemos con dos".

Sin scope creep: si aparece un bug, se corrige en commit aparte y **no** se re-corre el experimento.
Si el config necesita ajuste → ABORTAR y abrir ID nuevo.

## Reportar (formato obligatorio)

```
| Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
```

Más: media ± std · seeds positivos X/5 · CI bootstrap 95% (10.000 muestras) · t-test p-value ·
vs baseline V21.5b (+2.51%, 4/5) · vs buy-and-hold (-14.66%) · vs random (-4.12%).

Se anexa a `.claude/experiments/EXPERIMENT_LOG.md` y se pobla `_meta.results` del config.

## Veredicto

**Éxito** requiere ≥3/5 seeds positivos **y** CI bootstrap que excluye cero **y** PF > 1.05.
Si p > 0.05, escribe literalmente **"NO estadísticamente significativo"** — no lo suavices.

## Constraints (quant-constitution)

- **Cada versión, cada grid, cada gate mirado = 1 trial.** Regístralo en el HYPOTHESIS-REGISTRY
  del activo.
- **Prohibido elegir la mejor celda de un grid sobre el OOS.** Reporta la sensibilidad completa.
- **Sin DSR trial-aware > 0.95, no hay claim de edge.**
- Un diagnóstico sobre el OOS genera hipótesis para el período SIGUIENTE, no cambios evaluados en
  el mismo período.
- **Nunca** selecciones el modelo de producción por reward de evaluación (seed 456: eval=131,
  perdió -20.6%).
- CPU para PPO MlpPolicy; GPU solo RecurrentPPO.
- Con N < 20 trades: solo conteo y PnL. Nada de "Sharpe 19, p=0.000, 3 trades".
