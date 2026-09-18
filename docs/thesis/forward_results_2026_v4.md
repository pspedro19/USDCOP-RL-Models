---
kind: analysis
status: exploratory_forward
version: 1.0.0
last_verified: 2026-09-15
supersedes: docs/thesis/confirmatory_results_v4.md
code_anchors:
  - scripts/analysis/build_confirmatory_forward_specs.py
  - scripts/analysis/evaluate_confirmatory_forward_ppo.py
  - scripts/analysis/report_confirmatory_ppo.py
---

# Forward post-freeze 2026: PPO v4

Este carril usa únicamente observaciones posteriores al freeze, desde 2026-01-01 hasta
2026-09-10, que es el último día disponible en `seeds/latest/usdcop_m5_ohlcv.parquet`.
Se construyeron 162 sesiones con el scaler y el HMM ajustados exclusivamente sobre
2020--2022. No hubo refit, selección de hiperparámetros ni consulta al hold-out.

Portable estable: `65534cc4984cd2bb52bd7f42b845fcd5cb7be214123d154b5b8bf6c5ef2eb585`.
HMM fit range: 2020-01-08--2022-12-29. Macro: regla estricta T−1; no se imputaron
sesiones faltantes.

## Resultado descriptivo por semilla

| Configuración | Retorno medio | Retorno mediano | Sharpe medio | Sharpe mediano | Semillas positivas |
|---|---:|---:|---:|---:|---:|
| PPO régimen | −28,52 % | −26,89 % | −5,60 | −5,89 | 0/5 |
| PPO backbone | −40,93 % | −34,97 % | −6,49 | −5,88 | 0/5 |

Estos números son una lectura post-freeze parcial, no una licencia de rentabilidad. El
periodo fue examinado por otros diagnósticos históricos antes de este carril y aún no
existe un manifiesto de apertura única con baselines congelados, por lo que se etiqueta
`exploratory_forward`.

## Baselines del mismo motor

Los baselines se ejecutaron sobre las mismas 162 sesiones, el mismo portable estable y
la misma función `run_session`; no hubo ajuste de parámetros en 2026. Son descriptivos
del corte post-freeze y no convierten este carril en una apertura confirmatoria.

| Política | Retorno compuesto | Sharpe |
|---|---:|---:|
| Always flat | 0,00 % | 0,00 |
| Always long 1x | −29,92 % | −4,01 |
| Always short 1x | −10,57 % | −1,22 |
| Momentum 3 barras | −99,69 % | −27,73 |
| Mean reversion 12 barras | −91,23 % | −35,39 |
| Opening range 6 barras | −56,09 % | −8,02 |
| Random seed 42 | −99,71 % | −49,87 |
| Dos reglas por régimen | −20,84 % | −3,26 |

La comparación respalda únicamente una conclusión descriptiva: en este corte las
políticas PPO y los baselines no superan a `always_flat` después de costes. No se usa
para afirmar que ninguna política rentable exista en el espacio de acciones.

## Figuras

- [Capital](../../outputs/thesis-repair/confirmatory_v4_stable/forward_figures/01_capital_holdout_v4.png)
- [Drawdown](../../outputs/thesis-repair/confirmatory_v4_stable/forward_figures/02_drawdown_holdout_v4.png)
- [Semillas](../../outputs/thesis-repair/confirmatory_v4_stable/forward_figures/03_seed_variability_v4.png)
- [Stress de costes](../../outputs/thesis-repair/confirmatory_v4_stable/forward_figures/04_cost_stress_v4.png)

El resultado es consistente con el juez 2024--2025: ninguna semilla obtiene retorno
positivo después de costes. Para una conclusión confirmatoria de 2026 aún deben
congelarse y liquidarse los baselines, el ledger de trials y el tamaño muestral
preespecificado.
