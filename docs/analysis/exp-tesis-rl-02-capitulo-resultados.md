---
kind: analysis
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-12
supersedes: []
code_anchors:
  - outputs/thesis-repair/ppo_v2_diagnostic_summary.json
  - outputs/thesis-repair/baselines_holdout_v2.json
  - outputs/thesis-repair/baselines_holdout_dsr_v2.json
  - outputs/thesis-repair/results_v2_diagnostic/statistics_selection.json
  - scripts/diagnostics/summarize_baseline_dsr.py
---

# Capítulo de resultados — EXP-TESIS-RL-02

## Alcance

## Contraste estadístico adicional del brazo supervisado

En selección, el bootstrap estacionario pareado contra `always-flat` dio ΔSharpe `-9,729`,
IC95 % `[-12,261; -7,900]` y p=`0,0002` con 10.000 réplicas. El artefacto es
[supervised_v2_selection_inference.json](../../outputs/thesis-repair/supervised_v2_selection_inference.json).

Este capítulo presenta resultados reproducibles de la versión v2 del dataset y del entorno
de sesiones USD/COP. Las corridas PPO y los baselines fueron ejecutados con el mismo motor de
retornos y costos. La matriz PPO es retrospectiva diagnóstica; no es el juez confirmatorio
forward definido en el prerregistro.

Este capítulo no presenta un brazo supervisado confirmatorio: el clasificador logístico del
pipeline histórico usa otro contrato de datos y no se mezcla con la matriz v2. Si el título de
la tesis conserva “aprendizaje supervisado”, debe incorporarse como experimento separado o
enmendarse antes de la defensa.

## Integridad de datos

La auditoría sobre `seeds/latest/usdcop_m5_ohlcv.parquet` registra 100.674 barras, cero
duplicados, cero timestamps fuera de la grilla de cinco minutos y cero valores OHLC inválidos.
Se conservaron 1.457 sesiones completas de 60 barras. El macro contiene 12.882 fechas,
cuatro series completas, cero fechas duplicadas y cero valores numéricos inválidos. Las
variables macro se incorporan estrictamente con información disponible antes de la sesión.

## PPO

Se entrenaron dos configuraciones (`ppo_regime` y `ppo_backbone`) con las semillas
42, 123, 456, 789 y 1337, 300.000 pasos por corrida. La selección contiene 226 sesiones.

| Configuración | Retorno medio | Sharpe medio | MaxDD medio | Semillas positivas |
|---|---:|---:|---:|---:|
| PPO régimen | -26,61 % | -4,36 | -28,04 % | 0/5 |
| PPO backbone | -33,83 % | -7,78 | -34,23 % | 0/5 |
| Always-flat | 0,00 % | 0,00 | 0,00 % | — |

El bootstrap estacionario pareado produjo ΔSharpe = -4,360 para régimen frente a flat,
IC95 % [-6,890; -2,362], p = 0,0002. Para backbone produjo ΔSharpe = -7,782,
IC95 % [-10,473; -5,381], p = 0,0002. La configuración con régimen fue menos negativa
que backbone, pero ambas permanecieron por debajo de cero. El DSR de ambas familias no
superó 0,95 con 115 trials.

## Baselines

El hold-out contiene 570 sesiones. Los resultados netos son:

| Estrategia | Retorno | Sharpe | MaxDD |
|---|---:|---:|---:|
| Buy & hold intrasesión | -53,696 % | -2,813 | -54,791 % |
| Buy & hold overnight | -21,329 % | -0,693 | -32,078 % |
| Siempre corto | -45,571 % | -2,227 | -49,706 % |
| Momentum 3 barras | -100,000 % | -22,694 | -100,000 % |
| Mean reversion 12 barras | -99,930 % | -35,462 | -99,930 % |
| Opening range | -83,988 % | -6,201 | -84,255 % |
| Dos reglas por régimen | -34,118 % | -1,780 | -35,971 % |
| Always-flat | 0,000 % | suprimido | 0,000 % |

Los DSR calculados sobre los vectores diarios persistidos fueron 1,36e-7 para buy & hold
intrasesión, 1,26e-5 para siempre corto y 2,34e-4 para las reglas por régimen; ninguno
superó 0,95. Los ratios se suprimieron cuando el baseline tenía menos de 20 operaciones.

## Figuras

Las figuras PPO se generaron desde los artefactos v2:

- [Curvas de capital](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig01_curvas_capital_selection.png)
- [Drawdown](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig02_underwater_selection.png)
- [Sharpe móvil](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig03_sharpe_movil_selection.png)
- [Sensibilidad de costos](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig04_sensibilidad_costos_selection.png)
- [Acciones por régimen](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig05_acciones_por_regimen_selection.png)
- [Variabilidad entre semillas](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig06_semillas_selection.png)

## Interpretación

El patrón desarrollo positivo/selección negativo es consistente con sobreajuste temporal y/o
suboptimización. La mejora relativa del régimen no puede interpretarse como alfa porque su
retorno neto sigue siendo negativo. La convergencia de PPO en pruebas sintéticas demuestra
que el entorno puede aprender controles conocidos, pero no transfiere evidencia al mercado.

## Componentes adicionales y pendientes

## Brazo supervisado diagnóstico

Para resolver la discrepancia del título se ejecutó una única receta fija: `StandardScaler`
seguido de regresión logística balanceada (`C=1.0`), entrenada exclusivamente en desarrollo,
con umbrales congelados 0,55/0,45 para largo/corto. En selección (226 sesiones) obtuvo
`-55,08 %` compuesto y Sharpe `-9,73`; su DSR fue `0,0`. El replay retrospectivo del
hold-out (570 sesiones) obtuvo `-95,94 %`, Sharpe `-12,88` y DSR `0,0`. Estos números son
diagnósticos porque el hold-out ya se había abierto; no autorizan una afirmación confirmatoria.
Artefactos: [selección](../../outputs/thesis-repair/supervised_v2_selection.json),
[DSR selección](../../outputs/thesis-repair/supervised_v2_selection_dsr.json),
[hold-out retrospectivo](../../outputs/thesis-repair/supervised_v2_holdout_diagnostic.json),
[DSR hold-out](../../outputs/thesis-repair/supervised_v2_holdout_dsr.json).
El stress de costos de selección fue -55,08 %, -89,67 % y -97,67 % para 1x, 2x y 3x,
respectivamente; ninguna escala sobrevivió. El artefacto es
[supervised_v2_selection_stress.json](../../outputs/thesis-repair/supervised_v2_selection_stress.json).

No se han observado ledgers reales de DeepSeek o Azure en el repositorio. Por lo tanto, no se
reportan métricas, curvas ni conclusiones para LLM o híbrido. La afirmación correspondiente
queda pendiente hasta validar los ledgers, liquidar sus decisiones y ejecutar el juez forward.

## Conclusión

El resultado defendible de esta etapa es negativo y específico: con el dataset v2, las recetas
PPO congeladas y el contrato de costos utilizado, ninguna estrategia intradía supera a
abstenerse. Esto no prueba que USD/COP sea estructuralmente imposible ni que no exista otra
política rentable; prueba que las familias evaluadas no ofrecen evidencia de edge neto.
