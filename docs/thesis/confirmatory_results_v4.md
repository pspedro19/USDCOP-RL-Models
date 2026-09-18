---
kind: analysis
status: published
version: 1.0.0
last_verified: 2026-09-15
supersedes: docs/thesis/confirmatory_protocol_v4.md
code_anchors:
  - scripts/analysis/evaluate_confirmatory_ppo.py
  - scripts/analysis/report_confirmatory_ppo.py
  - scripts/analysis/check_confirmatory_artifacts.py
---

# Resultados confirmatorios PPO v4

Este informe fija lo que puede afirmarse con el paquete reproducible de PPO v4. El
entrenamiento se hizo únicamente sobre desarrollo (2020--2022), la selección 2023 se
usó para el diagnóstico descrito en el protocolo y el juez reservado es 2024--2025
(420 sesiones). El dataset portable utilizado por entrenamiento y evaluación tiene
SHA-256 `65534cc4984cd2bb52bd7f42b845fcd5cb7be214123d154b5b8bf6c5ef2eb585` (portable v4-stable).
Este portable conserva exactamente las mismas observaciones; solo normaliza las rutas
del manifiesto de identidad.

## Evidencia de ejecución

Se conservaron diez corridas independientes (dos configuraciones, cinco semillas),
cada una con 300 000 pasos efectivos, `identity_unchanged=true` y 226 sesiones de
selección. La compuerta `check_confirmatory_artifacts.py` verifica estos invariantes,
los 420 días del hold-out y la versión del informe estadístico.

## Juez 2024--2025

| Configuración | semillas positivas | retorno total medio | retorno total mediano | Sharpe medio | Sharpe mediano |
|---|---:|---:|---:|---:|---:|
| PPO régimen | 0/5 | −40,39 % | −38,17 % | −4,53 | −4,41 |
| PPO backbone | 0/5 | −43,44 % | −45,13 % | −4,20 | −3,95 |

Los retornos son compuestos y netos del contrato de costes. La dispersión entre
semillas se reporta explícitamente; no se presenta una cartera de semillas como si
fuera una ejecución única.

## Inferencia y límites

El DSR se calculó con 115 trials, que es el conteo heredado del registro y debe
reconciliarse antes de usarlo como cifra final de publicación. Con ese conteo, ninguna
semilla supera el umbral de 0,95: los DSR son esencialmente cero. Los contrastes de
regimen frente a backbone y frente a flat en el JSON son bootstrap exploratorios, no
confirmatorios, porque la selección 2023 ya fue observada y el número de semillas es
pequeño.

Por tanto, el resultado confirmatorio defendible es negativo para esta receta y este
contrato: no se encontró rentabilidad neta en USD/COP. Esto no prueba que ninguna
política rentable exista en el espacio de acciones. Sí descarta atribuir a esta versión
de PPO una señal operativa robusta bajo el protocolo v4.

## Figuras reproducibles

Las figuras se generaron directamente de `ppo_holdout_2024_2025_stable.json`:

- [Capital](../../outputs/thesis-repair/confirmatory_v4_stable/figures/01_capital_holdout_v4.png)
- [Drawdown](../../outputs/thesis-repair/confirmatory_v4_stable/figures/02_drawdown_holdout_v4.png)
- [Variabilidad por semilla](../../outputs/thesis-repair/confirmatory_v4_stable/figures/03_seed_variability_v4.png)
- [Stress de costes](../../outputs/thesis-repair/confirmatory_v4_stable/figures/04_cost_stress_v4.png)

Estas figuras cubren PPO v4; no representan todavía resultados confirmatorios de
DeepSeek, Azure ni del híbrido. Esos brazos requieren su propio freeze, ledger,
conteo de trials y juez forward antes de incorporarse a las conclusiones de la tesis.
