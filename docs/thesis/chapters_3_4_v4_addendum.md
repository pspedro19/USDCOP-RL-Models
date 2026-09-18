---
kind: thesis
status: ready_for_review
version: 1.0.0
last_verified: 2026-09-15
supersedes: docs/thesis/chapters_3_4.md
code_anchors:
  - config/research/thesis_confirmatory_v4.yaml
  - scripts/pipeline/rebuild_thesis_portable_v2.py
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/evaluate_confirmatory_ppo.py
  - scripts/analysis/run_thesis_llm.py
---

# Addendum a los capítulos 3 y 4: protocolo v4 y resultados ejecutados

Este addendum sustituye únicamente las afirmaciones experimentales incompatibles con
el protocolo v4. La memoria histórica se conserva y se etiqueta retrospectiva; no se
reutiliza para afirmar confirmación.

## Capítulo 3. Diseño experimental ejecutado

El activo primario es USD/COP en barras de cinco minutos, zona horaria
`America/Bogota`, sesión 08:00--13:00. La partición firmada es:

| Bloque | Fechas | Papel | Sesiones observadas |
|---|---|---|---:|
| Desarrollo | 2020-01-01--2022-12-31 | ajuste de scaler, HMM y PPO | 488 |
| Selección | 2023-01-01--2023-12-31 | diagnóstico de configuración | 226 |
| Hold-out | 2024-01-01--2025-12-31 | una sola mirada confirmatoria | 420 |
| Forward | 2026-01-01--2026-09-10 | posterior al freeze, lectura exploratoria | 162 |

La identidad del portable v4-stable es
`65534cc4984cd2bb52bd7f42b845fcd5cb7be214123d154b5b8bf6c5ef2eb585`.
Conserva exactamente las observaciones del bundle anterior; solo normaliza las rutas
del manifiesto para que el hash sea estable entre formas de ruta de Windows.
Macro se incorpora con regla estricta T−1, sin backfill, interpolación ni cero-fill.
Las series declaradas son Brent FRED, DGS2 FRED, DXY Investing instrumento 942611 e
IBR BanRep. La coincidencia numérica con las fuentes está verificada; la disponibilidad
histórica por vintage no está completamente demostrada y se conserva como limitación.

Se entrenaron dos familias PPO (`ppo_backbone` y `ppo_regime`) con cinco semillas
preespecificadas (42, 123, 456, 789 y 1337), 300.000 pasos por corrida y receta
`flat_init_no_turn`, validada previamente en fixtures sintéticas S1--S4. El conjunto
de entrenamiento no accedió al hold-out. La liquidación cobra el cierre terminal y
usa el mismo motor de costes para agente y baselines.

El brazo LLM se deja como exploratorio. Se generaron 13.334 contextos de selección y
24.780 contextos de hold-out retrospectivo ligados al hash v4; ninguna llamada nueva
se considera confirmatoria hasta fijar versión exacta del proveedor, prompt y
preregistro específico.

## Capítulo 4. Resultados

### PPO: juez 2024--2025

| Configuración | Retorno total medio | Retorno mediano | Sharpe medio | Semillas positivas |
|---|---:|---:|---:|---:|
| PPO régimen | −40,39 % | −38,17 % | −4,53 | 0/5 |
| PPO backbone | −43,44 % | −45,13 % | −4,20 | 0/5 |

El resultado es consistente entre semillas y no requiere promediar políticas para
obtener el signo negativo. Con el conteo provisional de 115 trials, el DSR de cada
semilla está por debajo de 0,95; ese conteo debe reconciliarse con el registro antes
de la versión final para publicación.

### LLM e híbrido: selección 2023, diagnóstico retrospectivo

| Brazo | Retorno compuesto | Sharpe | Max drawdown |
|---|---:|---:|---:|
| DeepSeek | −78,93 % | −22,78 | −78,82 % |
| Azure | −82,89 % | −14,84 | −82,82 % |
| PPO + DeepSeek | −69,24 % | −14,33 | −69,14 % |
| PPO + Azure | −70,59 % | −11,01 | −70,57 % |

Estas cifras no se mezclan con la tabla confirmatoria PPO: el portable histórico de
los ledgers LLM no coincide con la identidad calculada por HEAD y la liquidación tuvo
que usar una excepción retrospectiva explícita. Sirven para documentar el experimento,
no para declarar una ventaja general del proveedor.

### Criterio de interpretación

El retorno bruto positivo, cuando aparezca, no se denomina alfa por sí solo. Alfa en
esta tesis significa retorno incremental neto, ajustado por riesgo, costes, semillas y
el número total de trials. El resultado v4 no satisface ese criterio: rechaza la
rentabilidad neta de estas recetas bajo el contrato declarado.
