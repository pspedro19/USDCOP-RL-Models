---
kind: analysis
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-11
supersedes: []
code_anchors:
  - scripts/diagnostics/summarize_ppo_runs.py
  - scripts/analysis/thesis_statistics.py
  - scripts/presentation/generar_resultados_y_figuras.py
  - outputs/thesis-repair/ppo_v2_diagnostic_summary.json
  - outputs/thesis-repair/results_v2_diagnostic/statistics_selection.json
---

# EXP-TESIS-RL-02 — resultado PPO v2 diagnóstico

## Alcance y clasificación

Este informe presenta una matriz completa de **10 corridas diagnósticas retrospectivas**:
`ppo_regime` y `ppo_backbone`, cinco semillas (`42, 123, 456, 789, 1337`), 300.000 pasos
por corrida, entrenadas sobre desarrollo v2 y evaluadas en la selección 2023 (226 sesiones).
La matriz se inició antes de la firma del prerregistro v3 y, por constitución, permanece
clasificada como `diagnostic_retrospective`; no es el juez forward confirmatorio.

Artefactos de autoridad:

- [Resumen por semilla](../../outputs/thesis-repair/ppo_v2_diagnostic_summary.json)
- [Estadística de selección](../../outputs/thesis-repair/results_v2_diagnostic/statistics_selection.json)
- [Coherencia de resultados](../../outputs/thesis-repair/results_v2_diagnostic/coherencia_selection.json)
- [Tabla de desempeño](../../outputs/thesis-repair/results_v2_diagnostic/tablas/tabla_4_3_desempeno_selection.md)
- [Tabla de semillas](../../outputs/thesis-repair/results_v2_diagnostic/tablas/tabla_4_10_semillas_selection.md)

Los diez artefactos declaran `dataset_version=v2`, `timesteps=300000` y el portable v2. La
sanidad S1–S4 fue previa y sintética; no cuenta trials de mercado.

## Resultados por semilla

Retornos son compuestos netos del contrato de costos; Sharpe usa la anualización congelada
de 221 sesiones. Los valores de selección son los que se interpretan; desarrollo se muestra
solo para diagnosticar generalización.

| Configuración | Semilla | Desarrollo | Selección | Sharpe selección | Operaciones selección |
|---|---:|---:|---:|---:|---:|
| PPO régimen | 42 | −2,67 % | −24,44 % | −2,944 | 926 |
| PPO régimen | 123 | +4,40 % | −19,06 % | −2,027 | 593 |
| PPO régimen | 456 | +2,83 % | −34,66 % | −4,026 | 863 |
| PPO régimen | 789 | −3,15 % | −21,70 % | −2,324 | 1.109 |
| PPO régimen | 1337 | +9,50 % | −32,87 % | −3,671 | 534 |
| PPO backbone | 42 | +3,75 % | −37,90 % | −5,723 | 1.000 |
| PPO backbone | 123 | +15,67 % | −27,98 % | −3,578 | 827 |
| PPO backbone | 456 | +8,86 % | −21,83 % | −3,619 | 641 |
| PPO backbone | 789 | −3,91 % | −32,28 % | −4,763 | 1.265 |
| PPO backbone | 1337 | +4,69 % | −47,15 % | −5,033 | 1.377 |

Ninguna de las diez semillas fue positiva en selección. La media de semillas, que no se
interpreta como una política ejecutable individual, fue:

| Configuración | Retorno selección | Sharpe | Max drawdown |
|---|---:|---:|---:|
| PPO régimen | −26,61 % | −4,36 | −28,04 % |
| PPO backbone | −33,83 % | −7,78 | −34,23 % |
| Always-flat | 0,00 % | 0,00 convencional | 0,00 % |
| B1 pasivo | −20,15 % | −1,46 | −22,94 % |

## Contrastes y multiplicidad

El bootstrap estacionario pareado (10.000 réplicas, bloques 5–20) produce:

- `ppo_regime − always_flat`: ΔSharpe −4,360, IC95 % [−6,890; −2,362], p=0,0002.
- `ppo_backbone − always_flat`: ΔSharpe −7,782, IC95 % [−10,473; −5,381], p=0,0002.
- `ppo_regime − B1_pasivo`: ΔSharpe −2,899, IC95 % [−5,141; −0,860], p=0,0062.
- Ablación `ppo_regime − ppo_backbone`: ΔSharpe +3,422, IC95 % [+1,983; +4,993], p=0,0002.

El contraste de ablación indica que la configuración con régimen fue menos mala que backbone
en esta selección, no que sea rentable. DSR con `N=115` trials heredados/registrados es 0,0000
para ambas familias; ninguna supera el umbral 0,95. El PBO calculado sobre las 10 columnas de
semilla/configuración es 0,137; se reporta como diagnóstico del conjunto evaluado, no como
prueba de selección de hiperparámetros. White Reality Check y Hansen SPA se omiten porque el
universo no contiene un procedimiento de HPO comparable; el JSON declara la omisión.

Como control de generalización, en desarrollo las medias fueron +2,52 % (Sharpe +0,23) para
`ppo_regime` y +5,87 % (Sharpe +0,82) para `ppo_backbone`; sus DSR fueron 0,4064 y 0,7479,
respectivamente, ambos por debajo de 0,95. El PBO de desarrollo fue 0,504 y el informe lo
clasifica como `REJECT` por la tendencia del ganador in-sample a perder fuera de muestra. Estos
valores no se usan para elegir una receta: documentan la brecha desarrollo→selección.

## Figuras generadas desde datos reales

Todas están bajo [figuras v2 diagnósticas](../../outputs/thesis-repair/results_v2_diagnostic/figuras/):

- [Curvas de capital](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig01_curvas_capital_selection.png)
- [Underwater/drawdown](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig02_underwater_selection.png)
- [Sharpe móvil](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig03_sharpe_movil_selection.png)
- [Sensibilidad de costos](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig04_sensibilidad_costos_selection.png)
- [Acciones por régimen](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig05_acciones_por_regimen_selection.png)
- [Dispersión entre semillas](../../outputs/thesis-repair/results_v2_diagnostic/figuras/fig06_semillas_selection.png)

El generador verificó: 226 sesiones consistentes, 10/10 corridas presentes, `always_flat`
exactamente cero, DSR usando `N=115` y omisión SPA/Reality Check declarada.

## Conclusiones científicas permitidas

1. En esta representación v2 y bajo este contrato de costos, ninguna de las diez políticas
   PPO diagnosticadas supera a abstenerse en la selección 2023. El rechazo de rentabilidad de
   **esta receta y este bloque** está sobredeterminado por semillas y por bootstrap.
2. El régimen mejora estadísticamente a backbone en la selección, pero la mejora es relativa:
   ambas configuraciones tienen retorno y Sharpe negativos. No demuestra que el HMM aporte alfa.
3. El patrón desarrollo positivo/selección negativo en varias semillas es consistente con
   sobreajuste temporal y/o suboptimización; no permite atribuir causalmente el mecanismo sin
   experimentos de latencia, surrogate y descomposición adicionales.
4. La sanidad sintética demuestra que el entorno puede resolver controles conocidos y abstenerse
   ante ruido/costo; no transfiere rentabilidad al mercado USD/COP.
5. No es válido afirmar que una política rentable no existe en el espacio de acciones: `flat`
   existe y la evidencia solo muestra que estas dos recetas no la superaron.
6. No hay evidencia todavía sobre DeepSeek, Azure OpenAI ni el híbrido. Sus ledgers y resultados
   deben generarse por separado; no se puede completar esa conclusión con PPO solamente.

## Limitaciones y siguiente juez

La matriz es retrospectiva porque se ejecutó antes de la firma; el juez confirmatorio sigue
siendo forward desde el freeze, con el tamaño y una sola mirada definidos en el prerregistro.
Los costos intradía siguen siendo un contrato supuesto/cotas, no quotes bid/ask de venue. La
conclusión de esta página es por tanto: **PPO v2 no encontró rentabilidad neta en selección**,
no “USD/COP es estructuralmente imposible” y no “la tesis queda probada para LLM”.
