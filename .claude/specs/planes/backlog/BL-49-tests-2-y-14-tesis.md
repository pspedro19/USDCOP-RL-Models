---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-08-25
supersedes: []
code_anchors:
  - tests/regression/test_engine_parity_independent.py
  - tests/regression/test_research_features_are_causal.py
  - src/research/inference.py
  - src/research/session_gym.py
---

# BL-49 — Los dos tests de §13 que quedaron sin implementar

**Fuente**: [`planes/06-tesis-rl-llm-hibrido.md`](../06-tesis-rl-llm-hibrido.md) §13 ·
**Ola**: — · **Esfuerzo**: M · **Trials**: 0 (son comprobaciones, no hipótesis)

## Contexto

§13 declara *"Los 16 tests son irrenunciables"*. Al cerrar el brazo PPO quedaron **8 en verde**,
**3 N/A** (5, 11, 12 — caen con el LLM y el híbrido), **2 parciales** (1 y 6, cubiertos de
hecho por otros) y **2 sin implementar**. Estaban declarados como pendientes dentro del
documento y sin dueño en ninguna parte; esto los saca a la luz.

El test 7 sí se implementó, aunque no como decía el plan: en vez de `vectorbt` —que no está
instalado y añadirlo empeora la reproducibilidad sin mejorar lo que el test mide— se usó una
**implementación de referencia independiente** con otra formulación (caja y unidades frente a
pesos y retornos), que coincide a 1e-6 sobre 60 sendas aleatorias e incluye contraprueba.
Ver `tests/regression/test_engine_parity_independent.py`.

## Test 2 — shuffle

Cita literal de §13:

> **Shuffle:** con retornos aleatorizados **no existe ventaja positiva estable**; el desempeño
> es compatible con cero o con always-flat y no hay relación predictiva reproducible. *(No se
> exige que se parezca al control random: un PPO bien regularizado puede aprender a quedarse
> flat.)*

**No hay ninguna implementación reutilizable en el repo.** Lo único parecido es
`src/monitoring/multivariate_drift.py::_mmd_permutation_test`, que permuta pertenencia a grupo
para un p-valor de MMD en detección de drift — sirve como patrón, no como base. Todos los demás
`shuffle` del repo son el parámetro de barajado de `DataLoader`/`SessionTradingEnv`, otra cosa.

La maquinaria que sí encaja está en `src/research/inference.py`
(`stationary_bootstrap_indices`, `_bootstrap_matrix`, `paired_sharpe_test`): es remuestreo por
bloques, no permutación de etiquetas, pero es el módulo del carril donde vive este tipo de
contraste.

**Coste real**: el test exige *entrenar* sobre retornos aleatorizados, o sea 5 semillas más de
PPO (~2 h de reloj en Airflow). No es un test de segundos.

## Test 14 — etiqueta de subperíodo

Cita literal de §13:

> **Etiqueta de subperíodo:** toda celda anterior al hold-out procede de OOF y viene marcada

Y la regla que verifica, §11.9:

> Los subperíodos anteriores al hold-out (COVID 2020, ciclo macro 2022) se reportan
> **únicamente con predicciones OOF** de la CV purgada y etiquetados `OOF`; el hold-out se
> etiqueta `OOS estricto`. Nunca se ponen números in-sample y de hold-out en la misma columna
> sin etiqueta. Si un subperíodo carece de cobertura OOF suficiente, **la celda queda vacía**.

**Riesgo concreto que cubre**: las tablas 4.4 (`tabla_4_4_por_regimen_*.md`) ya se generan por
bloque y con sufijo, pero nada impide que alguien pegue una columna de desarrollo —que es
in-sample— junto a una de hold-out sin marcarla. Con el refit el problema se agrava: en las
corridas `*_refit_*` **selección también es in-sample**, y por eso esos JSON llevan
`in_sample: true` en `development` y `selection`. El test tendría que comprobar que el
generador respeta ese flag.

## Por qué no se hicieron aquí

El 2 cuesta dos horas de entrenamiento y sirve para reforzar un resultado que ya es un rechazo
decidido en los dos bloques (p < 0,0001, 0/10 semillas). El 14 protege una tabla que hoy se
genera automáticamente y sin mezclar bloques. Ninguno cambia una conclusión; los dos cierran
una deuda declarada.
