---
kind: roadmap
status: IMPLEMENTED
version: 1.3.0
last_verified: 2026-08-03
supersedes: []
code_anchors:
  - airflow/dags/asset_pipeline_factory.py
  - config/assets/pipelines.yaml
  - src/orchestration/dataset_uri.py
  - tests/unit/test_codex_fabric_contracts.py
---

# BL-35 — URIs de datasets + arista prohibida forecast→allocator en parseo

**Fuente**: plan 01 §3 / FABRIC §3.4, §9.8 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
No existe convención de URIs (asset://, strategy://, forecast://); la prohibición forecast→allocator no es verificable en el grafo.

## Qué falta exactamente
Esquema de URIs en los Assets de Airflow (shim 2/3); test de parseo: cualquier DAG que consuma forecast://*/prediction hacia book/exec ⇒ falla el parse + CI.

## Remediación lista para revisión cruzada (2026-07-29)

`asset_pipeline_factory._load_config()` distingue ahora indisponibilidad del
fichero o YAML ilegible de una violación del contrato: lo primero conserva el
warning/degradación histórica; cualquier `DatasetContractError` se propaga y
convierte la arista prohibida en error de importación del DAG. El nombre real
del tipo es `DatasetContractError` (la receta de remediación lo llamó
`DatasetEdgeViolation`, clase que no existe en el repositorio).

`config/assets/pipelines.yaml` declara seis aristas reales y no vacías para los
pipelines actuales. Se limitan a transporte `asset:// → artifact:// →
forecast://`; no deciden si Oro/BTC se comercializan como ML o reglas, decisión
de producto aún reservada al operador.

```bash
python -m pytest -q tests/unit/test_codex_fabric_contracts.py -k dataset
# 2 passed, 28 deselected
```

El test carga el módulo DAG real con stubs mínimos de la distribución Airflow,
inyecta un `forecast:// → exec://` y exige el error de contrato. También prueba
un config válido sin `dataset_edges`, para conservar compatibilidad del camino
feliz. Mutaciones:

- quitar el `except DatasetContractError: raise` ⇒ **1 failed / 1 passed**;
- retirar `dataset_edges` del YAML productivo ⇒ **1 failed / 1 passed**.

El 2026-08-03 se ejecutó dentro del scheduler real
`airflow dags list-import-errors`: terminó con exit 0 y salida `No data found`.
Claude cofirmó el resultado y su alcance: demuestra que los DAGs actuales
parsean, no que puedan ejecutarse con el esquema o los datos presentes.

El 2026-08-03 la aceptación se ejecutó también dentro del scheduler real. Un probe temporal con
`forecast://synthetic/prediction/v1 → exec://synthetic/orders/v1` apareció en
`list-import-errors` con `DatasetContractError`; Claude observó independientemente ese traceback
en `CLD-315`. Tras retirar ambos probes, `list-import-errors` volvió a `No data found`, no quedó
ningún DAG sintético en metadata, `git status --short -- airflow/dags` quedó vacío y ambos
filesystem devolvieron cero probes. La aceptación runtime y la limpieza quedan cofirmadas.

## Impacto frontend
Ninguno.

## Dependencias
BL-13; base para BL-28.

## Verificación
DAG sintético violador ⇒ import error visible en list-import-errors.

## Notas constitución
'Falla el parseo de DAGs y el CI' — la muralla en el grafo, no en la disciplina.
