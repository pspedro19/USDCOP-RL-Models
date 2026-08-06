---
kind: roadmap
status: PARTIAL
version: 1.1.1
last_verified: 2026-08-03
supersedes: []
code_anchors:
  - airflow/dags/asset_pipeline_factory.py
  - airflow/dags/fabric_factories.py
  - config/assets/pipelines.yaml
  - config/assets/fabric_factories.yaml
  - src/orchestration/factories.py
  - src/orchestration/semantic_diff.py
---

# BL-28 — Factories nuevas (data/strategy/forecast) + diff semántico

**Fuente**: FABRIC §13 + §28 E7 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-08-03)

El camino legacy sigue activo, pero el incremento `b18720d1` ya entregó en paralelo:

- especificaciones puras para generadores de data, estrategia, forecast y backfill;
- adapter Airflow con shim `Asset`/`Dataset`, pools, timeouts y aislamiento por sleeve;
- SSOT `fabric_factories.yaml` con separación ACTION/DIAGNOSTIC;
- comparación de bundles mediante `semantic_hash`, ignorando sólo campos volátiles declarados;
- backfill sin schedule y con `as_of` obligatorio.

La batería focal incluye candados directos para que un backfill sin `as_of` lance y para que
campos estructurales no volátiles produzcan hashes distintos. Esto hace falso el estado anterior
`PLANNED`; el BL es `PARTIAL`.

## Qué falta exactamente

- BL-17 ya está `IMPLEMENTED`; queda conectar los productores propios con identidad canónica real.
- Sustituir los `candidate_generator: null` del plan strangler por generadores ejecutables,
  capa por capa y sin apagar el camino legacy.
- Ejecutar ambos caminos contra el stack Airflow y registrar el diff semántico prospectivo.
- Cumplir el criterio E7 durante al menos dos semanas antes de retirar el camino viejo por activo.
- Mantener cualquier backfill en un DAG separado con `as_of` explícito.

## Impacto frontend
Ninguno directo.

## Dependencias
BL-17.

## Verificación

Verificación local ya disponible:

```bash
python -m pytest tests/unit/test_codex_fabric_contracts.py -q
```

Los casos `test_backfill_factory_requires_explicit_as_of` y
`test_semantic_diff_preserves_nonvolatile_factory_structure` cubren explícitamente las dos
propiedades anteriores. El conteo y resultado real de la suite se registra al ejecutar el gate;
no se infiere de una corrida previa.

Verificación de cierre pendiente: diff semántico verde ≥2 semanas antes de apagar el camino
viejo por activo, con productores reales e identidad de BL-17.

## Notas constitución
Backfill SIEMPRE en DAG aparte con as_of explícito.
