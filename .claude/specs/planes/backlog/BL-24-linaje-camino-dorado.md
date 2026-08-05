---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-05
supersedes: []
code_anchors:
  - airflow/dags/l0_macro_update.py
  - database/migrations/067_spx500_regime_macro_vars.sql
  - src/lineage/macro_revision.py
  - src/lineage/paper_path.py
  - scripts/diagnostics/verify_paper_lineage.py
---

# BL-24 — Linaje nodes/edges + camino dorado + revisiones tipificadas

**Fuente**: FABRIC §22 + §28 E6 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-08-05)
Entrega parcial. La migración 076 y `src/lineage/graph.py` definen el grafo. BL-24(A) ya integra
el writer transaccional de revisiones macro y su sello de verificación; la migración 086 está
aplicada. BL-24(C) añade un verificador persistente fail-closed y una CLI que distingue
`RESOLVED`, `BROKEN` y `ABSENT`.

El ledger servido real devuelve `ABSENT`, `coverage=0`, `verified=false` y exit code 2 porque aún
no declara IDs de linaje por estrategia. Por tanto no existe todavía un camino dorado real que
pueda contarse como verificado y el BL permanece `PARTIAL`.

## Qué falta exactamente
BL-24(B) debe persistir y servir IDs estables de señal, snapshot y barra L0 para al menos una
estrategia real, junto con sus nodos/aristas. Después, la CLI debe pasar de `ABSENT` a `RESOLVED`
contra ese mismo ledger. Una declaración parcial, una arista intermedia ausente o una ruta ambigua
debe producir `BROKEN`.

## Impacto frontend
Passport muestra linaje (BL-32).

## Dependencias
BL-17.

## Verificación
Camino dorado resuelve para 1 señal del paper ledger; LEGITIMATE_RELEASE no marca STALE histórico.

Estado 2026-08-05: el segundo criterio está verificado en PostgreSQL real mediante probe
rollback-only (historia y descendiente conservaron `VALID`; rollback limpio). El primero sigue
abierto: el ledger real devuelve `ABSENT`, no `RESOLVED`. Las pruebas focales del verificador y el
resolvedor suman 16 verdes; mutar `ABSENT` para que cuente como éxito produce dos fallos tanto en
biblioteca como en la CLI.

## Notas constitución
Screening consume as_released por defecto (PIT-correcta).
