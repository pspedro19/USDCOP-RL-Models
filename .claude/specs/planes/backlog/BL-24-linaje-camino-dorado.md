---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-08-05
supersedes: []
code_anchors:
  - airflow/dags/l0_macro_update.py
  - database/migrations/067_spx500_regime_macro_vars.sql
  - src/lineage/macro_revision.py
  - src/lineage/paper_writer.py
  - src/lineage/paper_path.py
  - src/forecasting/dataset_loader.py
  - scripts/pipeline/candidates_paper_ledger.py
  - scripts/diagnostics/verify_paper_lineage.py
  - usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json
---

# BL-24 — Linaje nodes/edges + camino dorado + revisiones tipificadas

**Fuente**: FABRIC §22 + §28 E6 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-08-05)
Entrega parcial. La migración 076 y `src/lineage/graph.py` definen el grafo. BL-24(A) ya integra
el writer transaccional de revisiones macro y su sello de verificación; la migración 086 está
aplicada. BL-24(C) añade un verificador persistente fail-closed y una CLI que distingue
`RESOLVED`, `BROKEN`, `ABSENT` y `UNAVAILABLE`.

BL-24(B) está aprobado bilateralmente (CLD-535/536): el ledger servido declara para una fila real
de `smart_simple_v11` IDs content-addressed de señal, snapshot consumido y barra L0. PostgreSQL
resuelve un camino único `paper_signal -> data_snapshot -> bar_l0` con `coverage=1` y
`verified=true`. El BL global permanece `PARTIAL` hasta que Claude y Codex auditen conjuntamente
si A+B+C satisfacen el cierre completo; este incremento no se convierte unilateralmente en flip.

## Qué falta exactamente
Auditar conjuntamente el cierre global después de A+B+C. B ya persiste y sirve IDs estables para
una señal real; la CLI pasa a `RESOLVED` contra el mismo ledger. Una declaración parcial, cero o
dos filas coincidentes, una arista ausente o una ruta ambigua producen `BROKEN`; una base no
alcanzable produce `UNAVAILABLE`, no un falso defecto de linaje.

## Impacto frontend
Passport muestra linaje (BL-32).

## Dependencias
BL-17.

## Verificación
Camino dorado resuelve para 1 señal del paper ledger; LEGITIMATE_RELEASE no marca STALE histórico.

Estado 2026-08-05: el segundo criterio está verificado en PostgreSQL real mediante probe
rollback-only (historia y descendiente conservaron `VALID`; rollback limpio). El primero también
resuelve en PostgreSQL real para v11 (`RESOLVED`, coverage 1); v12/v14 permanecen honestamente
`ABSENT`. Las mutaciones de timestamp 0/2, side e IDs falsos muerden. El trainer congelado quedó
byte-idéntico y los muros de manifiestos y catálogo están verdes; el re-registro 2.0.1 declara
provenance aditiva sin cambio de features y 0 trials.

## Notas constitución
Screening consume as_released por defecto (PIT-correcta).
