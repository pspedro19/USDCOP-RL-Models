---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - database/migrations/075_fact_position_pnl.sql
  - scripts/data/backfill_catalog_facts.py
  - src/metrics/engine.py
  - scripts/analysis/timing_ratio_oneoff.py
  - tests/unit/test_codex_safety_contracts.py
---

# BL-22 — fact_position / fact_pnl + identidad contable + timing_ratio persistido

**Fuente**: FABRIC §18 + §28 E3 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
PnL vive en bundles JSON por estrategia; no hay tablas de hechos ni descomposición beta/timing/carry.

## Qué falta exactamente
DDL §18.1 (env en PK: paper y live conviven), identidad contable como test de CI (|residual|/|gross| ≤ tol), timing_ratio con IC como métrica del catálogo.

## Impacto frontend
Production/Passport leen facts (v11-live vs v12-paper = una query).

## Dependencias
BL-17, BL-18, BL-21.

## Verificación
Identidad contable verde sobre el paper ledger anclado.

## Estado real verificado 2026-07-29

La migración 075 ya declara `fact.position`, `fact.pnl`, la vista agregada y
`fact.assert_pnl_identity`; el backfill legacy escribe hechos sin inventar
atribución (gross y residual `UNATTRIBUTED`). Esto todavía no equivale al cierre
integral.

La receta de revisión que pedía mover un paréntesis era incorrecta. Desde
`b18720d1`, `identity_error` ya es:

```text
ABS(calculated_residual - reported_residual)
```

No se cambió el DDL para hacer coincidir el código con una lectura equivocada.
En su lugar, el test nuevo extrae y **ejecuta la expresión de producción** sobre
dos casos con `calculated_residual = -5`:

- `reported_residual = -3` => error `2`;
- `reported_residual = 3` => error `8`.

Las expresiones `ABS(calculated) - reported` y
`ABS(calculated - reported)` intercambian esos resultados, por lo que el
oráculo no puede pasar por coincidencia de subcadenas.

```bash
python -m pytest tests/unit/test_codex_safety_contracts.py -k pnl_identity -q
# 2 passed, 17 deselected
```

Mutaciones ejecutadas:

1. cerrar `ABS` antes de restar el residual reportado:
   **1 failed / 1 passed** (`8 != 2`);
2. neutralizar la identidad con el prefijo `0 * ABS(...)`:
   **1 failed / 1 passed** (`0 != 2`).

El primer extractor del test empezaba dentro de `ABS` y el segundo mutante
habría sobrevivido; la auto-revisión movió el límite al inicio de la expresión
completa y volvió a ejecutar el mutante rojo. La migración 075 quedó restaurada
sin diff, SHA-256
`F02A51D90BC828DBF9F347FD518B67525B88FAE29E81AAA6C924975B9FE8144F`.

### Por qué sigue PARTIAL

- `strategy.timing_ratio` existe en el catálogo y el motor, pero el motor
  gobernado devuelve sólo el escalar: no publica intervalo de confianza.
- `timing_ratio_oneoff.py` sí calcula un IC por bloques, pero su propio payload
  declara `persisted: false`; no satisface “timing_ratio con IC persistido”.
- No se aplicó el plan Fabric a PostgreSQL vivo ni se ejecutó la identidad sobre
  el paper ledger anclado. Las pruebas PostgreSQL/CI amplias permanecen
  diferidas por orden del operador.

## Notas constitución
El PnL operativo NO es el NAV legal (plan 03 §3.4).
