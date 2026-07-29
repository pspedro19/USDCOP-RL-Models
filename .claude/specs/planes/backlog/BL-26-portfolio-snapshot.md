---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - scripts/analysis/book_construction.py
  - config/book/book_v1.yaml
  - src/portfolio/snapshot.py
  - tests/unit/test_codex_fabric_contracts.py
---

# BL-26 — portfolio_snapshot: barrera temporal del libro

**Fuente**: FABRIC §14.1 · **Ola**: 5 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
book_construction usa pesos ERC estáticos sobre trades históricos; no existe snapshot con cutoff ni políticas de faltante.

## Qué falta exactamente
Contrato + builder: cutoff explícito, accepted/stale/missing, max_age por sleeve, política de faltante declarada (USE_LAST_VALID sin max_age PROHIBIDO).

## Remediación lista para revisión cruzada (2026-07-29)

`PortfolioSnapshot` valida de nuevo su identidad al reconstruirse: exige cutoff
con zona horaria, calcula el hash semántico desde todos los inputs
materializados y exige que `snapshot_id` sea exactamente el UUIDv5 derivado.
El snapshot conserva en su identidad tanto `max_age` como la política de
faltante de cada sleeve. `USE_LAST_VALID_WITH_MAX_AGE` rechaza además una señal
que ya haya expirado al cutoff.

```bash
python -m pytest -q tests/unit/test_codex_fabric_contracts.py -k snapshot
# 5 passed, 24 deselected
```

Mutaciones ejecutadas y restauradas:

- omitir la recomputación del hash semántico: **1 failed / 4 passed**;
- omitir la comprobación del UUIDv5: **1 failed / 4 passed**;
- aceptar y normalizar un cutoff naïve: **1 failed / 4 passed**;
- omitir `fallback_signal.valid_until < cutoff`: **1 failed / 4 passed**.

El fichero completo conserva un fallo ajeno a BL-26 en el contrato de
`MetricEngine(annualization_by_asset=...)`; el carril focal de snapshot queda
verde y las cuatro mutaciones anteriores demuestran causalidad. El BL no se
cuenta como cerrado hasta revisión bilateral.

## Impacto frontend
Control Tower muestra el snapshot vigente.

## Dependencias
BL-15 (tipos), BL-17.

## Verificación
Libro con señal COP de hoy + Oro de ayer ⇒ rechazado sin políticas declaradas.

## Notas constitución
'La señal de hoy de SPX + la de ayer de Oro no es un libro, es una foto movida'.
