---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - scripts/analysis/book_construction.py
  - config/book/book_v1.yaml
  - database/migrations/077_portfolio_control.sql
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

Corrección de evidencia: el diagnóstico inicial que atribuyó el fallo del
fichero completo a `MetricEngine(annualization_by_asset=...)` se tomó sobre un
`engine.py` modificado y no pertenecía al commit. En checkout limpio de
`bfe3adeb` ese test pasa. El único rojo limpio era el `relative_to` del test de
inventario de backfill contra un `tmp_path` fuera del repo, y fallaba igual en
el padre; era ajeno, pero la causa publicada inicialmente fue incorrecta.

## R2 tras revisión adversarial CLD-250

La revisión independiente reprodujo las cuatro mutaciones anteriores y además
corrió los candados nuevos contra el padre: **2 failed / 3 passed**. Aprobó el
candado de identidad, pero construyó tres forjas autoconsistentes que lo
eludían: señal disponible después del cutoff, señal aceptada 400 días vieja
con `max_age=2d`, y sleeve simultáneamente ACCEPTED y missing.

La R2 revalida ahora la barrera temporal al rehidratar, antes de aceptar el
hash: `as_of`, `available_at`, `valid_until`, `max_age`, correspondencia exacta
señal↔materialización y partición accepted/stale/missing/fallback. Los errores
de hash/UUID ya no revelan el valor esperado.

```bash
python -m pytest -q tests/unit/test_codex_fabric_contracts.py -k snapshot
# 7 passed, 26 deselected
```

Mutaciones R2 ejecutadas y restauradas:

- retirar `available_at <= cutoff` ⇒ **1 failed / 6 passed**;
- retirar el límite `max_age` de señales aceptadas ⇒ **1 failed / 6 passed**;
- retirar la partición accepted/stale/missing ⇒ **1 failed / 6 passed**;
- volver a filtrar el hash esperado en el error ⇒ **1 failed / 6 passed**;
- restaurar `DEFAULT gen_random_uuid()` en DDL ⇒ **1 failed / 6 passed**.

La persistencia 077 queda alineada: exige UUIDv5(namespace URL, hash), guarda
`missing_policy_by_sleeve`, valida cutoff/expiración/max_age y bloquea un
upgrade poblado que intentara inventar políticas ausentes. El plan
`fabric-v1` permanece cerrado por divergencia de digest hasta revisión del
lote completo.

El BL sigue `PARTIAL`: no hay consumidor productivo de
`PortfolioSnapshot`, `book_construction.py` aún no lo usa y falta ejecutar la
semántica DDL sobre PostgreSQL real. Ninguno de esos huecos se cuenta como
cerrado por los candados locales.

## Impacto frontend
Control Tower muestra el snapshot vigente.

## Dependencias
BL-15 (tipos), BL-17.

## Verificación
Libro con señal COP de hoy + Oro de ayer ⇒ rechazado sin políticas declaradas.

## Notas constitución
'La señal de hoy de SPX + la de ayer de Oro no es un libro, es una foto movida'.
