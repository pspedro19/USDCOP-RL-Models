---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - services/common/metrics.py
---

# BL-09 — Ledger doble FT-/AT- global (registries/ledger.jsonl)

**Fuente**: plan 02 §4 / FABRIC §9.6-§9.7 + §28 E1 · **Ola**: 2 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Un HYPOTHESIS-REGISTRY por activo (COP 88, Oro 77, BTC 33, SPX 17) mezcla trials predictivos y económicos; no hay ledger global ni N_cluster/N_global maquinal.

## Qué falta exactamente
`registries/ledger.jsonl` append-only hasheado (una línea por trial FT-xxxx/AT-xxxx con familia, cluster, asset, cutoff, result, N×3). DSR×3 (family/cluster/global) reportados; el gobierno gatea con DSR_family. N_MAX=989 solo como cota de gasto, JAMÁS en el DSR.

## Impacto frontend
Futuro: Control Tower muestra N_global vs N_MAX (BL-32).

## Dependencias
—

## Verificación
CI: `trials_charged` de cada familia == conteo en ledger; suma == n_trials_total de los registries actuales.

## Notas constitución
Etapa 'irreparable-hacia-atrás': cada activo sumado con N fragmentado es deuda estadística sin refinanciación.
