---
kind: roadmap
status: IMPLEMENTED
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

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_trial_ledger.py -q
verde:   26 passed

muta:    scripts/validation/check_trial_ledger.py — `return errors` insertado en
         run_all_checks() tras el primer check
espera:  10 failed — un caso por cada check desconectado del gate
         "check_hash_chain NO esta cableado en run_all_checks(): sus violaciones no
          llegan al exit code del gate"

muta-2:  scripts/validation/check_trial_ledger.py — eliminada la comparacion
         `record["prev_hash"] != prev_hash` de check_hash_chain
espera:  3 failed — borrar filas del medio, insertar una fila fabricada con line_hash
         impecable pero prev_hash ajeno, y reordenar dos contiguas
```

**Historial honesto**: hasta el 2026-07-28 el agregador podía quedarse con **1 de 11 checks**
y la suite seguía verde, porque los ~20 tests llamaban a cada `check_*` directamente y nadie
verificaba el cableado con el `main()` que devuelve el exit 0/1. Y la cadena hash solo
detectaba la **edición** de una fila —que el `line_hash` ya caza solo—: supresión, inserción
y reorden pasaban en silencio.

### Cross-review CODEX — APROBADO (CXD-089, 2026-07-28)

Verificado contra el corte inmutable `cb1241b2`. CODEX **ejecuto las mutaciones**, no leyo el diff:
pristino y restaurado **26/26**, y las tres mutaciones mataron **10, 3 y 2** tests respectivamente,
con SHA de restauracion exacto.

Los 10 de la primera son el test parametrizado por **introspeccion** sobre las funciones `check_*`:
cae un caso por cada check que se desconecte del agregador. Ese numero es la medida de lo que
faltaba — antes de este cierre, `run_all_checks` podia quedarse con **1 de 11 checks** y la suite
seguia entera en verde.

## Notas constitución
Etapa 'irreparable-hacia-atrás': cada activo sumado con N fragmentado es deuda estadística sin refinanciación.
