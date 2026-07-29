---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - config/strategy_manifests/usdcop.yaml
  - scripts/pipeline/normalize_champions.py
---

# BL-13 — Campo surface en manifiestos/registry + normalize lo respeta

**Fuente**: plan 00 §2-§3 / FABRIC §9.2-§9.3 · **Ola**: 2 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Ni manifiestos ni registry.json declaran surface; la muralla es implícita (por convención de pipelines).

## Qué falta exactamente
`surface: action|diagnostic` en manifiestos + entradas de registry; normalize_champions rechaza CHAMPION para diagnostic; re-freeze consciente de manifiestos (campo aditivo, señal intacta).

## Impacto frontend
Registry API expone surface (futuros filtros de vistas).

## Dependencias
—

## Verificación
Test: entrada diagnostic con status CHAMPION ⇒ CI rojo.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_strategy_manifests.py
                          tests/regression/test_feature_contracts.py -q
verde:   49 passed, 1 xfailed

muta:    src/identity/source_hash.py::canonical_lf  ->  `return data`
         (normalización CRLF->LF eliminada)
espera:  8 failed, 41 passed — DOS ficheros de test distintos caen desde UNA sola línea de
         producción. Es el rojo que demuestra que la circularidad murió.
         restaurado: 49 passed, 1 xfailed

muta-2:  scripts/pipeline/normalize_champions.py — `_frozen_surfaces()` -> {}
muta-3:  quitar el `raise` de surface desconocido
muta-4:  borrar `surface` del registry.json que sirve el dashboard
espera-2/3/4: tres rojos, dos de ellos ejecutando el script end-to-end
              (`test_manifests_declare_action_surface`,
               `test_registry_carries_surface_and_diagnostic_never_visible`,
               `test_diagnostic_champion_forces_red_exit_and_archival`)
```

**Historial honesto**: la mitad `surface` de este BL **mordía de origen** (tres mutaciones,
tres rojos). Lo que **no** mordía es la otra mitad, y el defecto era grave: **la garantía era
CIRCULAR**. `_canonical_lf` y `_sha16` vivían DENTRO de
`tests/regression/test_strategy_manifests.py`; no existía ninguna función de producción que
computara ese hash, así que el test **se verificaba contra su propia implementación** y no
había NADA que mutar. Investigación previa a tocar: nadie escribe ni congela los manifiestos
por código (`normalize_champions.py` lee `surface` y `strategy_id` pero no computa hashes;
los `code_hash` están puestos a mano), y el único sitio de producción con el método era
`validate_feature_catalog.py`. La lógica subió a `src/identity/source_hash.py` (leaf sin
dependencias, junto a `canonical.py` y `fingerprints.py`) y el validador del catálogo DELEGA
ahí conservando sus nombres como alias: **una implementación, dos consumidores**.

**Deuda declarada y no arreglada**: no existe escritor/congelador de manifiestos —los tres
hashes se editan a mano—; `usdcop_v12/v14` declaran un `current_model_snapshot` rotativo SIN
`registered_in`; `forecast_trial_ids_legacy` no está tipado en ningún contrato ni spec; y
`test_feature_contracts.py:112` conserva su propia copia de `_sha16_lf` (tercera copia del
mismo concepto, candidata a delegar en `source_hash`).

## Notas constitución
Re-freeze de manifiesto = bump versión + nota (patrón refreeze_note_v8 existente).
