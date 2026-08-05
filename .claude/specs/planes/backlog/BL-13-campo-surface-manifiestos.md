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
verde:   48 passed, 2 skipped   (2026-08-05; los 2 skips son los artefactos H5
         gitignored, no un criterio esquivado)

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

**El verde de esta ficha fue FALSO durante una semana (medido 2026-08-05).** La cifra
declarada arriba se midió el 2026-07-28 y dejó de ser cierta **al día siguiente**: el commit
`73f8c9b0` (ON CONFLICT en los UPSERT H5) y `8f783d89` (bloque `governance:` de BL-16 en
`smart_simple_v1.yaml`) hicieron drift de los ficheros congelados, y la suite estuvo en
`4 failed, 44 passed` del 2026-07-29 al 2026-08-05 sin que nadie la volviera a correr. Nadie
mintió: la ficha decía la verdad **el día que se escribió**. La lección es que una cifra de
verde con fecha es evidencia que **caduca**, y esta ficha la citaba como si no. Cerrado con el
re-freeze consciente `4ed4a673` (0 trials: ninguna de las dos causas toca señal, leverage,
stops, TP/HS, PnL ni features; autorizado por el operador).

**Re-verificación del propio candado, hoy y no por cita (2026-08-05):** repetir el ataque es
lo que vale, así que se re-ejecutó la mutación en vez de confiar en el registro previo —
`_frozen_surfaces() -> {}` en `scripts/pipeline/normalize_champions.py` ⇒ **2 failed, 22
passed** (`test_diagnostic_champion_forces_red_exit_and_archival` +
`test_frozen_yaml_invalid_surface_exits_red_in_both_modes`), uno de ellos ejecutando el script
end-to-end; restaurado con `git checkout --` ⇒ árbol byte-idéntico. El candado de `surface`
sigue vivo un mes después de escribirse.

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

**Reparto de la deuda (2026-08-05) — para que el cierre no sea "declarar el hueco fuera de
alcance".** Los cuatro puntos de abajo se encontraron HACIENDO este BL, pero ninguno es un
criterio de este BL (`surface` en manifiestos + registry, normalize lo respeta, re-freeze
consciente). Cada uno queda con dueño explícito en vez de disolverse:

| Deuda | Dueño | Por qué no es de BL-13 |
|---|---|---|
| No hay escritor/congelador de manifiestos (hashes a mano) | BL-14 (CLAUDE) | El muro SÍ caza el drift a mano — verificado hoy por mutación; falta la ergonomía, no la garantía |
| `usdcop_v12/v14`: `current_model_snapshot` sin `registered_in` | BL-14 (CLAUDE) | Es linaje de componente, el objeto de BL-14 |
| `forecast_trial_ids_legacy` sin tipar en ningún contrato | BL-12 (CLAUDE, cerrado) → arrastre a BL-14 | Contabilidad FT/AT, no superficie |
| `test_feature_contracts.py:112` conserva su copia de `_sha16_lf` (3ª copia) | **CODEX, dentro de C032** | Es exactamente el defecto que BL-13 mató (una implementación, N consumidores) y C032 ya declara ese fichero en su impact map: arreglarlo ahí evita colisión de leases |

**Asimetría descubierta el 2026-08-05, declarada y NO parcheada**: en `usdcop.yaml` el bloque
`files:` cubre sólo los dos `.py`, así que el hash a nivel de manifiesto **no** cubre
`config/execution/smart_simple_v1.yaml`; una mutación económica real (`hard_stop_max_pct`
0.03→0.09) la caza **únicamente** el `spec_fingerprint` del componente. Si alguien borrase el
bloque `components:` de v11, el stop máximo se podría triplicar sin poner rojo nada.
`usdcop_v12`/`v14` sí incluyen su propio YAML en `files:`. Es asimetría de contrato, no de este
BL: pendiente de veredicto de CODEX en el cross-review de `4ed4a673`.

**Deuda declarada y no arreglada**: no existe escritor/congelador de manifiestos —los tres
hashes se editan a mano—; `usdcop_v12/v14` declaran un `current_model_snapshot` rotativo SIN
`registered_in`; `forecast_trial_ids_legacy` no está tipado en ningún contrato ni spec; y
`test_feature_contracts.py:112` conserva su propia copia de `_sha16_lf` (tercera copia del
mismo concepto, candidata a delegar en `source_hash`).

## Notas constitución
Re-freeze de manifiesto = bump versión + nota (patrón refreeze_note_v8 existente).
