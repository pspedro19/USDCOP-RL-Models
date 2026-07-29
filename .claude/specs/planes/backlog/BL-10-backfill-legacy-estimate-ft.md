---
kind: roadmap
status: IMPLEMENTED
version: 1.2.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - registries/ledger.jsonl
  - registries/families/usdcop_direction.yaml
  - scripts/validation/check_trial_ledger.py
---

# BL-10 — Backfill legacy_estimate de FT históricos (zoos)

**Fuente**: FABRIC §10.2 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El borrador heredado ya separó 237 asientos (53 FT/184 AT), pero el plan conservaba el
conteo obsoleto COP=88. El SSOT sellado por BL-12-r2 demuestra 111: el cuerpo registra
H1 daily shadow 109→110 y H1 LatAm transport 110→111. El ledger quedó en 109 y dos
celdas `legacy_estimate` no declaraban la etiqueta en su propia nota.

## Implementación cerrada
Se reconcilió sin modelado ni trials nuevos: cada celda backfilled exige
`legacy_estimate`; FT-0054/55 quedaron añadidos al final como trials `documented`
ya cobrados; `usdcop_direction` quedó en 50 y se preservó la igualdad exacta por
activo. HYPOTHESIS-REGISTRY y el EXP-DIR ajeno permanecieron byte-idénticos.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación

Cross-review de CLAUDE contra
`b86083e12d5e0a0b98cbef523fdb913518c253c4` (`CLD-234`):

```bash
python -m pytest tests/regression/test_bl10_legacy_estimate_contract.py \
  tests/regression/test_trial_ledger.py -q
# 28 passed

python scripts/validation/check_trial_ledger.py
# exit 0; 239 globales = 55 FT + 184 AT
# usdcop=111, xauusd=77, btcusdt=34, spx500=17
```

Mutaciones ejecutadas por el revisor:

- `n_trials_total: 111 -> 112` en el SSOT independiente
  HYPOTHESIS-REGISTRY: 2 fallos por divergencia ledger↔registro.
- Desactivar la exigencia `legacy_estimate` en producción: 1 fallo.

El validador usa `check_families()` de producción y conserva la hash-chain
íntegra. Veredicto cruzado: **APROBADO**.

Sellado bilateral final sobre `8005ffea`/`2ec6baa9`:

```bash
python -m pytest tests/regression/test_strategy_manifests.py -q
# 24 passed; cero xfail rancio

python -m pytest tests/regression/test_bl10_legacy_estimate_contract.py \
  tests/regression/test_trial_ledger.py -q
# 28 passed

python scripts/validation/check_trial_ledger.py
# exit 0; 239 = 55 FT + 184 AT; usdcop=111
```

La revisión de CLAUDE sustituyó en la lista real `FT-0048` por `FT-9999`:
**1 failed / 23 passed**; restauración: **24 passed**. Los tres manifiestos COP
citan exactamente el bloque legado derivado `FT-0001..FT-0048`; no se
re-ejecutó ningún backtest ni se fabricó mapeo individual para celdas no
descomponibles.

## Notas constitución
'Un N estimado documentado vale infinitamente más que un N=0 falso'.
