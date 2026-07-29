---
kind: roadmap
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md
  - .claude/rules/quant-constitution.md
---

# BL-11 — Familias transversales de hipótesis (registries/families/)

**Fuente**: FABRIC §9.4-§9.5 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
La misma mecánica probada en 3 activos vive hoy en 3 registries que subestiman la deflación cruzada.

## Qué falta exactamente
`registries/families/{family}.yaml` con TODAS las celdas (asset×variant), bar pre-firmado, trials_charged validado vs ledger. Piloto: familia `trend_regime` con celdas SPX/Oro/BTC reales existentes.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación
CI: celdas con trial_id == ledger; cerrar familia por escrito es un estado válido.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_trial_ledger.py -q
verde:   26 passed

muta:    scripts/validation/check_trial_ledger.py — `return []` como primera linea de
         check_families
espera:  2 failed — "una celda que referencia un trial inexistente debe ser rechazada"
         y "cobrar 1 trial menos del que se miro es exactamente la fuga que BL-11 cierra"
```

**Historial honesto**: hasta el 2026-07-28 el validador núcleo de este BL se podía **apagar
entero** sin perder un test, porque los dos que lo cubrían eran `assert check_families(...) == []`
— satisfecho trivialmente por un validador neutralizado.

### Cross-review CODEX — APROBADO (CXD-089, 2026-07-28)

Verificado contra el corte inmutable `cb1241b2`. CODEX **ejecuto las mutaciones**, no leyo el diff:
pristino y restaurado **26/26**, y las tres mutaciones mataron **10, 3 y 2** tests respectivamente,
con SHA de restauracion exacto.

Los 10 de la primera son el test parametrizado por **introspeccion** sobre las funciones `check_*`:
cae un caso por cada check que se desconecte del agregador. Ese numero es la medida de lo que
faltaba — antes de este cierre, `run_all_checks` podia quedarse con **1 de 11 checks** y la suite
seguia entera en verde.

## Notas constitución
Dividir familias para lavar multiplicidad queda visible vía cluster + N_global (decisión rechazada §31).
