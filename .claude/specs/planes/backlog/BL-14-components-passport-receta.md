---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - config/strategy_manifests/usdcop.yaml
  - scripts/pipeline/train_and_export_smart_simple.py
---

# BL-14 — Bloque components: receta congelada del predictor de v11

**Fuente**: plan 02 §1 / FABRIC §16 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
code_hash global del manifiesto cubre fuentes, pero no distingue el componente Ridge/BR ni registra sus model_snapshots semanales.

## Qué falta exactamente
Bloque `components:` (component_id, role=decision_input, spec_fingerprint de la RECETA, retrain_policy=weekly_expanding, current_model_snapshot rotativo, forecast_trial_ids heredados). CI: 'la receta está congelada; todo snapshot queda registrado' — nunca 'los pesos no cambian'.

## Impacto frontend
Passport (BL-32) lo muestra.

## Dependencias
BL-12 (herencia FT).

## Verificación
Manifest test extendido; snapshot semanal nuevo aparece con linaje.

## Notas constitución
Precisión FABRIC §16: v11 reentrena cada domingo — lo congelado es la receta.
