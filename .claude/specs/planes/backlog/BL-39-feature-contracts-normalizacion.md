---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/core/contracts/feature_contract.py
  - scripts/pipeline/train_and_export_smart_simple.py
  - src/forecasting/enhance_v2.py
---

# BL-39 — Feature contracts por estrategia-versión + normalización al artefacto

**Fuente**: Plan Consolidado §1.2-1.4 / DATA-STRATEGY §40-48 (D4) · **Ola**: 2-3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
config.feature_definitions (30) mezcla definición+normalización+código: z-scores con media/sigma HARDCODEADAS ((vix-21.16)/7.89 — dependen del período de entrenamiento), python_function/sql_formula como strings no autoritativos, source_table apuntando a tablas por renombrar. El contrato de 20 del RL es el patrón correcto a generalizar.

## Qué falta exactamente
Catálogo estable (feature_id, causality_policy, source_contract, transformation, lookback, code_reference+code_hash) en Git; feature_set(strategy_version, feature_id, order, required) por estrategia; normalization_snapshot_id → artefacto MinIO/MLflow (mean, std, training_cutoff, semantic_hash). La matriz por estrategia de §41-47 se convierte en fixture de CI (v11 = 25 feats; las rule-based declaran su set mínimo — MA200 solo close).

## Impacto frontend
La vista SHAP admin (BL-20) consume el feature_set versionado.

## Dependencias
Coordina con BL-14 (components) y BL-16. NO toca v11 en runtime: el snapshot actual se registra como legacy_v1 bit-idéntico.

## Verificación
Reproducir la señal v11 de la última semana desde feature_set+snapshot == bit-check; CI rechaza features sin causality_policy o sin prior de signo (Anexo A.4).

## Notas constitución
Cambiar UNA constante de normalización tras reentrenar sin snapshot versionado = leakage silencioso — exactamente lo que este BL elimina.
