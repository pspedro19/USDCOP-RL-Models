---
kind: roadmap
status: PARTIAL
version: 1.0.2
last_verified: 2026-08-05
supersedes: []
code_anchors:
  - config/features/feature_catalog.yaml
  - scripts/validation/validate_feature_catalog.py
  - tests/regression/test_feature_contracts.py
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

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_feature_contracts.py -q
verde:   24 passed, 2 skipped   (2026-08-05; NO "26 passed" — ver más abajo)

muta:    quitar el .shift(1) de un feature macro (fuga temporal T-1 pura)
espera:  4 failed — 2 muros de hash + los 2 muros nuevos de CAUSALIDAD

muta-2:  RE-REGISTRAR el hash del catálogo con la fuga dentro
         (5 ocurrencias, b601ae271c8e2b5f -> 45b16e0da0928467)
espera-2: 2 failed, 24 passed — los dos muros de hash VUELVEN A VERDE con la fuga dentro
         y los dos tests de causalidad SIGUEN ROJOS.
         Restaurado: producción y catálogo sin diff.
```

**Corrección 2026-08-05: tampoco hay evidencia LOCAL, y el número de arriba estaba mal.** Más
abajo esta ficha ya declaraba que el bit-check "skipea siempre en CI" y lo llamaba *evidencia
local-only*. Medido: **no existe esa evidencia local**. La corrida real aquí es
**24 passed, 2 skipped** —no las `26 passed` que esta ficha publicaba—, y los dos `skipped` son
`test_disk_artifacts_are_bit_identical_to_frozen_registration` y
`test_bitcheck_v11_signal_from_feature_set_plus_snapshot`, o sea la primera línea de esta
Verificación. No hay artefactos H5 **ni en el working tree ni dentro del contenedor de Airflow**
(`find /opt -name feature_cols_h5.json` ⇒ 0 resultados), y la metadata de Airflow sólo registra
ejecuciones de tres DAGs (`control_system_health`, `core_l0_01_ohlcv_backfill`,
`rbac_entitlements_daily`): H5-L3 nunca ha entrenado en este entorno.

La diferencia importa: "skipea en CI pero se verifica en local" describe una evidencia que existe
en otro sitio; "no hay artefactos en ningún sitio" describe una que **no se ha producido nunca**.
Lo que sostiene el `PARTIAL` es la mitad **contractual** del criterio —catálogo, priors de signo,
causalidad, muros de hash y la mutación de look-ahead—, que sí corre. Lo que convertiría el
bit-check en evidencia es una corrida real de `train_and_export_smart_simple.py` que produzca los
artefactos congelados; entonces se publica `26 passed` **medido**. Hasta entonces el skip se
declara y no se cuenta como verde (`K-051`).

### C032 — identidad por activo y serie física (2026-08-05)

C032 cerró el falso verde medido por CLD-503: la resolución global por `feature_id` ligaba
`close` de BTCUSDT, XAUUSD y SPX500 al contrato USDCOP en `cop_per_usd`. El catálogo v2 exige
`(asset_id, feature_id)`, `series_id`, fuente de mercado discriminada por activo y resolución
exact-one de todos los feature sets. DXY/WTI/VIX/UST10Y-2Y usan el `canonical_name` del SSOT macro;
la paridad física cubre unidad/fuente/transformación/código, pero excluye correctamente el prior
por consumidor y la materialización local `asbuilt_source`.

TDD medido: antes de implementar, **6 failed / 24 passed / 2 skipped**. Después: suite focal
**31 passed / 2 skipped** y validador CLI verde sobre el catálogo real. El estado sigue `PARTIAL`:
C032 no fabrica los artefactos H5 ausentes y los dos bit-checks continúan en skip explícito.

**Historial honesto**: hasta el 2026-07-28 el look-ahead **solo se detectaba por DRIFT DE
HASH**. Quitar el `.shift(1)` de un feature macro daba rojos que decían *"drifted from the
registered catalog hash"*; **re-registrando el hash, la suite volvía a VERDE con la fuga
dentro**. Es decir: ningún test detectaba el look-ahead por sí mismo. El muro nuevo es
semántico e independiente del hash: frame `MACRO_DAILY_CLEAN` sintético con UN salto de nivel
en T, y se exige `feature[T] == spread[T-1]` (el salto es invisible), `feature[T+1] ==
spread[T]` (la feature no está muerta) y la igualdad T-1 en TODAS las barras. Un tercer test
es un meta-assert: la sección **no puede leer** `_sha16` / `sha256_16` / `_catalog()`, o sea
que no puede volver a apoyarse en el hash.

**Cobertura declarada, no total** (escrito en el propio fichero, no es olvido): cubre
`rate_diff_ibr_ust2y` y `term_spread`. NO cubre `usdmxn/usdclp_ret_1d_lag`
(`include_xlead` default `False`, experimento pre-registrado) ni los cuatro `*_close_lag1` de
`dataset_loader.py` (el lag vive en otro módulo y necesita su propio fixture).

**La otra mitad de la Verificación declarada arriba —el bit-check v11— SKIPEA SIEMPRE en CI**
porque los `.pkl` están gitignored: es evidencia local-only, no de integración continua.

## Notas constitución
Cambiar UNA constante de normalización tras reentrenar sin snapshot versionado = leakage silencioso — exactamente lo que este BL elimina.
