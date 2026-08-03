---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Auditoría integral de datos, granularidad y features

## Resultado

La auditoría reproducible está en `evidence/market-data-statistics.json` y los
informes detallados en `PUBLIC-SNAPSHOT-STATISTICAL-AUDIT.md` y
`DATA-DESCRIPTIVE-MACRO-AUDIT.md`. Las cuatro series públicas son diarias,
sin timestamps duplicados; BTC cubre 2.392 filas, SPX500 1.643, USD/COP 1.703
y XAU/USD 1.645. Los gaps de calendario de SPX/XAU son esperables por sesiones,
pero deben validarse contra calendarios oficiales antes de generar labels.

Frecuencia observada: BTC `median_gap=1d/p95=1d` (24/7 diario); SPX500
`1d/3d` con 49 gaps >3d; USD/COP `1d/3d` con 1 gap >3d; XAU/USD `1d/3d`
con 48 gaps >3d. La tabla macro también es diaria (`1d`, p95≈1,8d), pero
contiene 522 gaps >3d por publicación/feriados y no debe rellenarse sin una
política as-of explícita.

Se observan colas pesadas y outliers: BTC retorno diario σ≈3,14% y kurtosis
≈11,57; SPX σ≈1,28% y kurtosis ≈13; XAU kurtosis ≈7,83. USD/COP tiene
volumen cero en el 100% de las filas y XAU presenta ceros ocasionales; por tanto
el volumen no puede usarse como feature sin un proveedor/liquidez alternativo.

## Features y macro

Los snapshots contienen OHLCV y disponibilidad, no un vector técnico completo.
RSI/ATR/ADX, retornos, volatilidad, régimen, calendario y features macro deben
calcularse dentro del pipeline con `shift(1)`/ventanas causales y tests de
look-ahead. Existe drift pendiente: SSOT declara 20 features mientras
`feature_config`/safeguards declaran 15. Las columnas macro tienen fechas de
publicación, pero no vintages históricas verificables; `available_at` de los
snapshots fue reconstruido, no es PIT real. `macro_shift=1d` no sustituye un
as-of vintage para indicadores revisables.

## Gate y acciones bloqueantes

El harness incluye ahora `market-data-statistics`; queda `BLOCKED` cuando la
auditoría es `REVIEW_REQUIRED`. La decisión agregada permanece `NO-GO` hasta:

1. incorporar datasets PIT con vintages/as-of y hashes por release;
2. reconciliar contrato de 15 vs 20 features y registrar esquema/versionado;
3. fijar calendarios por activo y política de gaps/feriados;
4. demostrar cálculo causal de features técnicas y macro, incluyendo tests de
   ausencia de leakage, y validar labels/OOS por activo;
5. sustituir volumen cero/proxy y documentar `adj_close`/corporate actions.

Comprobación ejecutada: `python -m pytest tests/unit/test_market_data_statistics.py -q`
(1 passed). El harness completo sigue correctamente en `NO-GO` por estos bloqueos
de evidencia externa, no por un fallo de código local.

## Reentrenamiento 2026

Se añadió `scripts/validation/retraining_readiness.py` y su prueba unitaria. El
check valida PIT, OOS, trials y estado de promoción para los cuatro activos; hoy
devuelve `NO-GO` porque esos artefactos aún no existen. Por ello no es válido
afirmar que las estrategias sean rentables ni que estén listas para reentrenar
2026. Los DAGs de Airflow existentes cubren adquisición, features, training,
backtest, deployment, inference, monitoring, drift, noticias y RBAC, pero deben
ejecutarse con datos PIT y conectarse a este gate antes de promover modelos.
