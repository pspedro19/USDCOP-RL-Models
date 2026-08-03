---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Auditoría de adquisición, seeds, backfill y backups — 2026-07-20

Se ejecutó `scripts/analysis/audit_acquisition_assets.py` en modo offline. El inventario encontró 33 artefactos Parquet y detectó implementaciones para TwelveData, MT5, BCRP, Suameca, scraping (incluido Investing/BanRep) y el adaptador público.

## Cobertura observada

| Artefacto | Filas | Cobertura observada | Estado |
|---|---:|---|---|
| `seeds/latest/btcusdt_daily_ohlcv` | 3.260 | 2017-08-17–2026-07-20 | presente |
| `seeds/latest/usdcop_m5_ohlcv` | 96.900 | 2020-01-02–2026-07-16 | presente; reconciliar con backup de 138.454 filas |
| `seeds/latest/usdcop_daily_ohlcv_full` | 1.213 | 2015-01-02–2019-09-20 | histórico parcial |
| `seeds/latest/usdcop_daily_ohlcv` | 1.661 | 2020-01-02–2026-07-16 | hueco entre fuentes a investigar |
| `seeds/latest/xauusd_daily_ohlcv` | 5.871 | 2004-01-02–2026-07-21 | presente |
| `seeds/latest/fx_multi_m5_ohlcv` | 134.450 | 2020-01-02–2026-07-16 | multi-activo; verificar partición |
| `seeds/latest/macro_indicators_daily` | 10.759 | manifest 1954-07-31–2026-01-22 | revisar tz/vintages |
| `data/backups/seeds/*` | 138.454 OHLCV / 10.882 macro | backup 2026-07-20 | hash en manifest |
| `data/snapshots/public_daily/*` | 1.643–2.392 por activo | 2020-01-01–2026-07-20 | `pit_vintage=false`, no promocionable |

## Verificación de métodos

La presencia de código no demuestra una ejecución exitosa. La auditoría confirma módulos para TwelveData, MT5, BCRP, Suameca, Selenium/scrapers y snapshots públicos, pero no encontró evidencia local de una corrida reciente con respuesta, conteo, checksum y latencia por proveedor. Las claves/API y conectividad no se prueban en este entorno.

## Bloqueos para aprobar

1. Ejecutar backfill real por fuente y guardar manifest por corrida: request, proveedor, símbolo, intervalo, filas recibidas/aceptadas/rechazadas, primer/último timestamp, latencia, error y SHA-256.
2. Reconciliar discrepancias de filas y fechas (`usdcop_m5` seed 96.900 vs backup 138.454; macro con errores de mezcla tz-naive/tz-aware).
3. Hacer restore drill en entorno aislado y comprobar hashes, conteos y consultas post-restauración; demostrar copia remota/offsite y RPO/RTO.
4. Validar calendario, duplicados, gaps, OHLC y freshness por activo/fuente; no rellenar gaps sin registrar método.
5. Para macro, conservar `release_date`/vintage as-of real; los snapshots públicos no son PIT.

**Decisión:** `REVIEW_REQUIRED` / no autoriza promoción ni afirmar rentabilidad. Evidencia estructurada en `evidence/acquisition-assets-audit.json`.
