---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Auditoría de adquisición, seeds, backfills y backups

Se ejecutó `scripts/analysis/audit_acquisition_backups.py` sobre `seeds/latest`
y `data/backups`. Se encontraron 29 ficheros Parquet legibles, 463.144 filas
agregadas y ningún fichero corrupto. El inventario cubre OHLCV diarios, M5/1H,
derivados cripto, macro diaria, snapshots de features y registros de ejecución.

El repositorio contiene adaptadores/código para TwelveData, MT5, scraping,
BCRP/Suameca y snapshots públicos. Esto demuestra presencia de implementación,
no una ejecución exitosa reciente ni disponibilidad de credenciales. Cada fuente
debe aportar en producción un manifiesto con request, timestamp de adquisición,
respuesta, hash, zona horaria, frecuencia y `available_at`.

La auditoría quedó en `evidence/acquisition-backups-audit.json` y valida lectura,
cobertura temporal, duplicados, timestamps inválidos, gaps, missingness y OHLC
inválidos. El inventario local pasa; la promoción sigue bloqueada porque los
backups/seeds no prueban PIT/vintages ni que los scrapers/APIs se hayan ejecutado
contra datos históricos reproducibles.

Prueba: `tests/unit/test_acquisition_backups_audit.py` (1 passed).
