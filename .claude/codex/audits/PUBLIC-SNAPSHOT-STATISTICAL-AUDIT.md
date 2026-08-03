---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Auditoría estadística de snapshots públicos

Generado automáticamente por `analyze_public_snapshots.py`. La fuente es pública diaria; no implica vintages PIT históricas.

## btcusdt

- Filas: 2392 | periodo: 2020-01-01 00:00:00+00:00 → 2026-07-19 00:00:00+00:00
- Granularidad: mediana 1.00 días, p95 1.00; gaps >3d: 0, máximo 1.0d
- Duplicados: 0 | faltantes: {'timestamp': 0, 'adj_close': 0, 'close': 0, 'high': 0, 'low': 0, 'open': 0, 'volume': 0, 'available_at': 0, 'asset_id': 0}
- Lag available_at: mediana 0.00d, mínimo 0.00d, fechas anticipadas: 0
- Retornos: media 0.001420, std 0.031404, min -0.3717, max 0.1875, skew -0.461, kurtosis exceso 11.567, outliers 3×IQR 40
- Volumen cero: 0 filas

## spx500

- Filas: 1643 | periodo: 2020-01-02 00:00:00+00:00 → 2026-07-17 00:00:00+00:00
- Granularidad: mediana 1.00 días, p95 3.00; gaps >3d: 49, máximo 4.0d
- Duplicados: 0 | faltantes: {'timestamp': 0, 'adj_close': 0, 'close': 0, 'high': 0, 'low': 0, 'open': 0, 'volume': 0, 'available_at': 0, 'asset_id': 0}
- Lag available_at: mediana 1.00d, mínimo 1.00d, fechas anticipadas: 0
- Retornos: media 0.000642, std 0.012767, min -0.1094, max 0.1050, skew -0.262, kurtosis exceso 13.241, outliers 3×IQR 24
- Volumen cero: 0 filas

## usdcop

- Filas: 1703 | periodo: 2020-01-01 00:00:00+00:00 → 2026-07-17 00:00:00+00:00
- Granularidad: mediana 1.00 días, p95 3.00; gaps >3d: 1, máximo 5.0d
- Duplicados: 0 | faltantes: {'timestamp': 0, 'adj_close': 0, 'close': 0, 'high': 0, 'low': 0, 'open': 0, 'volume': 0, 'available_at': 0, 'asset_id': 0}
- Lag available_at: mediana 1.00d, mínimo 1.00d, fechas anticipadas: 0
- Retornos: media 0.000041, std 0.009855, min -0.0438, max 0.0748, skew 0.720, kurtosis exceso 3.959, outliers 3×IQR 7
- Volumen cero: 1703 filas

## xauusd

- Filas: 1645 | periodo: 2020-01-02 00:00:00+00:00 → 2026-07-17 00:00:00+00:00
- Granularidad: mediana 1.00 días, p95 3.00; gaps >3d: 48, máximo 4.0d
- Duplicados: 0 | faltantes: {'timestamp': 0, 'adj_close': 0, 'close': 0, 'high': 0, 'low': 0, 'open': 0, 'volume': 0, 'available_at': 0, 'asset_id': 0}
- Lag available_at: mediana 1.00d, mínimo 1.00d, fechas anticipadas: 0
- Retornos: media 0.000660, std 0.011883, min -0.1137, max 0.0608, skew -0.766, kurtosis exceso 7.828, outliers 3×IQR 15
- Volumen cero: 9 filas

## Interpretación y bloqueos

- Las series son diarias, pero mercados distintos tienen calendarios diferentes; los gaps de fin de semana/feriados deben modelarse explícitamente.
- `available_at` es un lag conservador reconstruido, no evidencia PIT de revisiones. No usar para promoción productiva sin vintages verificables.
- Revisar outliers extremos y cambios de régimen; winsorización solo dentro del pipeline y documentada.
- USD/COP y XAU/USD presentan volumen proxy/no comparable; no interpretar como volumen negociado real.
- Próximo paso: añadir macro/tecnical features con sus propios `available_at`, medir missingness y ejecutar pruebas de leakage/paridad offline-online.