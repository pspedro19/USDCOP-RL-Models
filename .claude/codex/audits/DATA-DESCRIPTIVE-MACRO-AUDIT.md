---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Auditoría descriptiva de datos, features y macro — 2026-07-20

## Alcance

Se inspeccionaron los snapshots `data/snapshots/public_daily` de USD/COP, XAU/USD,
BTC/USDT y SP500/SPY, sus timestamps `available_at`, y los contratos
`config/experiment_ssot.yaml`, `config/feature_config.json` y
`config/economic_calendar.yaml`.

## Hallazgos por activo

| activo | filas | rango | frecuencia observada | gaps relevantes | missing | anomalías |
|---|---:|---|---|---:|---:|---|
| USD/COP | 1,703 | 2020-01-01–2026-07-17 | diaria | 1 gap >3 días (máx 5) | 0% OHLC; volumen 100% cero | volumen no informativo |
| XAU/USD | 1,645 | 2020-01-02–2026-07-17 | diaria calendario | 48 gaps >3 (máx 4) | 0%; volumen 0.55% cero | fines de semana/feriados mezclados |
| BTC/USDT | 2,392 | 2020-01-01–2026-07-19 | diaria 24/7 | ninguno | 0% | colas muy pesadas (kurtosis retorno 11.57) |
| SP500/SPY | 1,643 | 2020-01-02–2026-07-17 | diaria mercado NY | 49 gaps >3 (máx 4) | 0% | `available_at` cae en sábado para 2020-01-03; verificar calendario |

Retorno diario (media/volatilidad/skew/kurtosis): BTC 0.142%/3.14%/-0.46/11.57;
SPY 0.059%/1.28%/-0.26/13.11; USD/COP 0.004%/0.99%/0.72/3.96;
XAU 0.066%/1.19%/-0.77/7.83. Las colas invalidan supuestos gaussianos: usar
bootstrap bloqueado, CVaR y tests robustos.

## Granularidad y disponibilidad

Los cuatro archivos son OHLCV diario. No hay intradía para validar ejecución 5m,
spreads o slippage. `available_at` parece un lag sintético de un día (y para SPY
puede ser fin de semana), no una fecha de publicación/vintage del proveedor.
Por tanto los snapshots son válidos para smoke/backtest exploratorio, no para
evidencia PIT/OOS de producción. Debe conservarse `source_timestamp`,
`published_at`, `revision_id` y timezone del proveedor.

## Features y macro configuradas

El SSOT nuevo declara 20 dimensiones (18 market + 2 state): retornos/volatilidad,
RSI y tendencia, más DXY, VIX, EMBI, Brent, Treasury-10Y, USDMXN y spreads de
tasas. `feature_config.json`/`deployment_safeguards.yaml` todavía declaran 15
(13+2): existe drift de contrato que puede romper entrenamiento/inferencia.
Las macro diarias/mensuales deben unirse `as-of` por `available_at`; el
`macro_shift_days: 1` no sustituye vintages reales y puede ser insuficiente para
CPI, empleo, GDP, PCE o M2 revisados.

## Riesgos de leakage y calidad

1. `available_at` sintético y calendario incorrecto (SPY sábado) permiten usar
   información antes de su publicación.
2. Macro mensual con fecha de observación en vez de release date filtra revisiones.
3. `adj_close` de SPY incorpora dividendos futuros si no se fija versión PIT;
   separar price-return y total-return.
4. Volumen USD/COP todo cero y XAU con ceros deben convertirse en `NaN` y excluirse
   de features de liquidez; no imputar cero silenciosamente.
5. Gaps de mercado deben etiquetarse como sesión cerrada, no rellenarse con
   forward-fill para targets.
6. Features rolling (z-score 252, RSI) deben calcularse dentro de cada fold,
   después del corte temporal; nunca normalizar con todo el dataset.

## Acciones requeridas antes de aprobación

- Unificar SSOT a 20 features o revertir explícitamente a 15; regenerar hashes y
  contratos de inferencia.
- Añadir validadores de frecuencia/calendario por activo y regla de gaps.
- Sustituir `available_at` sintético por vintages PIT o marcar artefacto
  `research_only` automáticamente.
- Ingerir macro FRED/market con release vintages; probar `asof_join` y embargo.
- Ejecutar auditoría de missingness, PSI/KS y correlación/VIF por fold y régimen.
- Reportar métricas robustas (IC, MAE/RMSE/MASE, cobertura, CVaR, bootstrap) y
  sensibilidad a costos; bloquear promoción si cualquier contrato falla.

**Decisión:** datos completos a nivel de celdas, pero granularidad diaria y
lineage PIT insuficiente; estado `RESEARCH_ONLY / NO-GO` para producción.
