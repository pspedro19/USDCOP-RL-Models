---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Auditoría profunda de mercado canónico y propuesta cuantitativa

Fecha de corte: 2026-07-21  
Estado: verificado contra código, migraciones, PostgreSQL vivo, Airflow, respaldos y evidencia cuantitativa local.

## Veredicto ejecutivo

La capa de datos es materialmente buena y mucho más madura que la capa de alpha, pero todavía no permite declarar que los cuatro activos estén listos para una estrategia rentable en producción. El score global cercano a 9.7/10 describe integridad OHLC, frescura y disponibilidad; no demuestra disponibilidad point-in-time (PIT), ausencia de sesgo de selección ni rentabilidad OOS.

La afirmación más precisa es:

> Ingestión y almacenamiento están operativos; la semántica SPX intradía, la restauración cold-start, la cobertura PIT y la evidencia de alpha siguen bloqueando la promoción cuantitativa.

No existe una manera honesta de garantizar rentabilidad "a toda costa". Ese objetivo incentiva sobreajuste, apalancamiento y riesgo de ruina. El objetivo controlable debe ser maximizar supervivencia y retorno ajustado por drawdown, y promover únicamente cuando la evidencia neta de costos sobreviva fuera de muestra.

## Qué se verificó directamente

### DDL y objetos

Las migraciones `060` a `063` existen y están commiteadas:

- `060_market_canonical_views.sql`
- `061_market_wide_views.sql`
- `062_macro_monthly_views.sql`
- `063_native_multiframe_ohlcv.sql`

Los objetos principales existen en la DB viva. Los conteos cambian con la ingestión y por eso deben verse como snapshots, no como constantes de documentación.

| Objeto | Conteo vivo auditado | Conteo del resumen | Conclusión |
|---|---:|---:|---|
| `dim_asset` | 4 | 4 | coincide |
| `market_session_calendar` | 11,688 | 11,688 | coincide |
| `usdcop_m5_ohlcv` | 2,197,621 | 2,196,397 | +1,224 por ingestión posterior |
| `asset_native_ohlcv` | 299,886 | 299,826 | +60 por ingestión posterior |
| `asset_daily_ohlcv` | 54,093 | 54,090 | +3 por ingestión posterior |
| `market_ingestion_manifest` | 1,853 | 1,733 | +120 corridas |
| `market_macro_monthly_wide` | 282 | 282 | coincide |
| `market_macro_quarterly_wide` | 38 | 38 | coincide |
| `market_ohlcv_1h_agg` | 186,172 | ~10k | el resumen era COP-only o estaba obsoleto |
| `market_ohlcv_4h_agg` | 50,421 | ~3.8k | el resumen era COP-only o estaba obsoleto |

Desglose real de agregados M5:

| Símbolo | 1h | 4h |
|---|---:|---:|
| BTC | 78,128 | 19,548 |
| BRL | 21,966 | 7,056 |
| COP | 8,389 | 3,375 |
| MXN | 37,913 | 9,861 |
| XAU | 39,776 | 10,581 |

Las claves primarias están correctamente definidas en las tablas físicas:

- M5: `(time, symbol)`
- nativo: `(time, symbol, tf)`
- diario: `(time, symbol)`

### Integridad y calendario

- Incoherencias OHLC observadas en M5, nativo y diario: **0**.
- Timestamps futuros observados: **0**.
- El calendario NYSE reproduce el cambio DST auditado: **1,322** sesiones abren a 13:30 UTC y **687** a 14:30 UTC.
- La convención de instantes UTC más vistas COT es correcta conceptualmente; la zona de sesión pertenece al activo/calendario y no debe convertirse en una zona única de negociación.
- El DAG `l0_multiframe_catchup` mostró **3/3 corridas exitosas**: dos programadas y una manual.
- El manifiesto vivo tenía 1,846 corridas correctas y 7 con error: **99.622%** de éxito. Los siete errores no deben desaparecer del historial; deben tener clasificación, reintento y resolución trazable.

Una consulta analítica full-scan sobre duplicados/sesiones agotó la conexión antes de terminar. PostgreSQL siguió healthy. No se repitió una operación pesada sin índice. Las PK impiden duplicados de clave exacta, pero la unicidad por `session_date` con horas distintas debe permanecer como ratchet indexado.

### Scorecard reproducido

| Capa | Score |
|---|---:|
| 5m | 9.8 |
| 1h nativo | 9.9 |
| 4h nativo | 9.9 |
| diario | 9.7 |
| mensual nativo | 9.8 |
| mensual macro | 9.0 |

El score es válido como indicador operacional, pero oculta la principal limitación cuantitativa. Cobertura `available_at` observada:

| Capa/símbolo | Filas | `available_at` nulo | Lectura PIT |
|---|---:|---:|---|
| M5 total | 2,197,621 | 141,198 | incompleta; COP concentra la debilidad |
| nativo 1h/4h/1month | 299,886 | 0 | 100% estampado |
| diario BTC | 3,261 | 3,260 | prácticamente 0% PIT |
| diario COP | 9,445 | 1,683 | 82% PIT aproximado |
| diario XAU | 13,301 | 7,098 | 47% PIT aproximado |
| diario SPX500 | 1,644 | 0 | 100% estampado |
| mensual macro | 282 | cobertura efectiva 0% en scorecard | no apto aún para claim PIT |

`available_at` no prueba por sí solo que el timestamp sea históricamente correcto. Para macro se requiere vintage real o una regla conservadora verificable por serie, frecuencia, calendario de publicación y revisiones.

## Contradicciones y bloqueos encontrados

### P0 — SPX intradía no está realmente en el contrato wide

`asset_native_ohlcv` contiene SPY nativo en 1h y 4h:

- 1h: 11,283 filas
- 4h: 3,248 filas

Pero `market_ohlcv_1h_wide` y `market_ohlcv_4h_wide` no exponen columnas OHLCV para SPX. Sólo publican `status_spx500 = 'no_native_data'`. La migración `063` une COP, XAU y BTC e ignora SPY. Por tanto, la frase "wide native-first para los cuatro activos" es falsa en el estado vivo.

Corrección requerida:

1. Definir explícitamente si el activo económico `spx500` usa SPY total-return/executable, SPX price index o ambos con roles distintos.
2. Crear un mapeo `dim_asset_symbol(asset_id, provider, provider_symbol, role, valid_from, valid_to)`.
3. Hacer que 1h/4h wide resuelvan el alias aprobado y publiquen OHLCV, fuente, origen, disponibilidad y estado con contrato simétrico.
4. Añadir prueba DB-backed que falle si un activo declarado nativo queda sólo con `status_*`.

### P0 — El restore automático está sobreafirmado

Los respaldos de `data/backups/features` existen, tienen manifiesto y SHA-256. El snapshot auditado incluye 17 tablas y, entre otras:

- M5: 2,197,621 filas, ~53.2 MB.
- nativo: 299,886 filas, ~8.3 MB.
- diario: 54,093 filas, ~1.7 MB.
- manifiesto: 1,853 filas, ~0.2 MB.

Sin embargo, el arranque no prueba una recuperación completa:

- `init-scripts/26-restore-features.sh` aplica sólo migraciones 043–059, no 060–063.
- El seeder de M5 usa el backup/seed clásico, no el backup nuevo de features.
- No se encontró wiring de arranque que restaure `asset_native_ohlcv` y `market_ingestion_manifest` mediante `feature_data_backup.py --mode restore`.

Un backup sin restore drill no es todavía una garantía de recuperación. Gate requerido: base vacía → migraciones completas → restore → refresh de matviews → checksums/conteos → suite DB-backed, todo en un job reproducible.

### P0 — La cobertura PIT todavía bloquea ML promovible

BTC diario, XAU diario y macro mensual no satisfacen el contrato PIT necesario. No deben alimentar un retraining con claim causal/OOS hasta corregir o excluir las filas sin disponibilidad histórica defendible. Imputar `available_at` con la fecha de carga actual tampoco reconstruye un vintage.

### P1 — SPX500 y SPY están mezclados semánticamente

`dim_asset` mapea `spx500` a `SPX500`, mientras el nativo intradía y la historia diaria profunda usan SPY. La vista diaria canónica excluye la historia SPY desde 1993 y conserva aproximadamente 1,644 filas SPX500 desde 2020. Una estrategia de retorno no puede mezclar price index, ETF ejecutable y total-return sin declararlo.

Decisión recomendada:

- investigación/retorno: SPY adjusted/total-return, con costos y dividendos definidos;
- indicador de mercado: SPX price index;
- ejecución: instrumento/broker específico;
- cada rol con symbol alias y linaje propios.

### P1 — `annualization` no debe ser una propiedad única del activo

USDCOP tiene `annualization = 52`, que describe la cadencia semanal de una estrategia, no el reloj del activo. Ahora que existen M5, 1h, 4h y diario, esa columna puede producir Sharpe/volatilidad incorrectos.

La anualización debe resolverse desde el contrato estrategia-frecuencia: 52 para decisiones semanales, ~252 para diario de sesión, barras de sesión para intradía; BTC requiere 365 en el sleeve 24/7. La cartera debe convertir explícitamente los relojes antes de combinar riesgos.

### P1 — “256 tests verdes” no tiene evidencia reproducible en esta auditoría

La ejecución host de las pruebas estructurales relevantes produjo **7 passed, 12 skipped**. Los skips fueron pruebas DB-backed sin credenciales PostgreSQL en el host. Los archivos de pruebas no están montados en el scheduler para repetir allí la misma orden. Las verificaciones SQL directas cubrieron parte del hueco, pero no equivalen a una corrida persistida de 256 tests.

Se debe publicar un artefacto CI con commit SHA, entorno, lista de tests, passed/failed/skipped, duración y logs. `skipped` no cuenta como verde para un gate de datos.

### P2 — Inconsistencia documental en la reparación TZ

El test/fuente de remediación documenta **15,865** barras desplazadas +5h, 815 colisiones y 86 strays; el resumen declara **15,109** desplazadas. Debe reconciliarse con el backup/migration log original y fijar una sola cifra verificable. No afecta el estado OHLC vivo, pero sí la trazabilidad de una eliminación material.

## Evidencia de alpha disponible

La infraestructura está por delante del alpha. La evidencia vigente no permite llamar rentable a ninguna de las cuatro estrategias:

| Activo/track | Resultado atractivo | Motivo de no promoción |
|---|---|---|
| USDCOP `smart_simple_v11` | +18.73%, Calmar 3.895, Sharpe 2.292, 31 trades | DSR headline 0.6911, falla B1/B1-prime y PIT |
| BTC trend b2 | +793% full-history, Calmar 1.409 | B&H +1,419%, DSR 0.0467, falla OOS/edge y PIT |
| XAU dynamic exit | +5.8% | Calmar 0.0045, DSR 0.0518, PBO 0.7236, falla costos x2 y baselines |
| SPX regime gated | +43.5%, Calmar 0.644 | DSR 0.8804, pierde contra MA200/B&H, PBO ausente |

Dos hipótesis nuevas tampoco desbloquearon dirección:

- COP con cross-asset lead: DA 0.5686 → 0.5490; delta -1.96 pp; McNemar p=1.0.
- BTC con funding: DA 0.4986 → 0.4767; delta -2.19 pp; McNemar p=0.2153.

No se deben borrar esas variables: pueden aportar a volatilidad, sizing o stress. Sí se debe impedir que se vendan como edge direccional.

## Propuesta de estrategia multi-horizonte y multi-timeframe

### Regla de arquitectura

Multi-timeframe no debe significar cuatro señales competidoras por activo. Debe existir una decisión principal por activo y timeframes subordinados con funciones distintas:

1. **Contexto lento:** régimen y tendencia.
2. **Pronóstico de riesgo:** volatilidad/rango y cuantiles.
3. **Entrada/salida:** ejecución, liquidez, spread y sesión.
4. **Cartera:** sizing, correlación y límites.

Esto reduce trials, evita votos correlacionados y conserva trazabilidad.

### Estrategia recomendada por activo

| Activo | Decisión principal | Uso 5m/1h/4h | Target recomendado | Estrategia candidata |
|---|---|---|---|---|
| USDCOP | semanal sobre cierre completo | M5 para fills/HS/TP; 1h/4h para vol, rango y liquidez de sesión | dirección 5d sólo como contexto; vol/rango 1d y 5d para sizing | Ridge + BayesianRidge vivo sólo tras PIT macro; baseline trend/carry; ejecución 08:00–12:55 COT |
| BTCUSDT | diaria UTC, filtro semanal | 1h/4h para vol-of-vol, funding shock, OI/basis y ejecución | cuantiles de retorno absoluto/realized vol 1d/7d; dirección secundaria | blended trend + vol targeting; funding/OI/basis como modificadores de riesgo, no dirección hasta nueva evidencia |
| XAUUSD | diaria UTC | 1h/4h para Parkinson/range, gap y liquidez; 5m no es señal principal | vol/rango 1d/5d y régimen 20d | TSMOM combinado 63/126/252 + vol targeting; next-session; descartar dynamic-exit actual |
| SPX/SPY | diaria al cierre, ejecución next-open | 1h/4h sólo cuando el alias/contrato SPY esté corregido | vol 1d/5d/20d y drawdown regime | MA200 o blended trend simple + vol targeting; SPY total-return como investigación y benchmark |

### ¿Reentrenar dirección a siete semanas?

No como estrategia ejecutable en el estado actual.

Un target no solapado de siete semanas produce aproximadamente `52 / 7 = 7.4` observaciones independientes por año. Incluso seis años aportan sólo unas 45 decisiones efectivas, insuficientes para 25 features, varios modelos y corrección por múltiples trials. Usar labels solapados aumenta el N nominal, no la información, y exige purging, embargo y block bootstrap.

Uso aceptable del horizonte de siete semanas:

- escenario de volatilidad/rango y cuantiles;
- límite de riesgo y presupuesto de exposición;
- régimen lento o stress, no orden directa;
- reporte separado, sin mezclar su reloj estadístico con el horizonte diario/semanal.

Horizontes iniciales defendibles:

- retorno/dirección: 1d y 5d, con una sola salida operativa por activo;
- volatilidad/rango: 1d, 5d y 20d;
- régimen lento: 60–63 sesiones;
- siete semanas: investigación de riesgo hasta reunir suficiente forward.

## Protocolo de validación y promoción

### Checkpoint 0 — Reparar contratos de datos

- Exponer SPY/SPX correctamente en 1h/4h wide.
- Separar aliases/roles de símbolo.
- Resolver anualización por timeframe/estrategia.
- Automatizar restore 060–063 y ejecutar cold-start drill.
- Cero skips en ratchets DB-backed.

### Checkpoint 1 — PIT y linaje

- 100% de filas usadas por un experimento con `available_at` defendible.
- Vintages macro o lag conservador por serie documentado.
- Manifest inmutable con provider, query, checksum, ventana, fecha de conocimiento y código.
- Prohibir backfill revisado en folds históricos sin vintage.

### Checkpoint 2 — Feature/label contract

- Feature timestamp <= decision timestamp.
- Scaling/selección fit sólo sobre train.
- Labels con ejecución next-bar/next-open y costos realistas.
- Purge/embargo igual al máximo horizonte solapado.
- Un manifiesto por activo: features, target, reloj, costos, seeds y hash.

### Checkpoint 3 — Forecasting

Para dirección, DA es diagnóstica, no el criterio central. Reportar Brier/log-loss, calibración, balanced accuracy/MCC e IC contra baselines. Para volatilidad, reportar QLIKE, MAE/RMSE sobre log-vol, cobertura de cuantiles y comparación EWMA/HAR/naive.

No se compara sólo contra cero. Baselines obligatorios:

- B1 buy-and-hold o exposición natural;
- B1-prime misma volatilidad/exposición;
- regla dumb correspondiente (MA200, TSMOM simple, EWMA vol);
- costos x1, x2 y stress x3.

### Checkpoint 4 — Walk-forward trial-aware

- Ventanas expansivas/rodantes definidas antes de ejecutar.
- Purged/embargoed CV cuando hay labels solapados.
- DSR > 0.95 como gate primario de selección múltiple.
- PBO y estabilidad por folds/regímenes.
- Sin grid search sobre OOS ni cambio de tesis después de mirar el test.
- Calmar primario; Sharpe secundario; máximo drawdown, CVaR, turnover y capacidad siempre visibles.

### Checkpoint 5 — Paper forward

- Ledger inmutable de señal, disponibilidad, orden, fill, costo, slippage y decisión.
- USDCOP con reloj semanal + ejecución M5; BTC 24/7; XAU/SPX por sesión.
- Mínimo 30 oportunidades y cobertura de más de un régimen antes de dinero real; para estrategias lentas se requiere más tiempo, no observaciones artificiales.
- Breaker por drawdown, staleness, fuente, spread y divergencia backtest/serving.

### Checkpoint 6 — Cartera y promoción

- Sleeves por activo con vol targeting y límites de concentración.
- Covarianza shrinkage/Ledoit-Wolf; ERC/risk parity antes que optimización de media inestable.
- Conversión explícita de relojes 365/252/52.
- Promoción gradual: shadow → paper → capital mínimo → escalado condicionado.
- Rollback automático si integridad, PIT, latencia, slippage o drawdown exceden límites.

## Métricas que deben quedar automatizadas

### Datos

- completitud session-aware, frescura y latencia p50/p95/p99;
- PIT coverage por tabla/símbolo/feature;
- duplicados exactos y duplicados por session date;
- coherencia OHLC, timestamps futuros y barras fuera de sesión;
- divergencia entre proveedores y tasa de reparación/backfill;
- restore RTO/RPO y checksum post-restore.

### Modelos

- calibración, Brier/log-loss/MCC/IC para dirección;
- QLIKE, error de log-vol y cobertura de cuantiles para riesgo;
- drift de features/modelo y estabilidad por régimen;
- degradación contra baseline y champion/challenger;
- tamaño efectivo de muestra, no sólo filas nominales.

### Estrategia

- retorno neto, Calmar, Sharpe/Sortino, max DD, CVaR;
- DSR, PBO, estabilidad de parámetros y fold dispersion;
- turnover, slippage, capacidad y costos x1/x2/x3;
- captura de upside/downside y protección en crisis;
- paper-vs-backtest delta y atribución por sleeve.

## Definición de terminado

La capa puede llamarse productiva para investigación cuando se cierren SPX wide, aliases, PIT y restore. Una estrategia puede llamarse promovible sólo cuando, además, supera baselines y costos con DSR > 0.95, PBO aceptable, Calmar estable, paper forward suficiente, contrato de ejecución reproducible y límites de riesgo/rollback probados.

Hoy el mejor camino no es entrenar más dirección con los mismos datos. Es convertir la nueva granularidad en mejores pronósticos de volatilidad, rango, liquidez y sizing, mientras se corrigen PIT y semántica. El zoo direccional debe permanecer como superficie de transparencia hasta que una hipótesis nueva sobreviva el protocolo completo.
