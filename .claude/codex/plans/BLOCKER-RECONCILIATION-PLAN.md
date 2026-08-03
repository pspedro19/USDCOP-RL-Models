---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Plan definitivo de reconciliación de bloqueos

Aplicado con las skills locales `quant-algo-trading`, `data-recovery`,
`statistics-fundamentals`, `performance-metrics`, `dag-change`,
`approval-cycle` y `release-and-rollback`. La constitución cuantitativa del
repositorio prevalece sobre cualquier skill externa.

## Fase 0 — Baseline y trazabilidad (completada parcialmente)

- Congelar hashes de seeds, backups, snapshots y configuración.
- Ejecutar inventario de DAGs y registrar fuente, frecuencia, timezone, rows,
  gaps, missingness y OHLC inválidos.
- Crear un manifiesto por adquisición con request, response hash, `available_at`,
  latencia, reintentos, filas aceptadas/rechazadas y error.
- No considerar “adaptador presente” como “fuente ejecutada”.

## Fase 1 — Reconciliación de datos y recuperación

1. Diagnosticar primero freshness (OHLCV 3d, macro 7d, modelos 10d, noticias
   24h) y estado de DAG; no disparar backfill a ciegas.
2. Reconciliar `usdcop_m5` seed (96.900) contra backup (138.454): definir
   snapshot canónico, deduplicar por `(symbol,time)`, explicar filas excluidas y
   guardar diff firmado.
3. Reconciliar macro manifest (10.759) contra backup (10.882): normalizar UTC,
   separar frecuencia diaria/mensual/trimestral y conservar `release_date` y
   vintage.
4. Ejecutar restore drill empty-table-only y registrar checksum antes/después;
   probar copia offsite y recuperación.
5. Validar TwelveData, MT5, BCRP, Suameca y scraping en sandbox/credenciales
   reales; cada ejecución debe producir el manifiesto de Fase 0.

**Gate:** cero diferencias no explicadas, timestamps homogéneos, cobertura
completa por activo/frecuencia y restore verificado.

## Fase 2 — PIT y features causales

- Obtener vintages históricos reales; `available_at` reconstruido no es válido.
- Resolver drift del contrato de features (15 vs 20) y generar esquema versionado.
- Calcular retornos, RSI, ATR, ADX, volatilidad, régimen, calendario y macro con
  ventanas causales y `shift(1)` donde corresponda.
- Ejecutar anti-leakage por columna, ventana y label; congelar transformaciones
  del train para validation/OOS.

**Gate:** contrato único, no leakage, missingness/outlier policies aprobadas y
hashes reproducibles.

## Fase 3 — Evidencia cuantitativa

- Usar purged/embargoed walk-forward; nunca seleccionar parámetros mirando OOS.
- Evaluar B1 buy-and-hold, B1' exposure-matched, baseline ingenuo y coste ×2.
- Reportar Calmar como métrica primaria; Sharpe/Sortino secundarios.
- No reportar Sharpe/p-value con menos de 20 operaciones.
- Calcular DSR trial-aware > 0,95, PBO < 0,50, bootstrap/block-CI, drawdown,
  turnover, slippage y estabilidad por régimen.
- Emitir `oos_manifest`, `trial_count`, benchmark, costes y artefacto inmutable
  para cada uno de los cuatro activos.

**Gate:** `config/quant_evidence/assets.json` completo y estado `approved` solo
si supera todos los criterios.

## Fase 4 — DAG, retraining y operación

- Conectar adquisición → calidad → features → training → backtest → promoción →
  serving → monitoring → drift → retraining.
- Verificar registry, dependencias, sensores, horarios sin colisión e
  importabilidad; no activar DAGs H1 pausados ni DAGs RL deprecated.
- Ejecutar dry-run 2026, fallo/reintento, idempotencia, DLQ, alertas y rollback.
- Confirmar que `forecast_h5_l4b_production_deploy` solo corre tras Vote 2 humano.

## Fase 5 — Comercio, RBAC y release

- Ejecutar sandbox E2E de checkout, suscripción, webhook, replay, refund y
  chargeback.
- Probar permisos negativos por rol y entitlement por modelo.
- Verificar que deploy/canary workflows no se describan como infraestructura real;
  exigir evidencia Airflow y `summary.json`.
- Canary/testnet durante una semana antes de cualquier orden real.

## Criterio final

El harness solo puede pasar a `GO` cuando todas las fases tienen evidencia
firmada. Mientras falten PIT/OOS, sandbox PSP o restore drill, la decisión es
`NO-GO`; no se debe afirmar rentabilidad ni reentrenamiento 2026 aprobado.
