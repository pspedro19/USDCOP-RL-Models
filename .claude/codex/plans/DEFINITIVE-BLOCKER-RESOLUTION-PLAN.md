---
kind: roadmap
status: PAUSED
version: 1.0.0
last_verified: 2026-07-20
supersedes:
  - .claude/codex/plans/PRODUCTION-UNBLOCK-PLAN.md
code_anchors:
  - .claude/codex/harness/harness_engine.py
  - .claude/codex/evidence/harness-latest.json
  - config/quant_evidence/assets.json
  - database/migrations/059_checkout_order_ledger.sql
---

# Plan definitivo para eliminar los dos bloqueos externos

## Decisión central

No se intentará “aprobar” usando datos sintéticos, timestamps inventados o un mock presentado como
PSP real. Se construirán dos carriles de evidencia:

1. **Carril científico**: datos públicos/contractuales con snapshots inmutables, calendario de
   disponibilidad, revisiones y lineage; solo después se crea evidencia OOS.
2. **Carril comercial**: contract test local siempre reproducible + sandbox real del PSP con secretos
   efímeros, tenant de prueba y eventos reales; producción requiere ambos.

El harness agregado seguirá siendo `NO-GO` hasta que los dos carriles entreguen evidencia firmada.

---

## Carril A — datos PIT/OOS para los cuatro activos

### A0. Congelación y contrato

Crear un `DataSnapshotManifest` por activo con:

- `asset_id`, proveedor, símbolo y timezone;
- URL/endpoint y licencia;
- `retrieved_at`, `as_of`, `available_at_policy` y calendario;
- hash SHA-256 del raw y del normalizado;
- esquema, filas, rango temporal, duplicados, gaps y revisiones;
- transformaciones, imputaciones y feature hash;
- commit/config/container usados.

No se aceptan parquet sin raw original, checksum y política de disponibilidad.

### A1. Fuentes primarias y fallback

| Activo | Primaria | Fallback | Evidencia mínima |
|---|---|---|---|
| USD/COP | proveedor contratado/TwelveData con timestamp de publicación | seed actual solo para smoke | OHLCV + release lag + timezone COT |
| XAU/USD | proveedor contratado/TwelveData | fuente diaria secundaria cross-check | sesión metals, close convention y revisiones |
| BTC/USDT | Binance/CCXT spot histórico | proveedor secundario | 24/7, trades/candles, fees y funding si aplica |
| SP500 | SPY total-return o índice con dividendos | Stooq/Yahoo solo como cross-check | corporate actions, NY holidays, total-return policy |

Macro: FRED/series oficiales con `observation_date`, `realtime_start`, `realtime_end` cuando exista;
si una serie no tiene vintages, se etiqueta como `not_realtime_vintage` y no puede alimentar una
afirmación PIT fuerte.

### A2. Ingesta reproducible

Implementar un único `snapshot_ingest` parametrizado por `asset_id`:

1. Descarga raw a `data/raw/<asset>/<snapshot_id>/`.
2. Guarda headers, request parameters, provider response y timestamp de adquisición.
3. Normaliza a UTC internamente y conserva timezone de sesión.
4. Valida OHLC, monotonía, unicidad, calendario, gaps, stale bars y rangos.
5. Produce `available_at` desde fuente real o una política conservadora documentada.
6. Escribe `data/curated/<asset>/<snapshot_id>/` y manifest firmado.
7. Nunca sobreescribe snapshots; cada corrección crea una nueva versión.

### A3. Regla de disponibilidad

- `available_at` de una cotización no es automáticamente `timestamp`.
- Para mercado, usar timestamp de publicación confirmado por el proveedor o un lag conservador
  explícito y validado.
- Para macro, usar vintage/realtime fields; si no existen, aplicar lag publicado y marcar limitación.
- Features solo pueden leer filas con `available_at <= decision_time`.

### A4. OOS y estadística

Para cada activo, ejecutar:

- train/validation/test temporal sin reutilizar el test final;
- walk-forward purgado y embargo;
- benchmark naive/hold/MA200/vol-target según activo;
- retorno bruto y neto con costos 1x/2x/3x;
- MAE/RMSE/MASE y calibración para forecasting;
- Sharpe/Sortino/Calmar/max drawdown/CVaR/turnover/capacidad para estrategia;
- block bootstrap de alpha y drawdown;
- trial ledger completo, DSR y PBO;
- resultados por fold, seed, año y régimen.

### A5. Gates de promoción

Un activo pasa solo si:

- `data_pit=true`, raw+curated+manifest+hash presentes;
- leakage/parity/drift pasan;
- forecast supera baseline o estrategia supera benchmark neto;
- Sharpe OOS ≥ 0.5, retorno neto ≥ 0, drawdown dentro del límite;
- DSR > 0.95, PBO < 0.50, costos y capacidad declarados;
- el informe incluye todos los trials, no solo el ganador;
- revisión independiente y aprobación dual registradas.

Si una fuente no tiene vintages reales, el estado máximo es `research_validated`, nunca `production`.

---

## Carril B — pagos, suscripciones y PSP

### B0. Provider contract test local

Mantener `MockProvider` para probar deterministicamente:

- checkout y quote;
- amount/currency tampering;
- firma inválida;
- approved/declined/cancelled;
- replay/idempotencia;
- refund/chargeback;
- concurrencia;
- revocación de entitlements;
- auditoría y estados SQL.

El mock prueba lógica, pero nunca habilita `production=true`.

### B1. Sandbox real

Requisitos operativos:

- cuenta sandbox Wompi/PSP a nombre del proyecto;
- public key, events secret e integrity secret almacenados en secret manager/CI secrets;
- no commitear `.env`, claves ni payloads con PII;
- tenant/usuarios de prueba desechables;
- webhook HTTPS temporal o túnel controlado;
- IDs de pruebas correlacionados con `checkout_orders` y `billing_events`.

### B2. Matriz E2E

Ejecutar en sandbox:

1. checkout base;
2. checkout con uno y varios add-ons;
3. monto alterado;
4. moneda alterada;
5. firma alterada;
6. webhook duplicado;
7. webhook fuera de orden;
8. timeout/retry;
9. declined/error/voided;
10. refund;
11. chargeback;
12. renovación/expiración;
13. dos checkouts simultáneos;
14. usuario A intentando leer orden/entitlement de B.

Cada caso debe dejar request ID, provider event ID, reference, estado antes/después y audit row.

### B3. Gates comerciales

- Todas las transiciones son válidas según la máquina de estados.
- Exactamente un grant por evento aprobado.
- Exactamente una revocación por refund/chargeback.
- Monto de orden = monto proveedor = monto ledger.
- Moneda coincide.
- No hay grants cross-user/tenant.
- Replay produce respuesta idempotente sin doble cobro ni doble auditoría.
- Recibos/revenue/MRR concilian con el ledger.

---

## Seguridad, secretos y rollback

- Secret manager como única fuente de claves PSP/proveedores.
- Rotación y expiración antes de cada release.
- Redacción de tokens, emails y PII en logs/evidence.
- Webhook con límite de tamaño, timeout, replay window y comparación constante de firma.
- DB migrations backward-compatible y backup probado.
- Rollback de aplicación y migración ensayado en staging.
- Kill switch verificado antes de cualquier canary.

## Secuencia ejecutable

1. Aprovisionar credenciales sandbox y contrato de datos.
2. Ejecutar `snapshot_ingest` para los cuatro activos.
3. Validar manifests y actualizar `config/quant_evidence/assets.json` solo con hashes reales.
4. Ejecutar OOS por activo y publicar reportes inmutables.
5. Ejecutar contract harness + E2E PSP.
6. Ejecutar RBAC/BOLA, accessibility, load y production harness.
7. Generar `release-manifest.json` con todos los hashes y owners.
8. Reunión de aprobación dual.
9. Canary/paper period; solo después activar producción.

## Criterio final

El release solo cambia a `GO` cuando:

```text
all_internal_gates == PASS
all_asset_pit_oos_gates == PASS
provider_sandbox_matrix == PASS
security_and_slo_gates == PASS
release_manifest_signed == true
rollback_rehearsal == PASS
```

Sin las credenciales sandbox o una fuente PIT contractual, el estado correcto seguirá siendo
`BLOCKED_EXTERNAL_INPUTS`; no existe una implementación honesta que elimine esas dependencias
solo con código local.
