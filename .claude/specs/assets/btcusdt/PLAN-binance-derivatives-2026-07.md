# PLAN — Integrar data de derivados de Binance para mejorar el modelo de BTC

> **Objetivo:** dar al modelo de BTC su insumo faltante — **datos no-precio** (funding, open interest,
> long/short, liquidaciones) — para pasar de un seguidor de tendencia frágil (OOS-2025 −1.4%) a una
> estrategia con señal de **posicionamiento/sentimiento**. Aterriza la Fase 1 cripto-native del
> `IMPLEMENTATION_ROADMAP.md` y el P2 del `../../audit/STRATEGIC-ASSESSMENT-2026-07.md §4`.
>
> **Contexto (por qué):** `src/btc_strategy/indicators.py` es 100% precio; la migración 052 diseñó las
> tablas cripto-native pero **no hay extractor que las llene**. En cripto el alfa está en el
> posicionamiento de derivados y flujos on-chain, no en el precio. Esto es el desbloqueo.

---

## 0. Realidad de la API pública de Binance (lo que SÍ y NO tiene historia)

Todo lo de abajo es **público, sin API key** (las llaves guardadas solo suben el rate-limit / permiten
endpoints autenticados). **La restricción clave que define el plan:**

| Señal | Endpoint (público) | Historia disponible | ¿Backtesteable 2018-2026? |
|---|---|---|---|
| **Funding rate** | `fapi/v1/fundingRate` (8h, paginado por time) | **Profunda (~2019-09 → hoy)** | ✅ **SÍ** |
| Open Interest | `futures/data/openInterestHist?period=1d` | **Solo últimos 30 días** | ❌ solo forward |
| Long/Short account ratio | `futures/data/globalLongShortAccountRatio` | Solo 30 días | ❌ solo forward |
| Top-trader L/S ratio | `futures/data/topLongShortPositionRatio` | Solo 30 días | ❌ solo forward |
| Taker buy/sell vol | `futures/data/takerlongshortRatio` | Solo 30 días | ❌ solo forward |
| **Liquidaciones** | REST público **descontinuado**; WS `!forceOrder@arr` live | Ninguna histórica | ❌ forward (WS) o Coinglass (pago) |

**Consecuencia honesta:** solo el **funding rate tiene historia profunda** → es lo único que podemos
**backtestear en 2018-2026 hoy**. OI / long-short / taker / liquidaciones son **forward-only**: hay que
empezar a coleccionarlos ya para acumular historia, pero NO entran al backtest histórico todavía. El plan
se ordena por esa realidad — no prometemos backtest de lo que no tiene historia.

---

## Estado (2026-07-06)

| Sub-fase | Entregable | Estado |
|---|---|---|
| **1.1 Ingesta de datos** | Extractor `scripts/data/ingest_btc_derivatives.py` → `crypto_derivatives_daily` + seed parquet, integrado al DAG `asset_btcusdt_pipeline_weekly` (`l0c_ingest_derivatives`) | ✅ **HECHO** — 2492 días de funding (2019-09-10→2026-07-06) en BD; OI/LSR forward 30d; idempotente; backup en `feature_data_backup.py` |
| 1.2 Features | `funding_rate`/`z_funding` en `src/btc_strategy/indicators.py` | ⏳ pendiente |
| 1.3 Estrategia | `btc_trend_funding_s4` (B2 × funding-gate) | ⏳ pendiente |
| 1.4 Evaluación | correr trial + gate DSR/OOS | ⏳ pendiente |

> **1.1 tal como quedó:** funding history profunda (backtesteable) + basis (mark vs spot, ~980 días con
> mark disponible) + OI/long-short forward-only (30d, se acumulan semanalmente vía el DAG). Todo desde
> endpoints **públicos** de Binance USDT-M (sin key; `.env` solo sube rate-limit). Corre semanalmente
> antes del backtest de BTC.

---

## Fase 1 — Funding rate (backtesteable, EL primer entregable)

**Hipótesis:** funding persistentemente alto+positivo = longs sobre-apalancados/crowded → riesgo de
reversión; funding negativo = shorts crowded → suelo. Un `z_funding` extremo debe **reducir exposición**
(gate de posicionamiento), complementando el trend. Es señal **ortogonal** al precio.

### 1.1 Extractor — `scripts/data/ingest_btc_derivatives.py` (NUEVO, patrón de `ingest_btc_ohlcv.py`)
- Descarga funding histórico paginado: `GET fapi/v1/fundingRate?symbol=BTCUSDT&startTime&endTime&limit=1000`
  desde 2019-09 → hoy (8h → ~3 registros/día). Público, sin key; rate-limit suave.
- **Resample a diario** alineado al cierre canónico del seed (UTC 00:00): `funding_rate` diario =
  suma o media de los 3 fundings del día (documentar cuál; suma ≈ costo diario de carry).
- **Anti-leakage:** el funding de un día se conoce al final del período (último settle ~16:00 UTC) →
  `date` = día del evento (UTC), `published_at` = fin del día D (disponibilidad). El feature code hace
  **shift(1)** causal (igual que macro T-1) → el backtest nunca ve funding futuro.
- Escribe a **`crypto_derivatives_daily`** (migración 052 — columnas reales: `date, symbol,
  funding_rate, funding_zscore, open_interest, basis_annualized, long_short_ratio, liquidations_usd,
  source, published_at, updated_at`; PK `(date, symbol)`) vía UPSERT idempotente **con `COALESCE`** (un
  re-run forward no borra el funding histórico) + **seed parquet**
  `seeds/latest/btcusdt_derivatives_daily.parquet` (misma filosofía file-driven que los OHLCV). El
  extractor computa `funding_zscore` (rolling 30d) y `basis_annualized` (mark vs spot-close) además del
  `funding_rate` crudo. `liquidations_usd` queda NULL hasta Fase 3.
- Gate de calidad: reusar el patrón `src/data_quality/` (rango, dups, NaN, cobertura de fechas).

### 1.2 Features — `src/btc_strategy/indicators.py::build_daily_features` (aditivo)
Merge causal (`merge_asof` backward + `shift(1)`) del funding diario y derivar:
- `funding_rate` (diario, shift 1)
- `z_funding` = z-score de `funding_rate` sobre ventana (p.ej. 30d) — señal principal
- `funding_ma7`, opcional `funding_extreme = |z_funding| > 2`

Si el seed de derivados falta (fresh clone / rango sin cobertura), los features degradan a 0/NaN → la
estrategia sigue corriendo con solo-precio (graceful, como hoy).

### 1.3 Estrategia — `src/btc_strategy/strategies.py` (variante NUEVA, no tocar B1/B2/S3)
Nueva sub-estrategia **`btc_trend_funding_s4`** = B2 trend × gate de funding:
- `intent` = trend (SMA100+ADX, como B2)
- `size` = vol-target × **funding_gate**, donde `funding_gate = clip(1 − k·max(0, z_funding), floor, 1)`
  → reduce exposición cuando los longs están crowded (funding alto), mantiene en funding normal/negativo.
- Constantes (`k`, floor, ventana z) **fijas y documentadas** (no ajustadas al test).

### 1.4 Evaluación (gate honesto, ya instrumentado)
Correr `run_btc_pipeline.py` con la nueva estrategia añadida al set de trials. El gate:
- **DSR > 0.95** (deflado ahora por N=4 trials) **Y** OOS-2025 positivo **Y** le gana a B2 en Sharpe+Calmar.
- Comparar `btc_trend_funding_s4` vs `btc_trend_b2`: ¿el funding-gate mejora el OOS-2025 (−1.4%) y baja el
  MaxDD? Si **no** mejora OOS, se registra como REVIEW y no se promueve (honestidad — el funding puede no
  aportar; lo sabremos con evidencia, no con fe).

---

## Fase 2 — OI / Long-Short / Taker (forward-only: empezar a coleccionar YA)

- Extender `ingest_btc_derivatives.py` para bajar los `futures/data/*` (últimos 30d) y **hacer append
  diario** a `crypto_derivatives_daily` (OI, `long_short_ratio`, `taker_buy_sell_ratio`).
- **DAG diario** (o stage en `config/assets/pipelines.yaml` del BTC): correr una vez al día para acumular
  historia hacia adelante. En ~3-6 meses habrá suficiente para features + un mini-backtest reciente.
- Features futuros: `oi_change`, `z_oi`, `lsr` (long-short ratio), `taker_imbalance`. **No** entran al
  backtest 2018-2026 (sin historia); se usan primero en paper/monitor forward.

## Fase 3 — Liquidaciones (forward vía WebSocket, o Coinglass diferido)

- Colector WS `!forceOrder@arr` (proceso liviano) → agrega liquidaciones long/short diarias a
  `crypto_flows_daily`. Sin historia pública REST → forward-only. Alternativa: Coinglass (pago) si se
  justifica. **Diferido** hasta que Fase 1-2 muestren que los derivados aportan.

## Fase 4 — Régimen HMM on-chain (roadmap Fase 3, SPEC-01)

Con funding (+ OI/LSR cuando haya historia) alimentar el **HMM 4-estados** que reemplaza el clasificador
price-only. Es el paso que el `btc_exposure_s3` necesita para justificarse (hoy pierde vs B2). Requiere
además on-chain (MVRV/netflows) — proveedor aparte (BGeometrics/DefiLlama free-tier), fuera de Binance.

---

## Almacenamiento & claves (SSOT)

- **Tabla:** `crypto_derivatives_daily` (migración 052, ya aditiva). Keyed `(date, symbol)` → ETH/otros
  enchufan por symbol. Se llena best-effort (como el OHLCV multi-asset); serving real = seed parquet.
- **Seed:** `seeds/latest/btcusdt_derivatives_daily.parquet` (file-driven, restore-critical → trackear).
- **Claves Binance:** en `.env` (gitignored) — `BINANCE_API_KEY` / `BINANCE_API_SECRET`. **No requeridas**
  para market-data público; el extractor las usa solo si están presentes (headers) para más rate-limit.
  Para producción migrar a Vault (`VaultService`, AES-256-GCM) como el resto de secretos. **Rotar** las
  actuales (se compartieron en texto plano).

## Archivos a crear/tocar
- **Nuevo:** `scripts/data/ingest_btc_derivatives.py` (extractor funding + forward OI/LSR/taker).
- **Nuevo (seed):** `seeds/latest/btcusdt_derivatives_daily.parquet` (generado).
- **Editar (aditivo):** `src/btc_strategy/indicators.py` (features funding), `src/btc_strategy/strategies.py`
  (`btc_trend_funding_s4`), `scripts/pipeline/run_btc_pipeline.py` (añadir el trial + DSR/OOS ya presente).
- **Editar (config):** `config/assets/pipelines.yaml` (stage `l0c_ingest_derivatives`),
  `database/migrations/052_*` ya cubre la tabla.
- **Tests:** `tests/` — extractor (schema/anti-leakage shift), feature parity, gate.

## Gate de éxito (honesto)
`btc_trend_funding_s4` se promueve **solo si**: DSR > 0.95 (N=4 trials) **y** OOS-2025 positivo **y** bate a
B2 en Sharpe **y** Calmar. Si el funding no mejora el OOS, se documenta como “funding no aporta señal
backtesteable en BTC daily” — resultado válido, sin inflar. Primero medimos, luego prometemos.

## Riesgos / límites
- Solo funding es backtesteable hoy; OI/liquidaciones son forward-only (limitación real de Binance, no del
  plan). No prometer backtest de lo que no tiene historia.
- Funding diario resampleado desde 8h puede introducir ruido de zona horaria — alinear al cierre UTC 00:00
  del seed y validar con el `ohlcv_validators` pattern.
- Riesgo de sobre-ajuste al elegir `k`/ventana: fijar constantes a-priori, validar con DSR + OOS.
