# SPEC-13 — Integración escalable de BTC/USDT en el monorepo

| Campo | Valor |
|---|---|
| Estado | En curso (Fase 0 entregada) |
| Contrato | CTR-ASSET-PROFILE-001 · CTR-L0-CRYPTO-001 (migración 052) |
| Depende de | `_onboarding-playbook.md` (stages A–I, tests A1–F1) · `registry-lifecycle.md` (R1–R9) · migración 051 |
| Materializa | `config/assets/btcusdt.yaml`, migración 052, `tests/onboarding/test_asset_btcusdt.py`, extractores cripto |
| Análogo | Oro [`SPEC-12`](../../xauusd/specs/SPEC-12-scalable-registry-integration.md) |

## 1. Propósito

Esta spec es el **puente** entre la ciencia de la estrategia (`../design/`, autónoma del repo) y la
**columna vertebral multi-activo/multi-estrategia ya implementada y probada sobre USD/COP**. Define
cómo BTC se enchufa **sin duplicar infraestructura ni tocar lo que COP/Gold ya consumen**: qué se
reutiliza, qué es genuinamente nuevo, qué APIs/keys se necesitan, y qué decisiones abre el 24/7.

**Regla rectora (del onboarding-playbook):** un activo entra por **config + datos**. Si aparece
código copiado con `if symbol == "BTC"`, es una violación — el comportamiento va al `AssetProfile`.

## 2. AssetProfile — el keystone (Stage A, test A1) ✅

`config/assets/btcusdt.yaml` parametriza todo lo pegado a COP. Campos que difieren de COP/Gold:

| Campo | COP | Gold | **BTC** | Por qué |
|---|---|---|---|---|
| `asset_class` | fx | commodity | **crypto** | rama de validación del contrato |
| `session.mode` | exchange_hours | metals | **24x7** | nunca cierra |
| `session.timezone` | America/Bogota | UTC | **UTC** | cripto es UTC-native (carve-out data-governance) |
| `session.days` | [0-4] | [0-4] | **[0-6]** | incluye fin de semana → `is_24x7=true` |
| `session.open/close` | 08:00/12:55 | —/— | **null/null** | sin corte intradía (`_filter_session` no recorta) |
| `session.bars_per_day` | 60 | 288 | **288** | 24h×12 (re-medido por ingesta, test A3) |
| `session.bars_per_year` | 15660 | 72000 | **105120** | **288×365** — anualización Sharpe (NUNCA reusar COP/Gold) |
| `session.weekend_flat` | true | true | **false** | BTC mantiene posición el fin de semana |
| `session.forced_close` | friday | friday | **none** | exposición continua, no trade semanal |
| `pipeline_type` | ml_forecasting | rl | **rl** (motor de exposición) | no es forecasting |

**Anti-copia verificado por test:** A1 (perfil valida), C1 (`is_24x7`, `weekend_flat=false`, seed con
barras Sáb/Dom), B1 (ningún driver COP-only filtrado; presente ≥1 cripto-native + DFII10).

## 3. Datos sin silos — qué reutiliza y qué es nuevo

### 3.1 Reutiliza (0 tablas nuevas)
| Dato | Tabla existente | Cómo entra BTC |
|---|---|---|
| 5-min OHLCV | `usdcop_m5_ohlcv` (multi-par por `symbol`, migr. 040) | `symbol='BTC/USDT'`. El filtro de sesión COP **no** aplica (sesión viene del AssetProfile) |
| Daily OHLCV | `asset_daily_ohlcv` (multi-activo, migr. 051) | `symbol='BTC/USDT'`, instante UTC 00:00 |
| Macro FRED | `macro_indicators_daily` | DFII10 / DTWEXBGS / VIXCLS / M2SL **ya ingeridas** — se leen como drivers rodantes |

### 3.2 Genuinamente nuevo → **migración 052** (`CTR-L0-CRYPTO-001`, cripto-native, aditiva)
Ninguno existe en el repo; son datos que FX/Gold no necesitan. Todas keyed por `(date/time, symbol)`
→ un 2.º cripto (ETH) enchufa por `symbol`.

| Tabla | Contenido | Alimenta (design) |
|---|---|---|
| `crypto_onchain_daily` | MVRV-Z, NUPL, SOPR, Puell… (BGeometrics) | `z_ciclo` → régimen HMM (SPEC-01/03) |
| `crypto_derivatives_daily` | funding, OI, basis (Binance perp, **DATA-ONLY**) | `z_funding` (freno posicionamiento, SPEC-02) |
| `crypto_flows_daily` | ETF net flows (Farside, **D+1**), stablecoin supply (DefiLlama) | `M_liquidez` (SPEC-04) |
| `crypto_event_calendar` | gate discreto `G∈{1,.5,.25,0}` (LLM + vol-spike) | `G_eventos` (SPEC-05) |
| `crypto_exposure_signals` | salida del motor: `exposure∈[0,1]` + breakdown | auditoría / DSR (SPEC-06/11) |

> `published_at` separa la fecha del evento de su disponibilidad — **crítico para ETF flows (D+1)**:
> nunca se lee un flujo antes de que Farside lo publique (anti-look-ahead capa datos).

## 4. Fuentes de datos y API keys (Stage B — data engineering real)

Lo que sigue es **la razón por la que BTC no es "solo config"**: hay que construir extractores nuevos.

| Fuente | Provee | Auth / key | Costo | Notas |
|---|---|---|---|---|
| **Binance / CCXT** | 5-min + daily spot BTC/USDT; funding/OI/basis (perp) | API key opcional para OHLCV público; sí para rate-limit alto | Gratis | Canónico (ADR-0008). `ccxt` ya es dep común; extractor nuevo |
| **BGeometrics** | On-chain (MVRV-Z, NUPL, SOPR, Puell…) | free tier con API key | Gratis (limitado) | Sustituto libre de Glassnode |
| **Farside** | ETF net flows spot BTC | scrape (sin API oficial) | Gratis | **D+1**; `cloudscraper` como el extractor Investing del Oro |
| **DefiLlama** | Stablecoin supply agregado | pública, sin key | Gratis | Endpoint `/stablecoins` |
| **FRED** | DFII10, DTWEXBGS, VIXCLS, M2SL | `FRED_API_KEY` (ya en `.env`) | Gratis | **Reutiliza** el extractor macro existente |
| **TradingView / CryptoCompare** | BTC dominance, cross-checks | key CryptoCompare free | Gratis | Cross-check de precio 2013+ (deep history stitch) |
| **LLM (Azure/Anthropic)** | Clasificación de eventos → `G` | ya configurado (news/analysis) | Presupuesto existente ($1/día) | **Reutiliza** `LLMClient` + budget guard. Guard de contaminación (ADR-0011) |

> **Keys nuevas requeridas:** `BGEOMETRICS_API_KEY`, `CRYPTOCOMPARE_API_KEY` (ambas free-tier).
> `BINANCE_API_KEY`/`SECRET` opcionales (solo para subir rate-limit). Todas van a `.env` (gitignored)
> + Vault en producción. **NUNCA** commitear keys (regla `execution-bridge`).

## 5. Decisiones 24/7 (el break estructural — para el operador)

BTC rompe supuestos que COP (5h) y Gold (~23h weekday) comparten. Decisiones tomadas y abiertas:

| # | Tema | Decisión | Estado |
|---|---|---|---|
| D1 | **Barra canónica de decisión** | Cierre **UTC 00:00** diario (PRE-REGISTRATION §1, ADR-0008) | Cerrada |
| D2 | **Anualización** | **√365** (no √252, no 15660/72000) → `bars_per_year=105120` | Cerrada (test A3) |
| D3 | **Sin forced-close de viernes** | `forced_close: none`; exposición **continua** el fin de semana | Cerrada |
| D4 | **Semántica de ejecución** | El ejecutor semanal H5 (`WeeklyTPHSExecution`) **no aplica**. BTC usa un motor de **rebalanceo por bandas ±12.5%** (SPEC-06), no trades TP/HS | Cerrada (nuevo ExecutionStrategy) |
| D5 | **DAG schedule** | Los DAGs L0 COP corren en ventana de mercado 8-12 COT. BTC necesita ingesta **24/7** (o al menos diaria post-cierre UTC). La fábrica debe emitir un schedule cripto, no reusar el de COP | **Abierta** — flag al operador |
| D6 | **Live execution** | Fuera de alcance de esta entrega. Ruta rápida = web-only/paper (como Gold), evitando `forecast_h5_*` y el OMS live | Diferida |

> **D5 y D6 son las que requieren decisión del operador antes de producción.** El resto ya está
> resuelto en config/migración. Se sigue la "ruta realista más rápida" del onboarding-playbook:
> paper + visibilidad web primero; live después.

## 6. Registro dinámico + fábrica (Stage F — reutiliza lo de COP/Gold)

Idéntico al Oro: el runner publica bundles inmutables `(strategy_id='btc_exposure_s3', version, year)`
vía `register_bundle` (aditivo, no toca COP/Gold), el frontend arma solo el selector + replay leyendo
`/api/registry`, y el `chart_symbol='BTCUSDT'` del manifest etiqueta el chart (nunca "USDCOP"). La
maquinaria ya está probada (9 tests R sobre COP; Gold ya publicó 3 bundles). **BTC la reutiliza.**

## 7. Mapa de archivos (CREATE vs MODIFY)

**CREATE (nuevo, aditivo):**
- `config/assets/btcusdt.yaml` ✅ · `database/migrations/052_crypto_native_data.sql` ✅
- `tests/onboarding/test_asset_btcusdt.py` ✅ · `.claude/specs/assets/btcusdt/**` (este paquete) ✅
- (pendiente) extractores: `scripts/data/ingest_asset_ohlcv.py` (extender a Binance) + `scripts/data/ingest_crypto_native.py`
- (pendiente) módulo de modelado `src/btc_strategy/` (materializa `design/SPEC-01…12`)

**MODIFY (mínimo, aditivo — no romper COP/Gold):**
- `.claude/specs/architecture-overview.md` — registrar BTC como 3.er activo + migración 052
- `CLAUDE.md` / `_asbuilt-implementation.md` — fila BTC en la tabla per-activo (sesión/TZ/anualización)
- `_onboarding-playbook.md` — marcar BTC como 2.º caso del recorrido A–F

## 8. Criterios de aceptación (DoD de la integración)

- [x] A1: `config/assets/btcusdt.yaml` carga y `validate()==[]`.
- [x] B1: sin drivers COP-only; ≥1 cripto-native + DFII10 presente.
- [x] Migración 052 aplica de forma aditiva (no toca objetos existentes; hypertables best-effort).
- [ ] A2/A3/A4/C1: seeds ingeridos → rango, `bars_per_day==288` medido, UTC-native, barras de fin de semana presentes.
- [ ] Bundle BTC publicado en `registry.json` (activo `btcusdt`) sin regresión COP/Gold.
- [ ] Dashboard: selector muestra "Bitcoin · …", chart etiquetado **BTCUSDT**, replay con rango real.
- [ ] D5 (schedule 24/7) y D6 (live) resueltas o explícitamente diferidas por el operador.

## 9. Anti-patrones (NO hacer)

- **NO** crear una tabla `btc_m5_ohlcv` — reusar `usdcop_m5_ohlcv` por `symbol` (silo prohibido).
- **NO** anualizar con 15660/72000 (COP/Gold) — BTC es √365 → 105120.
- **NO** aplicar `forced_close`/cierre de viernes a BTC — exposición continua.
- **NO** leer ETF flows sin respetar `published_at` (D+1) — es look-ahead.
- **NO** copiar los thresholds de régimen de COP (Hurst 0.52/0.42) — re-fit sobre historia BTC.
- **NO** commitear API keys — `.env` gitignored + Vault.
