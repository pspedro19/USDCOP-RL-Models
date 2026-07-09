# BTC/USDT — Estrategia de Exposición Spot (paquete SDD)

Paquete de especificaciones para onboardear **Bitcoin (BTC/USDT)** como el **3.er activo tradeable**
del sistema (USD/COP fx · XAU/USD commodity · **BTC/USDT crypto**). Metodología **Spec-Driven
Development + TDD**: el activo entra por **config + datos**, no por código copiado.

> **Tesis (recordatorio).** En BTC la dirección de corto plazo es casi ruido. El alpha vive en
> **modular la exposición** — cuánto BTC spot tener (`exposure ∈ [0,1]`), no si comprar/vender —
> combinando ciclo on-chain (meses) + funding (días) + liquidez macro + un gate de eventos
> catastróficos. Spot-only: **elimina la liquidación por diseño**. El backtest espectacular es la
> señal de alarma, no el premio (DSR con conteo completo de trials, ver `design/`).

---

## Dos capas de este paquete (léelas en orden)

1. **`design/`** — la **ciencia de la estrategia** (movida desde `.claude/future plans/`, pre-registrada).
   Es autónoma del repo: define el motor de exposición, régimen HMM, gates y validación DSR.
   Entrada: `SPEC-00` (arquitectura de 5 capas) → `PRE-REGISTRATION.md` (constitución numérica) →
   `SPEC-01…12` + `adr/` + `config/`. **No se re-optimiza mirando resultados.**
2. **`specs/SPEC-13-scalable-integration.md`** — cómo esa ciencia **se enchufa al monorepo** existente
   (AssetProfile, reutilización de tablas, migración 052 crypto-native, registro dinámico, DAGs por
   fábrica, decisiones 24/7). Es el puente entre `design/` y la **columna vertebral multi-activo ya probada**.

Complementos: [`IMPLEMENTATION_ROADMAP.md`](./IMPLEMENTATION_ROADMAP.md) · [`IMPLEMENTATION_STATUS.md`](./IMPLEMENTATION_STATUS.md) · [`adr/ADR-log.md`](./adr/ADR-log.md)

---

## BTC se onboarda como un ACTIVO, no como un silo

Igual que el Oro (ver [`../xauusd/`](../xauusd/)), BTC reutiliza la maquinaria multi-activo /
multi-estrategia ya **implementada y probada sobre USD/COP** — no la copia:

1. **`AssetProfile`** (`config/assets/btcusdt.yaml`) parametriza todo lo pegado a COP: símbolo
   `BTC/USDT`, `chart_symbol BTCUSDT`, **sesión 24/7 UTC**, drivers macro cripto, régimen re-fit.
   El *keystone* que desbloquea el test de onboarding A1.
2. **Datos sin silos:** 5-min → `usdcop_m5_ohlcv` (multi-par por `symbol`), daily → `asset_daily_ohlcv`
   (migración 051). Lo **genuinamente nuevo** (on-chain, funding, ETF flows, eventos, señales de
   exposición) vive en tablas cripto-native → **migración 052** (`crypto_*`).
3. **Registro dinámico:** el pipeline publica bundles inmutables vía `register_bundle`; el frontend
   arma solo el selector Activo→Estrategia→Versión→Año + replay. Cero `strategy_id`/`symbol` hardcodeado.
4. **Fábrica de pipelines:** agregar BTC = una entrada de config → **0 DAGs nuevos** escritos a mano.

Reglas del sistema que este paquete respeta (en `.claude/rules/` + `.claude/specs/`):

| Regla / spec | Qué gobierna |
|---|---|
| [`assets/_onboarding-playbook.md`](../_onboarding-playbook.md) | Contrato `AssetProfile`, stages A–I, tests **A1–F1** |
| [`assets/_asbuilt-implementation.md`](../_asbuilt-implementation.md) | Sesión/timezone/anualización **por activo** (BTC = 24/7, √365) |
| [`platform/registry-lifecycle.md`](../../platform/registry-lifecycle.md) | Bundles inmutables, `registry.json`, replay, fábrica, tests **R1–R9** |
| [`.claude/rules/data-governance.md`](../../../rules/data-governance.md) | Golden rule TZ (BTC = carve-out UTC, no COT), UPSERT idempotente |
| [`.claude/rules/experiment-protocol.md`](../../../rules/experiment-protocol.md) | 5 seeds RL, 1 variable, validación estadística |

> **La diferencia dura vs COP/Gold:** BTC es **24/7** (sin sesión intradía, sin cierre de fin de
> semana, sin forced-close de viernes). Eso cambia anualización (√365), semántica de ejecución
> (exposición continua, no trade semanal) y añade datos cripto-native que **no existían** en el repo
> (funding, on-chain, ETF flows). Detalle y decisiones abiertas en SPEC-13 §24/7.

---

## Estado (honesto)

**BTC ONBOARDADO END-TO-END Y VISIBLE EN EL FRONT.** Ver [`IMPLEMENTATION_STATUS.md`](./IMPLEMENTATION_STATUS.md).
- **Fase 0** ✅ AssetProfile (`config/assets/btcusdt.yaml`), migración **052** crypto-native, tests A1/B1.
- **Fase 1** ✅ Data canónica real vía **Binance público (sin API key)**: daily 3,245 barras 2017→2026
  (cierre UTC 00:00) + 5-min 288/día con fines de semana. Tests A2/A3/A4/C1 verdes (7/7).
- **Fases 2/7/F** ✅ `src/btc_strategy/` (spot `exposure∈[0,1]`, √365) → backtest → **3 bundles
  publicados** al registro. **B2 Trend-follower: +351%, Sharpe 1.40, Calmar 1.83, MaxDD −11%, PROMOTE.**
  Dashboard arma solo el selector "Bitcoin" (chart **BTCUSDT**, replay 2018→2026). COP/Gold intactos.

**Gate honesto:** el motor S3 regime-gated **aún no supera** a B2 — necesita el HMM on-chain
(BGeometrics/funding), que entra por los extractores cripto-native pendientes → tablas 052. El
baseline B2 es el suelo publicado a batir. **Pendiente:** extractores cripto-native + refinamiento del
motor (Fases 3-6, 8-11 de `design/`).

---

*Aviso: diseño metodológico, no asesoría financiera ni promesa de rentabilidad. El trading de BTC
implica riesgo sustancial. Spot-only no elimina el riesgo de mercado, solo el de liquidación.*
