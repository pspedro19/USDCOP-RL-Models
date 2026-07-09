# BTC/USDT — Roadmap de Implementación (por fases, con gate)

> Construye por fases; no avances sin pasar el gate (tests verdes) de la anterior. Combina el
> **onboarding del activo** (stages A–F del playbook) con las **fases de modelado** de `design/`
> (SPEC-00 §7). El gate honesto final: S3 solo se promueve si le gana a **ambos** baselines OOS en
> Calmar Y Sortino, con DSR positivo (conteo completo de trials).

## ⚠️ Conclusión empírica (2026-07) — Fase 1 es el bloqueo crítico, no un detalle

Medición honesta tras agregar Deflated Sharpe + OOS-2025 (ver `../_ds-cycle-asbuilt.md §8`):
`btc_trend_b2` full-history **+351%, Sharpe 1.40, DSR 0.999** — pero **OOS-2025: −1.4%, Sharpe −0.05,
p=0.62**. El número grande lo hicieron los bull cycles 2017/2021; fuera de tendencia se "pica" en los rangos.

**Causa raíz = features.** `src/btc_strategy/indicators.py` es **100% precio** (SMA/ADX/ATR/Hurst/vol). En
cripto el alfa es mayormente **no-precio**: funding, open interest, liquidaciones (derivados) + netflows,
MVRV, supply de stablecoins (on-chain). La **migración 052 define esas tablas pero NO hay extractor que las
llene** (grep: cero scripts de ingesta on-chain/funding). El S3 regime-gated *no puede* superar a B2 porque
le falta justamente su insumo (el HMM on-chain). **Fase 1 no es "onboarding de datos" — es el desbloqueo del
alfa.** Sin ella, ningún modelo (ML o reglas) sobre features de precio va a ganarle de forma robusta al HODL
(vol diaria 68%, kurtosis 15.7).

**Recomendación de arranque (barata, alto-ROI):** empezar Fase 1 por el **extractor de derivados** (funding
+ OI + liquidaciones desde la API pública de Binance, sin key de pago) → poblar `crypto_derivatives_daily` →
features `z_funding`/`oi_change`/`liq_imbalance`. Eso mide si hay señal ortogonal real **antes** de gastar en
data on-chain cara. Si no se va a ingerir esa data, lo honesto es tratar BTC como **HODL + vol-targeting**
(`btc_hodl_b1`) y no implicar que hay un modelo predictivo. Detalle completo en
`../../audit/STRATEGIC-ASSESSMENT-2026-07.md §4`.

> **Plan de implementación concreto (2026-07): [`PLAN-binance-derivatives-2026-07.md`](PLAN-binance-derivatives-2026-07.md).**
> Aterriza esta Fase 1 con la realidad de la API: **solo funding tiene historia profunda (~2019→hoy) →
> es lo único backtesteable en 2018-2026**; OI/long-short/taker son **forward-only (30 días)** y las
> liquidaciones no tienen REST público (WS o Coinglass). Por eso el primer entregable es el
> `funding_rate → z_funding` + una sub-estrategia `btc_trend_funding_s4` con gate honesto (DSR>0.95 ∧
> OOS-2025+ ∧ bate a B2). Llaves Binance en `.env` (no requeridas para market-data público).

| Fase | Bloque | Entregable | Gate |
|---|---|---|---|
| **0** | Onboarding A | `config/assets/btcusdt.yaml` (AssetProfile) + migración 052 + test A1/B1 | **A1, B1 verdes** ✅ |
| **1** | Onboarding B (datos) | Extractor Binance/CCXT (5-min+daily) → seeds; extractores cripto-native (BGeo/funding/Farside/DefiLlama) → tablas 052 | **A2, A3, A4, C1 verdes** (seed en rango, 288 bars/día, UTC, fin de semana presente) |
| **2** | design SPEC-10 | Baselines **B1** (HODL vol-targeted) y **B2** (cripto reglas) — ANTES de todo lo demás | B1/B2 backtesteados con CostModel (SPEC-07) — sin benchmark no hay gate |
| **3** | design SPEC-01 | Régimen HMM 4-estados fit-congelado sobre on-chain | H-REG-01, H-REG-02 |
| **4** | design SPEC-02 | Posicionamiento (`z_funding`) | H-POS-01 |
| **5** | design SPEC-03 | Combinación en riesgo `R=0.7·z_ciclo+0.3·z_funding` | H-CMB-01, H-CMB-02 (ortogonalidad) |
| **6** | design SPEC-04 | Gate de liquidez `M_liquidez∈[0.5,1.25]` | H-LIQ-01, H-LIQ-02 |
| **7** | design SPEC-06 = **S3** | Motor de exposición (vol-target √365 + bandas ±12.5% + cap portafolio) | **H-ENG-01, H-ENG-02 (le gana a B1 y B2 en Calmar OOS)** |
| **8** | design SPEC-05 | Gate de eventos `G` (LLM + vol-spike breaker) | H-EVT-01, H-EVT-02 |
| **9** | design SPEC-08 = **S4** | Meta-labeling (XGBoost, solo frena) | H-MET-01, H-MET-02 |
| **10** | design SPEC-09 = **S5** | RL táctico PPO (±10% NAV, opcional, 5 seeds) | H-RL-01 |
| **F** | Onboarding F | Publica bundle → `registry.json` (activo `btcusdt`) → visible en `/dashboard` (chart BTCUSDT) | Selector arma solo, sin regresión COP/Gold |
| **11** | design SPEC-12 | Retiro + shadow + monitores de drift | Protocolo §14 firmado |

## Ruta realista más rápida (recomendada)

Igual que el Oro: **paper + visibilidad web primero**, evitando `forecast_h5_*` y el OMS live.
Orden mínimo para "BTC visible en el dashboard con un baseline honesto":

```
Fase 0 (hecha) → Fase 1 (datos) → Fase 2 (B1/B2) → Fase 7 (S3) → Fase F (publicar bundle)
```

Las fases 3–6, 8–11 refinan S3; se construyen solo si el activo justifica la inversión (S3 debe batir
los baselines primero). Live (D6) queda diferido hasta validar en paper una semana.

## Decisiones que requieren al operador antes de producción

- **D5** — schedule de ingesta 24/7 (la fábrica debe emitir un DAG cripto, no reusar la ventana COP 8-12).
- **D6** — ejecución live (spot en exchange cripto vs paper). Ver SPEC-13 §5.
- **Keys** — provisionar `BGEOMETRICS_API_KEY`, `CRYPTOCOMPARE_API_KEY` (free-tier) en `.env`/Vault.
