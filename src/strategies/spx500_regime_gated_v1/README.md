# `strategy/` — Estrategia SP500: el patrón B1 / B2 / S3

Capa de estrategia para `sp500-alpha-research`. Implementa las **tres políticas de
exposición** al S&P 500 y las corre contra los cuatro benchmarks obligatorios y los
gates del charter. Es **autocontenida y runnable hoy**: reusa los kernels de ciencia
ya implementados en la raíz del repo (`deflated_sharpe.py`, `pbo.py`,
`economic_metrics.py`, `gates.py`) sin duplicarlos ni tocar la suite `sp500.*` (que
sigue roja por diseño, red-first).

> **Honestidad primero.** La demo corre sobre **datos SINTÉTICOS** (`datagen.py`).
> Las métricas miden el *cableado del pipeline y la severidad de los gates*, **no alfa
> real**. Sobre ruido sintético lo correcto es que los gates rechacen a las candidatas
> (H6). Para evidencia real, enchufá SPY total-return + FRED (§4).

---

## 1. Las tres estrategias (patrón B1/B2/S3)

| ID (`strategy_id`) | Qué es | Hipótesis | Rol |
|---|---|---|---|
| `spx_hodl_b1` | HODL con vol-targeting (σ objetivo 10%) | **H1** | Baseline honesto — el listón real |
| `spx_trend_b2` | Trend follower: MA(200) **y** TSMOM 12-1, vol-scaled | **H3** | El que suele ganar |
| `spx_regime_gated_v1` | Tendencia (B2) × techo de exposición por régimen | **H2+H3** | La hipótesis a probar |

Se publican **las tres**. Regla del `quant-constitution`: si S3 no bate a B2 en Sharpe
y Calmar bajo **costos ×2** + DSR/PBO, entonces **B2 es la estrategia** (pasó con BTC).

### Decisión de diseño específica de SP500 (no heredada de COP)
- **El VIX es la propia volatilidad implícita del SPX** → usarlo como feature
  *predictivo* roza la circularidad. Aquí entra sólo como **etiqueta de estado**
  (régimen `high_vol`/`bull`) y como *haircut* de exposición, nunca como alfa direccional.
- El sesgo direccional lo aportan **tendencia** (MA200, TSMOM) y un **driver macro
  exógeno** (proxy NFCI/HY-OAS). Coherente con la premisa del charter: *gestionar
  exposición, no predecir dirección*.
- **Umbrales de régimen:** no se copian los de COP (0.52/0.42). Se parte de un pivote
  neutral explícito (`RegimeConfig`); ajustarlos sería un trial registrado.

---

## 2. Cómo correrlo

```bash
cd strategy
python run_strategy.py                       # demo end-to-end (data sintética)
python run_strategy.py --seed 7              # otra realización
python -m pytest test_strategy.py -c pytest.ini -q   # 10 tests, verdes hoy
```

> El `pyproject.toml` de la raíz fuerza pytest-8 + `--cov=sp500`; por eso los tests de
> esta capa usan `-c pytest.ini` (config local aislada).

---

## 3. Qué verifica (mapeo a los gates del charter)

El runner corre los 7 activos (4 benchmarks + 3 políticas) a costo base y ×2, y aplica
sobre la candidata S3:

| Gate | Qué exige | Kernel real usado |
|---|---|---|
| **G3** utilidad económica | `CER_gain vs VOL_TARGET_10 > 0` (γ=5) | `economic_metrics.cer_gain_bps` |
| **G4** significancia | `DSR > 0.95` con **N=989** trials **y** `PBO < 0.50` | `deflated_sharpe`, `pbo.cscv` |
| **G5** robustez de régimen | bate a vol-target en **≥3/4** regímenes (bull/bear/high_vol/sideways) | `metrics.regime_robustness` |
| **G6** supervivencia a costos | `break_even ≥ 3×2 pb = 6 pb` | `costs.break_even_cost`, `gates.gate_g6` |

Los regímenes se etiquetan **PIT** (estadísticas expanding, sin look-ahead), tal como
exige SDD-000 §6. El motor de backtest ejecuta **next-open** con el `shift(1)` aislado
en una sola línea (`engine._lag_weights`) y testeado.

**Resultado típico sobre data sintética:** S3 luce el mejor Sharpe pero **falla G4
(DSR≈0.89, PBO≈0.81) y G5 (2/4)** → veredicto **H6, NO PROMOTE**. Esa es la
demostración: el DSR con N=989 no aprueba un Sharpe de ~1.0 (Bailey-López de Prado).

---

## 4. Enchufar datos reales

`datagen.generate()` devuelve el contrato de columnas que consume el motor:
`open_to_open_return`, `close`, `vix`, `macro_stress`. Para correr con datos reales,
reemplazá esa fuente por un loader que devuelva el mismo DataFrame:

- **Precio:** SPY **total-return** (dividendos reinvertidos), NO `^GSPC` (SDD-000 §4:
  usar el índice price infla el alfa ~1.8%/año).
- **VIX:** `FRED:VIXCLS` (arranca 1990-01-02; define el inicio de muestra).
- **Macro (para el overlay de S3):** `FRED:NFCI`, `FRED:BAMLH0A0HYM2` (HY OAS),
  term spread `FRED:T10Y2Y` — con sus lags de publicación (ver `registry.py` raíz).
- **60/40:** añadir `IEF` para el benchmark `SIXTY_FORTY` real (hoy es proxy 0.6 const).

El resto del pipeline (regímenes, políticas, gates) no cambia.

---

## 5. Skills de la librería que fundamentan esto

De `FINAL SKILLS/`, el blueprint del gate de régimen es la cadena de `exposure-coach`:

- `01-market-regime/exposure-coach` — agrega breadth/régimen/flujo en un techo de exposición → `regime.exposure_ceiling`
- `01-market-regime/{macro-regime-detector, market-breadth-analyzer, uptrend-analyzer, market-top-detector, ftd-detector}` — los estados que alimentan el overlay
- `13-cross-asset-quant/xasset-alpha-engine` — trend + vol-targeting + purged CV + Deflated Sharpe + PBO (el núcleo quant)
- `04-backtesting-validation/{backtest-expert, volatility-modeling, statistics-fundamentals}` — robustez, vol realizada, resampling

> Nota: al onboardear SP500, los detectores US-equity (`ftd-detector`,
> `ibd-distribution-day-monitor`, monitores de distribución/breadth) pasan de ruido a
> candidatos reales del overlay de régimen.

---

## 6. Archivos

| Archivo | Contenido |
|---|---|
| `datagen.py` | Mercado sintético tipo SPY con regímenes (Markov 2 estados) |
| `regime.py` | Clasificador PIT de régimen + techo de exposición (`exposure-coach`) |
| `benchmarks.py` | Los 4 benchmarks obligatorios (SPY_TR, 60/40, VOL_TARGET_10, MA200) |
| `policies.py` | Las tres estrategias B1/B2/S3 |
| `costs.py` | Modelo de costos + break-even (G6) + 5 escenarios de estrés |
| `engine.py` | Motor next-open con el `shift(1)` aislado |
| `metrics.py` | Tabla de métricas + puentes DSR/PBO/régimen |
| `kernels.py` | Puente a los kernels de ciencia de la raíz |
| `run_strategy.py` | Runner end-to-end con veredicto de gates |
| `test_strategy.py` | 10 tests runnable (next-open, costos, régimen PIT, DSR con dientes) |
