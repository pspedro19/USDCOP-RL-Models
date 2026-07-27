---
kind: roadmap
status: PLANNED
version: 1.1.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - config/assets/spx500.yaml
  - config/assets/pipelines.yaml
  - config/strategy_manifests/spx500.yaml
  - scripts/pipeline/run_spx500_pipeline.py
  - src/strategies/spx500_regime_gated_v1/load_real.py
---

# Plan de rentabilidad SPX500 / SPY

> Decisión del operador: **Investing.com es la SSOT canónica de OHLC diario del S&P 500**.
> Objetivo: reemplazar el circuito sintético por uno causal daily, reparar los baselines y
> acumular forward next-open. SPX no recibe trials direccionales nuevos.

## 1. Estado honesto y retractaciones

- Existe un snapshot SPY previo, pero deja de ser la autoridad del nuevo track. Se conserva
  como evidencia histórica/contraste; no se mezcla con Investing dentro de una corrida.
- **El DAG científico no usa ese loader.** `run_spx500_pipeline.py` delega a
  `run_strategy.main()`, que llama `datagen.generate()`. El wiring actual prueba código,
  no mercado.
- El Calmar 1.641 atribuido a `dumb_ma200_always_on` queda **RETRACTADO** hasta recomputar:
  el adapter entrega `trend_on` sin el shift next-open y el evaluador descuenta los costos
  de la estrategia gated, no el turnover propio del baseline.
- `spx500_regime_gated_v1` tiene evidencia real solo por un adapter paralelo y permanece
  research-only. Su DSR no supera de forma robusta 0.95 y PBO no es evaluable con bundles
  publicados actuales.
- El registry declara 17 trials, pero su cuerpo conserva sumas 14/16. Se reconcilia antes de
  abrir cualquier resultado corregido.
- No hay instrumento de ejecución definido. La serie de Investing es un **price index** de
  investigación; no incluye dividendos y no se presenta como total-return ni como fill de SPY.

## 2. Contrato económico y multi-timeframe

| Rol | Símbolo objetivo | Uso |
|---|---|---|
| research/signal | S&P 500 price index de Investing | OHLC diario, MA200, señal y benchmark price-return |
| raw authority | payload diario Investing append-only | auditoría, checksum, parser y disponibilidad |
| execution | **TBD por operador** | CFD/ETF/futuro con tracking, costos, financiación y contrato propios |

| Tabla | Función permitida |
|---|---|
| 5m | `NOT_REQUIRED` para v1; solo se añade al homologar un broker |
| 1h | `NOT_REQUIRED` para v1 |
| 4h | `NOT_REQUIRED` para v1 |
| daily | única fuente de señal: cierre t y ejecución paper al open t+1 |

Reloj 252, sesión `America/New_York`, DST desde `market_session_calendar`. No se rellena un
feriado de SPX con un retorno BTC ni se usa COT como calendario del activo.

## 3. Plan por dependencias

### S0 — Reparar evidencia y runner (P0, 0 trials nuevos) — **DONE 2026-07-27**

| ID | Acción | PASS | Estado |
|---|---|---|---|
| S0.1 | Hacer que el entrypoint cargue `load_real()` | test monkeypatch demuestra que `datagen.generate()` no se invoca; falta de snapshot falla cerrado | ✅ (default real fail-closed; `--synthetic` es opt-in explícito) |
| S0.2 | Unificar engine de candidato y baselines | todos pasan por `BacktestEngine`, shift exactamente una vez, turnover/costos propios | ✅ (runner delega al motor de `profitability_evidence`; bit-check Calmar 0.6482/0.5393 MATCH exacto vs artefacto 2026-07-21) |
| S0.3 | Recomputar MA200 causal | posición de decisión → `weights_exec`; 3 bps propios ×1/×2/×3; B1 y B1′ en la misma ventana | ✅ (ERRATUM 2026-07-21 + re-verificado hoy) |
| S0.4 | Emitir erratum de evidencia | artefacto anterior marcado inválido, no sobrescrito; nuevo hash, código, datos y motivo | ✅ (2026-07-21) |
| S0.5 | Reconciliar 17 trials | front matter, cuerpo y evidencia coinciden; no se reduce por descubrir un bug | pendiente (contable, no bloquea S3) |

Snapshot SPY refrescado 2026-07-27 (1,648 filas, 2020-01-02→2026-07-24, sha `816448b4`;
el anterior respaldado). Evidencia full-window regenerada: DSR 0.873 (antes 0.8813 —
solo creció la ventana), mismo veredicto **research_only**, mismos gates fallados.

Corregir una medición defectuosa de una hipótesis ya registrada no crea licencia para una
hipótesis nueva. El resultado corregido pertenece a `H-SIMP-SPX-02`, conserva el conteo
conservador y no selecciona parámetros. Si se cambia MA/window/target, sí es otro trial.

### S1 — Contrato Investing daily (P0, 0 trials)

1. Configurar `investing_pair_id`, URL/referer, timezone y símbolo canónico en el AssetProfile.
2. Adaptar `_investing_daily()` para SPX en modo **authoritative/fail-closed**. El código actual
   la trata como cross-check best-effort y deja a TwelveData como autoridad; ese comportamiento
   no sirve para esta decisión.
3. Guardar el payload bruto de cada adquisición con `source_url`, `instrument_id`,
   `extracted_at`, rango pedido, checksum y versión del parser. No sobrescribir snapshots.
4. Normalizar `session_date_ny`, `bar_start_utc`, OHLC y volumen si existe. `available_at`
   será posterior al cierre publicado; una fecha reconstruida se etiqueta, no se llama vintage.
5. Prohibir fallback silencioso a TwelveData, Stooq, yfinance o datos sintéticos. Si Investing
   falla, el DAG queda stale/failed y conserva el último snapshot válido.
6. Validar duplicados, OHLC, huecos contra NYSE, cambios de escala, revisiones y divergencia
   entre payload nuevo y snapshot anterior antes del upsert.
7. La v1 usa **MA200 daily long/cash** y no consume VIX/NFCI/HY-OAS ni proxies. El regime-gated
   queda pausado hasta tener variables reales bajo un contrato separado.

**Gate S1:** 100% de barras usadas provienen de Investing, `available_at <= decision_at`,
price-return declarado y cero mezcla/fallback. La ausencia de intradía no bloquea v1.

### S1b — Variables reales de régimen: ENTREGADO 2026-07-27 (0 trials)

El "contrato separado" de variables que S1.7 exigía ya existe: **9 series FRED** ingresadas
al L0 (migración `067_spx500_regime_macro_vars.sql`, SSOT + `extractors/config.yaml`,
backfill completo 44,749 obs, mantenimiento automático vía `core_l0_04_macro_update`):
HY-OAS, curvas 10y-2y/10y-3m, breakeven 10y, real yield 10y (DFII10 — también frontera de
Oro), NFCI, STLFSI4, initial claims, regla de Sahm. VIX ya vivo vía investing (curl_cffi,
2026-07-27). Detalle, priors e historia por serie: evento de datos en
`HYPOTHESIS-REGISTRY.md` (spx500).

**Esto NO reabre el regime-gated por sí solo.** La secuencia legal sigue siendo: cerrar
S0 (evidencia real) → pre-registrar la familia de features de régimen COMPLETA (cada
celda = +1 trial), screening SOLO ≤2024 con purged K-fold, bar = batir al MA200 causal en
Calmar con costos propios → si una celda gana, UNA variante económica con juez forward.
Dirección/DA no se toca: sin lift vs baseline en ningún activo (constitución §1).

#### Contrato de ALINEACIÓN del dataset (directiva operador 2026-07-27)

**Regla**: el dataset de cualquier variante que consuma series externas empieza en
`max(min_fecha de las series ELEGIDAS)` y termina en `min(max_fecha)` — todas las filas
con TODAS sus series pobladas (las semanales/mensuales entran ffill-acotado + shift T-1).
Elegir series ES elegir ventana; se declara ANTES del screening. Disponibilidad medida
en DB (2026-07-27):

| Serie | Desde | Costo de incluirla |
|---|---|---|
| claims (ICSA) | 1967 | ninguno (pre-data SPX) |
| NFCI | 1971 | ninguno |
| curva 10y-2y | 1976 | ninguno |
| curva 10y-3m | 1982 | ninguno |
| STLFSI4 | 1993 | ninguno |
| **SPX oficial** | **1995-01-03** | — (el piso del activo) |
| real yield 10y / breakeven 10y | 2003 | pierde 1995-2002 (dot-com parcial) |
| DXY / VIX (investing en DB) | 2020 | **pierde los 3 osos** (backfill investing pendiente, 0 trials) |
| HY-OAS (licencia ICE) | 2023-07 | **mata la ventana** — solo forward-going |

La estrategia congelada actual consume SOLO precio (proxies derivados, shift(1)) ⇒ su
ventana alineada es el rango completo 1995→. VIX/DXY profundos vía backfill investing
es la pre-tarea de datos (0 trials) si la familia los quiere sin sacrificar osos.

### S2 — Separar los ciclos operativos (0 trials)

El DAG actual ejecuta L4 diariamente bajo un ID “weekly” y no tiene L5 de señal. El objetivo:

| Ciclo | Cadencia | Responsabilidad |
|---|---|---|
| L0 | diario post-cierre | ingesta SPY/SPX/macro, validación y manifest |
| L4 | event-driven por versión | replay/backtest, gates, bundles inmutables; no retestar cada día |
| L5 | diario post-cierre | calcular señal congelada y publicar orden paper next-open |
| L6 | diario + corte semanal | staleness, fill, PnL, DD, tracking y ledger |

`normalize_champions.py` no debe llamar campeón de producción a algo con bundle sintético.
Hasta cerrar S0/S1, SPX queda visible como experimental con `promotion_eligible: false`.

### S3 — Baseline contra gated, sin trial nuevo — **DONE 2026-07-27**

**Publicado**: `scripts/pipeline/publish_spx500_bundles.py` (invocado por el default real
de `run_spx500_pipeline.py` → el stage l4 del DAG factory publica end-to-end; inmutable
por (version, año) ⇒ re-runs = `immutable_hit`, idempotente). Bundles v1.0.0 con
`summary/trades/signals_{2025,2026}.json` para ambas estrategias + registry refresh;
`pipelines.yaml` verify exige las dos. Tests 31/31. Números (SPY total-return, motor
del ERRATUM; N<20 ⇒ solo conteo y PnL):

| Estrategia | 2025 | 2026 YTD | Lectura |
|---|---|---|---|
| `spx500_regime_gated_v1` | +7.68% (DD 6.19%, 5 seg.) | −0.57% (DD 5.44%, 3 seg.) | el gate pagó prima en años toro |
| `spx500_daily_ma200_v1` | +9.85% (DD 11.07%, 3 seg.) | +4.96% (DD 5.82%, 2 seg.) | **el baseline va ganando el forward-context** |
| B1 buy&hold | +16.34% | +7.70% | el listón real de SPX |

El histórico corregido es contexto; el forward decide (S4). Si el baseline no es batido,
el baseline ES la estrategia. `research_only` se mantiene (DSR 0.873 < 0.95).

Diseño original de la sección (referencia):

Después de S0, se publican dos bundles explícitos ya cubiertos por el registro:

- `spx500_daily_ma200_v1`: MA200 causal, sin vol target ni regime gate; candidato operativo
  simple y baseline a la vez;
- `spx500_regime_gated_v1`: contrato actual, con drivers reales o proxy declarado.

Se congela ambos en la misma fecha para `H-SIMP-SPX-02`. El histórico corregido es contexto;
el forward decide. Si el baseline no es batido en Calmar/MaxDD/costos, **el baseline es la
estrategia**. No se crea challenger adicional para rescatar la gated.

Comparación mínima:

- B1 del mismo S&P 500 price index de Investing, B1′ de exposición emparejada y cash;
- MA200 y gated con costo/turnover propios;
- costos ×1/×2/×3;
- upside/downside capture y 2020/2022 como estrés descriptivo;
- delta Calmar block-bootstrap, declarado no concluyente si IC95 incluye cero.

### S4 — Protocolo, broker y forward (0 trials)

Crear y firmar `WITHDRAWAL-PROTOCOL-SPX.md` antes de paper:

- estrategia(s)/hash, fecha de inicio y ventana ≥26 semanas;
- fill next-open, benchmark, tracking y costos del instrumento real;
- retiro por DD >15%, error de datos, divergencia o breaker;
- graduación solo con ≥30 oportunidades, más de un régimen, Calmar/MaxDD firmados y costos ×2;
- capital real únicamente después de homologar SPY/CFD/futuro, sizing unit-correct y una semana
  técnica sin errores. Si el instrumento es SPY, CFD, ES o MES, su tracking contra el índice
  Investing se mide explícitamente; nunca se usa el índice como precio de fill.

El ledger debe guardar señal del cierre, orden, open de fill, costo, fuente y revisión. Un
replay sin paper no cuenta como forward.

## 4. Integración al libro

SPX entra al ERC diario solo con retornos corregidos y hash congelado. El libro usa intersección
estricta con XAU/BTC, Ledoit-Wolf pasado, correlaciones co-activas y efectivo. Si cambia el
campeón SPX, el libro abre nueva versión; no reescribe su historia.

## 5. No hacer

- no citar Calmar 1.641 hasta cerrar S0.3;
- no usar el DAG sintético para publicar/promover;
- no mezclar Investing con Twelve/Stooq para rellenar fechas;
- no llamar total-return a la serie price-index de Investing;
- no barrer MA {150,200,250} ni vol targets para escoger ganador;
- no añadir horizonte de siete semanas ni otro modelo direccional;
- no sustituir VIX con realized vol sin etiquetar que es otro modelo;
- no ejecutar el índice SPX como si fuera un instrumento negociable.

## 6. Verificación y terminado

```powershell
python scripts/pipeline/run_spx500_pipeline.py --check
python -m pytest tests/integration/test_spx500_pipeline_config.py -q
python -m pytest tests/regression/test_spx500_integration.py -q
python -m pytest tests/unit/test_sp500_oos_gate.py -q
python scripts/pipeline/normalize_champions.py --check
python -m scripts.analysis.forward_tracker --report
```

Terminado significa: Investing daily fail-closed, raw snapshots auditables, MA200 causal,
L5/L6 paper operativos, protocolo firmado y forward suficiente. 5m/1h/4h no son requisito
para esta versión. Hasta entonces SPX es investigación, aunque el DAG esté verde.
