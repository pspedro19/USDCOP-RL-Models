# Pitch Global Minds · Script maestro

> Guion con cue sheet completo de las 8 escenas del pitch (75s, 1080×1080p, 30fps).
> Toda la narrativa va por **kinetic typography + SFX** (sin voz humana).
> Números auditados desde `src/data/metrics.ts` → `PITCH_METRICS`.

---

## P01 · Hook (0–150f / 0–5s)

**Fondo**: Radial gradient `bg.secondary → bg.deep` con luces cyan/purple
**Música**: Ambient pad arrancando (fade-in 20f)

| Frame | Contenido on-screen | SFX |
|-------|--------------------|-----|
| 0–40 | `"Febrero 2026."` typewriter (cyan sutil) | tick cada char |
| 40–100 | `"El mercado USD/COP cambió."` typewriter (gradient cyan→purple) | tick cada char |
| 80+ | Equity curve stroke-dashoffset arrancando (ambiental) | — |

Transición a P02: whoosh SFX (-10f antes del cambio)

---

## P02 · Inicio (150–390f / 5–13s)

**Stage 1 (150–245f)**: Video S01 login en pantalla completa con zoom-in sutil
**Stage 2 (235–390f)**: Hub con 6 tarjetas de módulos staggered

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 150 | Whoosh entrada | whoosh |
| 160–245 | Playwright S01 (typing creds) con brightness 0.85 + slight scale 1→1.05 | — |
| 235–280 | Typewriter `"Trading cuantitativo end-to-end"` (gradient) | tick |
| 285 | Sub `"6 módulos integrados"` | — |
| 290+ | 6 ModuleCards staggered (Inicio, Dashboard, Producción, Forecasting, Análisis, SignalBridge) con delay 8f cada una | — |

Transición: whoosh (-10f)

---

## P03 · Dashboard (390–840f / 13–28s) ⭐ escena money shot

**Stage 1 (390–475f)**: OOS badge intro
**Stage 2 (470–670f)**: Backtest replay
**Stage 3 (660–840f)**: KPI metrics count-up

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 390 | Whoosh entrada | whoosh |
| 394 | OOS Badge "Validación Out-of-Sample 2025" pulsante (verde) | — |
| 400+ | Texto: `"Datos nunca vistos por el modelo en entrenamiento."` | — |
| 470 | Fade a Playwright S03-replay (backtest animado bar-a-bar 2025) | — |
| 470+ | Lower third `"▶ Backtest replay · 2025 reproducido bar-a-bar"` (cyan) | — |
| 630 | Fade a dark background para el reveal de métricas | riser (build-up) |
| 660 | OOS Badge "Resultados 2025 OOS" reaparece | impact (boom) |
| 670 | MetricCounter `+25.63%` verde count-up durante 45f | tick×3 |
| 690 | MetricCounter `Sharpe 3.35` gradient cyan-purple | tick×3 |
| 710 | MetricCounter `p=0.006` cyan (`pvalue` format) | tick×3 |
| 800 | Subtítulo `"34 trades · 82.4% win rate · MaxDD 6.12%"` | — |

Transición: whoosh (-10f)

---

## P04 · Producción (840–1200f / 28–40s)

**Stage 1 (840–930f)**: Split screen — producción live vs regime badge
**Stage 2 (920–1080f)**: WeekBlockingBars staggered
**Stage 3 (1080–1200f)**: Comparación +0.61% vs B&H -2.82%

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 840 | Whoosh entrada | whoosh |
| 850 | Split screen: izq S05-production-live + badge "● LIVE · /production", der RegimeBadge "MEAN-REVERTING H=0.280" | — |
| 870 | Texto: `"El régimen gate detectó un mercado de reversión a la media."` | — |
| 920 | Fade a fondo uniforme | — |
| 925 | Título: `"Q1 2026 · Semanas bloqueadas por el gate"` | — |
| 930+ | WeekBlockingBars: 14 barras, 13 se tachan con X roja (stagger 8f) | pop × 14 (soft) |
| 1042 | Texto final: `"13 / 14"` (rojo) + `"no se operó"` | — |
| 1080 | Comparison: `Global Minds +0.61%` (verde) vs `Buy & Hold -2.82%` (rojo) | — |

Transición: whoosh (-10f)

---

## P05 · Forecasting (1200–1470f / 40–49s)

**Fondo**: S07-forecasting-zoo video con brightness 0.45 + zoom 1→1.12

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 1200 | Whoosh entrada | whoosh |
| 1204 | Typewriter `"Model Zoo · Forecasting"` (gradient cyan-purple) | tick |
| 1245 | MetricCounter `9 Modelos ML` cyan | — |
| 1260 | MetricCounter `63 Backtests WF` purple | — |
| 1275 | MetricCounter `7 Horizontes` green | — |
| 1300+ | 9 pills staggered (Ridge, BR, ARD, XGBoost, LightGBM, CatBoost, Hybrid XGB, LGB, CAT) con delay 4f | — |

Transición: whoosh (-10f)

---

## P06 · Análisis (1470–1740f / 49–58s)

**Fondo**: S06-analysis-chat video con brightness 0.55 + gradient izquierdo
**Foreground**: alineado a la izquierda

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 1470 | Whoosh entrada | whoosh |
| 1480 | Badge "🧠 IA Semanal" (purple) | — |
| 1490 | Typewriter `"Análisis macro generado"` (white) | tick |
| 1525 | Typewriter `"por GPT-4o."` (gradient purple-cyan) | tick |
| 1560+ | Stats triplete: `16 Semanas` / `128 Gráficos` / `13 Variables` | — |

Transición: whoosh (-10f)

---

## P07 · SignalBridge (1740–2100f / 58–70s)

**Stage 1 (1740–1880f)**: Execution dashboard S08 + título
**Stage 2 (1870–2010f)**: Airflow I04 + stats
**Stage 3 (2010–2100f)**: Security callout + kill switch

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 1740 | Whoosh entrada | whoosh |
| 1748 | Video S08-signalbridge con brightness 0.55 | — |
| 1748+ | Typewriter `"Ejecución automatizada"` (gradient green-cyan) | tick |
| 1785 | Sub `"MEXC · Binance · CCXT · 7 risk rules"` | — |
| 1870 | Fade a I04-airflow (Airflow home con DAGs) | — |
| 1878+ | Panel overlay con `37 DAGs · 25 Servicios · 43 Migrations` | — |
| 2010 | Fade a fondo uniforme | impact |
| 2015 | Kill Switch badge "🛑" rojo pulsante | — |
| 2030+ | 4 tarjetas: AES-256 / 9-check chain / Circuit Breaker / Paper→Live (stagger 6f) | — |

Transición: whoosh (-10f)

---

## P08 · Outro (2100–2250f / 70–75s)

**Fondo**: Radial gradient + gradient mesh cyan/purple

| Frame | Contenido | SFX |
|-------|-----------|-----|
| 2100 | Whoosh entrada | whoosh |
| 2105 | Impact boom suave | impact |
| 2100+ | BrandLogoReveal `"Global Minds"` (letter-physics stagger) | — |
| 2150 | `"SIGNALBRIDGE"` subtitle (staggered letters) | — |
| 2155 | DisclaimerRoll (fade in 20f) con texto legal | — |
| 2210 | `"globalminds.ai · 2026"` (cyan) fade in | — |
| 2220–2250 | Música fade-out | — |

---

## Cue sheet SFX consolidada

| Frame | SFX | Volumen | Duración |
|-------|-----|---------|----------|
| 140 | whoosh | 0.55 | 30f |
| 380 | whoosh | 0.65 | 30f |
| 630 | riser | 0.50 | 45f |
| 670 | impact | 0.75 | 60f |
| 690/710/730 | number-tick × 3 | 0.5 | 10f |
| 830 | whoosh | 0.60 | 30f |
| 930–1038 | pop × 14 | 0.25 | 8f |
| 1190 | whoosh | 0.60 | 30f |
| 1460 | whoosh | 0.55 | 30f |
| 1730 | whoosh | 0.60 | 30f |
| 2010 | impact | 0.55 | 60f |
| 2090 | whoosh | 0.60 | 30f |
| 2105 | impact | 0.40 | 60f |

**Ducking**: música baja a 30% durante impactos y risers (ventanas de 12f).

---

## Verificación números (paths auditados)

| Mostrado | Valor | Fuente |
|----------|-------|--------|
| +25.63% | `PITCH_METRICS.oos_2025.return_pct` | `summary_2025.json` |
| Sharpe 3.35 | `.sharpe` = 3.347 | idem |
| p=0.006 | `.p_value` = 0.0063 | idem |
| 34 trades | `.trades_total` | idem |
| 82.4% WR | `.win_rate` | idem |
| 6.12% MaxDD | `.max_dd_pct` | idem |
| +0.61% | `PITCH_METRICS.ytd_2026.return_pct` | `summary.json` |
| 13/14 | `.gate_blocked_weeks / .total_weeks` | derivado regime gate |
| -2.82% BH | `.buy_and_hold_pct` | `summary.json` |
| H=0.280 | `PITCH_METRICS.regime_gate.hurst_2026_q1` | `regime_gate.py` |
| 37 DAGs | `PITCH_METRICS.infra.airflow_dags` | audit 2026-04-16 |
| 9 modelos | `.ml_models` | idem |
| 63 backtests | `.backtests_run` | 9×7 horizonte |

---

## Safe zones respetadas

- Landscape 1920×1080: top/bottom 60px, sides 80px
- Texto mínimo: 28px (headline 72px, body 32px, caption 20px)
- Disclaimer P08: 20px pero en zona interior de 1200px width, line-height 1.6 = legible
