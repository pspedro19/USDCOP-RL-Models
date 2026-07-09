# SPEC-03 — Feature Engineering

## Propósito
Construir el conjunto **lean** de features (H1 técnicas + Daily de régimen + contexto macro rodante), todas **causales** y point-in-time correctas. Nada de 140 features: eso es carnada de overfitting.

## Features H1 (técnicas — para el estado del agente)
Todas calculadas solo con información hasta la barra actual (causales):

| Feature | Definición | Lib |
|---|---|---|
| `log_ret_1` | `log(close_t / close_{t-1})` | numpy |
| `rsi_14` | RSI(14) | pandas-ta |
| `z_sma` | `(close - SMA(n)) / std(n)`, n≈50 | numpy |
| `bb_pctb` | %B de Bollinger(20, 2) | pandas-ta |
| `macd`, `macd_sig` | MACD(12,26,9) línea y señal | pandas-ta |
| `atr_14` | ATR(14) normalizado por close | pandas-ta |

## Features Daily (para el clasificador de régimen — SPEC-04)

| Feature | Definición |
|---|---|
| `rvol_20` | std de log-returns diarios, ventana 20 |
| `atr_norm` | ATR(14)/close diario |
| `adx_14` | ADX(14) |
| `hurst` | Exponente de Hurst rodante (ventana 100–250). **Estimador ruidoso: suavizar con EMA y tratar como feature lenta, no switch.** |
| `z_sma50` | z-score de distancia a SMA50 |
| `macro_event_flag` | 1 si hay evento high-impact hoy/mañana |

**Estimador de Hurst:** R/S o DFA. Documentar cuál y su ventana. Suavizado EMA(≈10) sobre la serie de Hurst antes de usarla.

## Contexto macro (Daily) — correlaciones RODANTES, no supuestos fijos
La relación oro–DXY/tasas reales no es estable (2022–2025 la rompió). Se mide, no se asume:

| Feature | Definición |
|---|---|
| `corr_gold_dxy_90` | correlación rodante (90d) de retornos oro vs DXY |
| `corr_gold_real_90` | correlación rodante (90d) oro vs `DFII10` (cambios) |
| `dxy_trend` | signo/pendiente de SMA del DXY |
| `real_rate_level` | nivel de `DFII10` |
| `real_rate_chg` | cambio de `DFII10` (Δ n días) |
| `gold_trend` | SMA50 vs SMA200 del oro (regime de tendencia larga) |

El agente aprende **cuándo** la correlación importa según su valor vigente. Todo es contexto, nunca filtro duro.

## Normalización (anti-leakage)
- Escaladores (media/std o robust) se **ajustan SOLO en el tramo de train** de cada fold walk-forward, y se aplican a val/test.
- Para RL: usar `VecNormalize` de SB3 sobre observaciones, guardando las stats por fold (SPEC-08). En inferencia: `training=False`, `norm_reward=False`.
- Prohibido calcular estadísticas de normalización sobre el dataset completo.

## Alineación point-in-time
Las features Daily se adjuntan a H1 con el shift de 1 sesión definido en SPEC-02 §5. Las features H1 usan solo pasado. Test anti-look-ahead obligatorio.

## Interfaz
```python
# src/gold_rl/data/features/build.py
def build_h1_features(h1: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame: ...
def build_daily_features(daily: pd.DataFrame, macro: pd.DataFrame, cal: pd.DataFrame,
                         cfg: FeatureConfig) -> pd.DataFrame: ...
def assemble_state_frame(h1_feat, daily_feat, regime_labels) -> pd.DataFrame:
    """Une H1 + Daily(shift 1 sesión) + régimen. Salida = matriz de estado del entorno."""
```

## Criterios de aceptación
- [ ] Cada feature H1 y Daily calculada solo con pasado (test: recalcular en `t` con datos truncados a `t` da el mismo valor).
- [ ] Correlaciones rodantes presentes y con NaN correctos al inicio de la ventana.
- [ ] Hurst suavizado; test de que ventanas cortas no producen switches bruscos.
- [ ] Escaladores ajustados solo en train (test de leakage: stats de train no dependen de val/test).
- [ ] `assemble_state_frame` respeta el shift de 1 sesión (test point-in-time).
- [ ] Conteo total de features en el estado documentado y ≤ ~20 (lean).

## Dependencias
SPEC-02 (procesado). Consumido por SPEC-04, SPEC-05.
