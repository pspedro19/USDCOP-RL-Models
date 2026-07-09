# SPEC-06 — Capa de Riesgo (Vol-Targeting)

## Propósito
Módulo **determinista, puro y unit-testable** que convierte la dirección del agente en un tamaño de posición, aplicando vol-targeting, multiplicador de régimen, blackouts, flat de fin de semana y circuit breakers. **Aquí vive el alpha.** Vive DENTRO del entorno (SPEC-05).

## Función principal — position sizing

```python
def position_size(direction: int, st: RiskState, cfg: RiskConfig) -> float:
    if _is_blackout(st) or _is_weekend_flat(st) or st.breaker_active:
        return 0.0
    vol = max(st.realized_vol_h1, st.realized_vol_daily)     # manda la mayor
    raw = (cfg.target_vol / max(vol, cfg.vol_floor)) * cfg.base_size
    size = raw * _regime_mult(st.regime, cfg) * direction
    return float(np.clip(size, -cfg.max_leverage, cfg.max_leverage))
```

### Vol-targeting
```
tamaño = (vol_objetivo / vol_realizada) · base · mult_régimen · dirección
```
- `vol_objetivo`: anualizada, **10–15%**.
- `vol_realizada`: **EWMA de retornos H1** (half-life 20–50 barras), pero **capada por la vol Daily**: usar `max(vol_h1, vol_daily)`. Si la diaria explota y la H1 aún no lo refleja, manda la diaria.
- `vol_floor`: evita división por vol ≈ 0 (tamaños absurdos en calma extrema).

### Multiplicador de régimen (determinista, además de informar al agente)
| Régimen | mult |
|---|---|
| Compresión | 1.0 |
| Tendencia | 1.0 |
| Estirado | 0.5–0.75 |
| Event-driven | 0.25–0.5 |

## Circuit breakers (límites duros)
| Breaker | Regla | Acción |
|---|---|---|
| `MAX_RISK_PER_TRADE` | 2% del capital por trade | cap del tamaño |
| `MAX_DAILY_LOSS` | −5% en el día | flat + no operar resto del día |
| `MAX_DRAWDOWN` | umbral de DD de cuenta | **desactiva el sistema**, revisión manual |
| `MAX_POSITIONS` | exposición acumulada | rechaza nueva exposición |
| **`VOL_SPIKE`** | `ATR_h1 > k·EWMA(ATR)`, k≈3 | **flat + pausa N barras** |

**Vol-spike breaker** cubre los eventos **no calendarizados** (geopolítica, flash crashes, bancos centrales sorpresa). El oro es el activo geopolítico por excelencia; el calendario solo cubre lo programado. Este breaker es lo que atrapa lo demás.

## News blackout
- Alrededor de eventos USD high-impact (CPI, NFP, FOMC): **aplanar ±30–60 min** (config).
- **Regla de re-entrada:** no volver por reloj ("pasaron 60 min") sino por **normalización de vol**: `ATR_h1 < 1.5 · EWMA(ATR)`. Evita entrar en el pico de volatilidad post-release.

## Flat de fin de semana
- **Aplanar el viernes antes del cierre NY** (última hora de sesión). El oro gapea el domingo con noticias geopolíticas; un gap en contra no respeta stops.
- Beneficio secundario: elimina 2 swaps/semana. Si algún día quieres capturar drift de fin de semana, que sea decisión medida contra datos, no default.

## Configuración (`config/risk.yaml`)
```yaml
target_vol: 0.12            # 12% anualizado
base_size: 1.0
vol_floor: 0.02
max_leverage: 3.0
regime_mult: {COMPRESSION: 1.0, TREND: 1.0, STRETCHED: 0.6, EVENT: 0.35}
breakers:
  max_risk_per_trade: 0.02
  max_daily_loss: 0.05
  max_drawdown: 0.20
  vol_spike_k: 3.0
  vol_spike_pause_bars: 6
blackout:
  window_min: 45
  reentry_atr_mult: 1.5
weekend_flat:
  friday_close_hour_ny: 16
```

## Criterios de aceptación
- [ ] Funciones puras: mismo `RiskState` ⇒ mismo tamaño (sin efectos colaterales; test).
- [ ] Vol-targeting: sizing inversamente proporcional a vol (test con vol sintética alta/baja).
- [ ] `max(vol_h1, vol_daily)` aplicado (test: diaria alta reduce tamaño aunque H1 sea baja).
- [ ] Cada breaker se dispara en su condición y fuerza el efecto correcto (tests unitarios por breaker).
- [ ] Vol-spike breaker aplana ante salto de ATR sintético.
- [ ] Blackout aplana en ventana y re-entra solo cuando la vol normaliza (test).
- [ ] Weekend flat aplana el viernes (test).
- [ ] Multiplicador de régimen aplicado por estado (test).

## Dependencias
SPEC-04 (régimen), consumido por SPEC-05, SPEC-07.
