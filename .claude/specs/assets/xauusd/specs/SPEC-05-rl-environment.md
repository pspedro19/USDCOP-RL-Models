# SPEC-05 — Entorno RL (Gymnasium)

## Propósito
Definir `GoldTradingEnv`: el entorno donde el agente decide **dirección** y la **capa de riesgo (SPEC-06) vive dentro**, de modo que el agente entrena sobre el PnL real post-sizing/post-costos. Evita el train/serve skew.

## Contrato Gymnasium

```python
class GoldTradingEnv(gymnasium.Env):
    def __init__(self, state_frame: pd.DataFrame, risk: RiskLayer,
                 cost_model: CostModel, cfg: EnvConfig): ...
    def reset(self, *, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action: int) -> tuple[obs, reward, terminated, truncated, info]: ...
```

### Observation space
`Box(low=-inf, high=inf, shape=(W, F))` o aplanado — ventana de `W`=48–96 barras H1 × `F` features (SPEC-03), **más** el bloque de contexto no-secuencial (régimen one-hot, macro rodante, estado de cuenta) concatenado. Si se usa `RecurrentPPO`, la parte secuencial la maneja el LSTM (ver SPEC-08); pasar la ventana o dejar que el estado recurrente acumule — documentar la elección.

**Contenido del estado** (de SPEC-03 §assemble_state_frame):
- Features técnicas H1 (ventana): `log_ret_1, rsi_14, z_sma, bb_pctb, macd, macd_sig, atr_14`.
- Contexto Daily (point-in-time D-1): régimen one-hot (4), `corr_gold_dxy_90, corr_gold_real_90, real_rate_level, real_rate_chg, dxy_trend, gold_trend`.
- Estado de cuenta: `position ∈ {-1,0,1}`, `unrealized_pnl`, `bars_in_position`.
- Temporizadores: `bars_to_next_event`, `bars_to_weekend_close`.

### Action space
`Discrete(3)` → `{0: short(-1), 1: flat(0), 2: long(+1)}`. **Solo dirección.** El tamaño lo calcula la capa de riesgo.

### Reward — Differential Sharpe (Moody & Saffell)
Recompensa ajustada por riesgo, online, sobre el **retorno neto ya dimensionado y con costos** (spread + slippage + swap, SPEC-08 CostModel):

```python
# retorno del paso, post-sizing y post-costos:
r_t = size_t * direction_t * price_return_t - transaction_cost_t - swap_cost_t

# EMAs para Differential Sharpe (η ~ 1e-2 a 1e-3):
dA = r_t - A_prev
dB = r_t**2 - B_prev
denom = (B_prev - A_prev**2) ** 1.5
D_t = (B_prev * dA - 0.5 * A_prev * dB) / denom if denom > 1e-12 else 0.0
A = A_prev + eta * dA
B = B_prev + eta * dB

reward = D_t - lam_c * turnover_t - lam_d * dd_penalty_t - lam_f * flip_t
```
- `flip_t`: 1 si la acción invierte la posición (long→short o viceversa). Penaliza el churn direccional (modo de fallo clásico de RL en H1). Alternativa equivalente: cooldown de N barras tras cerrar (implementar una de las dos, no ambas).
- `dd_penalty_t`: función del drawdown corriente del episodio.
- Coeficientes `eta, lam_c, lam_d, lam_f` en `config/env.yaml`, tuneados por walk-forward.

### step() — secuencia por barra
1. Recibe `action` (dirección deseada).
2. Consulta SPEC-06: `size = risk.position_size(direction, state)` (aplica vol-targeting, multiplicador de régimen, blackouts, breakers, flat de fin de semana → puede forzar `size=0`).
3. Calcula `r_t` con `CostModel` (spread variable por hora + slippage + swap si cruza overnight).
4. Actualiza EMAs y computa `reward`.
5. Avanza una barra; `terminated` al fin del slice, `truncated` si un breaker duro apaga el episodio.
6. `info` expone: `size, direction, r_t, costs, regime, breaker_triggered, equity` (para logging y atribución SPEC-09).

### Episodio
Un episodio = un slice temporal del fold walk-forward (SPEC-09), recorrido barra a barra. `reset` reinicia equity, EMAs (A,B), posición y drawdown.

## Consistencia train/serve
El mismo `GoldTradingEnv` + `RiskLayer` + `CostModel` se usa en backtest y como motor de decisión en paper/live (SPEC-11), garantizando misma distribución. Cualquier lógica de riesgo que exista en live DEBE existir en el entorno de entrenamiento.

## Criterios de aceptación
- [ ] Pasa `gymnasium.utils.env_checker.check_env`.
- [ ] `reward` computado sobre retorno post-sizing/post-costos (test: costos afectan reward).
- [ ] Determinismo: mismo seed + misma secuencia de acciones ⇒ misma trayectoria (test).
- [ ] La capa de riesgo puede forzar `size=0` (blackout/weekend/breaker) y el reward lo refleja.
- [ ] `flip_penalty` (o cooldown) activo y testeado.
- [ ] Differential Sharpe: implementación verificada contra un caso analítico simple.
- [ ] `info` expone campos requeridos por SPEC-09.

## Dependencias
SPEC-03 (estado), SPEC-04 (régimen), SPEC-06 (riesgo), SPEC-08 (CostModel).
