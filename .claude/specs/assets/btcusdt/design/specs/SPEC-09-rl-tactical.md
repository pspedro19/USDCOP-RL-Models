# SPEC-09 — RL Táctico (LSTM→PPO) = S5 (opcional)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) — **opcional** |
| Versión | 1.0 |
| Depende de | SPEC-06 (exposición base), `btc_price_1h` + CVD/OFI + funding + contexto de gates |
| Materializa | `strategy/rl/` (entorno Gymnasium + política PPO) |
| Pre-registro | Guía v3 §4.5; ADR-0008 |
| Es | **S5** (S4 + RL táctico); **debe ganarse su lugar o se descarta sin duelo** |

## 1. Propósito y alcance

Una capa **táctica** que ajusta la exposición del motor en **±10 % NAV** como desviación,
sobre la barra horaria. Encoder LSTM (representación temporal) → política PPO. Es **opcional
y debe superar a S4 OOS** (mediana de seeds) para conservarse. La evidencia advierte que el
RL suele **perder** contra baselines en BTC (§4.1); por eso el gate es estricto.

## 2. Entradas (contrato)

| Entrada | Tipo | Descripción |
|---|---|---|
| Observación | vector | representación LSTM (retornos H1 a 1/5/10 pasos) + CVD/OFI + funding + estado de gates + vol |
| Exposición base | float | del motor (SPEC-06); el RL solo la desvía |

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Efecto |
|---|---|---|---|
| `tactical_shift_t` | discreto {−1, 0, +1} | — | mapeado a desviación acotada |
| `exposure_rl_t` | float | [0, 1] | `clip(exposure_base + shift·δ, 0, 1)`, con `|desviación| ≤ 10 % NAV` |

## 4. Interfaz (API)

```python
class TacticalRL(Protocol):
    def act(self, obs: np.ndarray, exposure_base: float) -> float:
        """Desviación táctica acotada ±10% NAV. exposure_rl = clip(base + shift, 0, 1)."""
```

Entorno: **la capa de riesgo va DENTRO del entorno Gymnasium** (no post-proceso), para evitar
el skew entre entrenamiento y ejecución. Recompensa = **Sortino diferencial post-costos**
(fees + slippage; funding solo señal).

## 5. Algoritmo / lógica

```
obs_t          = LSTM_encoder(H1 features)  ⊕  {CVD, OFI, funding, estado_gates, vol}
shift_t        = PPO_policy(obs_t) ∈ {−1, 0, +1}
δ              = 10% NAV  (desviación máxima)
exposure_rl_t  = clip( exposure_base_t + shift_t·δ , 0 , 1 )
reward_t       = Sortino_diferencial_post_costos           # dentro del entorno
# Entrenamiento: ≥ 5 seeds; se reporta MEDIANA e IQR (nunca la mejor corrida)
```

## 6. Invariantes y post-condiciones

- `|exposure_rl − exposure_base| ≤ 10 % NAV` (desviación acotada).
- `exposure_rl ∈ [0, 1]` (spot-only).
- El RL desvía; **no reemplaza** el motor.
- La capa de riesgo está dentro del entorno (no post-proceso).

## 7. Tests unitarios

- [ ] Desviación acotada: cualquier `shift` ⇒ `|exposure_rl − base| ≤ 10 %`.
- [ ] Rango: `exposure_rl ∈ [0, 1]`.
- [ ] Recompensa post-costos usa el CostModel (SPEC-07).
- [ ] Determinismo por seed (misma seed ⇒ misma política).

## 8. Tests de integración

- [ ] Entrenamiento con ≥ 5 seeds produce distribución de desempeño; se reporta mediana/IQR.
- [ ] La política no viola la desviación en ningún paso del histórico.

## 9. Test de hipótesis

**H-RL-01 — ¿S5 le gana a S4 en Calmar OOS (mediana de seeds)?**
- **H0:** mediana de Calmar OOS(S5) ≤ Calmar OOS(S4).
- **Estadístico:** bootstrap sobre la **mediana** de `ΔCalmar(S5, S4)` a través de ≥ 5 seeds.
- **Criterio:** IC 95 % de la mediana > 0. **Si no se rechaza ⇒ se descarta S5 sin duelo** y
  S4 (o S3) es la estrategia. Dado el prior (RL pierde en BTC), este es el resultado más
  probable — y es honesto.

## 10. Protocolo de backtest / validación

- Multi-seed (≥ 5), mediana/IQR — jamás la mejor corrida.
- Walk-forward con purga/embargo; recompensa post-costos.
- Reporte por sub-era; la historia H1 es corta ⇒ riesgo alto de sobreajuste, declarado.

## 11. Criterios de aceptación (DoD)

- [ ] H-RL-01 evaluado con ≥ 5 seeds; S5 solo se conserva si rechaza H0.
- [ ] Desviación acotada y rango verificados.
- [ ] Mediana/IQR reportados (no la mejor seed).

## 12. Anti-look-ahead / modos de fallo cubiertos

- **Multi-seed:** evita el sesgo de la corrida afortunada.
- **§7.2:** desviación acotada + spot-only (exposure ∈ [0,1]).
- **Skew train/ejecución:** riesgo dentro del entorno.

## 13. Parámetros pre-registrados

Desviación máxima ±10 % NAV; acción {−1,0,+1}; ≥ 5 seeds (mediana); recompensa Sortino
diferencial post-costos; LSTM→PPO (RecurrentPPO).
