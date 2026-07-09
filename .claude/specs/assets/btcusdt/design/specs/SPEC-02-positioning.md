# SPEC-02 — Posicionamiento del Crowd (z_funding)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | `btc_daily.funding` (Binance perp, D−1), `open_interest` (reciente) |
| Materializa | `strategy/positioning.py` |
| Pre-registro | Guía v3 §5.4, §6.2 |
| Rol | Componente del score interno (NO multiplicador directo); **casi solo reduce** |

## 1. Propósito y alcance

Producir `z_funding`, un score continuo del **apalancamiento del crowd** derivado del funding
rate de perpetuos. Entra al score interno de riesgo (SPEC-03) junto con `z_ciclo` — nunca
multiplica por separado, porque ambos miden "crowd estirado" (§7.1). El funding es **señal,
no costo** (núcleo spot-only, ADR-0008).

## 2. Entradas (contrato)

| Feature | Tipo | Unidad | Fuente | Disponibilidad | Rango |
|---|---|---|---|---|---|
| `funding` | float | tasa 8h | Binance `/fapi/v1/fundingRate` | D−1 | ~[−0.003, 0.003] |
| `open_interest` | float | USD | Binance (reciente) | D−1 | > 0 (histórico corto) |

Nota de series cortas: funding desde 2019-09. Antes de esa fecha `z_funding = 0` (neutro),
documentado — no se inventa.

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Semántica |
|---|---|---|---|
| `z_funding_t` | float | ~[−3, 3] | **positivo = riesgo** (longs apalancados); negativo = alivio (suelo) |
| `funding_pctl_t` | float | [0, 1] | percentil rodante 1a del funding |
| `crowded_flag` | bool | — | `True` si pctl > 0.95 sostenido ≥ 3 lecturas |

## 4. Interfaz (API)

```python
class PositioningModel(Protocol):
    def z_funding(self, daily_until_t: pd.DataFrame) -> float:
        """z-score con signo del percentil rodante 1a del funding. Solo info ≤ t."""
    def crowded(self, daily_until_t: pd.DataFrame) -> bool: ...
```

## 5. Algoritmo / lógica

```
funding_pctl_t = rolling_percentile_rank(funding_t, window=1y)           # [0,1], sin fuga
z_funding_t    = zscore(funding_pctl_t, window=1y)  * sign_convention     # positivo = caro/riesgo

# Semántica (priors):
#   pctl > 0.95 sostenido ≥ 3 lecturas  → longs amontonados → z_funding ↑ (riesgo de cascada)
#   funding negativo profundo en DEEP_VALUE → shorts pagando en el suelo → z_funding ↓ (alivio)
crowded_flag = (funding_pctl_t > 0.95) sostenido ≥ 3 lecturas
```

**Asimetría (§5.4):** el rol dominante de `z_funding` es **reducir**. Como su cola positiva
(euforia apalancada) es más frecuente y extrema que su cola negativa, al entrar al score
interno (SPEC-03) su efecto neto es defensivo — sin cap ad-hoc.

## 6. Invariantes y post-condiciones

- `z_funding` no usa datos > t.
- Antes de 2019-09, `z_funding = 0` exacto (no NaN que rompa el motor; neutro documentado).
- El signo es consistente: funding alto ⇒ `z_funding` alto ⇒ más reducción en SPEC-03.

## 7. Tests unitarios

- [ ] Funding constante ⇒ `z_funding` → 0.
- [ ] Un pico de funding aislado no activa `crowded_flag` (requiere ≥ 3 sostenido).
- [ ] Percentil rodante en el borde de la ventana no usa datos futuros (no-fuga).
- [ ] Pre-2019-09 ⇒ `z_funding == 0`.
- [ ] Signo: `funding` en máximo histórico ⇒ `z_funding` fuertemente positivo.

## 8. Tests de integración

- [ ] `crowded_flag` se activa en los picos de funding conocidos (p. ej. 2021-04, pre-caídas
      de apalancamiento) — sanity histórico.
- [ ] En capitulaciones conocidas (2022-11) el funding negativo produce `z_funding` negativo.

## 9. Test de hipótesis

**H-POS-01 — ¿el funding extremo precede a deterioro de riesgo?**
- **H0:** el CVaR (95 %) del retorno forward tras `funding_pctl > 0.95` es **igual** al del
  resto de los días.
- **H1:** el CVaR forward es peor (más negativo) tras funding extremo.
- **Estadístico:** diferencia de CVaR forward (H días), IC por bootstrap por bloques.
- **Criterio:** IC 95 % del delta **no cruza 0** (peor cola tras funding extremo).
- **Corrección:** 1 trial en el DSR.

## 10. Protocolo de backtest / validación

- El aporte real de `z_funding` se mide **dentro de SPEC-03/06 en aislamiento** (núcleo +
  z_funding vs. núcleo). Esta spec valida solo la señal, no la estrategia.
- Reporte por sub-era; dado que la serie arranca en 2019, la sub-era pre-2020 tendrá poco
  soporte — declarado.

## 11. Criterios de aceptación (DoD)

- [ ] H-POS-01 rechaza H0.
- [ ] Tests en verde; no-fuga auditado.
- [ ] Manejo documentado del hueco pre-2019.
- [ ] Aporte a S3 en aislamiento (verificado en SPEC-03).

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.1 doble conteo:** `z_funding` NO multiplica; se combina con `z_ciclo` en SPEC-03.
- **Capa datos:** percentil/z-score rodantes, funding D−1.
- **ADR-0008:** funding es señal, no costo (spot-only).

## 13. Parámetros pre-registrados

Percentil rodante 1a; `crowded` = pctl > 0.95 sostenido ≥ 3 lecturas; peso 0.3 en R (SPEC-03);
neutro 0 antes de 2019-09.
