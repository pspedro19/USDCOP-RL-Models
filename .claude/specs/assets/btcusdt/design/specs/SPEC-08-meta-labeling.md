# SPEC-08 — Meta-Labeling (exposición continua) = S4

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | SPEC-06 (cruces de banda), estado de todos los gates, `btc_price_1h` |
| Materializa | `strategy/meta_label.py` |
| Pre-registro | Guía v3 §4.4; ADR (López de Prado) |
| Es | **S4** (S3 + meta-labeling); el meta-modelo **solo frena** trades |

## 1. Propósito y alcance

Filtrar rebalanceos de baja calidad con un **meta-modelo** supervisado. Redefine
meta-labeling (que presupone señales discretas) para un **motor de exposición continua**: el
"evento etiquetable" es cada **cruce de banda de rebalanceo**. El meta-modelo predice *si el
ajuste va a funcionar*, no la dirección del precio. **Solo puede frenar** trades, nunca
crearlos ni ampliarlos.

## 2. Entradas (contrato)

| Entrada | Tipo | Descripción |
|---|---|---|
| Evento de rebalanceo | trigger | cruce de banda ±12.5 % NAV (SPEC-06) |
| Vector de estado | features | `z_ciclo`, `z_funding`, `M_liquidez`, `G_eventos`, régimen HMM, técnicos, vol realizada, magnitud+signo del ajuste |
| Trayectoria forward | pd.Series | precio post-costos hasta H o siguiente evento (para etiquetar) |

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Efecto |
|---|---|---|---|
| `p_success_t` | float | [0, 1] | probabilidad de que el rebalanceo mejore Sortino |
| `postpone_flag` | bool | — | `True` si `p < 0.45` ⇒ rebalanceo se **pospone** (re-evalúa al siguiente cruce) |

## 4. Interfaz (API)

```python
class MetaLabeler(Protocol):
    def label_event(self, state: dict, fwd_path: pd.Series, H: int = 20) -> int:
        """Triple-barrier adaptado: 1 si ΔSortino(con ajuste vs. sin ajuste) > 0 en H días."""
    def predict(self, state: dict) -> float:  # p_success ∈ [0,1]
    def should_postpone(self, p: float, threshold: float = 0.45) -> bool: ...
```

## 5. Algoritmo / lógica

```
# Definición del evento etiquetable (§4.4)
evento = cada cruce de banda de rebalanceo (|target − prev| > 12.5% NAV)   # la ÚNICA decisión real

# Etiqueta (triple-barrier adaptado)
label = 1 si  ΔSortino_realizado_post_costos( trayectoria_CON_ajuste
                                            , trayectoria_SIN_ajuste ) > 0
        evaluado en H = 20 días hábiles  (o hasta el siguiente evento si llega antes)

# Modelo: XGBoost / LightGBM sobre el vector de estado
p_success = model.predict(state)

# Uso en producción (ASIMÉTRICO — solo frena)
if p_success < 0.45:  posponer el rebalanceo   # no cancela: re-evalúa al siguiente cruce
# NUNCA crea ni amplía un trade
```

## 6. Invariantes y post-condiciones

- El meta-modelo **solo** puede posponer (reducir actividad), jamás crear/ampliar exposición.
- El embargo entre train y test es **≥ H** (para no filtrar la etiqueta).
- La etiqueta se computa post-costos (usa CostModel, SPEC-07).
- Sin evento de rebalanceo no hay predicción (el meta-modelo no actúa fuera de los cruces).

## 7. Tests unitarios

- [ ] `should_postpone(0.44)` = True; `should_postpone(0.46)` = False.
- [ ] `label_event`: ajuste que mejora Sortino ⇒ 1; que lo empeora ⇒ 0.
- [ ] El meta-modelo nunca produce una acción que aumente exposición (chequeo de rol).
- [ ] Etiqueta usa horizonte H y corta en el siguiente evento si llega antes.

## 8. Tests de integración

- [ ] Sobre el histórico, S4 hace **menos** rebalanceos que S3 (el meta-modelo frena algunos).
- [ ] El turnover de S4 ≤ turnover de S3 (consistente con "solo frena").
- [ ] Los rebalanceos pospuestos por S4 eran, en promedio, peores que los ejecutados
      (sanity del filtro).

## 9. Test de hipótesis

**H-MET-01 — ¿el meta-modelo discrimina señales buenas de malas?**
- **H0:** AUC OOS del meta-modelo = 0.5 (sin poder discriminativo).
- **Estadístico:** **test de DeLong** sobre el AUC OOS (con IC).
- **Criterio:** AUC > 0.5 con p < α. Si no ⇒ el meta-modelo no aporta ⇒ se descarta (queda S3).

**H-MET-02 — ¿S4 le gana a S3 en Calmar OOS?**
- **H0:** Calmar OOS(S4) ≤ Calmar OOS(S3).
- **Estadístico:** bootstrap pareado por bloques sobre `ΔCalmar(S4, S3)`.
- **Criterio:** IC 95 % > 0. Si no ⇒ **S3 ES la estrategia** (resultado honesto y aceptable).

## 10. Protocolo de backtest / validación

- Walk-forward con purga y **embargo ≥ H** (crítico: la etiqueta mira H días adelante).
- DSR cuenta el meta-modelo como trial.
- Reporte por sub-era; el meta-modelo puede aportar en unas eras y no en otras.

## 11. Criterios de aceptación (DoD)

- [ ] H-MET-01 rechaza H0 (AUC > 0.5) — si no, se descarta.
- [ ] H-MET-02 evaluado; S4 solo se adopta si gana a S3 OOS.
- [ ] Rol asimétrico (solo frena) verificado por test.
- [ ] Embargo ≥ H auditado (no fuga de etiqueta).

## 12. Anti-look-ahead / modos de fallo cubiertos

- **Fuga de etiqueta:** embargo ≥ H obligatorio.
- **§7.4 selección:** el meta-modelo es un trial contado en el DSR.
- **Rol asimétrico:** no puede inventar exposición (coherente con constitución §5).

## 13. Parámetros pre-registrados

Evento = cruce de banda ±12.5 %; H = 20 días; umbral p = 0.45 (solo frena); embargo ≥ H;
modelo XGBoost/LightGBM. Umbral no se optimiza sobre el test.
