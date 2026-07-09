# SPEC-04 — Clasificador de Régimen (Daily)

## Propósito
Producir un label de régimen diario ∈ {Compresión, Tendencia, Estirado, Event-driven} con **histéresis** (labels estables), disponible point-in-time para las barras H1 de D+1. El régimen alimenta el estado del agente (SPEC-05) y el multiplicador de tamaño (SPEC-06).

## v1 — Reglas transparentes (empezar AQUÍ)
Thresholds sobre features Daily (SPEC-03). Ejemplo de lógica (calibrar por percentiles históricos, no números mágicos):

```python
def raw_regime(row) -> str:
    if row.macro_event_flag:                      return "EVENT"
    if row.adx_14 > adx_hi and row.hurst > 0.55:  return "TREND"
    if abs(row.z_sma50) > z_hi and row.hurst < 0.45: return "STRETCHED"
    if row.rvol_20 < vol_lo and row.adx_14 < adx_lo: return "COMPRESSION"
    return prev_label  # zona ambigua: mantener el label previo (parte de la histéresis)
```
Umbrales sugeridos como percentiles rodantes (p.ej. `adx_hi`=p70, `vol_lo`=p30) para adaptarse a cambios de nivel del oro.

## Histéresis (OBLIGATORIA)
Sin histéresis el clasificador hace flip-flop en las fronteras y envía contexto ruidoso al agente.
- **Dwell mínimo 3–5 días:** no cambiar de label hasta que la nueva condición se sostenga ≥3–5 sesiones.
- **Confirmación de 2 días** consecutivos del nuevo régimen antes de conmutar.
- Implementar como máquina de estados sobre `raw_regime`.

```python
def apply_hysteresis(raw_labels: pd.Series, min_dwell=4, confirm=2) -> pd.Series: ...
```

## v2 — No supervisado (graduar solo si v1 muestra valor)
- **HMM** (`hmmlearn.GaussianHMM`, n_components=4) sobre el vector de features Daily. La **matriz de transición del HMM da la histéresis de forma natural** (probabilidad baja de saltar de estado). Requiere mapear estados latentes a los 4 regímenes interpretables (por sus medias de vol/ADX/Hurst).
- Alternativa: GMM (`sklearn.mixture.GaussianMixture`) + suavizado temporal.
- Reutilizar el know-how de clustering no supervisado del proyecto Tri-Score.

## Métrica propia: estabilidad de labels
El régimen es útil solo si es estable. Validar como métrica de primera clase:
- **Transiciones/año:** debe ser bajo (los regímenes del oro duran semanas). Alerta si > ~20/año.
- **Duración media por régimen** en días.
- **Sentido económico:** equity curve de un trend-follower simple debe ser mejor en `TREND` que en `STRETCHED` (sanity check).

## Salida
- `data/features/regime_daily/`: `session_date`, `regime` (categórico), one-hot, `regime_confidence` (si HMM).
- Point-in-time: aplicable a H1 de D+1 (shift en SPEC-02/03).

## Criterios de aceptación
- [ ] v1 produce los 4 labels; test sobre tramos conocidos (p.ej. 2020 crash → EVENT/STRETCHED; 2024 rally → TREND).
- [ ] Histéresis reduce transiciones vs `raw_regime` (test cuantitativo: transiciones_con_histéresis < transiciones_raw).
- [ ] Transiciones/año dentro de rango esperado (métrica reportada).
- [ ] Label point-in-time (test anti-look-ahead).
- [ ] (v2) Estados HMM mapeados a regímenes interpretables y documentados.

## Dependencias
SPEC-03 (features Daily). Consumido por SPEC-05, SPEC-06.
