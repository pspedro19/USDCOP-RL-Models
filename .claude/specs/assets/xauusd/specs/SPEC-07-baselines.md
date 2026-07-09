# SPEC-07 — Baselines Obligatorios (B1 y B2)

## Propósito
Definir los DOS benchmarks que el agente RL debe superar. Un solo baseline no basta: el oro tuvo un rally secular en 2024–2025 y *cualquier* sesgo long parece genio en ese tramo. Necesitamos separar "el agente aporta" de "el oro subió".

## B1 — Long-only vol-targeted (la vara del "no hiciste nada")
- **Siempre long** (`direction = +1`), tamaño por la MISMA capa de riesgo (SPEC-06): vol-targeting, límites, blackouts, weekend flat.
- Es buy & hold **bien gestionado**. Si el RL no le gana ajustado por riesgo, el agente no aporta *dirección* — solo captura beta del oro con pasos extra.

```python
class LongOnlyBaseline:
    def action(self, state) -> int: return 2  # long siempre; riesgo idéntico al del env
```

## B2 — Trend-follower Daily (la vara del "sistema simple honesto")
- Long cuando `close > SMA(100)` **y** `ADX(14) > 25`; flat en otro caso (opcional simétrico para shorts).
- Misma capa de riesgo, límites, blackouts, weekend flat.

```python
class TrendFollowerBaseline:
    def action(self, state) -> int:
        if state.daily_close > state.sma100 and state.adx14 > 25: return 2  # long
        return 1  # flat
```

## Ejecución
Ambos baselines corren por el **mismo harness de backtest** (SPEC-09) y el **mismo entorno/costos** que el agente — apples-to-apples. No reimplementar costos ni sizing: reutilizar SPEC-05/06.

## Regla de oro (gate de la Fase 6)
Si el agente RL (**mediana de ≥5 seeds**) no le gana a **AMBOS** baselines, OOS y ajustado por riesgo → el RL no aporta; el mejor baseline ES la estrategia. Y si **B2 no le gana a B1** → no hay edge direccional ni a nivel de reglas simples (información valiosa por sí misma).

## Criterios de aceptación
- [ ] B1 y B2 implementados como políticas que consumen el mismo `state` que el agente.
- [ ] Corren por el harness de SPEC-09 con costos/sizing idénticos (test de paridad).
- [ ] Reporte comparativo B1 vs B2 vs agente en las mismas ventanas OOS.
- [ ] Métricas ajustadas por riesgo con IC por bootstrap para los tres.

## Dependencias
SPEC-05, SPEC-06 (entorno/riesgo), SPEC-09 (harness).
