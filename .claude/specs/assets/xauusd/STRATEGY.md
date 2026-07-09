# Estrategia XAU/USD — Especificación Final v2 (Norte)

Documento canónico de la estrategia. Todas las specs (`specs/SPEC-XX`) implementan lo aquí definido. Si hay conflicto, este documento manda; propón cambios vía ADR.

**Timeframes:** H1 (ejecución) + Daily (régimen y dirección)
**Base:** extensión al oro del sistema RL de USD/COP ya validado out-of-sample.
**Filosofía central:** en oro la dirección es casi un volado (~50–53% de accuracy en la literatura honesta). Por lo tanto **el alpha vive en la gestión de riesgo y la adaptación al régimen, no en la predicción.**

## Principios no negociables (resumen ejecutivo)

1. **Separación dirección/tamaño.** El agente decide dirección `{short, flat, long}`; la capa de riesgo (determinista) decide tamaño.
2. **Riesgo DENTRO del entorno.** Vol-targeting, blackouts y límites viven dentro del entorno Gymnasium: el agente entrena sobre el PnL real post-sizing/post-costos. Evita train/serve skew.
3. **Régimen en Daily con histéresis.** 4 regímenes (Compresión, Tendencia, Estirado, Event-driven), dwell mínimo 3–5 días. Labels estables.
4. **Macro como correlaciones rodantes, NO supuestos fijos.** La relación oro–DXY/tasas reales no es estable (2022–2025 la rompió). Entra como feature medida.
5. **Costos completos:** spread variable + slippage + **swap overnight** dentro de recompensa y backtest.
6. **Multi-seed obligatorio:** ≥5 seeds, reportar mediana e IQR. La mediana debe ganar, no el mejor run.
7. **DOS baselines:** B1 long-only vol-targeted (captura beta del oro) y B2 trend-follower Daily. El RL debe ganar a ambos OOS.
8. **Blindaje estadístico:** purga+embargo, Deflated Sharpe con conteo real de trials, bootstrap CIs, atribución de PnL por régimen.
9. **Circuit breakers** incluyendo vol-spike (eventos no calendarizados) + flat de fin de semana.
10. **Onboarding como activo, no como silo (SPEC-12).** El Oro entra por `AssetProfile` (`config/assets/xauusd.yaml`) y publica sus backtests al **registro dinámico** con **versionado inmutable** — de forma **aditiva**, sin romper contratos ni consumos front↔back existentes. Reutiliza aprobación (Vote 2), replay y monitoreo del sistema; no duplica infraestructura. `XAUUSD` nunca se hardcodea.

## Recompensa (Moody & Saffell — Differential Sharpe)

```
reward_t = ΔSharpe_diferencial_t − λc·costo_t − λd·dd_penalty_t − λf·flip_penalty_t
```
computado sobre el retorno post-sizing, post-costos (spread + slippage + swap).

## Los 4 regímenes

| Régimen | Condición | Favorece | multiplicador_riesgo |
|---|---|---|---|
| Compresión | Vol baja, ADX bajo | Breakout | 1.0 |
| Tendencia | ADX alto, Hurst >0.5 | Continuación | 1.0 |
| Estirado | Z-score extremo, Hurst <0.5 | Reversión | 0.5–0.75 |
| Event-driven | Flag macro activo | Aplanar | 0.25–0.5 |

## Gate de decisión (Fase 6)

La **mediana de ≥5 seeds** del agente RL debe superar, out-of-sample y ajustado por riesgo (Sharpe/Sortino/Calmar con IC por bootstrap), a **B1 y B2**. Si no → el mejor baseline ES la estrategia; iterar features/reward antes de añadir complejidad. Si B2 no gana a B1 → no hay edge direccional ni a nivel de reglas.

> El documento completo v2 con todos los caveats (histéresis, correlaciones rodantes, swap, vol-spike breaker, flat de fin de semana, atribución por régimen, registro de experimentos) es la referencia extendida. Este resumen fija los invariantes que las specs deben respetar.

---
*Aviso: diseño metodológico, no asesoría financiera. El backtest espectacular es la señal de alarma, no el premio.*
