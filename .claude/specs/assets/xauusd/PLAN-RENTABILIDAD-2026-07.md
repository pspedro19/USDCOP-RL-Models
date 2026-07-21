---
kind: roadmap
status: PLANNED
version: 2.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/strategy_manifests/xauusd.yaml
  - src/gold_rl/strategies.py
  - scripts/pipeline/run_gold_pipeline.py
  - scripts/pipeline/generate_asset_weekly_forecast.py
---

# Plan de rentabilidad XAU/USD

> Objetivo: convertir `gold_trend_simple` en un track forward falsable, no volver a optimizar
> la señal histórica. El histórico es contexto; protocolo, broker y forward deciden.

## 1. Estado honesto

- Campeón: `gold_trend_simple`, regla fija de voto SMA 63/126/252 × vol target. No entrena.
- La familia ya acumula **77 trials centrales** y DSR histórico insuficiente para un claim de
  alfa. Se mantiene la abstención de nuevas hipótesis direccionales.
- La lectura atractiva OOS-2025 fue observada durante el desarrollo; 2026 live/contexto es
  negativo en el registry. Ninguna de las dos ventanas autoriza promoción.
- H-VOLF-01 mostró que HAR puede predecir vol mejor que persistencia, pero H-VOLE-01 no mejoró
  el sizing: floor/cap dominan. Predecir vol no equivale a monetizarla.
- Falta el contrato crítico: protocolo de retiro XAU firmado y paper-shadow MT5 con swap real.

## 2. Contrato congelado

| Elemento | Definición |
|---|---|
| Señal | al menos 2 de SMA {63,126,252}; long/flat |
| Sizing | `0.10 / rv20`, clip `[0.06, 1.5]` |
| Decisión | daily causal `t−1`, reloj 252 |
| Ejecución | siguiente sesión/barra; contrato final lo fija el broker MT5 |
| Costos actuales | 2 bps turnover + supuesto 2.5% anual de swap; se sustituyen por medición |
| Retraining | nunca; constantes fijas |

Uso multi-timeframe:

| Tabla | Función permitida |
|---|---|
| 5m | fill/spread/slippage si el feed del broker queda homologado; no señal |
| 1h | rango, gap, liquidez y realized vol para diagnóstico/paper |
| 4h | contexto de riesgo y control de sesión |
| daily | única decisión de exposición y replay oficial |

Las barras diarias se anclan en UTC antes de aplicar el cierre. Se prohíbe
`tz_convert(ET).normalize()` por el bug de “Sunday pile-up”.

## 3. Plan por dependencias

### X0 — Crear el juez antes del paper (0 trials)

Crear `WITHDRAWAL-PROTOCOL-XAU.md` y firmarlo antes del primer snapshot válido. Debe contener:

- `strategy_id`, manifest hash y broker/contrato;
- inicio por primer fill paper, ventana mínima 26 semanas y regla para pausas;
- retiro inmediato por DD >15%, divergencia de ejecución o violación de datos;
- revisión por 8 semanas con Calmar acumulado <0, sin retunear;
- graduación: Calmar >0.75, DD <12%, costos ×2 positivos, tracking paper/replay <2 pp
  promedio, ≥30 oportunidades y más de un régimen;
- firma humana y prohibición de relajar umbrales en drawdown.

Los valores son priors de riesgo; si el operador decide otros, deben quedar firmados antes
del dato y no pueden cambiar durante la ventana.

### X1 — Paper-shadow MT5 y financiación (0 trials)

| Evidencia | PASS |
|---|---|
| Mapeo de instrumento | símbolo, tamaño de contrato, tick, moneda PnL, horario, rollover y festivos |
| Señal→orden | decisión de cierre t ejecutada exactamente una vez en t+1 |
| Fill | precio, spread, slippage, rechazo y latencia persistidos |
| Swap | long/short, triple-swap y conversión medidos en statements |
| Replay | misma versión/costos; divergencia semanal explicable y <umbral firmado |

No hay live hasta una semana técnica sin violaciones y la ventana económica del protocolo.

### X2 — H-ROBUST-DECADES-XAU, falsificación (+1 trial)

Este test no elige parámetros ni promueve; intenta destruir la tesis con segmentos anteriores
al período de desarrollo.

Antes de abrir resultados se guarda un artefacto inmutable con:

- hash de código/manifest y parámetros idénticos al campeón;
- segmentos 1980–1989, 1990–1999 y 2000–2003;
- costos ×1/×2/×3, swap explícito y B1/B1′;
- definición única de barra, reloj 252 y bloque bootstrap 20d;
- prueba de que no se seleccionó ni excluyó un subsegmento después de mirarlo.

**PASS prefirmado:** Calmar >0, superior a B1′ y costos ×2 con Calmar >0 en al menos 2/3
décadas. **FAIL:** `gold_trend_simple` se degrada a contexto de régimen y se bloquea live.
PASS solo autoriza continuar paper; no prueba alfa ni sustituye el forward.

### X3 — Forward y operación (0 trials nuevos)

1. El DAG semanal conserva un solo campeón visible después de publicar la familia.
2. La inferencia semanal es una vista de la regla daily, no un modelo competidor.
3. El ledger registra paper YTD, retorno semanal derivado, replay semanal y divergencia.
4. A las 26 semanas se aplica el protocolo sin cambiar signal/sizer.
5. Solo tras graduar: capital mínimo, límites de notional y kill-switch; escalado por etapas.

## 4. Decisiones de no hacer

- no mover lookbacks ni transformar el voto a blend continuo;
- no usar Parkinson/Yang-Zhang dentro del sizer actual: H-VOLE-01 ya mostró que los clips
  absorben la mejora; reabrir exige primero una tesis independiente sobre la mecánica;
- no usar 1h/4h como votos de dirección;
- no publicar el Calmar OOS-2025 como expectativa forward;
- no ejecutar spot/CFD sin medir el swap del instrumento real.

## 5. Artefactos y verificación

```powershell
python scripts/pipeline/run_gold_pipeline.py
python scripts/pipeline/normalize_champions.py --check
python scripts/pipeline/generate_asset_weekly_forecast.py --asset xauusd --year all
python -m scripts.analysis.forward_tracker --report
```

Terminado significa: protocolo firmado, broker homologado, replay por décadas con veredicto,
ledger forward válido y retiro/escalado automatizable. El trial X2 es el único nuevo permitido
por este plan.
