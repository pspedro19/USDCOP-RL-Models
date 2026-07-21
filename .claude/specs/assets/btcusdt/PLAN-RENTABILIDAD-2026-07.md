---
kind: roadmap
status: PLANNED
version: 2.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/strategy_manifests/btcusdt.yaml
  - src/btc_strategy/strategies.py
  - scripts/pipeline/run_btc_pipeline.py
  - scripts/data/ingest_btc_derivatives.py
---

# Plan de rentabilidad BTC/USDT

> Objetivo: aceptar que el baseline vol-targeted es el campeón, iniciar un forward limpio y
> gastar como máximo un trial adicional solo si el M5 aporta una medición de riesgo realmente
> nueva. No se reabre dirección desde precio/funding.

## 1. Estado honesto

- Campeón: `btc_hodl_b1`, HODL spot × `0.30/rv20` clip `[0.30,1.0]`. No entrena.
- En OOS-2025 el campeón ganó +4.70% mientras las variantes perdieron aproximadamente 1.37%;
  que gane el baseline es una conclusión válida, no un fracaso que haya que ocultar.
- El protocolo actual apunta a `btc_trend_b2` y está sin firma. Por tanto, no existe un reloj
  forward válido para el campeón.
- Registry: **34 trials centrales**. Funding como freno y como predictor direccional dio NO;
  EWMA sizing y HAR-RV tampoco desplazaron a la persistencia. Esas vías quedan cerradas.
- `liquidations_usd` no tiene cobertura útil; OI/long-short son forward-only. No se construyen
  features históricas sobre nulls ni se backfillea una disponibilidad que nunca existió.

## 2. Contrato congelado y uso de frecuencias

| Elemento | Definición |
|---|---|
| Mercado | spot-only, long/flat, sin apalancamiento, 24/7 |
| Decisión | daily al cierre UTC, `shift(1)`, reloj 365 |
| Señal | intent constante 1.0; no hay claim direccional |
| Sizing | `0.30 / realized_vol_20`, clip `[0.30,1.0]` |
| Costos | 13 bps turnover del manifest; validar exchange/tier/slippage |
| Benchmark | B1 spot 1×, B1′ de exposición, HODL vol-targeted y cash |

| Tabla | Función permitida |
|---|---|
| 5m | realized range/vol, fills, spread y slippage; nunca voto direccional |
| 1h | shocks de volatilidad, vol-of-vol y ejecución |
| 4h | riesgo intermedio, funding/OI/basis agregados causalmente |
| daily | decisión oficial, replay, baselines y ledger |

El sleeve conserva 365. Cuando entra al ERC diario se observa en la intersección de días de
mercado del libro, pero no se reescribe su historia ni se forward-fillean fines de semana.

## 3. Plan por dependencias

### B0 — Corregir y firmar el protocolo (0 trials)

Reescribir `WITHDRAWAL-PROTOCOL-BTC.md` para `btc_hodl_b1` y congelar:

- estrategia, hash, spot venue, fees y ejecución;
- ventana única de 26 semanas desde el primer snapshot paper válido;
- graduación: Calmar ≥0.75, MaxDD ≤15%, costos ×2 positivos, tracking <2 pp promedio,
  cero violaciones de gate y ≥30 oportunidades;
- retiro inmediato: DD >20%, error de custodia/venue, datos inválidos o breaker;
- revisión por ocho semanas con Calmar acumulado <0, sin cambiar parámetros;
- firma humana no vacía.

Las métricas con N<20 trades se suprimen; la frecuencia diaria no convierte días flat en trades.

### B1 — Cerrar paper y ledger (0 trials)

1. El publisher de la familia no puede resucitar `btc_trend_b2`; el normalizador deja solo
   `btc_hodl_b1` visible.
2. La señal daily genera un fill paper next-bar con fee/slippage reales.
3. El ledger deriva `paper_week_pct` de snapshots YTD consecutivos y compara el mismo
   `strategy_id` contra replay.
4. Missing paper/replay no se registra como 0.0.
5. Dos semanas válidas consecutivas son el gate técnico; 26 semanas son el juez económico.

### B2 — Calidad PIT de intradía y derivados (0 trials)

| Serie | Gate |
|---|---|
| M5/1h/4h spot | instante UTC, barra completa, origen nativo/agregado, cobertura y checksum |
| Funding | `available_at` tras el settlement de 8h; no usar la observación del mismo cierre |
| OI/long-short | forward-only hasta acumular historia; no reconstruir vintages |
| Basis | definición de spot/perp, moneda, anualización y stale threshold |
| Liquidaciones | excluida mientras no tenga feed WS y cobertura declarada |

H-BASIS-01 permanece registrada pero inactiva hasta cumplir su gate de historia y motor. No
consume trial mientras no se abran resultados.

### B3 — H-VOL-02, condicional (+1 trial al abrir resultados)

Esta es una opción, no una obligación. Se cancela si B0–B2 no pasan o si el estimador no puede
especificarse sin escoger entre variantes después de mirar el juez.

- **Una variable:** reemplazar `realized_vol_20` por un único estimador spot intradía
  predefinido (Yang-Zhang si hay O/H/L causal y overnight coherente; realized-M5 si no).
- No se prueban ambos ni se barren ventanas. La elección se hace por contrato de datos antes
  de calcular performance y queda registrada.
- **Etapa estadística prefirmada:** QLIKE/MAE de vol contra `rv20` y EWMA, con block bootstrap.
- **Etapa económica:** solo existe si fue pre-registrada como parte del mismo trial; compara
  retorno del mismo `btc_hodl_b1` cambiando únicamente el estimador, con B1′ y costos ×2.
- **Juez:** forward posterior al freeze. El OOS-2025 ya influyó en la elección de esta vía y
  solo puede mostrarse como contexto.
- **PASS:** mejora QLIKE con IC95 y delta Calmar forward con IC95 >0, sin peor DD ni turnover
  fuera del presupuesto. Si cualquiera falla, `rv20` permanece y la vía vol queda cerrada.

No se promueve por predecir volatilidad: debe mejorar el resultado económico neto de costos.

### B4 — Acumular datos nuevos sin generar hypotheses (0 trials)

- Persistir OI, long/short y basis con manifest append-only.
- Publicar cobertura, staleness y revisiones cada semana.
- Evaluar H-BASIS-01 únicamente al alcanzar el umbral ya registrado; no “mirar mientras crece”.
- On-chain, opciones y liquidaciones requieren otro plan y otra fuente PIT; no se improvisan
  dentro de B3.

## 4. Decisiones de no hacer

- no reabrir funding como dirección o freno;
- no correr el zoo de nueve modelos ni horizonte de siete semanas;
- no comparar weekend 365 con un libro 252 sin conversión explícita;
- no usar M5 para multiplicar el N nominal de una estrategia daily;
- no cambiar simultáneamente estimador, clips y señal;
- no llamar alfa a HODL vol-targeted: es beta gestionada.

## 5. Verificación y terminado

```powershell
python scripts/pipeline/run_btc_pipeline.py
python scripts/pipeline/normalize_champions.py --check
python scripts/pipeline/generate_asset_weekly_forecast.py --asset btcusdt --year all
python -m scripts.analysis.forward_tracker --report
```

Terminado significa: protocolo de `btc_hodl_b1` firmado, paper/replay semanal comparable,
derivados honestamente clasificados PIT/forward-only y, como máximo, un H-VOL-02 con veredicto
inmutable. El baseline continúa si el trial da NO.
