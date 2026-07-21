---
kind: roadmap
status: PLANNED
version: 2.1.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/assets/pipelines.yaml
  - scripts/analysis/portfolio_daily.py
  - scripts/analysis/forward_tracker.py
  - airflow/dags/forward_ledger_weekly.py
  - scripts/diagnostics/generate_inventory.py
---

# Plan consolidado de rentabilidad — cuatro activos y libro

> Plan maestro ejecutable. Los planes por activo viven en
> `.claude/specs/assets/<asset_id>/PLAN-RENTABILIDAD-2026-07.md`.
> La rentabilidad no se puede garantizar: el objetivo controlable es preservar capital,
> obtener evidencia causal neta de costos y retirar rápido lo que no cumpla.

## 1. Decisiones irrevocables de esta versión

1. **No se reentrena dirección a siete semanas.** Nueve modelos por un horizonte nuevo serían
   63 miradas adicionales sobre una familia direccional ya cerrada; labels solapados no crean
   observaciones independientes.
2. **Una frecuencia, una función.** Daily/semanal decide; 1h/4h estima riesgo y rango; M5
   ejecuta. No habrá cuatro votos direccionales correlacionados por activo.
3. **Calmar y drawdown mandan.** Sharpe es secundario; con menos de 20 trades no se publica
   Sharpe ni p-value. Un claim de alfa exige DSR trial-aware > 0.95.
4. **El presupuesto es 0–3 trials nuevos, no una cuota que deba gastarse.** Cada trial tiene
   un gate de activación previo y un juez que no fue usado para formularlo.
5. **Los cuatro campeones siguen congelados.** Un cambio de señal, sizing, costos o ejecución
   crea otra versión, otro manifest, otro protocolo y otro reloj forward.

## 2. Correcciones sobre el plan v1

| Hallazgo verificado | Corrección del plan |
|---|---|
| El ledger ya contiene filas W30 | No se “abre”: se exige cobertura semanal consecutiva y unidades comparables; las filas W30 anteriores al schema `paper_week_pct` son historia, no evidencia de divergencia |
| `portfolio_daily.py` excluye COP por reloj | H-CARRY-01 no puede inclinar el ERC diario; primero debe existir un overlay semanal que combine el libro diario agregado a semana con el sleeve COP |
| `run_spx500_pipeline.py` delega al runner sintético | SPX queda P0 `pipeline_blocked`; ningún resultado del DAG cuenta hasta que el entrypoint use `load_real()` y falle cerrado si no hay feed PIT |
| El protocolo BTC apunta a `btc_trend_b2` | Se reescribe para `btc_hodl_b1` antes de iniciar su reloj; firma humana obligatoria |
| Oro no tiene protocolo de retiro | Se crea y firma antes de paper-shadow MT5 |
| El audit de restore precede al commit `2af48c2` | Wiring 060–063 y restore quedan como corrección implementada; el cold-start drill con artefacto sigue siendo gate de release |
| Los planes v1 tenían `last_verified: 2026-07-22` | Se corrige a la fecha real de verificación: 2026-07-21 |

## 3. Estado de verdad al corte

| Activo | Campeón congelado | Reloj | Estado máximo hoy | Bloqueador que decide |
|---|---|---:|---|---|
| USD/COP | `smart_simple_v11` | 52/año; ejecución M5 COT | production operativa, **edge no probado** | Corte A/B forward, PIT macro y swap tradeable |
| XAU/USD | `gold_trend_simple` | 252/año; daily t−1 | experimental | protocolo firmado, paper MT5 y comportamiento forward |
| BTC/USDT | `btc_hodl_b1` | 365/año; daily UTC | experimental | protocolo correcto firmado y 26 semanas forward |
| SPX500 | transición a `spx500_daily_ma200_v1` | 252/año; daily next-open | research only / pipeline blocked | Investing daily fail-closed, baseline causal, broker y forward |
| Libro | ERC diario XAU+BTC+SPX, efectivo remunerado | 252/año | `risk_controlled_book`, research only | criterios 1–6 medidos; falta forward válido y protocolo |

Los conteos de trials del front matter de cada registry son la autoridad actual: COP 58,
XAU 77, BTC 34 y SPX 17. Antes de cualquier ejecución se reconcilian las contradicciones entre
front matter, cuerpo y artefactos; nunca se reduce un conteo sin evidencia.

## 4. Contrato multi-timeframe: las cuatro tablas no son cuatro estrategias

| Contrato | Uso permitido | Gate común |
|---|---|---|
| `market_ohlcv_5m` | fills, spread, slippage, HS/TP y realized range intradía | sesión/calendario correcto, `available_at`, cero huecos marcados como flat |
| `market_ohlcv_1h` | volatilidad, rango, liquidez y shocks; nativo primero | alias de símbolo, origen nativo/agregado y barra completa |
| `market_ohlcv_4h` | contexto de riesgo intermedio y vol-of-vol | mismas reglas de 1h; no votar dirección por separado |
| `market_ohlcv_daily` | decisión causal principal, baselines y replay | cierre de sesión correcto, `t−1`, ejecución next-bar/next-open |

Reglas por activo:

- **COP:** daily/semanal forma la decisión; M5 gestiona la sesión 08:00–12:55 COT; 1h/4h
  son diagnósticos de riesgo hasta crear una versión nueva.
- **XAU:** daily UTC decide; 1h/4h pueden medir rango y ejecución; M5 no genera alfa.
- **BTC:** daily UTC decide; 1h/4h/M5 solo pueden mejorar medición de riesgo y ejecución.
- **SPX:** Investing.com es la SSOT del price index diario. La v1 usa MA200 long/cash y no
  requiere 5m/1h/4h ni proxies macro; cualquier instrumento ejecutable tiene tracking propio.

`closed != missing`: cada join consulta `market_session_calendar`; no se rellenan con cero ni se
hace forward-fill de retornos para forzar calendarios comunes.

## 5. Arquitectura objetivo de cartera: dos relojes, no una covarianza falsa

1. **Libro diario 252:** XAU+BTC+SPX se alinean por intersección; BTC conserva su sleeve
   365 internamente, pero sus retornos de cartera se observan solo en las fechas comunes.
   Covarianza Ledoit-Wolf con ventana estrictamente pasada, ERC, efectivo remunerado y breaker.
2. **Overlay semanal 52:** se agrega el retorno realizado del libro diario por semana y se
   combina con el sleeve COP semanal. No se forward-fillea COP a daily ni se anualiza BTC a 252
   antes de construir su retorno.
3. **Carry COP:** solo puede modular el sleeve COP o el overlay semanal; jamás los pesos diarios
   XAU/BTC/SPX. El swap real neto del broker es el gate previo.

El libro se vende, si gradúa, como **beta gestionada**, no como señal ni alfa. Su titular es
Calmar, DD y estrés; debe mostrar correlación incondicional y co-activa juntas.

## 6. Presupuesto cerrado de trials

| Slot | Hipótesis | Se activa solo si | Juez limpio | Resultado de fallo |
|---|---|---|---|---|
| T1 | `H-ROBUST-DECADES-XAU` | estrategia/hash y segmentos 1980–2003 quedan sellados antes de abrir resultados | replay de falsificación por décadas; no promueve por sí solo | degrada campeón a contexto y bloquea live |
| T2 | `H-CARRY-01` COP | swap neto observado ≥50% del carry teórico y overlay semanal implementado | forward posterior al freeze; **2025 no vuelve a ser OOS** | carry cerrado; v11 no cambia |
| T3 | `H-VOL-02` BTC | protocolo firmado, M5 PIT completo y especificación única sin barrer estimador/ventana | forward posterior al freeze; QLIKE primero, economía después solo si estaba prefirmada | persistencia `rv20` sigue siendo el estimador; vía cerrada |

SPX no recibe trial nuevo. Primero debe publicar el baseline MA200 ya observado, arreglar su
pipeline y dejar que el forward decida `H-SIMP-SPX-02`. Registrar un plan no consume trial;
abrir resultados sí.

## 7. Fases y gates

### Fase 0 — Gobierno y jueces (0 trials)

| ID | Acción | Artefacto / aceptación |
|---|---|---|
| G0.1 | Reescribir y firmar protocolo BTC para `btc_hodl_b1` | estrategia, hash, ventana 26 semanas, DD, tracking y firma no vacíos |
| G0.2 | Crear y firmar protocolos XAU, SPX y libro | ningún track paper/live sin regla de retiro previa |
| G0.3 | Reconciliar los cuatro registries | front matter = cuerpo = evidencia; `sigma_trials` real o grid conservador declarado |
| G0.4 | Congelar manifests y costo por instrumento | fees, spread, slippage, swap/financing, reloj y ejecución explícitos |
| G0.5 | Sanear ledger | schema versionado; no comparar YTD contra semana; cobertura y missingness visibles |

**Gate G0:** cero protocolo ambiguo, cero champion huérfano y cero conteo de trials contradictorio.

### Fase 1 — Datos y pipelines causales (0 trials)

| ID | P | Acción | PASS |
|---|---:|---|---|
| D1.1 | P0 | SPX usa `load_real()` end-to-end y prohíbe fallback sintético | test demuestra que falta de snapshot falla cerrado; artefacto declara `data_class` y PIT |
| D1.2 | P0 | Investing daily authoritative para SPX | raw snapshots, checksum/parser, OHLC NYSE, fail-closed y cero fallback/mezcla |
| D1.3 | P0 | Matriz PIT por fila/feature usada | 100% `available_at <= decision_at`; macro vintage o lag conservador por serie |
| D1.4 | P1 | Cold-start drill 060–063 + restore | commit SHA, RTO/RPO, checksums, counts y cero skips DB-backed |
| D1.5 | P1 | Ratchet de cuatro frecuencias | OHLC, duplicados, futuro, sesión, closed/missing, nativo/agregado y timezone |

**Gate D1:** ninguna estrategia consume una fila cuya disponibilidad o símbolo económico sea
ambiguo. Un score de calidad alto sin PIT no pasa.

### Fase 2 — Cerrar el ciclo operativo (0 trials)

| ID | Acción | PASS |
|---|---|---|
| O2.1 | SPX L0→L4→L5→L6 daily | Investing→MA200 causal→signal next-open→paper fill→ledger; ningún `datagen.generate()` |
| O2.2 | XAU/BTC publishers preservan campeón | `normalize_champions.py --check` pasa después de cada DAG |
| O2.3 | Medir swap/financing real COP/XAU/SPX | statements normalizados y costo neto reproducible |
| O2.4 | Forward ledger de los cuatro campeones | ≥2 semanas consecutivas con `paper_week_pct` y replay comparables, o missing explicado |
| O2.5 | Libro de dos relojes | daily ERC + overlay semanal, ambos sin ffill y con efectivo/costos explícitos |

**Gate O2:** el mismo hash de estrategia produce señal, paper fill, replay y fila forward.

### Fase 3 — Ejecutar solo trials activados

Orden: T1 → T2 → T3. Se detiene al primer gate de activación fallido; no se reasigna el slot
automáticamente a otra idea. Cada ejecución actualiza registry, log, params hash, costos ×1/×2/×3,
baselines y veredicto, incluso si da NO.

### Fase 4 — Forward, retiro y capital

Secuencia por track: `shadow → paper → capital mínimo → escalado`. El Vote 2 continúa siendo
humano. Requisitos mínimos:

- protocolo firmado antes del primer dato;
- cobertura de más de un régimen y ≥30 oportunidades; más tiempo si la estrategia es lenta;
- Calmar y DD dentro de protocolo, costos ×2 positivos y tracking paper/replay dentro del umbral;
- DSR > 0.95 solo cuando se afirma alfa; el libro usa la barra de ADR-0020 y lenguaje de beta;
- breaker por datos, staleness, spread, slippage, divergencia y drawdown probado.

COP conserva sus cortes firmados: A 2026-09-16 y B 2027-03-17. Los demás relojes empiezan en
la fecha de firma y primer snapshot válido, no en la fecha de este documento.

## 8. Verificación reproducible

```powershell
python scripts/diagnostics/generate_inventory.py --check
python -m pytest tests/regression/test_knowledge_frontmatter.py -q
python -m pytest tests/regression/test_knowledge_inventory.py -q
python scripts/pipeline/normalize_champions.py --check
python scripts/pipeline/run_spx500_pipeline.py --check
python -m scripts.analysis.forward_tracker --report
```

Para un release, los tests DB-backed deben ejecutarse en el entorno con PostgreSQL; `skipped`
no equivale a verde. El artefacto de cierre registra commit, entorno, passed/failed/skipped,
duración y logs.

## 8b. Estado de implementación v2.1.0 (fusión Claude+Codex, 2026-07-21)

Revisión cruzada final: los planes v1 (Claude) y v2 (Codex) convergieron; v2.1.0 registra lo
que quedó IMPLEMENTADO en la misma sesión, con cada corrección de Codex verificada en código
antes de aceptarse:

| Ítem del plan | Estado | Evidencia |
|---|---|---|
| S0.3/S0.4 — Baseline tonto con look-ahead (Calmar 1.641 retractado) | **HECHO** | Verificado en `profitability_evidence.py:357` (señal sin lag + costos ajenos, afectaba también a Oro); corregido (lag 1 barra + costo |dW|×tarifa); recomputado: ma200 1.641→0.5393 (el gated 0.6482 SÍ lo bate), sma_vote Oro→−0.032; erratum en registry SPX; artefactos viejos preservados como inválidos (2026-07-20), corregidos en 2026-07-21 |
| S0.1 — Runner SPX fail-closed | **HECHO** | `run_spx500_pipeline.py`: default = `load_real()` (verificado vivo: 1.644 filas reales, 923 pesos no-nulos); sintético solo con `--synthetic` explícito |
| G0.1 — Protocolo BTC reescrito para `btc_hodl_b1` | **HECHO** (pendiente FIRMA del operador) | `WITHDRAWAL-PROTOCOL-BTC.md` v2.0.0 |
| G0.2 — Protocolos XAU y SPX creados ex-ante | **HECHO** (pendiente FIRMA) | `WITHDRAWAL-PROTOCOL-XAU.md`, `WITHDRAWAL-PROTOCOL-SPX.md` (cubre los DOS bundles de H-SIMP-SPX-02, con la regla "el baseline ES la estrategia" pre-firmada) |
| Enmienda H-CARRY-01 (scope sleeve/overlay + juez forward + gate U1 bootstrap) | **HECHO** (0 miradas) | registry COP, sección ENMIENDA |
| G0.3 σ_trials / conteos · G0.5 ledger schema · D1.x PIT/alias · O2.x | ABIERTO | siguiente tanda |

## 9. Definición de terminado

El plan termina cuando los cuatro tracks tienen datos causales, pipeline real, champion y
protocolo coherentes; el ledger acumula sin unidades mezcladas; el libro respeta los dos
relojes; los ≤3 trials tienen veredicto inmutable; y cualquier uso de capital puede ser
retirado automáticamente por reglas firmadas. Un backtest más bonito no es terminado.
