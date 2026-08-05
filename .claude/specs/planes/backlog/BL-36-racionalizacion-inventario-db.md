---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - airflow/dags/l0_macro_update.py
  - services/signalbridge_api
---

# BL-36 — Racionalización del inventario DB (59 tablas → matriz de verdad aplicada)

**Fuente**: perfil de datos del operador (artifact 2026-07-27: 59 tablas, 964 cols,
2.82M filas, 7 hypertables) × FABRIC §7 (matriz de fuente de verdad) y §31 ("ningún
atributo con dos escritores") · **Ola**: 2-3 (decisiones antes de crear esquemas
nuevos) · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado contra el perfil)

**Núcleo VIVO y alineado (mantener tal cual):**
`usdcop_m5_ohlcv` (2.20M, multi-símbolo por diseño) · `asset_daily_ohlcv` (60K, 1979→)
· `asset_native_ohlcv` (286K, raw provenance 1h/4h/1mo) · `macro_indicators_{daily 34c,
monthly, quarterly}` · **`macro_indicators_pit` (216K — ES la rama as_released que
BL-24 formaliza: victoria de alineación)** · `macro_banrep_forwards_monthly`
(forward_rate 100% NULL por diseño: la señal vive en implied_dev) · `macro_remesas` ·
`crypto_derivatives_daily` (funding backtesteable) · `forecast_h5_*` (producción) ·
`dim_asset` (¡ya trae annualization 52/252/365 — embrión del asset registry FABRIC!) ·
`market_session_calendar` (per-asset 2020-2027) · `market_ingestion_manifest`
(provenance) · `news_*` vivas · `audit_log` · `sb_users/configs/trading_state` ·
esquemas cripto vacíos forward-only (event_calendar/flows/onchain/exposure — moratoria
BTC: se llenan solos, NO borrar).

**Redundancias detectadas (dos escritores potenciales / peso muerto):**
1. **BI star schema**: `bi.fact_{consensus,forecasts,inference_runs,model_metrics}` =
   0 filas (dims 9/7 pobladas). El forecasting es file-based; esto duplica el destino
   de BL-15 (`forecast_output`) y BL-18 (`metric_event`).
2. **Registro de experimentos paralelo**: `experiment_{runs,comparisons,deployments}`,
   `model_registry`, `metrics.model_performance` = 0 filas. FABRIC §7 asigna modelos a
   MLflow+MinIO; estas 5 tablas son un segundo registry jamás escrito.
3. **`macro_variable_snapshots`** (0 filas, 22c) vs `macro_indicators_pit`: el
   snapshot-para-LLM quedó huérfano (analysis es file-driven).
4. **`daily_analysis`/`weekly_analysis`** (0): declaradas "DB destino futuro" — dos
   verdades latentes con los archivos.
5. **OMS legacy**: `signals`, `trades_history`, `equity_snapshots`, `sb_signals`,
   `sb_executions` = 0 filas — se solapan con el destino de BL-21 (`exec.*`
   event-sourced) y BL-22 (`fact_position/fact_pnl`).
6. **SPY legado**: 8,428 filas en asset_daily + 403 en native (retirado por directiva
   2026-07-27; el lector ya no lo consume).

**Gaps de spine detectados por el perfil** (alimentan BL-17/24):
`asset_daily_ohlcv.available_at` 19.9% NULL · `usdcop_m5.available_at` 6.4% NULL.

## Qué falta exactamente

1. **Decisión por grupo, escrita en la matriz de verdad** (un escritor por atributo):
   - bi.fact_* → DEPRECATED (drop tras BL-15/18; dims se derivan de config YAMLs).
   - experiment_*/model_registry/metrics.model_performance → DEPRECATED; autoridad =
     MLflow + bundles inmutables (o adopción explícita como proyección — pero UNA cosa).
   - macro_variable_snapshots → DROP (pit es la vintage; analysis es file-driven).
   - daily/weekly_analysis → DEPRECATED hasta que exista escritor real (nota en spec).
   - OMS legacy → ABSORBIDAS por BL-21/22 con migración y drop posterior (jamás
     esquemas paralelos conviviendo).
   - SPY → quarantine por `source`/símbolo (excluir de lectores; conservar como linaje
     de los bundles v1.0.0; NO delete).
2. **Backfill etiquetado de `available_at`** (reconstruido = cierre+1d, marcado, no
   vintage) en daily/m5 — prerrequisito del spine (BL-17) y del camino dorado (BL-24).
3. RL (`inference_features_5m`, `trading.*`) → mantener esquemas, marcar PAUSED.

## Impacto frontend

Ninguno directo (las tablas redundantes tienen 0 filas y nada las lee). Indirecto:
evita que BL-32 (Passport/Control Tower) nazca leyendo dos verdades.

## Dependencias

Decide ANTES de ejecutar BL-15/18/19/21/22 (los destinos nuevos absorben, no conviven).

## Verificación

- Documento de decisión por grupo en la matriz de verdad (§7) + migración de drops.
- `SELECT` de disponibilidad: 0 lectores rotos tras cada drop (grep de cada tabla en
  airflow/ services/ scripts/ dashboard antes de tocar).
- available_at: % NULL → 0 en daily/m5 con etiqueta de reconstrucción.

## Entrega parcial 2026-07-28 (matriz producida, decisiones NO ejecutadas)

La **matriz de verdad** vive en `.claude/specs/platform/db-truth-matrix.md`
(CTR-DB-TRUTH-MATRIX-001), generada desde el repo por
`scripts/diagnostics/db_inventory_matrix.py` → `.claude/generated/db-inventory.json`.
**Sin acceso a la DB** (Docker parado; orden de no arrancar nada): filas/freshness solo
para las 18 tablas de los manifiestos de backup commiteados; todo lo demás marcado
`NO VERIFICADO`. Añade `staging_contract` (§7) y la semántica de 5 timestamps con el
mapeo del legado (§8). **12 decisiones D-01..D-12 quedan pendientes del operador** (§10);
ninguna se ejecutó, cero DDL. Correcciones al estado declarado arriba: `bi.fact_*` SÍ
tiene escritores/lectores en código; hay **dos** `model_registry` y **tres** tablas de
credenciales; las migraciones 070-073 ya crearon los esquemas destino, por lo que el gate
"decidir antes de crear" se convierte en un problema de convivencia (D-08).

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_db_truth_matrix.py -q
verde:   8 passed

muta:    .claude/specs/platform/db-truth-matrix.md — invertir la decisión sobre bi.fact_*:
         de "DEPRECATED / 7 ficheros la referencian"
         a  "AUTORITATIVA (escritor único) / 0 ficheros"
         — o sea, una afirmación de autoría exclusiva sobre un atributo que YA tiene cinco
         escritores medidos, más un conteo de referencias errado en siete
espera:  **3 failed, 5 passed** — medido y registrado el 2026-08-05; caen los tres muros
         (test_prose_reference_counts_match_the_measured_inventory,
          test_no_table_is_declared_sole_writer_of_an_attribute_it_shares,
          test_deprecated_tables_disclose_the_readers_that_still_exist)

muta-2:  borrar SOLO `7 ficheros referencian \`bi.fact_forecasts\`` de esa fila
         (el retiro se propone sin decir a quién rompe)
espera-2: **1 failed, 7 passed** — únicamente
          test_deprecated_tables_disclose_the_readers_that_still_exist

control: citar un sustituto CON lectores vivos en la prosa de la decisión
         (`autoridad = \`control.metric_event\` y \`bi.fact_forecasts\``)
espera-3: **8 passed** — citar a quién se CONSERVA no es proponer su retiro
```

**El candado iba ROJO POR LA RAZÓN EQUIVOCADA, y eso corroe igual que un verde falso
(2026-08-05).** Esta ficha declaraba `verde: 8 passed`; la corrida real daba **1 failed**:

```
L185: propone Eliminar para ['control.metric_event'] pero la fila no declara ningún
      conteo de referencias; lectores reales: control.metric_event <-
      ['airflow/dags/control_system_health.py']
```

La fila acusada es `| metrics.model_performance | … | Eliminar; autoridad =
control.metric_event |` — o sea, propone eliminar `metrics.model_performance` y cita
`control.metric_event` como la autoridad que **se conserva**. El test recogía los backticks
de **todas** las celdas, así que trataba al sustituto como si fuera el objeto del retiro.

Por qué apareció ahora y no en julio: el test sólo se queja si la tabla tiene **lectores
vivos**, y `control.metric_event` ganó el suyo el 2026-08-05 (`control_system_health.py`,
el consumidor de C031 que entregó CODEX). O sea: **un cambio correcto y ajeno puso rojo un
candado mío por un defecto de parsing mío.** Arreglado tomando el sujeto de la **primera
columna**, nunca de la prosa; el control de arriba fija la regresión para que no vuelva.

Un rojo por la razón equivocada es tan caro como un verde por la razón equivocada: enseña
a ignorar el candado, y el día que grite de verdad ya nadie lo mira.

**Historial honesto**: hasta el 2026-07-28 este BL tenía **CERO cobertura**. Se podía invertir
cualquier decisión de la matriz de verdad —incluido declarar dos escritores para el mismo
atributo, que es justo lo que FABRIC §31 prohíbe— y **ningún gate se movía**;
`grep db_inventory_matrix|db-truth-matrix|CTR-DB-TRUTH` encontraba solo el generador y prosa.
Ahora 8 tests contrastan **77 claims** (50 de columnas W/R + 27 en prosa) contra
`.claude/generated/db-inventory.json`, derivando el perímetro de lectores por glob sobre 1277
ficheros con **las regex del propio generador** (K-029, cero listas a mano).

**Dos límites declarados**: a nivel **atributo** es imposible —la matriz no tiene columna
atributo→escritor, así que se implementó a nivel **tabla**—; y *"una DEPRECATED no puede tener
lectores"* **saldría rojo en prístino**, porque §9 declara `bi.fact_*` DEPRECATED y a la vez
publica que 7 ficheros la referencian: eso es el estado honesto, no el defecto. Se convirtió
en: *un retiro propuesto sobre una tabla con lectores vivos debe DECLARAR su conteo de
referencias*.

**Incoherencias numéricas encontradas: cero.** La matriz estaba bien; lo que no había era nada
que lo comprobara.

## Notas constitución

Ningún drop toca datos con historia real irrecuperable (todas las candidatas están en
0 filas salvo SPY, que se conserva). La regla que gobierna: "exactamente una fuente
autoritativa por atributo" (§25) — este BL es su aplicación al inventario heredado.
