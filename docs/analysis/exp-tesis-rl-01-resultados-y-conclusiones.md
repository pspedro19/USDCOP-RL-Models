---
kind: analysis
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-11
supersedes: []
code_anchors:
  - scripts/presentation/generar_resultados_y_figuras.py
  - scripts/analysis/thesis_statistics.py
  - outputs/thesis/statistics_holdout.json
  - outputs/thesis-repair/sanity_S1_protocol.json
  - outputs/thesis-repair/sanity_protocol_v2.json
  - src/research/llm_trader.py
  - src/research/macro_asof.py
  - config/research/llm_thesis.yaml
---

# EXP-TESIS-RL-01 — resultados, figuras y conclusiones defendibles

Este documento es el índice reproducible de resultados de la tesis. Las cifras de USD/COP
provienen de los JSON publicados; las figuras son salidas del generador, no capturas editadas.
La versión corregida del entorno aún no tiene un juez confirmatorio: selección 2023 y hold-out
2024–2026 se habían abierto antes de las reparaciones y solo sirven como diagnóstico
retrospectivo.

## Resultado principal USD/COP

La reconciliación de fuentes avanzó con un artefacto separado:
[`macro_identity_research_v2.json`](../../outputs/thesis-repair/macro_identity_research_v2.json).
Brent (FRED DCOILBRENTEU), DGS2 (FRED DGS2), IBR overnight nominal (BanRep
`IRIBRM00/NR`) y DXY Investing (instrumento `942611`) coinciden con las fuentes declaradas
en el parquet v2; `all_declared_identities_honoured=true`. La coincidencia DXY es una prueba
de identidad/reproducibilidad con Investing, no una validación independiente contra ICE ni una
licencia de redistribución. BanRep documenta el flujo histórico `DF_IBR_DAILY_HIST`
para IBR diario en su [guía SDMX oficial](https://suameca.banrep.gov.co/archivos/webservices/documento_tecnico_ws_consumo_sdmx.pdf);
ICE describe DXY como un índice administrado por ICE Data Indices en su [catálogo oficial](https://developer.ice.com/fixed-income-data-services/catalog/ice-data-indices-currency-indices).

El carril LLM está preparado, pero no ejecutado: [`llm_thesis.yaml`](../../config/research/llm_thesis.yaml)
congela proveedor, modelo, prompt, temperatura, tamaño de salida y política de errores;
[`llm_trader.py`](../../src/research/llm_trader.py) registra hashes de prompt/respuesta y conserva
la exposición previa ante JSON inválido o timeout. No se leen archivos `.env`, no se imprimen claves
y no se generaron resultados LLM sin pre-registro y ledger.

En el hold-out histórico (584 sesiones), `ppo_regime` obtiene **−54,87 % neto compuesto**
frente a **0,00 % de `always_flat`**. Las cinco semillas son negativas: −46,01 %, −59,37 %,
−65,44 %, −47,79 % y −53,91 %. La media de semillas no se interpreta como una política
individual. La tabla completa está en
[`tabla_4_3_desempeno_holdout.md`](../../outputs/thesis/tablas/tabla_4_3_desempeno_holdout.md)
y el origen estadístico en
[`statistics_holdout.json`](../../outputs/thesis/statistics_holdout.json).

El bruto de costo cero es una cota contrafactual: +31,84 % compuesto para `ppo_regime`, pero
los costos realizados superan el bruto. Con ×2 y ×3 costos, el neto compuesto cae a −84,59 % y
−94,75 %. Esto permite rechazar la rentabilidad de **esta política bajo este contrato**, pero no
la rentabilidad estructural de todo USD/COP ni la existencia de otra política.

La descomposición atribuye aproximadamente +48,9 puntos al signo de la exposición diaria y
−21,0 puntos al timing intradía. Es una atribución ex-post, no una señal disponible a las 08:00.

## Figuras generadas

Las curvas y diagnósticos se pueden insertar directamente en la tesis:

- [Curvas de capital — hold-out](../../outputs/thesis/figuras/fig01_curvas_capital_holdout.png) y
  [selección](../../outputs/thesis/figuras/fig01_curvas_capital_selection.png).
- [Underwater — hold-out](../../outputs/thesis/figuras/fig02_underwater_holdout.png) y
  [selección](../../outputs/thesis/figuras/fig02_underwater_selection.png).
- [Sharpe móvil — hold-out](../../outputs/thesis/figuras/fig03_sharpe_movil_holdout.png) y
  [selección](../../outputs/thesis/figuras/fig03_sharpe_movil_selection.png).
- [Sensibilidad de costos — hold-out](../../outputs/thesis/figuras/fig04_sensibilidad_costos_holdout.png)
  y [selección](../../outputs/thesis/figuras/fig04_sensibilidad_costos_selection.png).
- La distribución de acciones por régimen de v1 se conserva como artefacto histórico, pero no
  se presenta como resultado v2: el replay v1 espera 39 features y el dataset corregido tiene 37.
  La figura v2 se regenerará tras entrenar modelos v2; `--allow-stale-artifact` no transforma un
  replay incompatible en evidencia nueva.
- [Dispersión entre semillas — hold-out](../../outputs/thesis/figuras/fig06_semillas_holdout.png)
  y [selección](../../outputs/thesis/figuras/fig06_semillas_selection.png).
- [Particiones](../../outputs/thesis/figuras/fig07_particiones.png), [Sharpe por régimen](../../outputs/thesis/figuras/fig08_sharpe_por_regimen_holdout.png),
  [sesión ejemplo](../../outputs/thesis/figuras/fig09_sesion_ejemplo_holdout.png),
  [frecuencia](../../outputs/thesis/figuras/fig10_frecuencia_holdout.png) y
  [bruto versus neto](../../outputs/thesis/figuras/fig11_bruto_vs_neto_holdout.png).

Las tablas de régimen, ablación, costos, DSR/PBO y descomposición están indexadas en
[`outputs/thesis/tablas`](../../outputs/thesis/tablas/tabla_4_1_datos.md).

La auditoría integral actualizada está en
[`integrity_current_20260911.json`](../../outputs/thesis-repair/integrity_current_20260911.json):
la compuerta de causalidad macro pasa, pero la evidencia sigue marcada como retrospectiva y
la igualdad con una fuente oficial de precios permanece `NOT_CERTIFIED`. El contraste diario
independiente TwelveData vs Investing para USD/COP está en
[`usdcop_daily_cross_source_v2.json`](../../outputs/thesis-repair/usdcop_daily_cross_source_v2.json):
1.727 fechas comunes, mediana de diferencia absoluta 0,07782 %, P95 0,551359 % y máximo
5,192771 %; `agreement_flag=OK`, sin afirmar igualdad tick a tick.

La política de merge quedó centralizada en [`macro_asof.py`](../../src/research/macro_asof.py):
features de mercado y HMM usan el mismo `merge_asof` estrictamente anterior a la apertura y
el mismo límite de frescura leído de `macro_availability.yaml`. La prueba contractual está en
[`test_macro_asof_contract.py`](../../tests/regression/test_macro_asof_contract.py); no se
interpolan observaciones ni se rellenan macro ausentes con cero.

La auditoría reproducible más reciente del contrato de frecuencia está en
[`data_contract_audit_current.json`](../../outputs/thesis-repair/data_contract_audit_current.json).
Sobre `seeds/latest/usdcop_m5_ohlcv.parquet` registra 100.674 barras, cero duplicados,
cero timestamps fuera de la grilla de cinco minutos, cero valores OHLC inválidos y 1.457
sesiones completas de 60 barras. Las sesiones incompletas se excluyen y se trazan, no se
rellenan silenciosamente. El macro tiene 12.882 fechas, cero fechas duplicadas, cuatro
series completas, cero valores numéricos inválidos y `same_day_values_usable=false`.

## Control científico del optimizador

El PPO original no resolvía el caso sintético trivial S1: 0/5 semillas quedaban planas sobre
ruido iid con costos positivos. La sonda estructural `flat_init` terminó con 5/5 semillas
planas (exposición media 0,019; 0,021; 0,041; 0,018; 0,024), abriendo únicamente la compuerta
S1. Esto prueba que el resultado histórico no puede atribuirse solo al mercado.

En S1--S4, una única receta `flat_init_no_turn` (sonda posterior declarada, sin penalización de turnover)
pasó el criterio de 4/5 en cada fixture a 100k pasos. El agregado
[`sanity_protocol_v2.json`](../../outputs/thesis-repair/sanity_protocol_v2.json) exige cinco
semillas, cero trials de mercado, hashes de entrada y marca explícitamente `market_evidence=false`.
En S2, la receta obtuvo a
100k pasos retornos netos medios de +7,39 %, +7,38 %, +7,51 %, +7,49 % y +7,45 % en las
semillas 42, 123, 456, 789 y 1337, respectivamente, frente a un oráculo +7,72 %. Las 5/5
semillas pasan S2; S3 también pasa en las cinco semillas y S1/S4 satisfacen el control de
abstención. Esto prueba que la receta puede resolver controles conocidos y no confundir ruido
con una posición sostenida; no es evidencia de mercado ni de rentabilidad.
Evidencia: [`sanity_S1_protocol.json`](../../outputs/thesis-repair/sanity_S1_protocol.json) y
los artefactos `outputs/thesis-repair/sanity/S2_flat_init_no_turn_seed{42,123,456,789,1337}.json`.

Como controles puntuales adicionales, S3/semilla 42 produjo +7,24 % neto medio con señal por
encima del costo, y S4/semilla 42 convergió a exposición 0,000 cuando la señal era menor que el
costo. Son mediciones de una semilla, no veredictos: el criterio formal exige 4/5 en cada
fixture y el agregado formal confirma ese criterio, separado de cualquier evidencia de mercado.

## Oro (XAU/USD)

El pipeline se ejecutó sobre 5.897 barras diarias reales (2004-01-02--2026-08-24) y dejó el
artefacto [`gold_diagnostic_2026-09-11.json`](../../outputs/thesis-repair/gold_diagnostic_2026-09-11.json).
Es una medición diagnóstica: `trials_charged=0`, `promotion_eligible=false` y el contrato
[`gold_cost_contract.yaml`](../../config/research/gold_cost_contract.yaml) sigue en
`PENDING_VENUE`, por lo que no se reporta como rentabilidad ejecutable. En la serie sin un
contrato de spread/fills verificable, el mejor resultado histórico fue `gold_trend_simple` (retorno
190,16 %, Sharpe 0,613, MaxDD -22,87 %); en OOS 2025 obtuvo 38,0 % y Sharpe 3,048. El DSR de
esa familia fue 0,9904, pero estos números no incluyen un costo de venue verificable y no autorizan
operar. El régimen-gated no supera simultáneamente a los dos baselines (Sharpe 0,459 frente a
0,521 y 0,357), así que tampoco hay evidencia de que el HMM agregue valor.

## Baselines v2 recalculados

Los baselines se recalcularon con el mismo motor de sesiones y costos. En el hold-out de
570 sesiones, `B1_buy_hold_1x` obtuvo retorno compuesto -53,696 %, Sharpe -2,813 y MaxDD
-54,791 %; `B1_passive_buy_hold_overnight` obtuvo -21,329 %, Sharpe -0,693 y MaxDD
-32,078 %; `NULL_A_short_1x` obtuvo -45,571 %, Sharpe -2,227 y MaxDD -49,706 %.
`always_flat` permanece en 0 %. Las reglas intradía de referencia fueron aún peores:
momentum -100,000 %, mean-reversion -99,930 %, opening-range -83,988 % y las dos reglas
por régimen -34,118 %. El artefacto reproducible es
[`baselines_holdout_v2.json`](../../outputs/thesis-repair/baselines_holdout_v2.json).

Estos números no prueban que el mercado sea imposible: indican que el contrato de costos y
la representación actual castigan fuertemente el turnover intradía. No se usa esta tabla para
seleccionar una estrategia después de observar el hold-out.
El DSR reproducible con `N=115` trials y anualización `sqrt(221)` está en
[`baselines_holdout_dsr_v2.json`](../../outputs/thesis-repair/baselines_holdout_dsr_v2.json):
buy-and-hold intrasesión = `1,36e-7`, siempre corto = `1,26e-5`, reglas por régimen =
`2,34e-4`; ninguno supera 0,95. Los ratios se suprimen para `always_flat`, overnight y
`B1_prime` porque tienen menos de 20 operaciones, conforme a la constitución.

## Conclusiones de tesis

1. Bajo los artefactos auditados, el PPO evaluado pierde contra abstenerse después de costos.
2. No está demostrado que aprenda una señal negociable: la fuga macro, la representación plana
   de velas y la sensibilidad a latencia invalidan esa lectura del v1.
3. La receta PPO tiene un fallo de optimización/objetivo observable en datos sintéticos; por eso
   no es válido afirmar que una política rentable no existe en el espacio de acciones.
4. La contribución defendible es metodológica: contratos causales, costos explícitos, múltiples
   semillas, hold-out único y publicación de negativos.
5. Una afirmación de rentabilidad para USD/COP u oro queda **pendiente** hasta reconciliar las
   fuentes macro, medir costos de venue y completar el juez forward pre-registrado; S1–S4 ya
   pasaron como controles sintéticos y no sustituyen ese juez.

## Reproducción

La auditoría también calcula una correlación diagnóstica con FRED `DTWEXBGS` (proxy amplio
del dólar; 0,969961 en 5.080 fechas comunes), pero ese proxy no certifica que la columna sea
el índice ICE DXY y no abre el gate.

```text
python scripts/presentation/generar_resultados_y_figuras.py
python scripts/analysis/thesis_ppo_sanity.py --fixture S1 --probe flat_init --seed 42 --timesteps 100000
python scripts/diagnostics/audit_research_data_contract.py
python scripts/pipeline/rebuild_thesis_portable_v2.py --macro data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet --macro-identity outputs/thesis-repair/macro_identity_research_v2.json
# validar contextos LLM sellados sin red
python scripts/analysis/run_thesis_llm.py --input-jsonl data/thesis/llm/contexts.jsonl --model-id deepseek-chat
# sellar una sola barra PPO desde un prefijo recibido; rechaza barras futuras
python scripts/analysis/run_ppo_stream_bar.py --session-date YYYY-MM-DD --bar-index 0 `
  --bars-path data/forward/prefix.parquet --model data/thesis/ppo/model.zip `
  --model-id model.zip --arm-id ppo_stream_v1 `
  --ledger data/forward/ledger/decisions.jsonl --state data/forward/state/YYYY-MM-DD.json
python scripts/analysis/settle_thesis_llm.py --ledger data/thesis/llm/deepseek_decisions.jsonl --block selection --output outputs/thesis/llm_selection_settlement.json
python scripts/presentation/generate_llm_figures.py --settlement outputs/thesis/llm_selection_settlement.json --output-dir outputs/thesis/figuras/llm
```

Para ejecutar llamadas reales, el operador debe cargar las credenciales en el entorno de su
proceso y añadir `--execute`; el CLI no lee `.env`, no imprime secretos y exige un proveedor
explícito. La salida confirmatoria solo se autoriza después de firmar el preregistro y abrir el
ledger correspondiente.

El exportador de contextos también rechaza el portable actual cuando su identidad no coincide
con el código (`portable dataset identity mismatch`). Esto es intencional: primero debe
reconstruirse el portable v2 con el gate macro completo; no se permite generar decisiones LLM
contra un dataset obsoleto ni usar `allow_stale` como atajo.

La auditoría de frecuencias queda reproducible en
[`research_data_contract_macro_v2.json`](../../outputs/thesis-repair/research_data_contract_macro_v2.json):
OHLCV USD/COP está en grilla efectiva de 5 minutos, el macro diario no tiene fechas duplicadas,
las cuatro columnas son numéricas y la política exige merge PIT. `complete_session_grid=false`
para el archivo histórico completo porque contiene sesiones parciales; la máscara de investigación
las excluye y conserva sus fechas en `dropped`, no las rellena.

El alias `python` puede fallar en Windows por el launcher; usar el intérprete Python 3.12
instalado en el entorno. No se deben reemplazar cifras ni figuras manualmente.
