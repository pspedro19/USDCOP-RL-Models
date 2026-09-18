---
kind: analysis
status: PARTIAL
version: 1.0.13
last_verified: 2026-09-14
supersedes: []
code_anchors:
  - src/research/observation_contract.py
  - src/research/regime5_bundle.py
  - src/research/regime5_hmm.py
  - src/research/research_readiness.py
  - scripts/data/build_research_regime5.py
  - scripts/diagnostics/freeze_thesis_evidence.py
  - scripts/presentation/build_research_grade_thesis.py
  - scripts/diagnostics/audit_thesis_e2e_status.py
  - scripts/data/export_research_bundle_v3.py
  - src/research/reporting_v2.py
  - src/research/macro_evidence.py
  - src/research/publication_join_v2.py
  - src/research/sanity_gate.py
  - src/research/llm_experiment_v2.py
  - src/research/alfred_vintages.py
  - scripts/diagnostics/audit_research_macro_vintages.py
  - src/research/retrospective_regime.py
  - src/research/vintage_evidence_gate.py
  - src/research/historical_hmm_audit.py
  - scripts/diagnostics/audit_hmm_representation.py
  - scripts/diagnostics/audit_hmm_2026_representation.py
  - src/research/live_spec.py
  - tests/regression/test_live_regime_contract.py
  - src/research/evaluation_mask.py
  - tests/regression/test_live_prefix_causality.py
  - src/research/llm_forward/settlement_accounting.py
  - src/research/llm_forward/settle_thesis.py
  - src/research/llm_forward/verify.py
  - tests/regression/test_forward_settlement_accounting.py
  - src/research/llm_forward/arms/ppo_arm.py
  - airflow/dags/research_forward_arms.py
  - tests/regression/test_forward_first_bar_hold.py
  - tests/regression/test_forward_dag_safety.py
---

# Cierre escalonado de la tesis: evidencia reproducida y límites

Implementación solicitada por el operador. USD/COP es el objeto de este cierre; oro,
nuevos algoritmos y cambios retrospectivos de la política quedan fuera. No se promete
rentabilidad ni se convierte un resultado negativo en evidencia de imposibilidad.

## Resultado entregable ahora

La entrega vigente es el [bundle v5](../../outputs/thesis-repair/research_grade_20260912_v5/results.md),
con [manifiesto](../../outputs/thesis-repair/research_grade_20260912_v5/manifest.json)
SHA `00fc276e22b62ffac144bca1abb6c468f15e272225c46743304c67fe120ae2da`.
Corrige únicamente la clasificación descriptiva de regímenes y su figura, con
[procedencia explícita](../../outputs/thesis-repair/research_grade_20260912_v5/regime_provenance.json).
Las series, métricas, stress, Sharpe móvil y calidad son byte-idénticos a v3; también
las otras siete PNG. Las estadísticas no cambian; `results.json` sólo difiere en la hora
de creación. Los SVG se regeneran con identificadores/metadata nuevos y no se afirma
identidad byte a byte. Las versiones anteriores se conservan, no se sobrescriben.

El [bundle histórico v2](../../outputs/thesis-repair/research_grade_20260912_v2/results.md)
contiene tablas, series diarias, ocho figuras PNG y sus versiones SVG. El
[manifiesto](../../outputs/thesis-repair/research_grade_20260912_v2/manifest.json)
identifica todos los insumos y salidas por SHA256. Hash del manifiesto:
`169fbe22f455483ae7b7863a32d414882844fb1b363d886bc336a72d1fb7923e`.

Tras limpiar el código de presentación se ejecutó una
[tercera reproducción](../../outputs/thesis-repair/research_grade_20260912_v3/manifest.json),
SHA `d8c317592323f5cbad46e8b34eff1e72d5e2904b7c234e948bd1908268024a21`.
Las series, métricas CSV, datos de Sharpe/costos/acciones/calidad y las ocho PNG son
**idénticos byte por byte a v2**. El [E2E final](../../outputs/thesis-repair/research_grade_e2e_20260912_final.json)
verifica esa reproducción y mantiene explícitamente abiertos los gates pendientes.

Los resultados corresponden a **selección 2023-01-03 a 2023-12-29, 226 sesiones**,
no a un hold-out nuevo. Se preservaron los artefactos anteriores en un
[snapshot](../../outputs/thesis-repair/archive/pre_research_grade_20260912_1934/manifest.json)
de 326 archivos. No se reemplazaron las decisiones, pesos, checkpoints ni datos históricos.
El capital inicial se expresa en **100.000 unidades de cuenta**: no es una cuenta real
USD certificada ni un contrato de CFD/NDF/futuros con conversión y financiación completas.

| Estrategia | Neto compuesto % | Sharpe √221 | MaxDD % |
|---|---:|---:|---:|
| No operar | 0,00 | N/A | 0,00 |
| Siempre corto intradía | −11,97 | −1,108 | −16,58 |
| B1 pasivo, sin costos | −20,15 | N/A | −22,94 |
| B1 largo por sesión | −32,54 | −3,533 | −33,20 |
| PPO backbone: cartera de cinco semillas | −33,09 | −8,564 | −33,18 |
| PPO régimen: cartera de cinco semillas | −35,52 | −6,818 | −35,94 |
| PPO mediana de pesos: componente exacto del híbrido | −36,67 | −5,188 | −37,73 |
| LogReg | −55,08 | −9,729 | −55,08 |
| Híbrido PPO + DeepSeek | −69,24 | −14,330 | −69,24 |
| Híbrido PPO + Azure | −70,59 | −11,006 | −70,59 |
| DeepSeek | −78,93 | −22,777 | −78,93 |
| Azure | −82,89 | −14,836 | −82,89 |

N/A no significa Sharpe cero. Flat tiene varianza cero; el pasivo es una sola operación
completa. El contrato del proyecto suprime Sharpe y p con menos de 20 operaciones.
No se confunden sesiones ganadoras con win rate por trade ni se calcula Profit Factor
sobre retornos diarios llamándolo estadística de operaciones.

## Conclusiones que resisten la reproducción

1. Todas las estrategias activas de la tabla perdieron bajo el contrato supuesto. No
   se encontró rentabilidad neta en este bloque. Eso no demuestra que no exista una
   política rentable, ni que PPO o los LLM fallen en todo mercado.
2. A costo cero, con **las mismas exposiciones congeladas**, PPO mediana compone
   **+10,0769 %**, DeepSeek **−14,6664 %** y Azure **−28,9768 %**. La primera política
   pierde su rendimiento positivo al cobrar costos; las otras dos ya pierden en bruto.
   Son propiedades de estas trayectorias, no identificación causal de un mecanismo general.
3. El bruto aritmético de PPO mediana es **+9,9007 %** y su IC bootstrap 95 % es
   **[−3,5122; +24,1400]**. No acredita alfa. No mezclar ese porcentaje sumado con
   retornos compuestos ni interpretar la suma de costos como porcentaje del capital inicial.
4. La ventaja puntual régimen/backbone en Sharpe de las carteras es +1,7455, pero
   el IC jerárquico pareado, remuestreando semillas y sesiones, es
   **[−2,0838; +4,4642]**. La mejora no queda establecida. Hay cinco semillas por
   configuración, no diez semillas por modelo ni diez réplicas independientes de mercado.
5. El filtro de acuerdo LLM empeora retrospectivamente la política de pesos medianos.
   DeepSeek avala 3.048 barras; Azure 3.455; coinciden en 1.842 de una unión de 4.661.
   La correlación entre pesos de los proveedores es 0,626. Comparten datos y prompt:
   **no son réplicas independientes**. Direccionalidad, exposición, timing y unidades
   ambiguas son explicaciones alternativas; no se demuestra una señal sistemáticamente invertida.
6. Los dos ledgers contienen 13.334 decisiones únicas, 226 sesiones completas y cero
   respuestas inválidas/no disponibles en el conjunto liquidado. Eso reproduce la tabla,
   pero **no recupera** dataset hash, respuesta/modelo servido, corte causal o request
   completo que no se guardaron originalmente. Los fallos reemplazados siguen archivados.

Los prompts originales eran numéricos, sin noticias. No permiten concluir sobre el valor
de FinMA-ES, sentimiento, noticias ni LLM financieros en general. Tampoco es científicamente
válido ordenar esta tabla como una ley «más sofisticación implica peor desempeño».

## Estadística, unidades y figuras

La media diaria neta pareada contra flat evita el Sharpe indefinido de flat. El reporte
publica IC percentiles y, separadamente, un contraste exploratorio de la media bajo H0
centrada, con bloques estacionarios de longitud media 5/10/15/20, B=10.000 y semilla
20260912. El ajuste Holm abarca la familia declarada completa de diez comparaciones.
Las colas percentiles descriptivas se etiquetan aparte: no se presentan como prueba exacta.
La mezcla de longitudes es una elección de diagnóstico, no una garantía universal de cobertura.

Siempre corto tiene p centrado ≈0,1794: es falso que todos los comparadores tengan el
mismo p≈0,0002. Con cero excedencias del estadístico bilateral centrado, se informa
`(0+1)/(10000+1)`, no p=0 ni una cota estadística sin explicar su construcción.
La corrección múltiple no convierte la selección histórica en confirmación prospectiva.

La incertidumbre entre ejecuciones RL debe acompañar al desempeño puntual; véase
[Agarwal et al., 2021](https://arxiv.org/abs/2108.13264). DSR requiere contabilidad
de selección adecuada, no un contador nuevo para cada reparación; véase
[Bailey y López de Prado, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
Aquí DSR queda **pendiente de reconciliación**, no aprobado con el N heredado de 115.
PBO se omite: las diez semillas/configuraciones en un bloque no reconstruyen el proceso
histórico de selección. No se añaden pruebas redundantes sólo para acumular siglas.

| Figura | Pregunta y precaución |
|---|---|
| [Capital](../../outputs/thesis-repair/research_grade_20260912_v5/01_capital.png) | ¿Cuánto perdió cada trayectoria? Composición neta e inicio visible. |
| [Drawdown](../../outputs/thesis-repair/research_grade_20260912_v5/02_drawdown.png) | ¿Cuánto cayó desde el máximo? Incluye la pérdida de la primera sesión. |
| [Sharpe móvil](../../outputs/thesis-repair/research_grade_20260912_v5/03_sharpe_movil.png) | Ventana retrospectiva de 60 sesiones; no prueba secuencial ni nuevo test. |
| [Costos](../../outputs/thesis-repair/research_grade_20260912_v5/04_costos.png) | Cero/×1/×2/×3 sobre pesos reales congelados; no exposición media sintética. |
| [Acciones por régimen](../../outputs/thesis-repair/research_grade_20260912_v5/05_acciones_regimen.png) | Cinco IDs descriptivos: quinto posterior recuperado como residuo; 19 sesiones reclasificadas. Figura v2/v3 retirada como clasificación válida. |
| [Semillas](../../outputs/thesis-repair/research_grade_20260912_v5/06_semillas.png) | Cinco ejecuciones por PPO. No se fabrican semillas LLM. |
| [Bruto y costos](../../outputs/thesis-repair/research_grade_20260912_v5/07_bruto_costos.png) | Sumas diarias explícitas, distintas de la rentabilidad compuesta. |
| [Calidad OHLC](../../outputs/thesis-repair/research_grade_20260912_v5/08_calidad_ohlc.png) | Cambio de representación por año. Estructura correcta no implica OHLC ejecutable. |

Las ocho imágenes salen de cálculos con Matplotlib; no se generaron números o curvas
mediante IA gráfica. Hay SVG para publicación y archivos de datos para comprobar cada panel.

## Implementación y criterios de cierre

| Bloque | Implementado | Aún no certificado |
|---|---|---|
| Preservación | Snapshot con hashes y rechazo de sobrescritura/rutas sensibles | Reconstrucción completa desde proveedores y entorno histórico |
| Reportes | Contabilidad independiente, drawdown, métricas, bootstrap, ocho figuras | Inferencia confirmatoria y mercado ejecutable |
| Identidad macro | Cuatro series comparadas nuevamente con capturas crudas: cero discrepancias; flags antiguos no bastan | Disponibilidad y vintages históricos; independencia de fuentes |
| Dataset | Identidad v3; carga histórica por SHA; export transaccional; rechazo de K mayor que slots | Resolver contrato HMM K=5 frente a cuatro slots antes del nuevo freeze e integración |
| Frecuencias | Unión forward por publicación y primera recepción, revisiones y frescura por serie | Integración con un ledger de publicaciones reales de todas las fuentes |
| PPO | Constructor único; S1–S4 5/5 train/unseen; 20 checkpoints reproducidos exactamente | Prueba de aprendizaje de mercado: la sanidad no demuestra alfa |
| LLM piloto | Tres variantes, dos proveedores, presupuesto durable, configuración explícita y sellos | Tarifas/deployment verificados, stream futuro, ledger de 20 sesiones |
| E2E | Separa reproducción retrospectiva, ingeniería y evidencia confirmatoria | Ningún PASS global ni publicación/promo automática |

La identidad del portable antiguo **debe dejar de coincidir** cuando cambia el código o el
contrato. No se altera su hash para hacerlo pasar. Se puede reproducir este capítulo desde
el snapshot; eso es distinto de entrenar de nuevo desde los crudos y obtener el mismo modelo.
El antiguo rebuild escribe el portable antes de exportar scaler/HMM. El nuevo
[exportador](../../scripts/data/export_research_bundle_v3.py) publica tres componentes en
un directorio exclusivo y el manifiesto al final; una interrupción conserva un paquete
incompleto que ningún consumidor puede aprobar. Su loader verifica el SHA esperado antes
de deserializar, identidad actual, escaladores, probabilidades, covarianzas y particiones
ordenadas/disjuntas. No retargetea los consumidores live ni parchea el histórico en sitio.

El intento real de exportar la caché local SHA
`937649f5fcf513668040d9557c9664487c5c3e5c3b398521ca55d0f34bc0040e`
se detuvo con `cache was not built under the current v3 input contract`, antes de crear
el destino. No se presenta un export de fixtures unitarias como dataset real aprobado.

### Incompatibilidad adicional encontrada al reconstruir de verdad

La suite global reconstruyó después la caché mutable mediante su fixture de integración.
Se exportó un directorio NUEVO, sin sustituir scaler, HMM ni portable históricos. Ese
[candidato](../../outputs/thesis-repair/research_dataset_bundle_v3_20260912/manifest.json)
tiene 488/226/570 sesiones, pero el HMM ajustado sobre desarrollo eligió **K=5** y el código
del dataset guarda `probs[:N_REGIMES]` con **cuatro slots**. La dimensión total sigue siendo 37;
un test que sólo cuente columnas no detectaría el cambio semántico.

La [medición preservada](../../outputs/thesis-repair/research_dataset_bundle_v3_20260912/INCOMPATIBLE_WITH_FROZEN_SCHEMA.json)
detecta masa posterior omitida mayor que 1e-6 en 103/488 sesiones de desarrollo,
80/226 de selección y 160/570 del hold-out; la suma registrada llega a cero.
Se inspeccionaron **contextos**, no retornos de nuevas estrategias. Omitir una categoría
podría ser una codificación de referencia válida si se declarara; **no es el contrato
congelado de cuatro posteriores completos** que utiliza este sistema.

El candidato permanece como diagnóstico no utilizable. Exportador y loader ahora exigen
`K <= slots`, probabilidades entre 0 y 1, suma `1 ± 1e-6` y padding cero para K menor.
El constructor `build_research_data` también rechaza K mayor que los slots inmediatamente
después del ajuste HMM, antes de construir spreads o features. No es posible evitar el
control del exportador usando la ruta de entrenamiento que reconstruye la caché.
La carga corriente lo rechaza por cambio del exportador; incluso desactivando sólo ese
chequeo de actualidad, lo rechaza por K=5. No se lo usó en PPO, LLM ni forward.
No se fuerza K=4, se amplía el esquema ni se renormaliza después de ver desempeño:
la representación debe acordarse y congelarse antes del siguiente experimento.
**Corrección de nuestra afirmación anterior:** el portable histórico v2 archivado también
declara **K=5**, no K=4. Su SHA es
`7f332df17a2492a0533f713bc143316fec9b9eadedcfbdad59d5eb0a36b7b0b5` y referencia
el HMM `0a4b9ec5bf0db632d75dce73f3dfe5f2dadf1389aaa5cd8070c49c62680f736e`.
El JSON global denominado `_v2` con K=4 no describe ese portable. La frase anterior
de este informe y de nuestros mensajes de coordinación queda expresamente retirada.

Los cuatro componentes archivados son finitos y están en [0,1]. Al sumarlos en float64,
78/226 sesiones tienen déficit mayor que 1e-6; el exceso máximo es sólo 2,98e-8.
La cuenta 80 reportada antes utilizaba acumulación float32 cerca del umbral: no se
confunden ambas precisiones. Cuatro coordenadas de un simplex de cinco estados **permiten
recuperar la quinta** como `p4 = 1 - sum(p0..3)`. No se demuestra pérdida inevitable de
información ni que PPO fuera incapaz de inferir shock. El defecto demostrado afecta al
contrato descrito y al argmax usado para las figuras.

Reconstruir el residuo, sin cambiar las cuatro coordenadas, cambia la clasificación de
19/226 sesiones: los conteos pasan de `[20,27,115,64]` a `[14,27,106,60,19]`.
La comprobación independiente con spreads por estado reproduce el spread guardado con
error máximo 1,15e-7 en selección. Es clasificación **descriptiva retrospectiva**, no
refit, reparación de los inputs de entrenamiento ni recomputación independiente del HMM.
Los pesos, costos cobrados y retornos anteriores no se modifican por reclasificar el gráfico.

No extendemos este hallazgo al EXP-TESIS-RL-01 original: el portable actual sin sufijo
tiene conteos 488/226/520, distintos de 499/234/584 originales, y metadatos añadidos después.
Se necesita el artefacto histórico o una cadena verificable ejecución→dataset/HMM→checkpoint
para afirmar qué K se utilizó en aquellas corridas.

Para macro, identidad numérica y disponibilidad son verificaciones diferentes. La
[documentación oficial de ALFRED](https://fred.stlouisfed.org/docs/api/fred/alfred.html)
distingue los valores publicados y revisados a lo largo del tiempo. Un CSV diario actual,
aunque coincida exactamente con FRED, no demuestra qué vintage estaba disponible en 2023.
La nueva unión forward utiliza `max(publication_at, first_seen_at) < cutoff`; nunca fabrica
ambas horas desde una fecha de observación. Una serie trimestral no hereda la frescura diaria.

La [reconciliación numérica nueva](../../outputs/thesis-repair/research_grade_macro_20260912/identity_live_network.json)
se ejecutó con descargas de referencia reales y luego volvió a analizar sus crudos. SHA256:
`400ba9a5833f4829f91a471c644e0409cd185910d9dc95ded31867c3762e716e`.

| Serie declarada | Observaciones comunes | Discrepancias | Máxima diferencia absoluta |
|---|---:|---:|---:|
| Investing DXY 942611 | 1.747 | 0 | 0 |
| FRED Brent DCOILBRENTEU | 9.973 | 0 | 0 |
| FRED DGS2 | 12.566 | 0 | 0 |
| BanRep IBR overnight | 4.558 | 0 | 0 |

Son conteos del archivo completo por serie, **no sesiones del experimento**. El primer
intento fallido por conexión quedó preservado; sólo el retry con capturas y recálculo
soporta este PASS. La compuerta estricta acepta esta evidencia y rechaza el certificado
antiguo de flags sin comparaciones. El [E2E v3](../../outputs/thesis-repair/research_grade_e2e_20260912_v3.json)
confirma macro numérico y sanidad, pero continúa bloqueando la identidad live antigua.

El control USD/COP diario conocido reportó 1.727 fechas comunes, mediana 0,07782 %,
P95 0,551359 % y máximo 5,192771 %. Es coherencia diaria histórica, no prueba de bid/ask,
independencia intradía, spreads realizables ni fills. Investing DXY contra un parquet
construido con Investing comprueba reproducción de la fuente declarada, no independencia.

## Auditoría adicional de versiones históricas: ALFRED

La continuación del goal implementó un [auditor de vintages](../../scripts/diagnostics/audit_research_macro_vintages.py)
y su [parser estricto](../../src/research/alfred_vintages.py), separados de los datos de
entrenamiento. La cohorte viene de las mismas 226 fechas del bundle v3; el macro y la regla
de frescura se leen de los objetos del snapshot, comprobando sus hashes. No se eligen
fechas por retornos, no se reajusta HMM y no se recalcula ninguna política.

Para cada sesión d se solicita a ALFRED un vintage fechado d−1, por separado para DGS2
y DCOILBRENTEU, con ventana de observación de 60 días naturales. Se comparan **fecha y
valor** del nivel DGS2 seleccionado por T−1 y de **ambos extremos** del log-retorno Brent.
Hay categorías separadas para coincidencia, revisión, ausencia, contexto local obsoleto,
observaciones fuera de la consulta y error de captura. La salida nunca interpreta que una
ausencia pruebe indisponibilidad universal a las 08:00.

La [documentación oficial de ALFRED](https://alfred.stlouisfed.org/help) explica que sus
fechas pueden provenir de la publicación original, del proveedor o de la incorporación a
FRED; las observaciones se incorporan habitualmente dentro de un día hábil. Por tanto un
vintage no acredita una recepción histórica de nuestro pipeline, ni una captura exactamente
a las 23:59. La fecha de observación tampoco equivale a la de publicación: los
[avisos oficiales H.15](https://www.federalreserve.gov/feeds/h15.html) documentan omisiones
y correcciones de tasas en agosto y septiembre de 2023.

Los CSV exigen la cabecera exacta `SERIE_YYYYMMDD`, fechas ordenadas sin duplicados,
valores finitos o faltantes explícitos y ninguna observación posterior al vintage. Se
archivan respuesta, URL pública sin credenciales, timestamps de captura actuales y fuente
del parser/runner. El replay offline exige el SHA externo del manifiesto y verifica
freeze, registros individuales y crudos antes de recalcular. No hay fallback a FRED actual.

**Alcance:** entradas macro directas del archivo conservado con el experimento. No
certifica por sí solo linaje macro→checkpoints, trayectoria de transformaciones HMM,
vintages DXY/IBR, disponibilidad intradía ni ingesta histórica. Las métricas y figuras
anteriores se conservan sin modificación. La identidad numérica aprobada por el E2E
anterior no se convierte en un certificado de disponibilidad.

Pruebas del nuevo auditor: **39 passed**; lint de los tres archivos nuevos pasó con
`--no-fix`. El [conjunto integrado](../../outputs/thesis-repair/research_pit_integrated_20260912.xml)
de investigación, nuevo auditor, front matter y layout dio **1.272 passed**, un warning
`asyncio_mode`. No es una nueva corrida de toda la regresión global: sus fallos abiertos
siguen declarados. El revisor adversarial corroboró guardas específicas contra hashes,
no certificó capturas antes de que existieran.

### Captura completa y replay independiente de la red

La [captura final](../../outputs/thesis-repair/alfred_selection_20260912/report.json)
terminó con **452/452 snapshots capturados y parseados, cero errores**. Su
[manifiesto](../../outputs/thesis-repair/alfred_selection_20260912/manifest.json) tiene SHA
`f1b903ae5e9716d444ea8c11e691987746e320bf5f9ba584996c6430be031c66`.
El [replay offline](../../outputs/thesis-repair/alfred_selection_20260912_replay/report.json)
reprodujo exactamente las 452 filas, sin llamadas de red. Manifiesto de replay SHA
`4464a3b652455f8c39a82721d9dae3c9974597e297c2c8b1b2b0fae120ac6d44`.

| Serie | Sesiones | Coincidencia de todos los períodos seleccionados con el vintage d−1 | Al menos un período seleccionado ausente |
|---|---:|---:|---:|
| DGS2 | 226 | 0 | 226 |
| DCOILBRENTEU | 226 | 0 | 226 |

Ejemplo verificable, sesión 2023-01-03: T−1 selecciona DGS2 del 30 de diciembre,
4,41 %, pero el vintage ALFRED del 2 de enero termina el 29, 4,34 %. Para Brent,
T−1 utiliza 29/30 de diciembre, 80,96/82,82 USD; el vintage termina el 27.
Esto demuestra una incompatibilidad con **ese vintage previo**, no disponibilidad
universal a las 08:00. No se afirma que los 226 valores carecieran de publicación en
todo proveedor, ni se elige un lag ad hoc para mejorar los retornos.

El gate de causalidad por fecha T−1 no basta como certificado point-in-time. Para cerrar
el próximo dataset se necesita una regla de disponibilidad de la fuente concreta con
vintages, retrasos y recepción documentados. No se inventan timestamps históricos.
Los dos crudos, parser, código capturador y respuestas quedan conservados por hash.

La corrección de figuras añadió 20 pruebas: el
[conjunto integrado final](../../outputs/thesis-repair/research_pit_regime_integrated_20260912.xml)
dio **1.292 passed**, un warning `asyncio_mode`; incluye las 39 de vintages y 20 de
regímenes. Ruff `--no-fix` pasó sobre los seis paths de este incremento. No se suman
repeticiones anteriores ni se presenta ese conjunto como toda la suite global.

El [E2E sobre el bundle vigente](../../outputs/thesis-repair/research_grade_e2e_20260912_v5.json)
confirma `retrospective_results_reproduced=true`, pero mantiene `engineering_ready=false`
y `scientific_closure_ready=false`. Su control general de vintages sigue PENDING: esta
auditoría parcial, acotada a dos series y al vintage previo, no lo convierte en PASS.
Los [gates de conocimiento](../../outputs/thesis-repair/research_pit_knowledge_20260912.xml)
dieron 1.100 passed; inventory, índices, enlaces y grafo pasaron. El ledger pasó su
consistencia estructural, **no** una reconciliación de los trials nuevos (USD/COP sigue 115).

Archivos de este incremento: nuevos `src/research/alfred_vintages.py`,
`scripts/diagnostics/audit_research_macro_vintages.py`,
`tests/regression/test_research_macro_vintages.py`,
`src/research/retrospective_regime.py` y `tests/regression/test_research_retrospective_regime.py`;
modificados el generador de figuras, este informe y los tres canales de coordinación
`LEASES`, `CODEX-STATUS`, `INBOX-CLAUDE`. Ningún archivo eliminado o renombrado.
La guía de visualización influyó en añadir denominadores, porcentajes, tramas y notas
legibles a la figura corregida; los datos y métricas se calcularon en código.
La revisión independiente final de v5 verificó los 25 hashes de artefactos, las
30/30 filas de acciones desde pesos/ledgers y la igualdad de 594 campos numéricos de
resultados frente a v3. No ejecutó nuevas estrategias ni certificó la disponibilidad
histórica. El chequeo final de enlaces resolvió los 914 enlaces internos, sin roturas.

**Siguiente decisión de diseño, no aplicada:** pre-registrar una representación que
conserve todos los estados elegidos por BIC exclusivamente en desarrollo, y congelar K,
orden, dimensión y scaler antes del próximo entrenamiento. Esto requiere aprobación;
la corrección descriptiva del gráfico no modifica por sí sola el contrato de entrenamiento.
Además siguen pendientes disponibilidad temporal de las cuatro series, tratamiento
homogéneo de OHLC/rangos, contabilidad de trials, costos ejecutables y piloto/forward.
No se reentrena sobre la representación rechazada para completar artificialmente el E2E.

### Integración C040: el E2E revalida los crudos

La [compuerta nueva](../../src/research/vintage_evidence_gate.py) está cableada en el
auditor E2E, con ACK real de Claude (CLD-745), sin auto-ACK. El
[resultado vigente v7](../../outputs/thesis-repair/research_grade_e2e_20260912_v7.json)
recalcula desde las capturas: 452 snapshots, 226 sesiones y 452 discrepancias con el
vintage previo. El diagnóstico incluye explícitamente `proves` y `does_not_prove`.
Sigue sin certificar disponibilidad a las 08:00, recepción, DXY/IBR, transformaciones
HMM, linaje de entrenamiento, precios ejecutables ni confirmación prospectiva.

El llamador debe proporcionar captura, bundle y **ambos hashes externos** juntos.
Se validan manifiesto, freeze, código congelado, registros, CSV, macro archivado,
contrato de frescura y cohorte completa. Un bundle de figuras más nuevo puede referir
la captura sólo si conserva exactamente cohorte e insumos; también se verifica el
bundle original. El resumen guardado se compara con el recalculado, excluyendo sólo
su hora de generación. No basta reescribir un booleano o rehashear un resumen falso.

La revisión independiente encontró y reprodujo dos huecos: una ventana menor que los
60 días declarados y JSON con claves duplicadas en los insumos. Ambos tests fallaron
antes del arreglo; ahora cada registro exige `start_date = vintage_date - 60 días`
y se rechazan JSON ambiguos/no finitos antes del loader histórico. Las fuentes ALFRED
congeladas no se editaron. El resultado guarda también SHA del verificador y del runner
E2E; así se distingue la compuerta reforzada de la versión preliminar v6.

Las [pruebas focales finales](../../outputs/thesis-repair/research_vintage_gate_20260912_final.xml)
dieron **53 passed** (27 compuerta, 8 interfaz E2E, 18 evidencia). Ruff de cuatro paths
pasó. El revisor volvió a ejecutar las 452 capturas reales sin red y corroboró que todas
usan la ventana declarada; no evaluó nuevas estrategias. El conjunto 1.325 pasó antes
de añadir los dos tests adversariales, por lo que no se lo usa como prueba de esos arreglos.
La [regresión integrada posterior](../../outputs/thesis-repair/research_evidence_gate_integrated_20260912_final.xml)
terminó con **1.327 passed**, cero fallos y un warning de configuración `asyncio_mode`.
Incluye ambos guards y los hashes del verificador/runner emitidos por la versión final;
no se suma a los grupos anteriores ni sustituye la suite global pendiente.
El E2E v7 tiene SHA256
`a7293a829498b158f3e34c335a6f5333e0d5eba2d1a05814118cc0cee2b8e977`.
El [gate de conocimiento posterior](../../outputs/thesis-repair/research_evidence_knowledge_20260912_final.xml)
dio **1.100 passed**, con el mismo warning; se solapa con las 1.327, no se suman.
Inventory, índices, enlaces, grafo y contabilidad estructural del ledger pasaron.
El monitor de mensajes terminó normalmente; no queda vigilancia perpetua implícita.

Nuevos archivos: `src/research/vintage_evidence_gate.py` y
`tests/regression/test_research_vintage_evidence_gate.py`. Modificados:
`scripts/diagnostics/audit_thesis_e2e_status.py`, `tests/regression/test_thesis_e2e_status.py`,
este informe y coordinación (`CONTRACTS`, `LEASES`, `CODEX-STATUS`, `INBOX-CLAUDE`).
Capturas, modelos, tablas y figuras v5 permanecen intactos; no se alteró el registro.

## Corrección del join prospectivo entre frecuencias (C041)

El alcance C040 recibió revisión por hash/contenido de Claude en CLD-746. Ese ACK no
se presenta como una nueva reejecución de todas las pruebas por su parte.

La [unión por publicación](../../src/research/publication_join_v2.py) conserva dos tiempos:
publicación declarada por la fuente y primera recepción registrada. Un dato sólo entra
si ambos son estrictamente anteriores al corte. Entre revisiones ya conocidas selecciona
la última publicación del período, no simplemente el paquete que llegó más recientemente.

Se reprodujeron defectos reales del componente, **no se demostró que afectaran los
resultados históricos**: un denominador de 2020 con numerador de 2026 figuraba como
disponible; se aceptaban booleanos/valores flotantes como conteos de observaciones;
un período posterior a la fecha COT podía tener edad −1 y entrar como válido; el
cociente de dos niveles finitos podía desbordar antes de aplicar el logaritmo.
También se aceptaban fechas de texto ambiguas y faltaban columnas cuando no había contexto.

La corrección, con ACK de alcance de Claude CLD-747, exige identificadores y políticas
explícitos, rechaza booleanos como niveles y fechas sin formato ISO inequívoco, y mantiene
un esquema de salida estable aun vacío. Usa diferencia de logaritmos con resultado finito.
La frescura ya declarada se aplica a **los dos operandos**, sin elegir otro umbral.
Si el anterior es obsoleto, devuelve `STALE_PREVIOUS_OBSERVATION_PERIOD` sin nivel ni
retorno utilizable; conserva fechas, hashes y edades para diagnosticar la exclusión.
Un período futuro en el calendario COT también produce exclusión, no un valor operativo.

Las [pruebas antes de corregir](../../outputs/thesis-repair/research_publication_join_red_20260912.xml)
dieron **20 failed, 7 passed**. Tras el arreglo, las 27 pasaron; la
[cobertura ampliada](../../outputs/thesis-repair/research_publication_join_multifrequency_20260912.xml)
dio **40 passed**, incluyendo mezcla diaria/semanal/mensual/trimestral, unidades distintas,
fronteras de frescura, faltantes, llegada tardía, revisiones fuera de orden y diez
reordenaciones que conservan paridad entre procesamiento completo y por prefijos.
Son fixtures de software: **no son observaciones de mercado ni evidencia de rentabilidad**.

Una segunda revisión adversarial encontró cuatro escapes adicionales: unidades nulas
con dtype `string`, niveles nulos con `Float64`/`Int64`, duraciones convertidas a precios
y un entero aceptado como hash. Los [tests de nulos y duraciones](../../outputs/thesis-repair/research_publication_join_nullable_red_20260912.xml)
fallaron en **7 casos** y pasaron en 2; el [test de hash numérico](../../outputs/thesis-repair/research_publication_join_hash_red_20260912.xml)
también falló antes de corregir. Se exigen ausencia explícita de nulos, tipos no temporales
y hash de texto, sin depender del comportamiento de `.all()` que omite `pd.NA`.
Después, la [suite ampliada](../../outputs/thesis-repair/research_publication_join_hardened_20260912.xml)
dio **50 passed**. Ruff pasó sobre ambos archivos; no se formatearon fuentes congeladas.
La [integración final](../../outputs/thesis-repair/research_publication_join_integrated_20260912_final.xml)
dio **194 passed**, con un warning `asyncio_mode`; el
[gate de conocimiento](../../outputs/thesis-repair/research_publication_knowledge_20260912.xml)
dio **1.100 passed**, sin sumar grupos solapados. La suite global histórica no se reejecutó.
Revisión independiente: 50 casos parametrizados ejecutados directamente en memoria,
incluidos los cuatro escapes, y DataFrames de entrada sin mutación. SHA del módulo
`4281bc3a6e888942bfad847a557f37756bdfb70263660b246656ff8ad44af844`;
SHA de tests `84abee3b7777f5c10dbd96430264ed05db4839e7db8d33f6da4ac4f9a7f67caf`.

Límites conservados: el join no demuestra autenticidad del hash recibido, no captura por
sí solo publicaciones reales y no valida que los dos períodos sean consecutivos según
la frecuencia de la fuente. El calendario de cada serie y la separación admisible entre
períodos deben declararse antes del freeze; no se infieren mirando datos o PnL. Las filas
futuras malformadas invalidan el ledger completo conforme al rechazo global existente;
la invariancia por prefijos se afirma únicamente para entradas válidas. No se introdujo
cuarentena silenciosa, se interpolaron valores ni se cambió a publicaciones originales
en lugar de revisiones conocidas al corte.

Este componente sigue **sin productor prospectivo provisionado ni integración certificada
con el dataset/modelo futuro**. Sus tests no cierran el gate de disponibilidad histórica.
Se modificaron sólo el módulo, su test, este informe y los canales de coordinación;
no se alteraron las cuatro series macro, HMM, modelos, cifras, figuras ni registro de trials.
El monitor de mensajes de este bloque se detuvo deliberadamente verificando PID e instante
de inicio; se limpió su archivo PID temporal, sin eliminar datos ni logs. No queda vigilancia
en segundo plano implícita tras la entrega.

### Comprobación de la sensibilidad OHLC enviada por Claude

Se inspeccionaron el [runner de Claude](../../scripts/diagnostics/ohlc_representation_counterfactual.py)
y su [artefacto](../../outputs/thesis-repair/ohlc_representation_counterfactual.json), SHA
`9a7e553cec2cae39a1b5a4eee860f5b83a9c5b6367be7eb011ee1ab450ae3999` y
`d380bd7b058c5d7a460ba8950cc3ca65c4b8aa44d6486fcb4874ba6df7a2469f` respectivamente.
La raíz verificó el seed y recalculó en memoria el agregado diario, true range y media
móvil de 14 días con mínimo 5, sin importar ese runner ni cargar HMM o macro. Coincidieron
los **42 campos numéricos**, diferencia máxima 0,0. Es una comprobación separada de los
números, no un replay independiente archivado ni otro experimento de rentabilidad.

El ratio de medias de `atr_norm` original/aplanado menos uno es 18,5299 % en 2026;
el ratio de medianas es 17,8798 %. La diferencia nula en 2021 comprueba que aplanar
barras ya planas no las cambia. **No identifica la fracción causal atribuible al proveedor**,
ni descarta por sí sola influencia de outliers, ni demuestra un efecto sobre el posterior.
El diagnóstico agrega 260 fechas en 2023, no las 226 sesiones de la tabla económica.
La cadencia y la representación futura siguen siendo decisiones pre-registradas pendientes;
no se modifican a partir de este contraste. Los hashes, fechas exactas y rechazo de overwrite
del runner se solicitaron a su dueño en CXD-861; no se editó código ajeno.

## Sanidad: aislada y sin cifras de mercado simuladas

Se congeló una nueva batería S1–S4, cinco semillas, 100.000 pasos solicitados por ejecución,
con la receta `flat_init_no_turn`; no se busca otra si falla. Entrenamiento: 500 sesiones
sintéticas; evaluación: 100 sesiones de entrenamiento y 100 nuevas generadas con semillas
disjuntas. La regla exige al menos cuatro de cinco semillas aprobadas **en ambos conjuntos**.
S2 exige costos exactamente cero y al menos 70 % del oráculo en el conjunto evaluado.

El control se ejecuta bajo las versiones realmente instaladas y conserva código, receta,
costos, datos, pasos efectivos, checkpoint y normalizador por hash. No demuestra retrospectivamente
que el entrenamiento antiguo usara la misma versión de las librerías. Un PASS en ruido
significa «cerca de flat dentro de las tolerancias congeladas», **no cero pérdidas** ni
óptimo exacto. Las pérdidas residuales se publican. El agregado no puede aprobarse editando
un booleano: la compuerta verifica métricas, oráculo, datos, modelo y manifiestos.

La [batería completa](../../outputs/thesis-repair/sanity_research_grade_20260912/protocol.json)
terminó con **5/5 semillas aprobadas en cada S1–S4**, en ambos conjuntos. Cada corrida
ejecutó 102.400 pasos efectivos. En S2, el rendimiento unseen fue 98,999 %–99,343 % del
oráculo. S1 mantuvo exposición media absoluta de 0,076 %–2,619 % y S4 de 0,305 %–1,941 %:
queda explícito que la abstención no fue perfecta.

El [replay independiente](../../outputs/thesis-repair/sanity_research_grade_20260912/policy_replay.json)
recargó los 20 checkpoints y sus normalizadores: **2.000 sesiones nuevas, 118.000 decisiones,
diferencia máxima diaria 0,0**. No volvió a entrenar. El bucle evaluador es distinto, pero
comparte el motor contable; no se presenta como réplica completamente independiente del software.
Hashes: protocolo `33873c610252bb674879bb89e89272875d4d67b24062f8b9d4e62edac8fc21e9`,
replay `9332ecbf2ca0c2ae6e264bc3c66fb091eea2427a7cc6d75d8b1a7a558ca1186a`.

## Piloto LLM aprobado, no ejecutado retrospectivamente

Presupuesto adicional autorizado: **USD 100 conjunto**, no por proveedor. DeepSeek es
principal; Azure es robustez. L0 conserva el contexto numérico anterior; L1 añade el diccionario
de representación, escalas y unidades sin cambiar valores; L2 añade estado propio y costo.
L2 es una intervención conjunta estado+costo: **no identifica por separado sus dos efectos**.
Esa limitación debe permanecer en el pre-registro; separarlos requeriría otro contraste aprobado.
Comparaciones L1−L0 y L2−L1, mismas primeras 20 sesiones futuras del calendario congelado.
No se selecciona proveedor ni variante por resultados y no se llena el cupo con fechas pasadas.

La memoria original incluye otras hipótesis sobre noticias/FinMA-ES: quedan pendientes,
no sustituidas silenciosamente por este experimento numérico. El piloto es descriptivo;
no se confunde con el juez forward confirmatorio de tamaño/potencia pre-registrados.
Las decisiones no son fills: falta una liquidación con precio realmente posterior a recepción,
latencia, política de fallos, costes observables y cierre terminal antes de reportar dinero realizado.

El [piloto implementado](../../scripts/analysis/run_thesis_llm_pilot_v2.py) conserva la
aprobación y distingue la provisión pendiente. Requiere tarifas con evidencia archivada,
deployment/modelo servido esperado, calendario, diccionario ligado al scaler y snapshots
del productor prospectivo. Separa cierre de vela, recepción y creación del contexto: una
vela recibida dos segundos después del cierre es admisible antes del deadline; una vela
futura o recibida tarde no lo es. Request, contexto y estado quedan archivados antes de la API.
Cambios de modelo, uso inválido o exceso de presupuesto activan un bloqueo durable conjunto.
Sus primeras **41 pruebas** pasaron sin llamadas reales. C042 amplía la cobertura a
**115 pruebas**, descritas abajo. No se han gastado créditos del piloto.

### C042: admisión monetaria corregida, provisión aún pendiente

El 13 de septiembre se corrigió un defecto previo al primer gasto: la validación de
tarifas aceptaba un dominio oficial de **otro proveedor**; la propia fixture Azure
citaba DeepSeek. Ahora se exige proveedor reconocido y evidencia bajo su dominio:
DeepSeek para DeepSeek y Microsoft para Azure. OpenAI Docs se usó para contrastar la
documentación de `gpt-4o-mini`, **no para asignar su precio de OpenAI a Azure**.

También se rechazan límites fraccionarios o booleanos, metadatos de modelo/API vacíos
o no textuales, hashes numéricos y un freeze sin zona horaria. La reserva se calcula
con aritmética racional exacta y techo a micro-USD: evita que la precisión ambiental
de `Decimal` redondee hacia abajo antes del techo. No cambia presupuesto, cohortes,
semillas, muestreo, prompts, modelos ni ninguna cifra histórica. El caso de control
de tarifa ficticia conserva exactamente `4_686_960` micro-USD de reserva **simulada
para el test**, no gasto ni cotización de los proveedores.

Evidencia de software, con datos controlados exclusivamente en tests:

- [TDD previo](../../outputs/thesis-repair/research_pilot_admission_red_20260913.xml):
  **64 fallos y 45 pases** antes del arreglo; entre ellos proveedor equivocado,
  coerción de tipos, tiempo ingenuo y techo decimal insuficiente.
- [Primera corrección](../../outputs/thesis-repair/research_pilot_admission_green_20260913.xml):
  **109 pases**.
- [Verificación ampliada](../../outputs/thesis-repair/research_pilot_admission_final_20260913.xml):
  **115 pases**, un aviso `asyncio_mode` desconocido; Ruff de los dos archivos PASS.
  Estos conteos se solapan: no se suman como ejecuciones independientes.
- [Integración](../../outputs/thesis-repair/research_pilot_integrated_20260913.xml):
  **268 pases**, incluye los 115 del piloto, join causal, macros, export y auditor E2E.
- [Conocimiento](../../outputs/thesis-repair/research_pilot_knowledge_20260913.xml):
  **1.100 pases**; inventario, índices, enlaces y grafo también pasan sus CLI.
  No se reejecutó la suite global; sus 8 fallos y 6 errores anteriores no se borran
  por estos verdes focalizados.

Fuente: `llm_experiment_v2.py`, SHA
`ca9c5368820d5fc7a69189612fb22247e8562f31fe05618e781fafd9290131c6`.
Tests: `test_research_grade_llm_pilot.py`, SHA
`24040e6db2cb6e885f0c08fba99982d3494cb20d884eb4196223b7bfd5e29d9c`.
ACK de alcance Claude CLD-751 anterior a la modificación del módulo.
El reviewer independiente ejecutó **74 casos de admisión en memoria**, todos PASS,
sobre esos mismos hashes, sin archivos, ledger ni API. Es una revisión acotada;
no se atribuye al reviewer la ejecución completa de pytest ni un aval de mercado.

**Límite que sigue abierto:** un hash acredita los bytes archivados, no que el
importe elegido corresponda a esos bytes, al modelo servido, región o modalidad.
El validador no interpreta ni autentica semánticamente una página de precios.
La revisión de esa correspondencia sigue pendiente y no se declara solucionada
con C042. Tampoco una cadena `expected_served_model` demuestra por sí sola que
el proveedor mantenga inmutables los pesos detrás de ese identificador.

La comprobación offline del CLI con el config vigente retorna `BLOCKED`, exit **2**:
`calendar, scale dictionary, current pricing and immutable freeze required; no API called`.
No hay ledger SQLite nuevo del piloto. Se preserva la configuración pendiente;
no se incorporan las tarifas ficticias de los tests ni se inventa un deployment.

La [revalidación E2E offline del 13 de septiembre](../../outputs/thesis-repair/research_grade_e2e_20260913.json)
vuelve a comprobar el bundle v5 y la captura ALFRED con sus hashes externos.
Resultado: `retrospective_results_reproduced=true`, `engineering_ready=false`,
`scientific_closure_ready=false`. El exit 0 significa **informe generado**, no
experimento aprobado. Las figuras y las liquidaciones históricas no se regeneraron
ni modificaron; las pruebas de C042 no convierten esos resultados en confirmatorios.

### Identidad de los proveedores: no migrar modelos por alias

La [página oficial de DeepSeek abierta durante esta revisión](https://api-docs.deepseek.com/quick_start/pricing/)
mostró `deepseek-flash` asociado a `DeepSeek-V4.1-Flash` y `deepseek-v4-pro` a
`DeepSeek-V4-Pro-0813`; no incluyó `deepseek-chat` en esa tabla. Un resultado del
buscador de la misma fuente describía un alias anterior y una deprecación en julio:
**no se lo acepta como descripción vigente**. La consulta actual tampoco demuestra
qué modelo respondió a las llamadas históricas. No se ha llamado a la API para
resolverlo ni se ha sustituido V3 por otra familia.

La [documentación OpenAI de GPT-4o Mini](https://developers.openai.com/api/docs/models/gpt-4o-mini)
enumera el snapshot `gpt-4o-mini-2024-07-18`. Eso **no certifica** el deployment del
operador en Azure ni su facturación. Consultar documentación oficial no equivale
a provisionar o congelar el experimento. Antes de nuevas llamadas se necesita:

1. Decisión explícita sobre modelo/versión del futuro piloto, separada de la identidad
   no recuperada de los brazos históricos; no elegir según el mejor resultado previo.
2. Metadatos no secretos de Azure: origen del endpoint, deployment, versión API y
   correspondencia con el modelo servido; evidencia de su tarifa aplicable.
3. Fuentes de precios archivadas y revisadas para ambos proveedores, calendario y
   diccionario del dataset aprobado; después freeze, no antes.
4. Productor causal de contextos y liquidación posterior a recepción. Un ledger de
   decisiones completo no acredita fills ni rentabilidad ejecutable.

Las API keys no sustituyen ninguno de esos requisitos. `.env` y los secretos no se
leyeron. Las claves se consumen sólo desde el entorno del transporte autorizado.

## HMM histórico: paridad y sensibilidad de representación C043

El [diagnóstico v3](../../outputs/thesis-repair/hmm_representation_20260913/diagnostic_v3.json)
SHA `c50bf861614c0ba26af29482ab37efe72a5d08cb235ac6e89d8c63077e884930` reproduce las cuatro coordenadas de régimen float32
guardadas en las **226 sesiones de selección**. Error absoluto máximo
`2,973822776919377e-8`, frente a tolerancia declarada `1e-6`; ninguna fecha
excluida y ninguna coordenada fuera de tolerancia. Esto verifica el vector
archivado de cuatro coordenadas, **no un posterior completo K5 que se hubiera
guardado**. El cálculo nuevo sí utiliza los cinco estados del HMM congelado.

El [wrapper exclusivamente histórico](../../src/research/historical_hmm_audit.py)
verifica el SHA externo del snapshot, sus objetos, el enlace SHA portable→modelo,
la identidad declarada y metadata, las covarianzas full y el código de inferencia.
No cambia el loader actual ni su candado de identidad. El portable antiguo no
contiene `identity_manifest`: la concordancia comprobada no reconstruye todo
el linaje del entrenamiento ni acredita el vínculo a checkpoints PPO.

Las máscaras se reconstruyen desde sus exclusiones y se comprueban contra hashes:
1.361 sesiones válidas, de las cuales 1.261 son válidas para entrenamiento.
Los outliers sospechosos se conservan en la primera máscara, como en el contrato
histórico. No se usa el calendario actual para reinterpretarlas. El filtro empieza
en la primera observación completa, **2019-12-24**, no al comenzar el fit;
usa 774 observaciones completas hasta 2023-12-29. Se conserva la agregación sobre
sesiones válidas, el descarte conjunto de nueve features, el calentamiento de
60 observaciones y el desplazamiento sobre el índice limpio. Por eso la sesión
2023-01-03 recibe el posterior de 2022-12-29, no el del día calendario anterior.

Sólo después de pasar la paridad se ejecuta la intervención diagnóstica
`O=H=L=C`: cierres idénticos, mismo HMM y misma historia limpia completa.
No se entrenan modelos, calculan nuevos retornos ni eligen hiperparámetros.

| Medición pareada en selección | Resultado |
|---|---:|
| Sesiones comparadas | 226 / 226 |
| Cambios del estado de máxima probabilidad | 5 (2,21 %) |
| Distancia de variación total media | 0,0146174 |
| Mediana / percentil 95 | 0 / 0,00299355 |
| Distancia máxima | 0,8994581 |
| Fallbacks numéricos / covarianzas con jitter | 0 / 0 |

Variación total es `sum(abs(p_original-p_aplanado))/2`, entre 0 y 1;
no es retorno ni porcentaje de pérdidas explicado. Cambia el estado dominante
el 27-jun, 29–31-ago y 11-dic de 2023. Los identificadores de estado son únicos:
las tres etiquetas «intermedio» no se confunden entre sí. La mediana cero y el
máximo alto describen una sensibilidad heterogénea; **no procede resumirla como
efecto uniforme ni atribuir causalmente una fracción al proveedor**.

La cohorte tiene 13.560 barras y **89,60 %** ya eran OHLC planas. No se extrapola
esta sensibilidad a 2026 ni al forward, cuya representación puede ser distinta:
su efecto **no se midió en C043**. Tampoco se afirma que no existan coordenadas
históricas de otros bloques; simplemente no se auditaron aquí. Tres cambios en
fechas consecutivas son una descripción temporal, no una demostración de mecanismo
causal o de ausencia de ruido. Claude CLD-756 recomputó los estadísticos desde
las filas del artefacto, no desde precios/source; su revisión tiene ese alcance.

Un revisor independiente reprodujo los posteriores con una recursión forward
propia en dominio logarítmico; discrepancia máxima menor de `2,8e-15` frente
al JSON inicial. Fueron probes en memoria, no otra suite pytest. Identificó tres
guardas faltantes, corregidas: enlace SHA portable→modelo, exigencia de covarianza
full y serialización como null del error de una fila no finita. La suite focal
terminó con **44 passed**; incluye enumeración exhaustiva de caminos latentes
en una fixture pequeña, invariancia por extensión futura, objetos corruptos,
cohortes desalineadas y bloqueo del contrafactual si falla el baseline.
Las fixtures numéricas son tests de software; los 226 resultados del diagnóstico
proceden de precios históricos reales archivados, transformados explícitamente.

La [integración final](../../outputs/thesis-repair/hmm_integrated_final_20260913.xml)
terminó en **312 passed**, incluyendo estas 44 pruebas; no se suman como grupos
independientes. La [comprobación de conocimiento](../../outputs/thesis-repair/hmm_knowledge_final_20260913.xml)
registró **1.102 passed**. Ruff pasó sobre helper, runner y test nuevos; inventario,
índices, enlaces y grafo pasaron. La suite global previamente documentada con
8 fallos y 6 errores **no se reejecutó** ni se declara reparada por C043.

Se conservan las advertencias de autocorrelación indefinida en sesiones constantes;
esas observaciones aparecen en las fechas descartadas del JSON. No se rellenan
con ceros para conseguir paridad. La prueba comprueba el cálculo histórico, no
su disponibilidad point-in-time. Los joins macro históricos se preservan aquí
y **no se sustituyen** por el nuevo helper de publicación: hacerlo mezclaría
la reparación prospectiva con la reproducción.

Reproducción, con salida nueva:

```text
python scripts/diagnostics/audit_hmm_representation.py --snapshot outputs/thesis-repair/archive/pre_research_grade_20260912_1934/manifest.json --expected-snapshot-sha 3caf361570a2ed104bd8ff118ce55989eb9b2b0ccb3105877d056ba987194ea1 --output outputs/thesis-repair/hmm_representation_NEW/diagnostic.json
```

Este cierre diagnóstico **no aprueba K/esquema futuro**, no promueve un modelo
ni levanta los bloqueos PIT, costos ejecutables, trials o forward. Tampoco cambia
las figuras v5 ni las conclusiones económicas retrospectivas ya publicadas.

## HMM en 2026: diagnóstico separado C044 y figura reproducible

C044 examina **todas las 150 sesiones de 2026 del hold-out archivado**,
2026-01-02 a 2026-08-24, sin elegir fechas por su resultado. El
[reporte v4](../../outputs/thesis-repair/hmm_2026_20260913/v4/diagnostic.json)
SHA `02a50b59a8952abc182bb38c9be37f34d82349b304a2bb113a37567d82d0e944` y el
[manifiesto](../../outputs/thesis-repair/hmm_2026_20260913/v4/manifest.json)
SHA `3f27066896faf295605beb4fe3885b3564fe144b92732b5f4f486b90fde6152c` son nuevos; no sobrescriben C043, figuras v5 ni modelos.

La paridad pasa en **150/150 sesiones**: máximo error absoluto
`2,966715006991194e-8` sobre las cuatro coordenadas float32 archivadas,
con tolerancia predeclarada `1e-6`. El historial limpio contiene **1.344**
observaciones desde 2019-12-24; no se reinicia el filtro en 2026. La primera sesión
recibe el posterior de 2025-12-30. Se preservan máscara, orden de features, parámetros,
calentamiento y desplazamiento; no hubo jitter ni fallback. Como en C043, reproducir
fechas anteriores no acredita disponibilidad oficial point-in-time.

Después de esa paridad se aplica la misma intervención sobre **todo el prefijo**:
aplanar OHL al cierre de cada barra, conservando cierres y parámetros K5. El resultado
usa la misma historia y todas las fechas, sin intersección silenciosa:

| Medición descriptiva | Selección 2023 | Cohorte 2026 |
|---|---:|---:|
| Sesiones / barras | 226 / 13.560 | 150 / 9.000 |
| OHLC originalmente planas | 89,60 % | 0 % |
| Cambios de estado dominante | 5 | 2 |
| TV media | 0,0146174 | 0,0219426 |
| TV mediana | 0 | 0,00000127448 |
| TV percentil 95 | 0,00299355 | 0,12358086 |
| TV máxima | 0,89945812 | 0,70009694 |

Cambian el estado dominante el **14 de enero y el 13 de agosto de 2026**.
No se observó una alteración generalizada de esa categoría bajo esta intervención
del modelo congelado; sí existe una cola de cambios de probabilidad apreciables.
**No basta contar argmax para afirmar robustez**: la distribución completa aporta
información distinta. Tampoco prueba calidad correcta del proveedor, que ATR sea
prescindible, que otro ajuste del HMM fuera insensible ni que éste explique las
pérdidas de PPO. No se evaluaron acciones o retornos de estrategias.

La [figura PNG](../../outputs/thesis-repair/hmm_2026_20260913/v4/sensitivity.png),
su [versión vectorial SVG](../../outputs/thesis-repair/hmm_2026_20260913/v4/sensitivity.svg),
las [filas CSV](../../outputs/thesis-repair/hmm_2026_20260913/v4/posterior_rows.csv)
y la [descripción textual](../../outputs/thesis-repair/hmm_2026_20260913/v4/sensitivity.txt)
salen del mismo reporte validado. El panel A compara distribuciones empíricas completas
y el B muestra cada sesión de 2026; líneas distintas y cuadrados evitan depender
sólo del color. Se aplicó la guía de visualización `ui-ux-pro-max`, limitada a
legibilidad y representación, sin escoger datos o umbrales. La inspección visual
detectó un pie solapado en v1: corregido en v2. La revisión CLD-761 motivó
un recuadro ampliado para TV entre 0 y 0,05, conservando el eje completo y
exactamente las mismas observaciones, sin filtrar ni renormalizar la CDF.
Se conservan v1-v3; v4 añade esa ampliación y una etiqueta compatible con lint.
La ampliación mejora legibilidad, no constituye un contraste estadístico entre años.

**El hold-out se ha examinado.** Se declaran `holdout_inputs_read=true`,
`market_returns_derived=true`, `strategy_returns_evaluated=false` y
`claim_preserves_unseen_holdout=false`. Retornos y volatilidades forman parte de
las entradas del HMM; además, ver distribuciones OOS puede motivar cambios de diseño
sin calcular un score económico. Por eso no procede la exención «no se gasta el
hold-out» propuesta y retirada en CLD-759/760. Las decisiones motivadas aquí necesitan
un período posterior como juez. Este diagnóstico no crea una política FT/AT nueva
ni certifica una exención universal del registro de ensayos.

Reproducción independiente del reviewer: forward propio en dominio logarítmico,
discrepancia frente a v1 inferior a `1,8e-15`, sin escrituras ni pytest. La raíz
ejecutó primero **63 pruebas focales aprobadas**, de las cuales 19 son C044 y 44 C043.
La prueba adicional del recuadro falló antes de implementarlo y comprueba
que éste conserva todas las coordenadas de la distribución completa.
La integración ampliada a **357 pruebas pasó**, incluidos los manifiestos de estrategia.
La primera integración tuvo 355 PASS y 1 FAIL por `git ls-files` (exit 128,
propietario distinto en el sandbox); la repetición con `safe.directory` limitado
al proceso dio 356 PASS, antes de añadir la prueba del recuadro. No se cambió
la configuración global de Git ni se debilitaron los tests.
Nueve mutaciones demostraron primero que el publicador podía aceptar datos
incoherentes; ahora recalcula TV y argmax, comprueba simplex, fechas, resúmenes,
alineación, fallbacks y que no cambie la referencia. Si falla la paridad sólo se
publica el diagnóstico fallido, sin figura. Un directorio existente nunca se
sobrescribe. Se conservan las advertencias históricas de autocorrelación indefinida.

Verificación final v4: **1.166 PASS**, desglosados en 1.102 de conocimiento,
layout, espejos y registro, más 64 focales HMM (20 C044 y 44 C043).
Estos grupos se solapan con la integración de 357; no son 1.523 pruebas distintas.
Ruff de los dos archivos C044, inventario, índices, enlaces (947) y grafo pasan.
Se verificaron los cinco hashes del manifiesto v4 y la igualdad exacta de cohorte,
entradas, paridad y resultados entre v1-v4; los bytes del runner/helper/reporte C043
permanecen intactos. CLD-762 verifica hashes, cifras e imagen v4, no los tests.
No se reejecutó la suite global ni se repararon sus fallos previos. La comprobación
Git global falló al recorrer paths ajenos por permisos/LFS; no acredita un árbol
global limpio. No hay commit ni aprobación bilateral contra un commit sellado.

```text
python scripts/diagnostics/audit_hmm_2026_representation.py --snapshot outputs/thesis-repair/archive/pre_research_grade_20260912_1934/manifest.json --expected-snapshot-sha 3caf361570a2ed104bd8ff118ce55989eb9b2b0ccb3105877d056ba987194ea1 --selection-report outputs/thesis-repair/hmm_representation_20260913/diagnostic_v3.json --expected-selection-sha c50bf861614c0ba26af29482ab37efe72a5d08cb235ac6e89d8c63077e884930 --output outputs/thesis-repair/hmm_2026_REPRO_NEW
```

El ejecutable [C044](../../scripts/diagnostics/audit_hmm_2026_representation.py) y sus
[pruebas](../../tests/regression/test_historical_hmm_2026.py) son nuevos y están
separados del runner C043, cuyos bytes se comprueban contra el reporte de referencia.
El avance cierra esta medición pendiente de representación, **no** la elección
del esquema futuro ni los gates de fuentes PIT, costos, trials, LLM o forward.

## C045: contrato numérico en ambos constructores live

Se corrigió [live_spec](../../src/research/live_spec.py), conservando sus cambios
preexistentes. El productor del dataset ya rechazaba un HMM con más de cuatro estados,
pero los consumidores de sesión completa y parcial recortaban el posterior a cuatro
componentes. Un modelo explícito con cinco probabilidades válidas podía atravesarlos.
Las identidades congeladas obsoletas bloqueaban la carga por defecto, no ese caso.

Ahora ambas rutas rechazan K no entero, booleano, fuera de `[1, N_REGIMES]`, vectores de
forma/tipo incorrectos, probabilidades no finitas o fuera de `[0, 1]`, y masa distinta de
uno con tolerancia absoluta `1e-9`, relativa cero. No recortan, rellenan datos desconocidos,
normalizan ni ajustan el HMM. El padding de estados no usados para K menor que cuatro
se mantiene; los valores válidos conservan exactamente el contexto float32 y el spread.
`session_spread`, las clases serializadas y los exportadores no cambiaron.

La [regresión nueva](../../tests/regression/test_live_regime_contract.py) usa fixtures
aisladas de contrato: **no son precios simulados presentados como resultados de tesis**.
Antes del arreglo, la [tanda adversarial](../../outputs/thesis-repair/live_contract_adversarial_red_20260913.xml)
dio **62 failed, 12 passed**. Tras el primer guard, 74 passed. Una segunda revisión
encontró que `np.asarray` convertía silenciosamente `[True, 0, 0, 0]` y descartaba una
máscara de valores desconocidos. Los [cuatro casos adicionales](../../outputs/thesis-repair/live_contract_coercion_red_20260913.xml)
fallaron antes de validar la salida original del modelo. El
[resultado focal final](../../outputs/thesis-repair/live_contract_final_20260913.xml)
es **78 passed**; la [integración acotada](../../outputs/thesis-repair/live_contract_integration_20260913.xml)
es **448 passed**, incluyendo esos 78, con nueve warnings registrados. No se suman como
526 pruebas independientes. La protección contra booleanos mixtos se verifica antes de
la conversión de listas/tuplas; no permite recuperar tipos ya perdidos por una conversión
realizada aguas arriba.

Hash SHA256 del source al cierre C045: `8494bb339563cf89a3053b7643b932ac24e4f1d4714025bfbe7931cc52b35211`.
Hash del test al cierre C045: `2b6f2fa4504280557b947e83af74fdfa462b165dfcf53392abccbf04a446ff8e`.
Ruff del test pasa. El source conserva **seis avisos preexistentes**, reproducidos
contra el objeto exacto del snapshot `f7edb1bd08b0e32dd3408bcc7e22f70da7a98fb1987146ad4dfea1d871e77b63`:
dos RUF002, I001, F401, RUF100 y UP037; no se afirma lint global verde. El primer intento
por stdin alteró los caracteres Unicode por la codificación de PowerShell y mostró cuatro;
se repitió con salida UTF-8 y se verificaron los seis originales.

La revisión independiente final verificó los dos escapes, cinco vectores válidos,
ausencia de mutación y ausencia de slices mediante probes en RAM y AST; no ejecutó
pytest ni certificó el forward. Los
[gates documentales](../../outputs/thesis-repair/live_contract_knowledge_20260913.xml)
terminaron con **1.102 passed**. Inventario, índices, 954 enlaces internos, grafo y
`git diff --check` acotado pasaron. No se reejecutó la suite global ni se declaran
resueltos sus fallos anteriores.

**Límites al cierre C045:** esas pruebas validan el contrato numérico, no toda la causalidad
del flujo streaming. Los builders todavía consultaban `build_mask()` sobre el dataset
global; una sesión parcial podía quedar fuera de esa máscara hasta completarse. C046,
documentado a continuación, corrige esa dependencia; no declara operable todo el forward.
Tampoco se reconstruyeron cachés para poner verdes pruebas que implicitamente entrenan.
La comprobación directa de los loaders sigue rechazando scaler y HMM por identidad
obsoleta, ahora frente a `f78f7fbc67325881832caed67248335c8c41786b0cb4bafdd05f1e2ab8021836`.
El HMM congelado conserva SHA256 `0a4b9ec5bf0db632d75dce73f3dfe5f2dadf1389aaa5cd8070c49c62680f736e`.
No cambian K/esquema, configuración de entrenamiento, resultados ni figuras históricos.

C045 se propuso inicialmente como contrato; antes de editar source se aclaró como
**C-EXEMPT, reparación de implementación de una restricción existente**, bajo lease propio.
No se recibió ACK de Claude para C045 ni se inventa una aprobación bilateral o un commit.
El objetivo completo permanece PARTIAL; este incremento no certifica rentabilidad,
calidad point-in-time ni disponibilidad de una política futura ejecutable.

## C046: admisión causal de prefijos y formatos de datos

Se separaron las reglas de evaluación histórica de la admisión de barras recibidas hoy.
`build_mask()` delega ahora en un evaluador privado de DataFrame sin cambiar sus reglas.
Los consumidores `build_live_spec`, `build_live_spec_partial` y `session_spread` usan
exclusivamente la historia acotada que recibieron, sin volver a consultar una máscara
global. El parcial añade la sesión actual sólo después de validar el prefijo, aunque
esa sesión no exista aún en el seed. No exige saber cómo terminará la sesión.

La admisión exige OHLC finito, positivo y coherente, instantes con zona conocida,
símbolo USD/COP y grilla ordenada y contigua desde las 08:00 COT. UTC y COT se convierten
por instante; no se inventa la zona de un timestamp naïve. Un prefijo desordenado se
rechaza, no se ordena silenciosamente: el caller puede usar su última fila para sellar
la decisión. Si falta `symbol`, el constructor explícitamente USD/COP lo asigna antes
de concatenar; de otro modo la columna del histórico introducía NaN y borraba el prefijo.
No se fabrican barras faltantes ni OHLC ausentes.

**Invariancia histórica comprobada antes y después:** `build_mask().to_dict()` completo
conserva SHA256 `b18bcd8e1dcaaea7762de6c71105f6403dbe3bf28be0e2e5e425f10b6a8fae24`
(JSON con claves ordenadas y separadores compactos). Siguen 1.361 sesiones válidas,
1.261 de entrenamiento y las mismas exclusiones, porcentajes planos y fuente.
Hash del conjunto válido: `55cb1f75a4a0850d2c8fa2e1614a44b55a87ef3e7b084ad8448c9ba058fe9e30`.
No se aprovechó la extracción para cambiar el criterio histórico `n >= 60`, ordenar
retornos para outliers o reabrir la selección. El reviewer verificó también la
equivalencia AST del cuerpo de reglas.

Las [pruebas de prefijo](../../tests/regression/test_live_prefix_causality.py) ejecutan
las fórmulas reales de features y de observaciones HMM. Las fixtures usan parámetros
no ajustados y no representan una estrategia rentable. Además, una prueba lee
**cotizaciones y macro reales de 2023-06-01**, retira del seed todas las barras de hoy
y exige igualdad exacta entre el cálculo completo y los prefijos de 1, 11 y 59 barras:
mercado, contexto, cierres y spread. No es un replay del PPO publicado ni prueba de
disponibilidad histórica macro. Entradas reales:

- M5 SHA256 `3ea04d48360db8f355fccb17c8d77a8d1aff3ce3785b87d452aff79b080bc5a0`.
- Macro v2 SHA256 `8a4c70bfa8771c457dc4a4747e824f072334cc4758e10e28d73bc0b798ac932c`.

**Fallos propios y de revisión conservados:** la [primera tanda ampliada](../../outputs/thesis-repair/live_prefix_red2_20260913.xml)
dio 28 failed/4 passed. El primer arreglo pasó 118 pruebas, pero el
[contraste con datos reales](../../outputs/thesis-repair/live_prefix_real_red_20260913.xml)
reveló dos fallos: zonas conocidas mezcladas y comparación que confundía precisión de
almacenamiento microsegundos/nanosegundos con instantes distintos. Ambos se corrigieron;
se prueban `s/ms/us/ns`. El reviewer encontró conversiones de tipos no monetarios:
booleanos/complejos y fechas/duraciones nativas se aceptaban como precios. Se conservaron
los [cinco fallos de escalares](../../outputs/thesis-repair/live_prefix_price_red_20260913.xml)
y [dos de columnas temporales](../../outputs/thesis-repair/live_prefix_datetime_red_20260913.xml)
anteriores al rechazo por tipo, antes de `pd.to_numeric`.

Resultado [focal final](../../outputs/thesis-repair/live_prefix_strict_20260913.xml):
**137 passed = 51 prefijo + 78 C045 + 8 máscara**, con cinco warnings.
[Integración](../../outputs/thesis-repair/live_prefix_integration_20260913.xml):
**507 passed**, incluyendo esos 137, con trece warnings. No se suman tandas solapadas.
El reviewer final hizo 21 probes en RAM sobre tipos, formatos, zona e inmutabilidad;
no ejecutó pytest ni certificó el forward. Ruff de ambos tests pasa; los sources
conservan once avisos preexistentes (seis live, cinco máscara), no un lint global verde.
Los [gates documentales C046](../../outputs/thesis-repair/live_prefix_knowledge_20260913.xml)
terminaron con **1.102 passed**; inventario, índices, enlaces, grafo y diff-check
acotado pasaron. No se volvió a correr la suite global ni se ocultaron sus fallos previos.

Hashes de implementación C046:

- `live_spec.py`: `dff78d4f6cd3bd744081fac1e965a88821737b4982961405400f8cabb817a9d4`.
- `evaluation_mask.py`: `fc824269a59bd9bd7f3b4c22e1455b142c954de8f1a7f54deb84c57bfca1bfe1`.
- `test_live_prefix_causality.py`: `164e3c41a5038124e3beeb0c64257717d35947a9bdef07636c3f6194d36b16d9`.
- `test_live_regime_contract.py`: `0ebc96c1548b553a0941a5c8006eef182e7a6f3b8db6ee3e5d16eaf8b8cb73d2`.
  Sólo se completó su fixture OHLC y se cambió el punto de inyección de la máscara.

**Qué sigue sin resolverse:** el calendario vigente no acredita cobertura futura completa;
su lista adicional colombiana de 2026 termina en agosto. No se cambiaron esos días ni
la máscara histórica para esconder el problema. Al cerrar C046 el runner todavía debía demostrar
actualización correcta de `unrealized/drawdown`, sellado posterior a la inferencia y
ausencia de llamadas a `build_live_spec()` completo en el brazo de primera barra.
Este constructor valida un prefijo suministrado; no acredita por sí solo cuándo se
recibió o se cerró cada barra. Esos son gates de integración pendientes, no éxito implícito.

Los loaders reales mantienen `identity mismatch` frente a
`7c42aba78f09bd08f6dfdc0923f70542fd3e96305b785655306700aa54fde89f`.
No se reajustaron HMM/scaler, no se cambiaron K/esquema, checkpoints, resultados,
figuras ni registro de trials. El [E2E de cierre](../../outputs/thesis-repair/live_prefix_e2e_20260913.json)
revalida los artefactos retrospectivos; no certifica entrenamiento o forward de una nueva
versión: `retrospective_results_reproduced=true`, `engineering_ready=false` y
`scientific_closure_ready=false`. C046 es reparación de implementación C-EXEMPT bajo lease propio, sin ACK
de Claude inventado y sin commit/push. El objetivo completo continúa PARTIAL.

## C047: estado secuencial, reloj y liquidación del adaptador streaming

Este incremento repara ingeniería del carril research; **no entrena modelos ni genera
resultados económicos nuevos**. El runner anterior actualizaba exposición y contadores,
pero nunca `unrealized` ni `drawdown`. También copiaba la recepción al timestamp de emisión,
y el CLI pasaba el timestamp de apertura M5 como cierre. Los defectos se reprodujeron
antes de corregirlos; no se presentan las pruebas de control como trades reales.

El [runner persistente](../../src/research/llm_forward/stream_runner.py) reconstruye
el estado previo a cada decisión con los pesos del ledger y los cierres del prefijo.
Utiliza `bar_cost`, volatilidad causal y retornos simples de la misma contabilidad del gym.
Para decidir en `b`, sólo liquida los intervalos `0..b-1`. Drawdown conserva la convención
de entrenamiento: acumulado aritmético menos máximo, no drawdown de equity compuesta.
La entrada y la permanencia se reinician ante cualquier cambio de peso, incluida una
reducción del mismo signo. El costo de la decisión actual y el cierre terminal no entran
anticipadamente en la observación.

La caché representa el instante posterior a decidir, sin fabricar el cierre siguiente.
Antes de continuar se compara íntegramente contra la historia reconstruida; se validan
cadena, secuencia, IDs, modelo declarado, pre-registro, spread, pesos discretos y hashes
de las observaciones anteriores. Un cambio de features/contexto pasado no se acepta en
silencio. Se rechazan también fechas de prefijo distintas, booleanos usados como contadores,
metadatos temporales contradictorios y retrocesos del reloj entre decisiones.

El [sellador](../../src/research/llm_forward/arms/ppo_stream.py) exige timestamps con zona y
la grilla `08:00 COT + (b+1)*5 min`; comprueba cierre ≤ recepción ≤ inicio de inferencia.
`emitted_at_utc` registra el final de la inferencia con precisión de microsegundos.
Terminar exactamente en el límite siguiente o después produce un flag tardío, no un
sello válido ni un flat inventado. Las acciones fraccionarias, texto, booleanos, vectores
multivalor y observaciones no finitas se rechazan. El [CLI](../../scripts/analysis/run_ppo_stream_bar.py)
convierte explícitamente el timestamp de apertura L0 a cierre sumando cinco minutos,
y captura la recepción antes de construir features o cargar el modelo.

La prueba del recorrido completo descubrió otro defecto: el fallback de `dict.get`
evaluaba una clave inexistente incluso cuando el agregado tenía su propio flag.
Se corrigió sólo esa expresión en [liquidación](../../src/research/llm_forward/settle_thesis.py).
La prueba llega desde las 59 decisiones a un ledger de liquidación y verifica idempotencia.
**El esquema antiguo de ese ledger todavía no conserva todos los componentes netos**;
la reparación del fallback no constituye un export financiero completo.

Persistencia: escritores cooperantes se excluyen mediante un lock del ledger. El temporal
del estado se vacía con `fsync`; un `PermissionError` de reemplazo atómico se reintenta hasta
cinco veces con pausas 0,025/0,05/0,1/0,2 s. Nunca se repite el append, se borra el estado
anterior ni se degrada a escritura no atómica. Un fallo permanente conserva la decisión
ya escrita y bloquea el reinicio por discrepancia con la caché; un lock abandonado requiere
revisión del operador. No se afirma atomicidad de dos archivos ni exclusión frente a
escritores que no respeten este protocolo.

Evidencia de pruebas:

- [RED inicial](../../outputs/thesis-repair/stream_red_20260913.xml): 54 fallos y 5 pases.
- [Primer arreglo](../../outputs/thesis-repair/stream_green_20260913.xml): 55 pases y 7
  errores `WinError 5` al reemplazar el estado. El [retry escalado](../../outputs/thesis-repair/stream_retry_20260913.xml)
  produjo 57 pases y 5 errores iguales; no se atribuye la causa a OneDrive sin evidencia.
- [Ataques adicionales](../../outputs/thesis-repair/stream_review_red_20260913.xml):
  7 fallos y 2 pases antes del guard semántico/retry. El [RED de liquidación](../../outputs/thesis-repair/stream_command_red_20260913.xml)
  dio 1 fallo y 2 pases. Todos se conservan, no sólo los reportes favorables.
- [Focal final](../../outputs/thesis-repair/stream_final_20260913.xml): **77 passed**.
- [Integración](../../outputs/thesis-repair/stream_integration_20260913.xml): **599 passed,
  1 deselected**, incluyendo los 77. El test excluido usa el scaler/HMM histórico incompatible;
  no se reajustó ni se eludió su guard para producir un verde. No es una nueva suite global.

Los [gates de conocimiento](../../outputs/thesis-repair/stream_knowledge_20260913.xml)
terminaron en **1.102 passed**. Inventario, índices, enlaces, grafo y `git diff --check`
acotado pasaron. Ruff pasó sobre los siete archivos de implementación/tests revisados;
`settle_thesis.py` conserva cuatro avisos previos (RUF002 e I001), reproducidos exactamente
contra el objeto archivado SHA `869e01bcb4a21176cbf2c9de46f17007a26b48a783bf570f688e56aef23265c8`.
No se presenta el lint global como verde. Los warnings de pytest fueron configuración
`asyncio_mode` desconocida y autocorrelación de series planas en diagnósticos históricos.

Además de los cuatro archivos de implementación enlazados arriba, se crearon
[tests de estado](../../tests/regression/test_stream_state_parity.py) y
[tests de sellado](../../tests/regression/test_ppo_arm_per_bar_sealing.py), y se actualizaron
las fixtures de [runner persistente](../../tests/regression/test_live_session_runner.py) y
[runner PPO](../../tests/regression/test_ppo_stream_runner.py). Este informe y los canales
propios de coordinación registran el trabajo. No se renombraron ni eliminaron datos,
modelos o código, ni se hicieron commits/push; los cambios previos del worktree se preservan.

Diez sendas de control comparan las **590 observaciones** contra `SessionTradingEnv`, con
reinicios entre barras y liquidación a tolerancia 1e-12: cinco usan cierres de control y
cinco cierres reales del seed para 2023-06-01. Mercado/contexto son fixtures sin ajuste,
las acciones son prefijadas y no se evalúa un PPO entrenado. Los cinco números de semilla
identifican sendas de prueba, no nuevos experimentos. La revisión independiente en RAM
comprobó 295 vectores contra otra reconstrucción del estado y reprodujo los escapes
semánticos; no constituye doble firma de pytest ni ACK de Claude.

Hashes del código: runner `44552a72658dc05cf7ec8d0ca2abb6457d5d3873557bdd00e6b7e5eb4dc5c5f8`,
sellador `35c4b8050a66a03289dd3f65895c758aae8af5ec8dc4f49ce7f7fc15a3836145`,
CLI `02cfe2b8ad17f118a2801abea5d3a0411b498cb0879cd7fc68ac140a0cd37228`,
liquidación `40a383a13fed496e16da672e09521396591a6e5f5c30380ca69b518efe17d387`.

El [E2E actualizado](../../outputs/thesis-repair/stream_e2e_20260913.json) conserva el SHA
`a7293a829498b158f3e34c335a6f5333e0d5eba2d1a05814118cc0cee2b8e977`: reproduce los
artefactos retrospectivos, pero mantiene `engineering_ready=false` y
`scientific_closure_ready=false`. No certifica disponibilidad PIT, calendario futuro,
identidad binaria del checkpoint, autenticidad externa del reloj, persistencia antes del
deadline ni fills a precios ejecutables. Un flag de fin de inferencia no prueba ninguno
de esos hechos. Se preservan modelos, schema, series, resultados y figuras anteriores.
Sin nueva comunicación de Claude desde CLD762, el trabajo es independiente en modo
degradado; **C047 es progreso verificado, no cierre del objetivo ni aprobación bilateral**.

## C048: contabilidad persistida y admisión de resultados

**Incremento de ingeniería, no nuevas rentabilidades ni cierre científico.** El motor
calculaba bruto, costos y neto, pero el ledger conservaba únicamente `signed_return`
bruto. La corrección añade `SettlementRecord.accounting` opcional al final del contrato;
no cambia campos posicionales ni reescribe registros históricos. Ausente o `None`
significa **neto desconocido**, nunca cero y nunca un alias del bruto.

El [nuevo módulo](../../src/research/llm_forward/settlement_accounting.py) conserva:

- Cierres de 60 barras, 59 posiciones y contribuciones brutas, 60 costos incluido el cierre;
  bruto, costo total, neto, costo terminal, cambios de posición y turnover con cierre.
- Unidades explícitas: retorno decimal y precio/spread COP por USD; comisión y slippage
  empleados, hashes canónicos de insumos, hashes del motor y del contrato de costos.
- Referencias ordenadas a las decisiones originales. El lector verifica ambas cadenas,
  vuelve a vincular referencias y reconstruye posiciones y contabilidad; verifica también
  fecha, ID, precios extremos, número de barras y bruto exterior redondeado a ocho decimales.
- Costos API separados en USD, con número de decisiones con/sin información. Una suma
  parcial no se presenta como costo total y un importe reportado no se presenta como factura.

El [liquidador](../../src/research/llm_forward/settle_thesis.py) valida el lote antes
de escribir y serializa los escritores cooperantes. Una sesión incompleta, tardía o
abstenida no produce retorno cero. Se rechazan números no finitos, booleanos/cadenas
coercibles, sendas vacías o fuera del espacio de acción, índices ambiguos, mezcla de
identidades, spreads ausentes/inválidos, claves JSON duplicadas y suplantación de un
agregado interno. El resumen de la decisión debe corresponder a la primera posición.
El lector comparte la misma admisión de costos que el escritor.

El [verificador](../../src/research/llm_forward/verify.py) distingue decisiones,
sesiones-brazo admisibles, liquidaciones con neto verificado y registros antiguos sin
neto. No interpreta hashes cambiantes de observación como versiones cambiantes de un
prompt. Una identidad provider/model/prerregistro distinta bajo el mismo brazo sí exige
separar tratamientos. No genera Sharpe ni p-values desde estas pruebas pequeñas.

La compatibilidad temporal distingue el cutoff **planificado** (puede ser la apertura)
de la publicación de cada documento realmente incorporado: éste debe preceder al cutoff
y ya existir al emitir la decisión. Se comprueba el fin de inferencia, no se rescatan
filas tardías por un flag verdadero. El flag opcional `None` permite la regla diaria
anterior sin confundirlo con `False`; las decisiones por barra exigen su propia cronología.

Evidencia TDD y de revisión:

- [Admisión inicial: 43 fallos y 6 aciertos](../../outputs/thesis-repair/settlement_red_20260913.xml)
  antes de corregir; luego [49 aciertos](../../outputs/thesis-repair/settlement_admission_20260913.xml).
- [Payload ausente: 19 fallos / 50 aciertos](../../outputs/thesis-repair/settlement_payload_red_20260913.xml);
  [lector: 3 fallos / 73 aciertos](../../outputs/thesis-repair/settlement_reader_red_20260913.xml).
- El revisor independiente encontró contradicciones del resumen, JSON ambiguo y una
  duplicación de candidato. [Seis fallos reproducidos](../../outputs/thesis-repair/settlement_review_red_20260913.xml)
  se convirtieron en controles de rechazo. También se preservan los dos fallos de
  [cutoff/documentos](../../outputs/thesis-repair/settlement_cutoff_red_20260913.xml) y los dos de
  [conteo admisible sin spread](../../outputs/thesis-repair/settlement_coverage_red_20260913.xml).
- La integración inicial produjo [690 aciertos y un fallo](../../outputs/thesis-repair/settlement_integration_20260913.xml):
  `git ls-files` salió con código 128 por distinta identidad del propietario del checkout.
  La [repetición final](../../outputs/thesis-repair/settlement_final_20260913.xml) da **691 aciertos**
  con `safe.directory` únicamente en el proceso, sin modificar Git global. Incluye los
  92 controles del archivo nuevo. Un test queda deseleccionado porque intenta cargar
  artefactos congelados con identidad obsoleta; no se ajustan ni se re-vinculan para
  obtener un verde. No se reejecutó toda la suite global.
- [Contratos y espejos: 102 aciertos](../../outputs/thesis-repair/settlement_contracts_20260913.xml).
  El procedimiento local `contract-change` determinó que este ledger interno no tiene
  espejo TypeScript: no se modificaron APIs, RBAC ni contratos del dashboard.
- [Conocimiento: 1.102 aciertos](../../outputs/thesis-repair/settlement_knowledge_20260913.xml).
  Inventario, índices, enlaces internos y grafo: PASS. Ruff en los seis archivos de
  código/tests intervenidos: PASS. Estas suites se solapan, no se suman como pruebas únicas.

Los controles usan sendas conocidas y fixtures, incluido el cierre de precio constante
con costo analítico independiente, no modelos entrenados nuevos. La integración con el
runner conserva las comprobaciones con precios reales de C047 sin convertir sus acciones
de control en resultados de PPO. Se prueba escritura, lectura, repetición idempotente y
compatibilidad de una fila antigua sin `accounting` seguida de una fila nueva.

**Límites deliberados:** el hash del vector de cierres no prueba timestamps ni bid/ask
ejecutable; un hash calculado al liquidar no prueba que las tarifas estuvieran congeladas
al decidir; un reloj local no acredita sellado durable externo. El payload lo declara
`paper_assumed_costs` y no habilita preparación científica. La reproducción exige el
motor/contrato archivado correspondiente cuando cambian sus hashes, no una reinterpretación
silenciosa con código actual. No hay medición de fills, nuevos trials, entrenamiento ni APIs.

El [E2E de este incremento](../../outputs/thesis-repair/settlement_e2e_20260913.json) tiene SHA
`a7293a829498b158f3e34c335a6f5333e0d5eba2d1a05814118cc0cee2b8e977`, idéntico al anterior:
resultados retrospectivos reproducidos, `engineering_ready=false` y
`scientific_closure_ready=false`. Las figuras y resultados del bundle v5 no cambian.
PIT, costos observables, calendario futuro, K/esquema, identidad de modelos y reconciliación
de trials siguen pendientes. C048 usa auto-ACK **sólo aditivo** tras el plazo del protocolo;
al cierre de ese incremento no existía respuesta nueva de Claude ni aprobación bilateral
contra commit. Posteriormente CLD-763/764 dieron ACK de alcance limitado, sin certificar
las tandas de pruebas de Codex.

Archivos del incremento: nuevo módulo de contabilidad y su archivo de 92 controles;
modificados schema, liquidador, verificador, fixture de paridad y este informe, además
de coordinación y evidencia nueva en `outputs/thesis-repair/settlement_*20260913*`.
Sin commit/push ni modificación de resultados, modelos, datos o registros de trials.
Hashes de implementación: contabilidad `99a317077fc8358604361dee99c636029c55a6fb1e127df7592bb12a2872f1d6`,
schema `1a375f7d41f9c5eac9c5b5d9d0cdb4be3df9441c7af4a3b4b37be4e56cab25d1`,
liquidador `e840f6291bf126e3bfeb4499428d7b8e0b0617a8badab4b0399b67203837385e`,
lector `b829c952a8164d713ec4dcb0591d48dac7355f63da519aa805bd638549dd7dcc`.

## C049: primera barra, modalidad única y fallos visibles del DAG

Incremento de ingeniería con ACK de alcance CLD-765, no experimento nuevo. El
[productor](../../src/research/llm_forward/arms/ppo_arm.py) ya decide con un prefijo
de **una barra cerrada**, sin pedir la sesión completa para obtener el spread. El campo
opcional `decision_schedule=first_bar_hold` identifica una inferencia y 59 pesos constantes;
`None`/ausente mantiene el legado. No se reescribió ningún registro histórico.

Se cerraron los tres defectos adicionales identificados en la revisión:

1. **Doble conteo:** productor, liquidador y auditor rechazan modalidades distintas para
   un mismo brazo. Una decisión sostenida y 59 decisiones nativas no son dos resultados
   independientes del mismo brazo/día. No se atribuye un factor universal de inflación.
2. **Prefijo incorrecto:** fecha distinta/ausente y contadores booleanos o flotantes se
   rechazan antes de inferir. Una barra de otra sesión no puede alimentar esta decisión.
3. **Escritura concurrente:** el productor reutiliza el lock cooperativo del runner,
   verifica la cadena antes de inferir y reconoce un reintento válido sin segunda inferencia
   ni modificación de bytes. Un lock ocupado no se roba ni se elimina automáticamente.

El [DAG](../../airflow/dags/research_forward_arms.py) ahora lanza error si la liquidación
o auditoría no devuelve exactamente el entero cero. No acepta `False` ni `0.0` como éxito.
También bloquea su cohorte nativa de 59 decisiones **antes de llamar al LLM**, pues sigue
teniendo un solo despacho. No se eliminó un brazo, no se activó el DAG ni se cambió su cron.
Se aplicaron `dag-change` y `contract-change`: comprobación de consumidores, campo aditivo,
propagación de fallos y nota en el calendario operativo; no hay espejo TS de este ledger.

**La reserva de precio importa:** una M5 con apertura 08:00 COT cierra 08:05. Emitir antes
de 08:10 no acredita haber ejecutado al C0 conocido a las 08:05. La contabilidad aún asigna
el tramo C0→C1 bajo un precio proxy histórico; no se cambió el lag ni la estrategia para
mejorar el resultado. `information_edge` declara PAPER, sin fill ejecutable ni acuse durable.
Que las entradas sean prefijos no elimina por sí solo este supuesto de ejecución.

Pruebas ejecutadas en el worktree, sin modelos entrenados ni APIs:

- [Rojo inicial](../../outputs/thesis-repair/first_bar_red_20260913.xml): **11 failed, 27 passed**.
- Revisiones discriminantes: [3 fallos](../../outputs/thesis-repair/first_bar_review_red_20260913.xml)
  y [5 fallos](../../outputs/thesis-repair/first_bar_prefix_red_20260913.xml), corregidos.
- [Integración final](../../outputs/thesis-repair/first_bar_final_20260913.xml):
  **757 passed, 1 deselected, 13 warnings**, incluidos los **66 controles nuevos** de
  [modalidad/productor](../../tests/regression/test_forward_first_bar_hold.py) y
  [tareas del DAG](../../tests/regression/test_forward_dag_safety.py).
- [Contratos/registro DAG](../../outputs/thesis-repair/first_bar_contracts_20260913.xml):
  **87 passed, 1 skipped**. El omitido es importabilidad: Airflow completo no está instalado
  localmente. Docker tampoco tiene disponible su engine, incluso en la comprobación
  escalada; no se validó el scheduler. `py_compile` y lint de los ocho paths pasan.
- [Gobernanza](../../outputs/thesis-repair/first_bar_knowledge_20260913.xml):
  **1.102 passed**; inventario, índices, enlaces, grafo y `git diff --check` acotado pasan.
  No se detectaron enlaces internos rotos. La comprobación del seed, HMM y scaler conserva
  sus hashes previos. El monitor acotado terminó a las 15:17:33 UTC, sin dejar un worker vivo;
  sus logs se conservan. No se requiere otra revisión ni delegación para este cierre.

Las tandas se solapan: no deben sumarse como pruebas independientes. La extracción y
ejecución de funciones del DAG mediante AST prueba sus callbacks, **no importa el DAG en
Airflow ni demuestra orquestación real**. No se repitió la suite global de todo el repo.

**Prueba excluida, identificada exactamente:**
`tests/regression/test_forward_arms_parity.py::test_forward_settlement_reproduces_the_thesis_numbers[42]`.
Toma fechas, pesos y spread del JSON, pero compara `settle_session` con `run_session`
del motor **actual**, no contra los escalares históricos publicados. Su nombre/docstring
prometen más de lo que comprueba. Aborta por identidad congelada antes de comparar.
No se la cuenta como verde ni como reproducción histórica; no se modificó esa prueba
durante C049 para resolver artificialmente la discrepancia.

La **paridad vivo-batch sigue NO EJECUTABLE** con el HMM K=5 y cuatro slots, además del
scaler incompatible. Es un bloqueo de arranque explícito, no una invitación a cambiar hashes.
Requiere decisión del operador sobre K/esquema, nuevos artefactos/versionado y paridad antes
de cualquier nuevo sello. También faltan fuente viva puntual, calendario y orquestación59.

El [E2E final](../../outputs/thesis-repair/first_bar_e2e_20260913.json), SHA
`a7293a829498b158f3e34c335a6f5333e0d5eba2d1a05814118cc0cee2b8e977`, reproduce el
bundle v5 y mantiene `engineering_ready=false` y `scientific_closure_ready=false`.
No se generaron figuras nuevas ni se alteraron los resultados existentes. Bruto positivo
observado no establece alfa; disponibilidad histórica, costos ejecutables, modelos LLM y
trials siguen pendientes. No se afirma haber descartado toda fuga o fallo de aprendizaje.

Hashes de implementación: productor `681744f1ccedc4551a3642dc298395d345d02e1064e645b4ccaa6875890931d4`,
schema `4b5f6a371fa14e5b87e48acf65986df2b93f938800d92be52e99ef75582a7971`,
contabilidad `24bc1d7a7ab44e04e97e9ba90942f2ec353b99969ca3215d44327c27c5baac65`,
liquidador `4ce9928e7fe344c15b259d03d11a665961be4514def50e6beac374b4b1c31493`,
auditor `ba1d3371cdfeeba16654711aab35b65c304b50c56cbf453e4820d7f7e9e90fa6`,
DAG `02bc6250a8155f155c03494d5d6704ee25edcec168ee9318f3f9c07d80312c07`.
Archivos modificados: esos seis, el informe y la nota en `elite-operations.md`;
nuevos: los dos archivos de pruebas y salidas `first_bar_*20260913*`. Coordinación
append-only/heartbeat aparte. Sin commit/push, sin cambios de datos/modelos/registries.
Por instrucción del operador se cierra este incremento sin más delegaciones ni experimentos;
la tesis completa y el forward no se marcan terminados.

## Gobernanza, trials y siguiente aprobación

El registro sigue en 115 y no fue editado. El
[borrador de reconciliación](../../.claude/coordination/briefs/H-TESIS-RL-02-contabilidad.md)
identifica PPO v2, reglas y supervisado, y propone cargar también los cuatro brazos LLM/híbridos.
La etiqueta retrospectivo **no exonera** una comparación mirada. El total candidato 126 es
una propuesta condicionada a deduplicación, parentesco FT/AT, sensibilidades y revisión del operador;
no se usa aquí como N certificado. No se reinicia el universo de deflactación al crear v2.

La coordinación con Claude usa inbox append-only y leases. CLD-731 retiró las afirmaciones
de independencia, alfa demostrado y señal invertida; CLD-732 corroboró hashes y cifras de v1,
no todas las figuras ni la reconciliación macro. CLD-733 aceptó sustituir la prueba frágil de
cadenas de código por comportamiento del constructor. La revisión posterior de Claude también
reconcilió las series y contrastes v2, retirando «todos p=0,0002». No revisó aún los crudos
macro ni todas las tablas de calidad/regímenes. Es revisión con alcance, no doble firma global.
CLD-737 aceptó la medición del constructor en el runtime nativo y reconoció que algunos
sellos anteriores habían sido escritos sin leer el reloj; se preservan con la retractación.
La comprobación coincidente en dos versiones no prueba invariancia en todas las versiones.

### Reconciliación del registro: hallazgos verificados, sin editar asientos

La revisión independiente y la lectura de la raíz confirman el ledger SHA
`c708de2312287b6e739767a02d39daeb5e8564c2fcc972a9456ea1b13c8c65e0`:
115 asientos USD/COP, 51 FT y 64 AT, sin IDs repetidos. Son 108 `legacy_estimate`
y 7 `documented`, pero **109** tienen ambos hashes nulos: las 108 estimaciones más
FT-0049, documentado como H-VOLF-01. Por eso es incorrecto llamar legacy a las 109.
Los bloques estimados ya incluyen procedencia y `cell_kind: estimated_block`; no se
necesita reetiquetarlos para aparentar una mejora de trazabilidad.

Seis filas tienen ambos hashes no nulos. En AT-0185/0186, `data_hash` contiene el
prefijo de máscara `f6958be1b59e9767` completado con 48 ceros: **no acredita un SHA256
completo del dataset**. Tampoco procede deduplicar por igualdad de código/datos:
distintas configuraciones o brazos pueden compartirlos. La unicidad de IDs se comprobó;
el posible solapamiento de unidades estimadas requiere reconstrucción documental.

El YAML prospectivo v2 declara scope `family`, cluster y clave distintos de v1 y una
arista sibling unilateral. El validador estructural pasa porque sólo agrupa claves
iguales; ese verde no valida el linaje. La propuesta inicial de cambiar a `cluster`
se retiró: puede reducir el universo a dos asientos frente a los 115 del activo que
usa hoy la tesis. No se aplicó ningún cambio a familias ni registros.

**126 continúa siendo propuesta**, no conteo certificado. Antes de cobrar se debe
conciliar hipótesis/configuración, artefacto evaluado, ventana, primera mirada, asiento
previo o nuevo y motivo del cargo. Una repetición congelada no es automáticamente
otra hipótesis; la etiqueta retrospectiva tampoco exonera una variante nueva mirada.
Ningún claim debe utilizar un universo inferior al total vigente del activo; familia,
cluster y total global pueden informarse por separado, no sustituirlo silenciosamente.

## Reproducción local y definición de terminado

Desde la raíz, con el intérprete Python del entorno de investigación:

```text
python scripts/presentation/build_research_grade_thesis.py --snapshot outputs/thesis-repair/archive/pre_research_grade_20260912_1934/manifest.json --output outputs/thesis-repair/research_grade_reproduction_NEW
python scripts/diagnostics/audit_thesis_e2e_status.py --bundle outputs/thesis-repair/research_grade_20260912_v5 --macro-report outputs/thesis-repair/research_grade_macro_20260912/identity_live_network.json --sanity-report outputs/thesis-repair/sanity_research_grade_20260912/protocol.json --vintage-capture outputs/thesis-repair/alfred_selection_20260912 --expected-vintage-sha f1b903ae5e9716d444ea8c11e691987746e320bf5f9ba584996c6430be031c66 --expected-bundle-sha 00fc276e22b62ffac144bca1abb6c468f15e272225c46743304c67fe120ae2da --output outputs/thesis-repair/research_grade_audit_NEW.json
python scripts/analysis/run_thesis_llm_pilot_v2.py --validate-only
```

Cada salida NEW debe ser inexistente. El primer comando regenera resultados a partir de los
archivos preservados, sin APIs, reentreno ni evaluación del hold-out. El segundo escribe un
informe de estados: exit 0 significa que **generó el informe**, no que todos los gates pasan.
El tercero permanece cerrado mientras falten los insumos prospectivos y nunca carga `.env`.

Validación focal efectivamente ejecutada: **57 passed** de métricas/preservación/publicación
y gates; suite de sanidad/constructor/contabilidad **80 passed** y replay **10 passed**.
Son grupos con posibles solapamientos: no se suman como una suite única.
La primera suite nueva integrada terminó con 170 passed; la siguiente, con 177.
Tras añadir también el rechazo directo en el constructor del dataset, la
[suite integrada final](../../outputs/thesis-repair/research_grade_tests_20260912_final2.xml)
terminó con **178 passed**, incluyendo datos, export, piloto, sanidad y reportes; único
warning de configuración `asyncio_mode` no reconocido.

La primera suite global terminó con **2.231 passed, 66 failed, 87 skipped, 1 xfailed,
6 errors**. No se publica como verde. Al resolver `safe.directory` sólo en el proceso de
diagnóstico, el subconjunto afectado terminó en **220 passed, 3 failed, 51 skipped**.
Los tres fallos restantes de ese subconjunto son control BL08 de historia Git, un subprocess
de monitor con `WinError 1312` y falta de `dill` en H5. Separadamente, los tests live que
dependen del scaler/HMM global antiguo siguen rechazando su identidad: no se falsifican hashes
para aprobarlos. La prueba de presencia de `.env.example` tampoco certifica el entorno;
no se leyó ese archivo. El módulo Ruff faltaba en Python312, pero se encontró posteriormente
el ejecutable separado. El primer lint del paquete detectó 53 avisos, corregidos en los
archivos nuevos. La verificación `check --no-fix` de los trece archivos revisados pasó;
**no se formatearon las fuentes congeladas de sanidad ni el parser macro** y sus hashes
continuaron idénticos. La regeneración posterior comprobó la equivalencia numérica y gráfica.
El lint separado de `dataset.py` conserva un aviso C416 preexistente, ajeno al guard;
no se afirma que todo el repositorio haya pasado lint.
La última revisión amplió el lint a doce paths y encontró un import sin ordenar en el
test de evidencia macro. Se corrigió sólo ese import; `check --no-fix` pasó. Su rerun
inicial dio `8 passed, 16 errors` por `WinError 5` en el temporal de Windows, no por
aserciones. El retry autorizado, con temporal nuevo acotado al proyecto, terminó con
**24 passed**. No se suman estas pruebas repetidas a las 178.

La [regresión global final registrada](../../outputs/thesis-repair/research_grade_full_regression_20260912.xml)
terminó en **2.352 passed, 8 failed, 6 errors, 88 skipped, 1 deselected y 1 xfailed**.
El deselected fue deliberado: la prueba que abre `.env.example` está fuera del acceso
permitido a secretos. No se interpreta un skipped como comprobación realizada.
Los ocho fallos son BL08/historia, H5 sin `dill`, monitor `WinError 1312` y cinco tests
de paridad/live con identidad congelada obsoleta; los seis errores de setup son HMM portable
por esa misma identidad. Se preserva el reporte completo, no sólo el conteo favorable.
Esta corrida global fue anterior al último guard directo de `dataset.py`; no se volvió
a ejecutar completa después. Las 178 pruebas focales finales sí incluyen ese guard.

## Archivos del cambio y alcance de la entrega

Nuevos módulos: [reportes](../../src/research/reporting_v2.py),
[unión por publicación](../../src/research/publication_join_v2.py),
[evidencia macro](../../src/research/macro_evidence.py) y
[piloto LLM](../../src/research/llm_experiment_v2.py).
Nuevos entrypoints: [snapshot](../../scripts/diagnostics/freeze_thesis_evidence.py),
[tablas y figuras](../../scripts/presentation/build_research_grade_thesis.py),
[replay de sanidad](../../scripts/diagnostics/verify_sanity_policy_replay.py),
[export transaccional](../../scripts/data/export_research_bundle_v3.py) y
[piloto prospectivo](../../scripts/analysis/run_thesis_llm_pilot_v2.py), con
[configuración propia](../../config/research/llm_pilot_v2.yaml).

Se modificaron `src/research/{cost_model,dataset,ppo_recipe,sanity_gate,session_env,session_gym,synthetic_sessions}.py`,
`scripts/analysis/{thesis_ppo_sanity,thesis_train_ppo}.py`,
`scripts/data/build_research_macro.py` y
`scripts/diagnostics/{verify_macro_declared_identity,audit_thesis_e2e_status}.py`.
Las regresiones nuevas son los diez archivos `tests/regression/test_research_grade_*.py`;
se actualizaron los tests existentes de compuerta, fixtures distintas, contrato de semillas,
sanidad, constructor y estado E2E que los nuevos controles sustituyen.

Documentación/coordinación: este informe, índice `docs/analysis/README.md` generado,
`CODEX-STATUS`, `INBOX-CLAUDE` y `LEASES`, más adendas append-only al `EXPERIMENT_LOG`.
No se eliminaron ni renombraron archivos de código, datos o modelos; sólo se limpió el PID
temporal del monitor propio al detenerlo. No se hicieron commits/push desde esta ejecución,
y no se sobrescribieron resultados/checkpoints/ledgers/fuentes históricos. La caché mutable
de investigación sí fue reconstruida por la fixture global, y ese hecho se declara arriba.
Inventory, índices, enlaces, knowledge graph y ledger estructural pasaron sus comprobaciones;
ningún enlace interno roto fue detectado. Los cambios ajenos del worktree se preservaron.

**Estado final científico: PARTIAL.** Falta cerrar disponibilidad/vintages, representación HMM/integración,
trials, costos ejecutables y replicación futura. La tesis sí puede presentar este capítulo
como evidencia retrospectiva negativa, con sus limitaciones; no como experimento confirmatorio
terminado ni como demostración de rentabilidad o de su imposibilidad.

## C050 — Versión de cinco espacios, 2026-09-14 UTC

**Estado: incremento técnico implementado; plan completo PARTIAL.** El operador aprobó
cinco espacios para régimen, mantener la selección BIC solamente en desarrollo,
entregar tesis retrospectiva y piloto prospectivo por separado, y ejecutar por gates.
Esto no autoriza forzar K=5, reutilizar un checkpoint incompatible ni inventar
publicaciones históricas para permitir un entrenamiento.

### Lo implementado y su alcance

- [Contrato nuevo](../../src/research/observation_contract.py) y
  [esquema independiente](../../config/research/observation_regime5_v1.json):
  `research_regime5_v1`, 38 entradas; las primeras 37 conservan su orden,
  la última es `p_regime_4`. K=2..5 conserva toda la masa y sólo añade ceros
  después de K. Se rechazan booleanos, máscaras, valores no finitos y posterior
  inválido; no se recorta ni renormaliza una salida para hacerla caber.
- [Bundle nuevo](../../src/research/regime5_bundle.py): parámetros, esquema y
  observaciones NPZ sin pickle ejecutable, exportación exclusiva, manifiesto de
  componentes/código e identificación explícita de versiones. Un checkpoint se
  carga sólo con hashes de bytes, metadatos, bundle y esquema coincidentes.
  No se migró ningún checkpoint histórico.
- El mismo builder proporciona sesión completa y prefijos; las pruebas comparan
  las 59 observaciones con el gym y calculan las features batch por otra ruta.
  Historial futuro o macro posterior no puede cambiar una observación previa.
  El gym rechaza mezcla de versiones y conserva contabilidad/cierre terminal.
  `strip_regimes` elimina ahora todos los slots de la versión declarada, no
  sólo los últimos cuatro. El trainer legacy **rechaza** la versión nueva:
  no constituye una autorización de reentrenamiento.
- [HMM versionado](../../src/research/regime5_hmm.py): el PPO y el HMM consumen
  los mismos operandos macro del join de publicaciones observadas. Se exige
  período anterior a la sesión y `max(publication_at, first_seen_at) < cutoff`;
  disponibilidad de ambos operandos y staleness por serie son explícitos.
  Un hash de fuente no autentica por sí mismo esos timestamps.
  Se mantiene BIC K=2..5, default K=3 e histéresis estrictamente mayor a 10.
  Se registra BIC/loglik/convergencia/iteraciones/covarianza por candidato.
  En esta nueva ruta el ranking diagonal acepta la representación expandida
  K×D×D de hmmlearn y produce una varianza por estado; el módulo viejo no se cambia.
- Preparación ejecutable mediante
  [CLI](../../scripts/data/build_research_regime5.py) y
  [configuración](../../config/research/research_regime5_v1.yaml). Es una
  **preparación, no un SSOT de experimento firmado ni un ejecutor de training**.
  Las funciones de ajuste en desarrollo y exportación están implementadas;
  no se han ejecutado sobre datos de mercado en este incremento.

### Regresión encontrada y corregida, sin cambiar los hashes esperados

La primera integración produjo **783 PASS, 3 FAIL y 16 ERROR**:
`ValueError: historical source binding mismatch: src/research/regime_hmm.py`.
Mi cambio inicial al módulo histórico invalidaba la reproducción byte a byte.
La [salida fallida](../../outputs/thesis-repair/regime5_20260914_integration1.xml)
se conserva. Restauré exclusivamente mi delta sobre ese módulo y aislé la ruta
nueva en `regime5_hmm.py`; **no actualicé los pins de la auditoría**.
La [revisión dirigida posterior](../../outputs/thesis-repair/regime5_20260914_historical_fix.xml)
pasó 79 pruebas. Una prueba adicional exige igualdad exacta entre ambas fórmulas
de mercado del HMM cuando reciben los mismos operandos macro.

El test anteriormente excluido se llama ahora
`test_current_settlement_matches_current_engine_on_archived_weights[42]`.
Compara dos rutas actuales usando precios reales y pesos archivados; ya no intenta
construir un HMM que no necesita, ni afirma reproducir cifras históricas.
Por separado, el [test de cifras publicadas](../../tests/regression/test_regime5_historical_boundary.py)
fija el manifiesto v5 por SHA, verifica tablas/figuras y recompone los retornos
desde las series originales. Un artefacto original ausente no se fabrica.

Verificación C050: **804 PASS, sin exclusiones**, en la
[integración completa final](../../outputs/thesis-repair/regime5_20260914_integration_identity.xml);
**1.128 PASS** en [conocimiento y contratos](../../outputs/thesis-repair/regime5_20260914_knowledge.xml).
La repetición final, después de completar las dependencias de identidad, volvió a dar
804 PASS. El hash incluye también las constantes de dataset y el parser de costos.
Además, [tests/contracts](../../outputs/thesis-repair/regime5_20260914_contracts.xml)
pasó 58 pruebas. Estas tandas se reportan por separado, no se suman como controles
independientes. No se ejecutó la suite global del repositorio ni un DagRun real.
Ruff pasó en los 16 archivos de código/tests intervenidos (no se maquilla el lint
basal del HMM histórico restaurado). Inventario, índices, enlaces y grafo pasaron.
Los avisos de NumPy por autocorrelación de series constantes y el aviso de
`asyncio_mode` siguen reportados; no son resultados estadísticos.
Las pruebas con precios/señales construidos son controles de software; el checkpoint
SB3 de prueba se inicializa, serializa y lee, **sin llamar a learn()**.
No constituye una corrida PPO ni evidencia de rentabilidad.

### Resultado del gate, no una lista de deseos

El [reporte de preparación final](../../outputs/thesis-repair/regime5_20260914_preparation_post_review.json)
reproduce los escalares de la tabla y verifica las ocho figuras antiguas:
`retrospective_delivery_ready=true`. Verifica también esquema nuevo e identidad
del snapshot M5. Mantiene `training_ready=false`, `pilot_executed=false` y
`profitability_established=false`; la CLI termina en no-cero deliberadamente.
SHA del reporte: `6149d5efefbddadd0b12d1fda7221fe49d7fc05b1042a4352c514cd2e7af2f90`.
El validador independiente del piloto también devolvió BLOCKED:
`calendar, scale dictionary, current pricing and immutable freeze required; no API called`.

Para repetir la preparación, sin entrenamiento ni llamadas:

```powershell
python scripts/data/build_research_regime5.py --output outputs/thesis-repair/regime5_operator_readiness.json
```

El HMM histórico de código conserva SHA
`7dd697b105f43b02d587d34434ac92f427811ecc3a8229d5395c3189061d0caa`.
También se verificaron intactos el seed M5, el HMM congelado y el scaler congelado.
El esquema nuevo tiene SHA de archivo
`deab891afe3707fc2d43164312ff925e88f1a207c2594281a7a7922253f1d203`;
su hash de contrato canónico es distinto del hash de serialización del archivo.

Queda fuera, y por qué:

1. Dataset de mercado nuevo: faltan ledger de publicaciones/first-seen autenticado,
   políticas por serie y el bundle final resultante. La unión temporal está probada;
   la disponibilidad histórica no se inventó ni se declaró certificada.
2. Diez corridas PPO: faltan sanidad vigente para 38 entradas, freeze completo de
   experimento/receta y reconciliación del registro por el operador. El reporte
   sintético viejo falla por identidad de código obsoleta; no se cambió su hash
   para reutilizarlo.
3. Piloto DeepSeek/Azure: conserva su configuración separada y sigue requiriendo
   identidad pública exacta, tarifas, calendario, diccionario y runtime congelados.
   No se leyó .env ni se llamó a proveedores.
4. Afirmación ejecutable: no hay nuevos bid/ask o fills verificados; las figuras
   históricas siguen siendo diagnósticas bajo costos supuestos, no forward real.

No se generaron nuevas curvas que aparenten resultados de esta versión. El paquete
retrospectivo enlazado se preserva con sus fechas/semillas/limitaciones originales.
No se tocaron datos fuente, artefactos/modelos antiguos ni registros de trials.
No hubo commit/push, review bilateral ni delegaciones; C050 es evidencia local de
worktree, no un estado DONE del protocolo ni cierre científico del goal.

Archivos nuevos: `src/research/{observation_contract,regime5_bundle,regime5_hmm,research_readiness}.py`,
`scripts/data/build_research_regime5.py`, `config/research/{research_regime5_v1.yaml,observation_regime5_v1.json}`
y `tests/regression/test_regime5_{contract,bundle,adversarial,historical_boundary}.py`.
Modificados: `session_gym.py`, `live_spec.py`, `publication_join_v2.py`,
`llm_forward/{arms/ppo_stream,stream_runner}.py`, `thesis_train_ppo.py` y
`test_forward_arms_parity.py`; este informe y los canales de coordinación
`CODEX-STATUS`, `LEASES`, `CONTRACTS` e `INBOX-CLAUDE`.
No se eliminaron archivos. El módulo histórico `regime_hmm.py` quedó restaurado
al estado que tenía antes de esta tarea, conservando los cambios previos del usuario.
