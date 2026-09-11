---
kind: audit
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-09-10
supersedes: []
code_anchors:
  - scripts/diagnostics/audit_thesis_rl_integrity.py
  - src/research/features.py
  - src/research/dataset.py
  - src/research/session_gym.py
  - src/research/session_env.py
  - scripts/analysis/thesis_statistics.py
---
# EXP-TESIS-RL-01: auditoría independiente de datos, PPO y evidencia científica

Base auditada: `c43b72f673b7a2314410be4749982092b336b7d7`. Mediciones: 10 de septiembre de 2026, America/Bogota. El estado IMPLEMENTED corresponde a esta auditoría, no a la corrección de los defectos. No se entrenaron modelos ni se eligieron variantes nuevas. Se inspeccionaron retrospectivamente datos y resultados ya publicados; estos períodos no recuperan independencia para investigaciones futuras.

Entregables: [evidencia con hashes](exp-tesis-rl-01-evidence-2026-09-10.json) y [diagnóstico reproducible](../../scripts/diagnostics/audit_thesis_rl_integrity.py). El texto pegado por el operador fue contrastado independientemente; su enlace externo no pudo recuperarse. Las réplicas de políticas con macro desplazada y latencia citadas allí no tienen artefactos entregados y no se certifican aquí.

## 1. Veredicto y alcance de la conclusión

**Esta receta de PPO perdió dinero bajo la contabilidad publicada. No se ha demostrado que USD/COP sea imposible de negociar con beneficio, ni que estos datos permitan entrenar y evaluar una política ejecutable sin sesgos.** Hay errores de disponibilidad temporal y objetivo de entrenamiento comprobables, cambio de representación de precios, mezclas de instrumentos macro y problemas en la interpretación estadística.

Tampoco se puede defender el informe original sin corregirlo. La H1 registrada tenía otro comparador; el holdout ya estaba parcialmente observado; bruto y neto se expresaron con convenciones distintas; parte del stress no utilizó la estrategia; y una supuesta caída de correlación no ocurrió en los artefactos. La contribución científica defendible es una evaluación negativa de una implementación concreta, con auditoría de sus límites y un protocolo de réplica corregida. Una auditoría negativa no demuestra ausencia de una señal latente ni fracaso universal del algoritmo.

### Resultados económicos reproducidos

Todas las cifras de esta tabla usan la media diaria de cinco semillas por configuración. Una media de retornos no es una política de consenso de acciones; tampoco equivale a la semilla típica.

| Bloque y configuración | Sesiones | Bruto: suma de retornos | Bruto: compuesto | Neto: compuesto |
|---|---:|---:|---:|---:|
| Selección, PPO régimen original | 234 | No recalculado aquí | No recalculado aquí | −29,851878 % |
| Holdout, PPO régimen refit | 584 | +27,949692 % | +31,844923 % | −54,865770 % |
| Holdout, PPO backbone refit | 584 | +31,510752 % | +36,574335 % | −56,436397 % |

Los diez resultados netos por semilla del holdout son negativos. El bruto compuesto tampoco es positivo en todas: backbone semilla 456 tiene **−0,021146 %**, aunque su suma aritmética sea +1,269209 %. La diferencia es arrastre por volatilidad, no un error de resta.

Reprecio de las mismas acciones publicadas, sin entrenar ni seleccionar, mediante `net_d(k) = gross_d − k × cost_d`:

| Configuración | Costos ×1 | Costos ×2 | Costos ×3 |
|---|---:|---:|---:|
| PPO régimen, retorno compuesto | −54,865770 % | −84,586632 % | −94,749123 % |
| PPO backbone, retorno compuesto | −56,436397 % | −86,146026 % | −95,607540 % |

Este stress conserva posiciones; no describe una política reentrenada ante otros costos. Reproduce el fracaso económico observado, pero no calibra un mercado real.

## 2. Checklist de datos e ingeniería

PASS significa que se verificó la propiedad concreta. No extiende la certificación a otras capas. PENDIENTE significa evidencia insuficiente, no necesariamente un defecto del dato.

| Control | Estado | Evidencia y límite |
|---|---|---|
| Identidad del material auditado | PASS | Hash de cada input del diagnóstico, del propio script y commit base en el JSON. No demuestra qué archivo exacto se usó originalmente antes de su incorporación a git. |
| Tipos numéricos OHLC | PASS | 99.714 filas; cero OHLC nulos, infinitos, no positivos o incompatibles con máximo/mínimo. |
| Duplicados, timezone y grilla | PASS estructural | Cero duplicados `(time,symbol)`, cero fuera de sesión y grilla de cinco minutos; timezone America/Bogota. No prueba que el proveedor sellara apertura o cierre de barra. |
| Integridad de sesión | PARCIAL | Máscara verifica conteo mínimo y rango horario; no valida por sí sola exactamente 60 timestamps únicos, OHLC o todas las reglas prometidas. |
| Calendario | PARCIAL | Exclusión colombiana implementada; unión Colombia/EE. UU. de la spec no implementada. 2023-07-04 es válido. El calendario correcto debe pertenecer al instrumento y venue. |
| Calidad de OHLC consistente entre bloques | FALLA | O=H=L=C en 82,413703 % global; 100 % durante 2020–2022 y 0 % durante 2026. |
| Volumen | EXCLUIDO justificadamente | Todos los volúmenes son cero. Esto indica ausencia de volumen informado; no que no hubiera negociación. |
| Confirmación independiente con diario | FALLA como evidencia externa | Los 1.723 OHLC diarios coinciden exactamente con agregar el mismo M5; el productor diario deriva de la tabla intradía. |
| Igualdad con una fuente oficial | NO CERTIFICADA | El M5 no fue reconciliado contra ticks/cotizaciones oficiales independientes. TRM y cierre intradía son objetos diferentes. |
| Raw descargado → números limpios | PENDIENTE | Inputs configurados de fusión/scrapers no están materializados en esta copia; no hay conciliación completa de strings originales, rechazos y valores finales. |
| Separadores decimales/miles | RIESGO de parser | `errors='coerce'` permite pérdidas silenciosas; el helper no interpreta de forma general coma decimal/miles. No se demostró qué filas de la tesis fueron afectadas. |
| Identidad económica de macro | FALLA/PENDIENTE por serie | Brent tiene un parche spot sobre futuros. DGS2 tiene fuente primaria Investing y no coincide exactamente con FRED. DXY permite fallback Fed Broad, índice distinto; uso efectivo del fallback no reconstruido. |
| Disponibilidad macro antes de operar | FALLA comprobada | Perturbar valores fechados en d cambia el contexto de d; no hay `available_at` por observación ni desplazamiento suficiente por publicación. |
| Relleno de macro ausente | FALLA de integridad | Archivo inexistente devuelve todas las features macro en cero, sin fallo ni indicador de ausencia. |
| Escalador ajustado solo en desarrollo | PASS reproducible | Media y escala recalculadas con desarrollo coinciden exactamente con frozen y portable. |
| Dataset portable vs features recalculadas | PASS reproducible | Diferencia máxima 0 en mercado y macro para los tres bloques, usando float32 como entrenamiento. Esto también reproduce la fuga, no la corrige. |
| Saturación de features | FALLA de estabilidad | Parkinson y Garman–Klass son cero en todo el desarrollo efectivo y saturan ±5 en 42,625571 % del holdout. |
| Macro con escala comparable | PARCIAL | Tres features macro sin estandarización, de magnitud menor. No estandarizar no prueba por sí mismo fracaso; evaluarlo exige un experimento separado. |
| Ventanas intradía | DISCREPANCIA de contrato | Las features calculadas sobre sesiones concatenadas incluyen el salto nocturno; 1.378 primeras barras tienen logret_1 distinto de cero. Es pasado disponible, pero no coincide con “solo retornos intra-sesión”. |
| Faltantes/outliers | PARCIAL | Sesiones incompletas se excluyen; controles de outlier prometidos no están implementados. NaN/Inf de features se vuelven cero. Falta bitácora de causas y afectación. |
| Trazabilidad de sesiones descartadas | FALLA de reporte | Máscara 558/235/584; entrenamiento efectivo 499/234/584. Desarrollo empieza 2020-05-07; falta el shock inicial COVID. El JSON nuevo enumera todas las fechas perdidas. |
| Frescura para entrenar ahora | NO CERTIFICADA | Inputs terminan el 2026-08-24. Sirven como snapshot histórico declarado; no se certifican como feed fresco para entrenamiento/inferencia actuales. |

### El cambio de representación no es pequeño

| Año | Barras | O=H=L=C |
|---|---:|---:|
| 2020 | 14.887 | 100 % |
| 2021 | 13.811 | 100 % |
| 2022 | 15.118 | 100 % |
| 2023 | 15.556 | 89,9781 % |
| 2024 | 15.517 | 83,5793 % |
| 2025 | 14.814 | 76,8462 % |
| 2026 | 9.540 | 0 % |

Sobre las sesiones efectivamente construidas, Parkinson/Garman–Klass saturan 0 % en desarrollo, 11,225071 % en selección y 42,625571 % en holdout. El “21 %” global del informe diluye el problema del bloque evaluado. No puede recuperarse un máximo/mínimo no observado mediante limpieza; hace falta otra fuente o una representación común que no dependa de esa información ausente.

No todos los indicadores ATR son idénticos a Parkinson: true range también usa cambios del cierre previo. No se justifica eliminar indiscriminadamente todos los ATR por el solo diagnóstico de OHLC plano. El conjunto común de features se debe diseñar por semántica y congelar antes de evaluar rentabilidad.

### Fuentes oficiales: diferencia numérica no siempre significa mal parseo

El [productor del seed diario](../../airflow/dags/l0_seed_backup.py), líneas 257–285, agrega el intradía. Su igualdad exacta con el M5 es conciliación interna. La [TRM según BanRep](https://www.banrep.gov.co/es/glosario/tasa-cambio-trm) deriva de operaciones del día anterior; no debe exigirse que sea igual al último cierre de cinco minutos. Para cada contraste hay que igualar instrumento, fecha de observación/validez, unidad, metodología y vintage.

La [configuración macro](../../config/macro_variables_ssot.yaml) declara Investing como fuente primaria de la columna llamada DGS2 y FRED como fallback. Cotejos puntuales con la [tabla oficial DGS2](https://fred.stlouisfed.org/data/DGS2):

| Fecha | CLEAN | FRED DGS2 |
|---|---:|---:|
| 2020-01-02 | 1,571 | 1,58 |
| 2025-09-24 | 3,598 | 3,57 |
| 2025-09-25 | 3,663 | 3,64 |
| 2025-12-19 | 3,485 | 3,48 |
| 2025-12-22 | 3,509 | 3,44 |

El yield de un instrumento cotizado no equivale necesariamente al Treasury constant maturity de FRED. Antes de llamarlo DGS2 se debe resolver esa identidad. El H.15 tiene además su propia hora de publicación, posterior a la apertura colombiana; una fecha diaria no es un sello de disponibilidad. [Federal Reserve, H.15](https://www.federalreserve.gov/releases/h15/).

El [corrector de Brent](../../scripts/ops/fix_brent_corrupt_block.py), líneas 50–56, declara que injerta spot FRED en futuros. El [backup de reparación](../../data/backups/macro_fixes/brent_corrupt_20250925_20251219.csv) contiene 59 fechas y coincide exactamente con CLEAN en esas celdas. El cotejo oficial de [DCOILBRENTEU](https://fred.stlouisfed.org/data/DCOILBRENTEU) confirma los extremos reemplazados: 2025-09-25=70,48 y 2025-12-19=61,35. Fuera del tramo, 2025-09-24 CLEAN=69,31 vs spot=69,64; 2025-12-22 CLEAN=61,58 vs spot=62,22. Se corrigió una corrupción grande, pero falta gestionar el cambio de instrumento y sus retornos de frontera.

El fallback DXY→DTWEXBGS tampoco es una equivalencia: [ICE DXY](https://www.ice.com/forex/usdx) y [Fed Broad](https://fred.stlouisfed.org/series/DTWEXBGS) tienen canastas/metodologías distintas. La posibilidad está en configuración; la ausencia de procedencia por celda impide establecer cuándo actuó. El [IBR de BanRep](https://www.banrep.gov.co/es/glosario/indicador-bancario-referencia-ibr) se publica durante la sesión colombiana; deben identificarse plazo, tasa nominal/efectiva y hora histórica de publicación antes de usarlo a la apertura.

### Limpieza y causalidad: lo que sí se probó

La [limpieza macro](../../data/pipeline/04_cleaning/run_clean.py) usa coerción numérica y precedencia de archivos recientes sobre históricos. Sin raw original y conteos de rechazo no se puede certificar preservación completa. Hay 693 fechas de sábado/domingo desde 2020 en CLEAN; ese hecho no prueba interpolación incorrecta, porque otros productores también escriben el archivo. Se requiere procedencia por celda y límite de edad del ffill por serie, nunca bfill o interpolación con observaciones futuras.

Prueba en memoria sobre 2023-06-15, ejecutando la función real [attach_macro_features](../../src/research/features.py), líneas 263–291:

```text
dxy_ret_prev publicado por la función = -0.007882044864882502
retorno DXY de observaciones estrictamente anteriores = -0.0038984891835504927
al cambiar solo la fila macro de ese mismo día:
  max_abs_change_context = 0.09531017980432499
  causality_gate_pass = False
archivo macro ausente -> contexto todo cero = True
```

El nombre `ret_prev` no corresponde al valor calculado. `merge_asof(backward)` incluye igualdad de fecha; no convierte fecha de observación en fecha de publicación. Incluso un `shift(1)` puede ser insuficiente para una serie publicada con retraso mayor o revisada: el criterio operativo es `available_at <= decision_cutoff`, respetando la versión que entonces existía.

La caída 2022-06-01 de 3.976,50 a 3.784,00 en una barra (−4,840941 %) pasa la máscara. Es una anomalía para investigar, no una licencia para borrar pérdidas grandes. Debe contrastarse con raw independiente antes de decidir si es mercado o error.

## 3. PPO: errores demostrados y causas aún no identificadas

**Reward terminal:** [SessionTradingEnv.step](../../src/research/session_gym.py), líneas 145–185, informa el resultado de `run_session` con cierre cobrado, pero devuelve el reward de barra sin ese costo. El [test actual](../../tests/regression/test_session_gym_parity.py), línea 84, exige ese desajuste. Contabilidad compartida y pruebas verdes no garantizan alineación del objetivo.

**Precio del cierre terminal:** [run_session](../../src/research/session_env.py) pasa `c[:59]` a [session_costs](../../src/research/cost_model.py). El costo final usa c58 y sigma58, aunque la cronología declara liquidación en c59. Fixture sintética: todos los cierres en 4.000 salvo el último en 4.400, exposición +1 y spread 3:

```text
terminal_actual = 0.0005
terminal_con_precio_y_sigma_de_barra59 = 0.003205913352872407
suma_reward_sin_escala = 0.09950000000000009
daily_return_reportado = 0.09900000000000009
```

La fixture aísla el índice erróneo; no estima su impacto real. La réplica independiente sobre las acciones guardadas encontró un efecto pequeño del índice de cierre, distinto de la omisión sistemática en reward. No atribuirle por sí solo la pérdida total.

**Optimización:** flat está en el espacio y obtiene cero. Eso prueba que existe una política sin pérdidas; no demuestra una política estrictamente rentable. Los PPO negativos no maximizan el retorno económico realizado, pero el optimizador persigue reward descontado, normalizado y con entropía. Identificar el mecanismo de churn requiere un experimento controlado.

La [receta](../../scripts/analysis/thesis_train_ppo.py) fija γ=0,98, ent_coef=0,01 y reward×100; VecNormalize usa el mismo γ, por lo que no hay desalineamiento 0,99/0,98 en ese wrapper. `0.98**58 = 0.3098221`: existe descuento temporal; su efecto causal sobre las acciones es una hipótesis. Las penalizaciones κ_turn y λ_dd prometidas no están implementadas. La pérdida in-sample no basta para culpar exclusivamente al algoritmo.

Los [índices de entrenamiento](../../data/thesis/ppo/index.json) y [refit](../../data/thesis/ppo/index_refit.json) también corrigen generalizaciones del informe: régimen original tiene 4/5 semillas positivas en desarrollo, backbone original 1/5; todos pierden selección. Todos los refit pierden desarrollo; backbone refit tiene dos semillas positivas en selección in-sample (+0,846958 % y +3,233195 %).

**HMM:** la implementación por última posterior del prefijo es causal y desplaza el contexto al día siguiente. Dos comparaciones prefix/prefix de sus tests son tautológicas, aunque otro control sí lo distingue del smoothing global. Los tests sintéticos del HMM portable pasan. Eso no valida las tres features macro, que no están cubiertas por el test de mercado.

**Reproducibilidad del dataset:** la cache de [dataset.py](../../src/research/dataset.py) identifica máscara y nombres/grupos del schema, no todo el código de transformación ni el contenido macro/OHLC. Un arreglo semántico puede conservar esa clave y cargar un dataset antiguo. El portable no incorpora una comprobación completa de identidad al leer. Una corrección debe generar una versión nueva con hashes de inputs, fórmulas, partición, scaler y HMM; no reusar silenciosamente los checkpoints de 39 features por tener la misma dimensión.

**Forward:** [live_spec](../../src/research/live_spec.py) exige sesión completa; [ppo_arm](../../src/research/llm_forward/arms/ppo_arm.py) genera las acciones de toda la sesión y después fija `sealed_before_open=True`, igualando cutoff/open a `now`. No basta permitir una barra: hacen falta decisiones y sellos por barra, estado persistente y verificación de disponibilidad. La fecha derivada de logical_date en el [DAG](../../airflow/dags/research_forward_arms.py) merece prueba con data_interval_end; no se ejecutó un DagRun real en esta auditoría. Los registros stub retrospectivos no constituyen evidencia de negociación forward.

## 4. Correcciones necesarias en metodología y redacción

1. **H1 cambió de identidad.** El [prerregistro](../../.claude/specs/planes/06-PRE-REGISTRATION.md), línea 119, dice PPO régimen frente a `baseline_matched`. [RESULTADOS](../../.claude/specs/planes/06-RESULTADOS.md), línea 119, llama H1 a PPO frente a flat. El supervisado no se construyó. Escribir “subdesempeño frente a flat”; no “rechazo de la H1 originalmente registrada”.

2. **La independencia del holdout es limitada.** [partition.yaml](../../config/research/partition.yaml) y [apertura](../../outputs/thesis/holdout_opening.json) declaran que 2025 ya fue examinado por H5. El histórico git de firma/partición/apertura los incorpora en septiembre; fechas internas de agosto son declaraciones, no prueba externa del orden. La apertura guarda nombres de modelos, no sus hashes y el de la firma. Hay rutas de reevaluación y fuerza. Se puede reconocer la intención de prerregistro, no certificar una apertura única blindada de todo acceso. DSR no restaura independencia a un holdout observado.

3. **La correlación no cayó de 0,97 a 0,58.** Los valores recalculados y los JSON originales son selección **0,5852953457**, holdout **0,5821436208**. Retirar explicaciones de potencia/refit basadas en aquella caída.

4. **p no es cero medido con precisión infinita.** La réplica del bootstrap reproduce cero excedencias entre 10.000 simulaciones para PPO régimen frente al flat convencional. Reportar ese hecho y el IC; no afirmar precisión arbitraria `p<0,0001`. Revisar la construcción del p y su resolución Monte Carlo. Sharpe de flat igual a cero es convención; matemáticamente su varianza cero deja el cociente indefinido.

5. **Incertidumbre del algoritmo omitida.** La réplica de H2 da selección ΔSR=2,119757, IC [0,238054; 4,086521], p=0,0276; holdout ΔSR=0,431870, IC [−0,882825; 1,860640], p=0,5358. Estos contrastes usan la media de cinco series. No incorporan toda la incertidumbre de reentrenar PPO. Un bootstrap jerárquico debe conservar dependencia temporal y emparejamiento de fechas; cinco semillas permiten describir dispersión, no certeza sobre cualquier entrenamiento futuro.

6. **Fuga común no garantiza H2 invariante.** Las dos configuraciones pueden explotar o interactuar con la macro de forma distinta. Quitarla exige medir ambas en un protocolo legítimo. El resultado negativo histórico se conserva como descriptivo; no se hereda su p ni su comparación al sistema corregido.

7. **PBO reproduce, con alcance distinto.** 0,116 en selección y 0,2113 en holdout provienen de CSCV sobre diez políticas por semilla. Mide selección hipotética entre esas políticas; no certifica el procedimiento candidato/OOF prometido. No es correcto decir que el número carece de significado, ni usarlo como prueba del procedimiento que no se ejecutó.

8. **Trials:** hay 115 documentados. Cadencias/sensibilidades y otros intentos requieren conciliación con las reglas del activo. “N≥125” no está demostrado por un inventario verificable. DSR≈0 para estos retornos no establece imposibilidad del mercado; cualquier futuro claim positivo requiere un N actualizado y reconciliado.

9. **La atribución diaria es retrospectiva.** Se reproduce +48,907890 puntos direccionales y −20,958197 puntos de residuo para régimen, usando `mean(w_d) × sum(r_intradia_d)`. Pero esa exposición media incorpora acciones posteriores al inicio del día. No prueba que pudiera escogerse a las 08:00 ni que una regla diaria replique el componente. El mapeo de E7 derivado de una figura observada es una hipótesis para datos futuros.

10. **Costos y microestructura:** alfa 0,6676 y costo 2,5505 son por unidad de turnover, no por operación. La comisión aislada 0,5 sí es menor que ese alfa; falla al agregar otros costos. “1 pip=1 COP” es la unidad interna: antes de convertir un quote externo hay que conocer tick/pip contractual del venue. No está verificada una convención universal de 0,01 COP ni un rango universal de spread CFD. Una autocorrelación negativa es compatible con microestructura; sin bid/ask y tipo de precio no demuestra bid-ask bounce ni llena órdenes.

11. **Evidencia auxiliar incompleta:** `statistics_development` etiqueta backbone_mean5 pero contiene una semilla; `decomposition_selection` corresponde a un diagnóstico refit, no a la selección original. El nuevo script evita mezclarlos. Los `statistics_*.json` contienen literales NaN; la nueva evidencia se serializa con `allow_nan=False`.

## 5. Qué causa qué: resultado de la auditoría

| Nivel causal | Comprobado | Lo que aún exige intervención controlada |
|---|---|---|
| Observación | Macro del mismo día llega al agente; fuentes macro pueden cambiar de identidad; OHLC cambia de representación. | Cuánto del bruto desaparece al corregir cada componente. |
| Objetivo | Reward no cobra liquidación terminal; objetivo descontado/entropía difiere del retorno final. | Qué parte del churn proviene de reward, normalización, exploración o presupuesto. |
| Evaluación | Costos publicados superan el bruto; stress original usa otro P&L; suma/compuesto se mezclaron. | Rentabilidad neta con costos y fills medidos en un venue alcanzable. |
| Inferencia | Comparador H1 cambiado; medias de semillas sustituyen parte de la incertidumbre; holdout parcialmente observado. | Una conclusión confirmatoria sobre la implementación corregida en datos nuevos. |

La causa inmediata de la pérdida contable es que los costos acumulados superan el bruto. La auditoría no permite asignar porcentajes causales de esa pérdida a cada defecto ni concluir que corregirlos produzca beneficio. Mejorar datos puede incluso reducir el bruto si elimina fuga o artefactos.

## 6. Programa científico propuesto, sin ejecutarlo ni gastar trials aquí

El orden separa reparación de medición, sanidad del aprendizaje y búsqueda de señal. Las decisiones de modelado se pre-registran y se contabilizan antes de mirar resultados. No se heredan como gratuitos los grids descritos en el informe.

| Estudio | Diseño concreto | Criterio para continuar | Qué permite concluir |
|---|---|---|---|
| A. Libro de datos | Por observación: instrumento, proveedor, payload/hash, valor raw, regla de parseo, valor normalizado, unidad, observed_at, published_at, ingested_at, vintage y calidad. Reconciliar las series usadas, no todo el catálogo. | Rechazos y transformaciones contabilizados; diferencias explicadas por identidad/fecha; ningún input futuro ni relleno silencioso. | Dataset auditable. No rentabilidad. |
| B. Costo y posibilidad de ejecución (E1) | Captura de bid/ask y recepción en un instrumento realmente accesible; contrato, tick, tamaño y comisión explícitos. Medir por hora, día y tamaño; separar spread cotizado, efectivo y slippage. | Cobertura/ausencias y cuantiles reportados. Un piloto de 20 sesiones es descriptivo; no garantiza cubrir regímenes extremos. | Si existe un problema de trading ejecutable y qué costos modelar. |
| C. Sanidad contable/causal | Tests adversariales de macro en d/futuro, cierre en c59, igualdad reward económico antes de shaping, futuros que no cambian pasado, cache invalidada y replay streaming. Fixtures con precio constante, tendencia conocida y sin señal. | Cero violaciones de identidad/causalidad; en precio constante con costo positivo, flat es el óptimo económico conocido. | Entorno correcto antes de estudiar el mercado. |
| D. Sanidad PPO (E6 corregido) | Solo desarrollo/sintéticos, cinco semillas. Objetivo base congelado; después una variable por contraste (entropía, normalización o presupuesto), con todas las variantes declaradas y registradas. | Aprender la solución conocida en el entorno sintético y explicar las curvas de aprendizaje; fallo implica investigar receta/implementación. | Capacidad del algoritmo bajo esa receta. No ausencia de alfa del mercado. |
| E. Réplica sin fugas (E2) | Nueva identidad de datos/schema/modelo; correcciones de ingeniería documentadas. Describir resultados por semilla y cartera por separado; ejecución y lags predefinidos; tests de información estrictamente anterior. | Evaluación exploratoria histórica identificada como tal; decisión confirmatoria solo en nuevo período sellado, con costos de B. | Si la señal sobrevive en una implementación válida. |
| F. Baselines y horizonte (E3/E7) | Reglas fijas y supervisado matched con información/costos/horario equivalentes. Horizonte y abstención fijados antes de evaluación; calibración de probabilidades dentro de desarrollo. | Superioridad frente a flat, exposición emparejada y regla simple, con IC, todos los intentos y stress ×2. | Si RL aporta frente a soluciones más sencillas. |
| G. Forward de medición (E8) | Sellar cada decisión con información realmente disponible; registrar abstenciones, acciones, quotes y fills, incluso días sin operaciones. | Integridad del ledger y cobertura; tamaño final obtenido de potencia/MDE predefinidos, dependencia y frecuencia operativa. | Datos nuevos para la prueba; no beneficio garantizado. |

Modificaciones al programa original:

- No usar selección/holdout ya observados como prueba nueva de una corrección motivada por ellos. Un replay corregido allí es diagnóstico retrospectivo; el juez confirmatorio es futuro.
- Un surrogate intradía debe declarar qué preserva: suma simple no preserva necesariamente retorno compuesto. La permutación debe reconstruir un camino de precios factible y mantener contexto/calendario; un residuo no nulo no demuestra automáticamente señal o fraude.
- Eliminar umbrales universales sin potencia calculada: ni 85 sesiones hacen toda hipótesis imposible, ni 250 la vuelven suficiente. Mantener la prohibición del repo de Sharpe/p con menos de 20 operaciones; distinguir sesiones y operaciones.
- No financiar Transformers, RL offline o ejecución pasiva antes de tener datos y ejecución observables. Tampoco declarar esos métodos imposibles en general. Un book negociable, fill y selección adversa son requisitos empíricos para evaluar órdenes limitadas.
- Una estrategia diaria basada en el componente ex post +49 % o en un régimen escogido mirando figuras no está validada. Debe congelarse y probarse en un período nuevo.
- Ningún éxito exploratorio se promociona por esta auditoría. La [constitución](../../.claude/rules/quant-constitution.md) exige trials completos, DSR, baselines y stress; no es un mecanismo para rescatar un holdout contaminado.

## 7. Referencias científicas y cómo se aplican

Estas fuentes orientan el diseño; ninguna demuestra que PPO sea rentable o no en USD/COP.

| Fuente primaria | Aplicación concreta en la tesis |
|---|---|
| Henderson et al., [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) | Variabilidad por semilla, hiperparámetros e implementación: publicar dispersiones y condiciones completas; evitar inferir el algoritmo desde una corrida o una media. |
| Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | Estimadores robustos e intervalos en RL con pocas corridas. Adaptar el remuestreo a fechas financieras dependientes; no tratar las observaciones semilla×día como independientes. |
| Patterson et al., [Empirical Design in Reinforcement Learning](https://arxiv.org/abs/2304.01315) | Separar variabilidad, selección de hiperparámetros y comparación final; usar problemas ilustrativos con solución conocida para distinguir fallos de implementación y de aprendizaje. |
| Bailey y López de Prado, [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Ajustar claims de rendimiento por selección y no normalidad con universo de intentos justificado. DSR no corrige datos futuros ni precios no ejecutables. |
| Bailey et al., [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | Construir CSCV sobre el conjunto de candidatos y procedimiento que se pretende evaluar; especificar qué significa seleccionar una columna. |
| Roll, [A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1984.tb03897.x) | La covarianza serial puede reflejar costos bajo supuestos; una autocorrelación negativa aislada en cotizaciones indicativas no identifica el mecanismo ni mide fills. |

Texto defendible para la tesis:

> En los artefactos auditados, las políticas PPO evaluadas presentan rendimiento neto negativo frente a la abstención bajo el modelo de costos especificado. La atribución del rendimiento bruto a una señal negociable no está establecida, debido a problemas de disponibilidad temporal, heterogeneidad de la representación OHLC y falta de validación de ejecución. La auditoría identifica además divergencias entre objetivo entrenado, protocolo registrado y evaluación publicada. Estos resultados delimitan la implementación estudiada y motivan una réplica corregida; no demuestran imposibilidad de rentabilidad del activo ni de aprendizaje por refuerzo en general.

## 8. Verificación y reproducción

Diagnóstico de datos/resultados, sin entrenamiento:

```powershell
python scripts/diagnostics/audit_thesis_rl_integrity.py --output outputs/thesis-audit-reproduction.json
```

Elegir una ruta nueva: el script no sobreescribe evidencia. `exit 0` indica que terminó de medir; el propio resultado registra `macro causality gate=False`. El JSON adjunto conserva versiones, hashes, fechas descartadas, tasas de clipping, resultados por semilla, alineación de fechas y errores de paridad. No carga checkpoints; el portable se lee con whitelist de clases de datos.

Salida de la generación final:

```text
Evidence saved: docs\analysis\exp-tesis-rl-01-evidence-2026-09-10.json
Rows=99714; macro causality gate=False
exit 0
```

Tests existentes ejecutados aisladamente, sin cargar fixtures de entrenamiento:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest --confcutdir=tests/regression -p no:cacheprovider -o addopts='' tests/regression/test_session_env_and_costs.py tests/regression/test_session_gym_parity.py -q
python -m pytest --confcutdir=tests/regression -p no:cacheprovider -o addopts='' tests/regression/test_regime_portable.py -k 'not real_sessions' -q
python -m pytest --confcutdir=tests/regression -p no:cacheprovider -o addopts='' tests/regression/test_research_features_are_causal.py tests/regression/test_evaluation_mask.py -q
```

```text
28 passed, 1 warning in 3.76s
4 passed, 1 deselected, 1 warning in 2.26s
16 passed, 1 warning in 2.66s
warning: PytestConfigWarning: Unknown config option: asyncio_mode
```

Estos 48 tests verdes no certifican la ausencia de los defectos encontrados: algunos no cubren macro y otro codifica el reward sin cierre. La advertencia procede de desactivar plugins automáticos en la ejecución aislada. No se ejecutó la prueba de HMM que requiere refit ni un DagRun real.

Los contrastes bootstrap/PBO fueron recalculados separadamente por el revisor estadístico sobre arrays publicados, reproduciendo sus cifras con salida 0; el diagnóstico adjunto no los recalcula y no los presenta como inferencia corregida. Su import convencional falló por `ModuleNotFoundError: No module named 'psycopg2'`; la réplica numérica utilizó módulos puros cargados aisladamente. Esa réplica no certifica el runtime integral de la aplicación.

Gates finales, después de regenerar los dos índices afectados:

```text
generate_inventory.py --check: inventory OK
generate_doc_indexes.py --check: document indexes OK
check_knowledge_links.py: links OK
check_knowledge_graph.py: knowledge graph OK
pytest gobernanza: 1092 passed, 1 warning in 11.20s
ruff check --no-fix scripts/diagnostics/audit_thesis_rl_integrity.py: All checks passed!
```

La corrida de gobernanza incluyó `test_knowledge_frontmatter`, `test_knowledge_inventory`, `test_knowledge_autoload_budget`, `test_scripts_layout`, `test_contract_mirrors`, `test_knowledge_links` y `test_knowledge_graph`, con las mismas opciones de aislamiento arriba. Baseline previo al informe: esos gates también pasaban. Al añadirlo, el gate de índices devolvió `DOCUMENT INDEX DRIFT` para `docs/INDEX.md` y `docs/analysis/README.md`; se regeneraron mediante la herramienta oficial y el check posterior pasó. Ruff encontró detalles de estilo durante la elaboración del diagnóstico, corregidos antes de generar la evidencia final y su hash.

Hubo un intento inicial de pytest con 8 errores `PermissionError [WinError 5]` en temporales; la repetición autorizada resolvió esos errores. El arranque de Python dentro del sandbox también requirió ejecución autorizada fuera de él. Ninguno se presenta como test verde. Los revisores independientes de datos y estadística contrastaron el informe y sus poblaciones; esa revisión no constituye una aprobación de producción.

## 9. Fuera de alcance y entrega

No se corrigieron los modelos/configuraciones congelados ni las fuentes; se preservó el experimento para auditarlo. No hubo commit, push, modificación del registro de trials, cambio de contratos ni promoción. No se obtuvieron credenciales de broker ni historial bid/ask completo, no se certificó la cadena raw completa y no se ejecutaron E1–E8. El desajuste del objetivo de entrenamiento está identificado; su efecto sobre el aprendizaje y la existencia de una alternativa rentable permanecen abiertos.

Archivos creados: este informe, el JSON de evidencia, el diagnóstico y el índice generado `docs/analysis/README.md`. Modificados: `docs/INDEX.md`, `.claude/coordination/CODEX-STATUS.md` y `.claude/coordination/LEASES.md`. Sin renombrados ni eliminación de datos/modelos del usuario. Las referencias relativas del informe pasaron el gate de enlaces; los índices conectan el informe al árbol documental. Las cifras citadas sin réplica de políticas se identifican como no certificadas, y las recomendaciones metodológicas son propuestas, no resultados nuevos.
