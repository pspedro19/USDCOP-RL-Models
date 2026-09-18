---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-14
supersedes: []
code_anchors:
  - scripts/presentation/build_research_grade_thesis.py
  - scripts/presentation/build_thesis_manuscript.py
  - src/research/reporting_v2.py
---

# 3. Diseño e implementación

## 3.1. Arquitectura experimental y alcance de esta edición

El sistema experimental transforma observaciones de mercado en posiciones normalizadas y liquida sus consecuencias económicas mediante un motor común. La cadena consta de fuentes de precios y contexto macroeconómico, preparación temporal de variables, políticas de decisión, registro de posiciones y evaluación. La separación entre estos componentes permite distinguir un defecto de información de un defecto contable: una predicción puede ser causal y perder dinero, mientras que una curva rentable puede estar construida con información que no existía cuando se habría tomado la decisión.

La presente edición describe el conjunto de experimentos que puede reconstruirse desde el paquete retrospectivo v5. El objeto principal son las 226 sesiones comprendidas entre el 3 de enero y el 29 de diciembre de 2023, identificadas como selección. No se cambia su nombre a hold-out por disponer ahora de correcciones. Las modificaciones realizadas a partir de la inspección del período impiden utilizarlo como confirmación independiente del diseño corregido. La finalidad de esta edición es presentar evidencia económica y limitaciones verificables, no restaurar retrospectivamente un experimento que no se ejecutó bajo las condiciones originalmente propuestas.

El PDF original y el documento de presentación sirven como antecedentes de estructura y propósito. El segundo declara expresamente que sus cifras eran sintéticas. Por ello, ni las rentabilidades positivas ilustrativas ni sus pruebas estadísticas se conservan como resultados. La sustitución afecta también a la metodología: se documenta lo realmente ejecutado, en vez de mantener descripciones que implicarían la existencia de corpus, validaciones o variantes no respaldadas por los registros.

| Componente | Propuesta original | Evidencia utilizada en esta edición |
| --- | --- | --- |
| Evaluación final | Últimos doce meses, representados como 2025 | Selección 2023; diagnóstico retrospectivo |
| Lenguaje | Noticias y datos numéricos; DeepSeek y Llama local | Decisiones históricas DeepSeek y Azure; prompt numérico |
| Híbrido | Modulación por sentimiento FinMA-ES | Veto por coincidencia de dirección con PPO |
| PPO | Red y presupuesto ilustrativos, diez semillas | Receta registrada flat_init_no_turn; cinco semillas por configuración |
| Regímenes | Tres categorías semánticas | HMM histórico de cinco estados y cuatro coordenadas almacenadas |
| Estadística | Amplia batería prevista | Contrastes e intervalos efectivamente calculados y reservas explícitas |

La existencia de un servicio o de un módulo de código no se considera evidencia de su participación en un experimento. Para esa atribución se requieren entradas, parámetros y resultados vinculados. Tampoco se equipara una prueba unitaria aprobada con un despliegue operativo. Este criterio gobierna las afirmaciones sobre ingeniería a lo largo de la memoria.

FIGURE: figures/10_arquitectura.png | Figura 3.1. Flujo lógico de la evaluación realmente documentada: fuentes, contexto, políticas y motor común. La figura no atribuye noticias a los prompts ni certifica un despliegue en vivo.

## 3.2. Fuentes, identidad y calidad de representación

El precio se expresa como pesos colombianos por dólar estadounidense. Una subida representa encarecimiento del dólar frente al peso; una posición larga obtiene contribución positiva cuando aumenta esa cotización. La fuente intradía utilizada en el carril evaluado procede del histórico de TwelveData. Las referencias a HistData o Dukascopy en la propuesta no se transforman aquí en una afirmación de cobertura observada. Una fuente adicional sólo integra la evidencia cuando existe un archivo identificable que pueda contrastarse.

Los datos se organizan en barras de cinco minutos y una ventana de sellos entre 08:00 y 12:55 en America/Bogota. La normalización de zona horaria precede a la identificación de la sesión. Se distingue el sello de la barra de la hora en que fue recibida y de la hora a la que habría terminado una inferencia. El backtest basado en cierres no demuestra que un operador pudiera ejecutar a ese mismo cierre después de observarlo: falta la reconstrucción de latencia y precios disponibles para una operación real.

Los controles estructurales comprenden unicidad temporal, grilla de cinco minutos, coherencia de apertura/máximo/mínimo/cierre, calendario y cobertura de sesiones. Sin embargo, satisfacer desigualdades OHLC no implica disponer de velas informativas. La descripción del histórico muestra que todas las barras de 2020, 2021 y 2022 tienen O = H = L = C. La fracción es aproximadamente 89,98 % en 2023, 83,58 % en 2024, 76,85 % en 2025 y 0 % en 2026. Son propiedades del archivo examinado, no estimaciones de liquidez del mercado completo. El cambio de representación limita la comparación de indicadores basados en rangos entre bloques.

Las variables Parkinson y Garman–Klass fueron retiradas del esquema corregido frente al esquema anterior. Su eliminación evita alimentar el modelo con magnitudes que eran degeneradas en un bloque y no en otro, pero no convierte automáticamente al resto de indicadores en representaciones homogéneas. ATR y demás features deben interpretarse en el contexto de esa calidad de precio. El volumen ausente o sin información no puede utilizarse para inferir liquidez ni se reemplaza por un volumen negociado ficticio.

Un informe de contraste diario TwelveData–Investing contiene 1.727 fechas comunes, mediana de diferencia absoluta relativa de 0,07782 %, percentil 95 de 0,551359 % y máximo de 5,192771 %. El máximo se conserva porque una medida central pequeña no descarta observaciones problemáticas. Esta evidencia respalda coherencia aproximada a frecuencia diaria, no identidad completa ni validación independiente de las 59 transiciones intradía de cada sesión. La TRM y una cotización de cierre no deben exigirse iguales sin reconciliar previamente definición, fecha efectiva y ventana de cálculo.

## 3.3. Integración de frecuencias y disponibilidad macroeconómica

El contexto macro combina cuatro identidades de series para construir tres variables: retorno previo de Brent, retorno previo de DXY y diferencial IBR–DGS2. Brent se identifica como la serie spot DCOILBRENTEU de FRED; no se intercambia silenciosamente por un futuro. DGS2 corresponde al rendimiento de deuda del Tesoro estadounidense a dos años, expresado en porcentaje. DXY se conserva con la identidad Investing declarada, sin sustituirlo por un índice amplio del dólar. IBR corresponde a la referencia declarada de Banco de la República. Coincidencia económica aproximada no equivale a identidad de instrumento.

| Entrada | Frecuencia de origen | Transformación y requisito temporal |
| --- | --- | --- |
| Precio USD/COP | Cinco minutos | Features calculadas sobre prefijos observables |
| Brent | Diaria | Log-retorno de dos observaciones admisibles anteriores |
| DXY | Diaria | Log-retorno de dos observaciones admisibles anteriores |
| IBR y DGS2 | Diaria | Diferencia de tasas con unidades reconciliadas |
| Posterior HMM | Contexto diario | Filtrado y uso causal; no suavizado con observaciones futuras |

La regla correctiva del carril histórico usa una unión temporal hacia atrás que excluye coincidencias de fecha exacta. Por tanto, una observación fechada el día d no ingresa a la apertura de la sesión d. El módulo compartido ordena series, aplica el límite de antigüedad definido en configuración y devuelve ausencia cuando no hay una observación admisible. No se utiliza una interpolación que consulte el futuro ni se considera que un cero artificial equivale a una macro desconocida. El límite histórico usa cómputo de días de lunes a viernes; no debe describirse como calendario de publicaciones completo si no incluye sus feriados específicos.

La regla T−1 constituye una protección contra cierres del mismo día, pero no prueba por sí misma disponibilidad histórica. Es necesario distinguir fecha del período observado, publicación, primera recepción y versión o vintage. Una serie revisada puede contener hoy un valor distinto del que existía entonces. Asimismo, comparar DXY contra la misma fuente con la que se construyó el parquet verifica reproducibilidad de extracción, no independencia. Estas reservas permanecen abiertas y limitan cualquier atribución de alfa al contexto macro.

La preparación posterior de 38 entradas incorpora una interfaz explícita de publicaciones y primera recepción. Esa preparación no se utiliza para recalcular las curvas principales de esta memoria ni se presenta como certificación del conjunto histórico. Se preservan ambas versiones para no atribuir al modelo evaluado observaciones que nunca recibió.

## 3.4. Representación, entrenamiento y regímenes

El esquema evaluado contiene 37 entradas: 25 variables de mercado y tiempo, cinco variables de estado de posición, tres variables macro y cuatro coordenadas de régimen almacenadas. Entre las primeras se encuentran retornos a distintos retardos, retorno acumulado de sesión, volatilidad realizada, ATR, distancias a medias exponenciales, MACD, pendiente, RSI, z-score y codificación horaria. El estado de posición identifica exposición previa, PnL no realizado, permanencia, drawdown y cambios. Se evita confundir las variables observadas con un conjunto de noticias o sentimiento que no formó parte de estos registros.

El escalado se ajusta en desarrollo y se congela para aplicar la misma transformación a selección. El recorte de observaciones a un rango acotado limita magnitudes extremas, pero no es una demostración de calidad: una feature mal representada puede saturarse de forma sistemática. Las ventanas de retornos deben excluir el movimiento overnight según el contrato intradía; las features de nivel tienen una semántica distinta que debe mantenerse documentada. El libro de exclusiones y calentamiento constituye parte de la población efectiva, no un detalle administrativo.

En los artefactos del carril corregido se distinguen 488 sesiones de desarrollo y 226 de selección. La regresión logística registra 28.792 ejemplos de entrenamiento, equivalentes a 488 × 59 transiciones. Este conteo no autoriza afirmar que el modelo se entrenó en todos los episodios históricos originalmente mencionados: el calentamiento y las exclusiones pueden dejar fuera eventos que permanecen en el archivo fuente. La memoria no publica resultados del bloque de 570 sesiones reservado en el portable como si fueran una evaluación nueva de estos brazos.

El HMM estima estados latentes, no etiquetas económicas conocidas. La evidencia histórica declara cinco estados, mientras que el contexto almacenaba cuatro coordenadas. Para describir acciones se recuperó la quinta mediante el residuo de la suma, con tolerancia numérica explícita y sin renormalizar las anteriores. Esa reconstrucción cambia la etiqueta de máxima probabilidad en 19 sesiones. Sólo tiene uso descriptivo: no modifica la política, el PnL ni demuestra cuánto perdió el PPO por esa representación. Los estados se denominan R0–R4; no se fuerzan a las tres categorías del gráfico ilustrativo.

PPO se implementa con Stable-Baselines3, siguiendo una arquitectura de política y valor separadas de dos capas de 256 unidades con activación Tanh. Los registros económicos corresponden a cinco semillas por configuración: 42, 123, 456, 789 y 1337. La configuración régimen incorpora las coordenadas disponibles y backbone las pone a cero. No se presenta esta ablación como eliminación de todas las variables de posición.

| Parámetro | Valor documentado para el carril evaluado |
| --- | --- |
| Timesteps registrados | 300.000 por corrida |
| Learning rate; n_steps; batch | 0,0003; 4.096; 128 |
| Épocas; gamma; GAE lambda | 10; 0,98; 0,95 |
| Clip; coeficiente de entropía | 0,20; 0,01 |
| Red de política y valor | [256, 256], Tanh |
| Receta identificada | flat_init_no_turn |
| Sesgo inicial hacia flat | Logit 3,0, aplicado según metadata |
| Penalización adicional de turnover | Kappa = 0 |
| Normalización de recompensa | Activada; observaciones no normalizadas por VecNormalize |

La tabla distingue parámetros definidos en el código de receta de metadatos directamente guardados por corrida. Los registros conservan identificador de receta, presupuesto, semilla y campos efectivos relevantes; no contienen toda la procedencia moderna exigida a un checkpoint nuevo. La reproducción del retorno guardado no demuestra, por sí sola, que cualquier instalación actual reproduzca bit a bit el entrenamiento histórico. Por esa razón se preserva la reserva de procedencia del modelo aun cuando la contabilidad de sus posiciones coincida.

## 3.5. Entorno y contabilidad económica

Una sesión contiene 60 cierres y 59 retornos operables. En cada decisión b la política selecciona una exposición w dentro de {−1; −0,5; 0; 0,5; 1}. La exposición aplicada en b multiplica el retorno simple entre C_b y C_(b+1). La sesión comienza y termina sin exposición; la liquidación final genera un costo adicional sobre el último precio. Incluir ese costo tanto en la recompensa como en la evaluación evita entrenar un objetivo que concede una salida gratuita.

El motor utiliza las siguientes identidades. Para una sesión d: g_d = suma_b w_(d,b) × (C_(d,b+1) / C_(d,b) − 1); c_d = suma de cargos de cambios de posición, incluido el cierre; n_d = g_d − c_d. El capital entre sesiones sigue E_d = E_(d−1) × (1 + n_d). Dentro de la sesión las contribuciones se suman bajo la convención del simulador; no se inventa una segunda capitalización intrabar. El retorno bruto compuesto se obtiene de otro recorrido, producto_d (1 + g_d) − 1.

El costo de un cambio de exposición se modela como |Δw| × [(spread/2 + comisión)/precio + 0,1 × volatilidad]. La comisión declarada es 0,5 COP por USD por lado y los niveles de spread de referencia son 2, 3 y 6 COP por USD, según el nivel usado por el motor. Los nombres históricos que contienen “pip” no se trasladan sin explicación a una cotización de broker. En esta memoria la unidad del contrato es COP por USD. No se afirma que esos valores sean spreads efectivos medidos ni límites matemáticos de los costos reales.

Para representar la curva se normaliza el capital inicial a 100. El motor conserva además cantidades en unidades de cuenta, pero la ausencia de un contrato completo de financiación y conversión impide declarar automáticamente que representan dólares de una cuenta real. El drawdown incorpora el capital inicial al máximo acumulado: DD_d = E_d / max(E_0,…,E_d) − 1. Ello evita omitir una pérdida desde la primera sesión.

La recompensa escala por 100 la contribución neta, incluida la salida terminal, y la receta aplica normalización de recompensa. La configuración evaluada no utiliza una penalización positiva de drawdown ni de turnover adicional al costo. Describirla como una recompensa con esas penalizaciones activas, simplemente porque existen parámetros opcionales, sería incorrecto. La evidencia sintética ayuda a comprobar ese entorno, pero sus pases antiguos no certifican automáticamente un contrato de observación que cambió después.

## 3.6. Brazos supervisado, LLM e híbrido

La referencia supervisada emplea StandardScaler y LogisticRegression con C = 1, class_weight = balanced y semilla 42. Se ajusta en desarrollo con etiquetas de signo del siguiente retorno y se decide largo cuando la probabilidad estimada es al menos 0,55, corto cuando es como máximo 0,45 y neutral en el intervalo restante. No es una etiqueta consciente de costos: clasificar el signo puede inducir operaciones cuyo movimiento esperado no cubre el cargo de transacción. El mismo motor liquida sus posiciones.

Los brazos LLM corresponden a decisiones históricas registradas con los proveedores DeepSeek y Azure. Cada uno contiene 13.334 decisiones únicas, equivalentes a 226 × 59, y cubre las 226 sesiones de la cohorte liquidada sin respuestas inválidas ni indisponibles dentro de esa cohorte. Esto no mide la tasa de éxito de todas las llamadas originales: las correcciones y reintentos preceden al conjunto final conservado. Tampoco demuestra que se hubiera respetado el corte temporal en una inferencia en vivo.

Los registros históricos carecen de campos completos de procedencia original, entre ellos la identidad exacta del modelo servido y el identificador de respuesta. Por ello, los nombres de proveedor se conservan en las figuras, pero no se eleva un alias a garantía de una versión inmutable. El prompt usado para estos resultados era numérico, sin noticias ni explicación suficiente de las escalas, posición y costos. La metodología de la memoria original con hasta ocho documentos, FinMA-ES y Llama local no se considera ejecutada por disponer de un diseño escrito. No se completan datos de muestreo o deployment ausentes mediante suposiciones.

El híbrido parte de la mediana por barra de las exposiciones de las cinco políticas PPO régimen. Conserva esa exposición sólo cuando el LLM tiene posición no nula del mismo signo; en otro caso la fija a cero. Formalmente: w_h = w_PPO si ambos signos no nulos coinciden; w_h = 0 en caso contrario. No utiliza confianza como ponderación ni amplifica posición mediante un sentimiento u_t. La evaluación conserva esta regla aunque resulte desfavorable; modificarla después de ver selección introduciría un diseño distinto.

Un veto no garantiza menor costo. Puede interrumpir una posición larga, cerrarla y reabrirla repetidamente. Reducir barras expuestas y reducir transacciones no son equivalentes. Esta propiedad se examina mediante episodios de posición y turnover, no inferida de la palabra “filtro”. Para aislar el efecto del veto, la referencia correcta es PPO pesos medianos, no el promedio diario de retornos de cinco políticas, porque son objetos económicos distintos.

## 3.7. Referencias y convenciones de evaluación

No operar constituye el cero económico, pero su desviación estándar es nula: su Sharpe es indefinido y se muestra NA. La referencia larga intradía y la siempre corta usan los mismos 59 intervalos y la misma liquidación diaria que los agentes. El pasivo cierre a cierre se conserva sólo como contexto sin costos y con tenencia overnight; no es un baseline intradía plenamente comparable. Momentum aparece en la propuesta, pero no se agrega una curva sin artefacto alineado a esta cohorte.

Se informan retorno compuesto, drawdown, bruto, costos y Sharpe con reloj de 221 sesiones por año. El Sharpe es secundario a la evaluación económica y no permite ignorar el signo del neto. En caso de menos de 20 operaciones o varianza cero se suprime su interpretación inferencial. Para series que sólo conservan conteos mínimos se explicita la cota, en lugar de presentar sesiones como si fueran necesariamente operaciones completas.

El Profit Factor y el porcentaje de operaciones ganadoras requieren PnL por operación cerrada. No se derivan de porcentajes de sesiones positivas. Las columnas no estimables permanecen NA en los datos exportables. Tampoco se construyen Sortino, Calmar o DSR para rellenar celdas a cualquier precio: cada métrica se publica bajo su definición y con las reservas de información correspondientes.

## 3.8. Incertidumbre y reproducibilidad

El análisis utiliza bootstrap estacionario por sesiones, con 10.000 réplicas, semilla de remuestreo fija y longitudes esperadas de bloque 5, 10, 15 y 20 alternadas en el conjunto de réplicas. Los intervalos son percentiles del procedimiento combinado, no cuatro confirmaciones independientes ni la selección del bloque más favorable. Su validez depende de que el remuestreo represente razonablemente la dependencia y heterogeneidad de los retornos. No corrige la inspección retrospectiva del período.

Para comparar configuraciones PPO se incorpora variabilidad entre semillas y temporal mediante remuestreo jerárquico emparejado. Para el bruto de la política mediana se condiciona en la política ya fijada y se remuestrean sesiones: ese intervalo no integra toda la incertidumbre del entrenamiento ni de la selección de familias. Se distingue el intervalo de suma bruta del de bruto compuesto, calculando el estimando correspondiente en cada réplica.

Los contrastes contra no operar se formulan sobre diferencia de retorno medio diario, no restando un Sharpe cero ficticio. La tabla estadística distingue el cálculo centrado bajo la hipótesis nula de la cola del bootstrap percentil. La corrección de Holm abarca la familia retrospectiva explícitamente evaluada, no todos los intentos históricos del activo. El registro de trials permanece pendiente de reconciliación; el conteo heredado de 115 no se certifica como actualizado. PBO se omite porque una matriz de semillas no reconstruye el procedimiento histórico de selección.

La entrega incluye un manifiesto con hashes de entradas, código, manuscrito, tablas y figuras. El generador rechaza artefactos fuente alterados, fechas desalineadas, identidades bruto−costo≠neto y destinos ya existentes. Los mismos datos alimentan los valores numéricos y los gráficos. Esta vinculación reduce divergencias documentales, aunque no autentica retrospectivamente una publicación macro ni una solicitud de proveedor cuya evidencia original no se conservó.

## 3.9. Frontera con el siguiente ciclo

La preparación de 38 entradas permite representar hasta cinco posteriores sin truncación, con selección de K sobre desarrollo. El nuevo contrato requiere identidad de dataset, artefactos, publicaciones, políticas de antigüedad y checkpoint compatibles. No se migran pesos históricos mediante relleno silencioso. El entrenamiento futuro exige controles de sanidad vigentes sobre esa representación, congelamiento de receta y cierre del registro de intentos.

Esta memoria no depende de que ese ciclo futuro encuentre rentabilidad. Su evidencia central queda cerrada como resultado retrospectivo versionado. La confirmación requerirá datos posteriores al congelamiento, registro de decisiones antes del retorno y costos observables. Las claves de proveedores, la existencia del servicio o la aprobación del operador no sustituyen esos requisitos científicos.

# 4. Ensayos y resultados

## 4.1. Cohorte y lectura de las figuras

Todos los resultados centrales usan las mismas 226 fechas de selección 2023. Los datos de mercado y las decisiones conservadas son reales; la ejecución y los costos son simulados. Las curvas no muestran un escenario sintético de presentación. Tampoco describen una cuenta operativa ni constituyen un forward: los agentes se evalúan sobre una historia que ya había sido examinada durante la investigación.

La figura 4.1 reúne seis perspectivas complementarias. Capital y drawdown describen rendimiento y trayectoria de pérdida. Sharpe móvil muestra variación temporal bajo una ventana fija, no una serie de pruebas independientes. La sensibilidad a costos mantiene congeladas las posiciones. Las acciones por régimen describen el comportamiento condicionado a categorías reconstruidas. La última figura muestra semillas reales de PPO, sin asignar “semillas” ficticias a los proveedores LLM.

FIGURE: figures/00_resumen_seis_paneles.png | Figura 4.1. Resumen de resultados reales: selección 2023, 226 sesiones. Capital normalizado, costos supuestos y alcance retrospectivo. Los estados R0–R4 son una clasificación descriptiva; no equivalen al esquema de tres regímenes del ejemplo sintético.

## 4.2. Comparación económica global

La tabla 4.1 presenta bruto y neto bajo la misma convención compuesta. Se conserva la distinción entre carteras de cinco políticas y política de pesos medianos. Ninguno de los brazos activos obtuvo retorno neto positivo bajo el contrato aplicado. Este resultado no elimina diferencias entre enfoques: identifica cuál perdió antes de pagar costos, cuál tuvo bruto positivo y cuánto modificaron las fricciones el resultado.

{{TABLE_GLOBAL}}

Tabla 4.1. Rendimiento global reproducido. El asterisco del pasivo indica tenencia cierre a cierre sin costos: contexto, no una política intradía comparable. NA significa métrica no informativa o suprimida; no significa cero. La tabla incluye todos los brazos de la comparación principal, no sólo los favorables al argumento PPO.

La regresión logística produjo un bruto compuesto de +91,99 % y un neto de −55,08 %. Es un resultado material que no debe omitirse al destacar el PPO. Un bruto tan elevado, obtenido bajo señales y precios históricos cuya ejecutabilidad no está certificada, exige controles temporales y microestructurales antes de atribuirlo a ventaja predictiva. No se presenta como campeón alternativo: su neto también es negativo y no se ha realizado aquí una selección de estrategias.

La referencia siempre corta obtuvo +13,54 % bruto y −11,97 % neto. Su bruto supera al de PPO mediano, aunque sus exposiciones y riesgo no estén emparejados. Por tanto, ganar antes de costos no basta para concluir que el agente añade alfa. El resultado de no operar es 0 % y constituye el límite económico disponible en el espacio de acción; perder frente a él no demuestra que una política rentable sea imposible.

FIGURE: figures/01_capital.png | Figura 4.2. Curvas de capital neto de los brazos principales. Inicio normalizado a 100; la tabla global conserva los comparadores adicionales para evitar una selección visual de resultados.

FIGURE: figures/02_drawdown.png | Figura 4.3. Drawdown obtenido de las mismas curvas, incluyendo el capital inicial en el máximo acumulado. No representa un límite de riesgo futuro.

## 4.3. Qué significa el rendimiento bruto positivo del PPO

La política mediana de PPO obtuvo +10,08 % bruto compuesto durante el período. Su suma de retornos brutos diarios fue +9,90 %. Son dos descripciones distintas de la misma serie; la diferencia resulta de componer retornos entre sesiones. Ninguna equivale al porcentaje de datos utilizado, a la tasa de acierto ni a un alfa anual ajustado por factores.

La comparación bruta elimina costos de la liquidación de posiciones ya congeladas. No representa un PPO reentrenado sin costos: en tal caso las decisiones podrían cambiar. Se trata de un contrafactual contable preciso, que sirve para localizar cuánto de la pérdida proviene de las fricciones supuestas sin confundirlo con un experimento de aprendizaje distinto.

El intervalo publicado de la suma bruta es [−3,51 %; +24,14 %]. La reproducción adicional de esta edición calcula el intervalo del bruto compuesto, {{COMPOUND_CI}} %, usando las mismas réplicas por sesiones y aplicando capitalización dentro de cada réplica. Ambos deben leerse como intervalos exploratorios condicionados en las posiciones guardadas, no como certificaciones de habilidad del procedimiento de entrenamiento.

El hallazgo positivo es que esta secuencia concreta de posiciones acumuló ganancias antes de fricciones. La evidencia que falta es establecer que ese valor excede lo explicable por exposición, azar, selección y calidad de información. Esta distinción permite reconocer una observación favorable sin exagerarla. No haber demostrado alfa tampoco demuestra que el alfa verdadero sea exactamente cero; significa que las observaciones disponibles no zanjan la cuestión.

La referencia de exposición emparejada B1′ no está reconstruida en esta entrega como una política de mecánica intradía equivalente. Tampoco se estima aquí un alfa por regresión multifactorial ni se realiza un nuevo replay con latencia. Son controles pendientes: la comparación con siempre corto no permite suplirlos ni aislar por sí sola habilidad de timing. Se conserva esta limitación antes de interpretar el bruto, en lugar de esconderla en las recomendaciones futuras.

FIGURE: figures/07_bruto_neto.png | Figura 4.4. Ganancia bruta y resultado neto, con comparación adicional de LogReg y referencias. La separación de curvas no se interpreta como una resta de porcentajes acumulados.

FIGURE: figures/09_incertidumbre_bruto.png | Figura 4.5. Intervalos de suma bruta y retorno compuesto calculados sobre sus estimandos respectivos. La línea cero permite identificar directamente el límite de la evidencia.

## 4.4. Semillas y variación temporal

{{TABLE_SEEDS}}

Tabla 4.2. Las diez corridas corresponden a dos configuraciones y cinco semillas por configuración, no a diez semillas de un mismo modelo. No se selecciona la mejor corrida para representar al algoritmo. El promedio de retornos de cinco políticas constituye una cartera distinta de operar la mediana de sus pesos.

La dispersión muestra que la receta no determina una única trayectoria. Una semilla con bruto favorable no justifica retirar las demás del análisis. Para el contraste régimen frente a backbone, la diferencia puntual de Sharpe de las carteras es aproximadamente +1,745, pero su intervalo jerárquico [−2,084; +4,464] contiene cero. No se conserva la afirmación de mejora robusta basada únicamente en un p-value anterior que ignoraba parte de la incertidumbre.

FIGURE: figures/06_semillas.png | Figura 4.6. Distribución de Sharpe neto con las cinco semillas observadas en cada configuración. Los puntos son corridas reales; las cajas no crean observaciones adicionales.

El Sharpe móvil se calcula sólo después de completar 60 sesiones. Las ventanas solapadas están fuertemente relacionadas y sirven para describir cuándo cambió la relación media/volatilidad, no para declarar repetidas confirmaciones. En un contexto de pérdidas persistentes y poca dispersión diaria pueden surgir valores negativos muy grandes: su magnitud no se interpreta como precisión predictiva ni como evidencia equivalente a una prueba causal.

FIGURE: figures/03_sharpe_movil.png | Figura 4.7. Sharpe móvil de 60 sesiones bajo el reloj de anualización del experimento. No operar no se dibuja con Sharpe cero porque su varianza es nula.

{{TABLE_QUARTERS}}

Tabla 4.3. Descripción de los cuatro trimestres de 2023 para PPO mediano. Se incluyen todos, sin elegir retrospectivamente el mejor. Las sumas brutas son aditivas entre trimestres; los retornos compuestos no se suman. Esta tabla no sustituye una validación en subperíodos independientes ni permite atribuir desempeño a COVID o a 2022, bloques no evaluados aquí.

## 4.5. Fricciones y costo de equilibrio

{{TABLE_COSTS}}

Tabla 4.4. Sensibilidad del retorno compuesto al multiplicador de costos para posiciones congeladas. ×0 es bruto, no una estrategia entrenada con ejecución gratuita. ×2 y ×3 multiplican los costos contabilizados de la estrategia, no un PnL sintético basado en exposición media.

El PPO mediano pasa de +10,08 % bruto a −36,67 % neto con el costo base. Su suma de costos diarios equivale a 55,16 puntos porcentuales, pero no debe restarse directamente de +10,08 % para obtener el neto compuesto. La identidad exacta opera primero por sesión: n_d = g_d − c_d; sólo después se compone el capital. Esa diferencia de convenciones explica por qué algunas formulaciones anteriores de “alfa menos peaje” eran aritméticamente engañosas.

FIGURE: figures/04_costos.png | Figura 4.8. Sensibilidad a fricciones contabilizadas, manteniendo decisiones y precios. No hay una cotización medida detrás de cada multiplicador.

Para cuantificar la distancia al equilibrio se resuelve, para cada trayectoria congelada, suma_d log(1 + g_d − k c_d) = 0 en el intervalo k ∈ [0;1]. La solución del PPO mediano es k = {{BREAK_EVEN_MEDIAN}}. Esto equivale a reducir aproximadamente {{COST_REDUCTION_MEDIAN}} % el costo base aplicado, manteniendo inalteradas todas las posiciones y precios. No implica que un venue alcanzable pueda ofrecer esa reducción, ni incorpora cambios en fills, latencia o selección adversa.

{{TABLE_BREAK_EVEN}}

Tabla 4.5. Raíz contable del multiplicador de costo en [0;1]. “Sin raíz” significa que no existe un equilibrio identificable en ese dominio bajo la trayectoria fija, o que el costo es nulo y no permite identificar un multiplicador. No significa que se haya probado imposibilidad universal de una estrategia rentable. Los brazos con bruto negativo no se rescatan simplemente reduciendo sus costos a cero.

## 4.6. Qué muestran los LLM y el híbrido

DeepSeek perdió −14,67 % bruto compuesto y Azure −28,98 % antes de costos. Sus pérdidas netas fueron −78,93 % y −82,89 %. A diferencia del PPO mediano, estos brazos no presentaron en esta cohorte una trayectoria bruta agregada positiva que las fricciones eliminaran después. Eso caracteriza la interfaz y el contexto empleados; no demuestra que una familia completa de modelos carezca de información financiera.

Los híbridos tampoco mejoraron la referencia correcta: la política PPO mediana. El híbrido DeepSeek obtuvo −69,24 % neto y el de Azure −70,59 %, frente a −36,67 % del PPO mediano. La cohorte del primero conservó una contribución bruta sumada de −12,35 % y rechazó posiciones con contribución +22,25 %. El segundo conservó −23,41 % y rechazó +33,31 %. Las partes suman el bruto original; son una partición retrospectiva del PnL de PPO, no retornos compuestos aditivos.

La interpretación causal debe ser más limitada que “el LLM está invertido”. En Azure, 3.428 de las 3.455 barras aceptadas eran largas y sólo 27 cortas. La asociación negativa puede estar confundida por composición direccional. Los dos proveedores comparten período, contexto y diseño de prompt, y sus pesos muestran dependencia; no constituyen dos replicaciones independientes del mercado. Utilizar el conjunto rechazado como nueva estrategia y evaluarlo en el mismo bloque sería otra forma de selección retrospectiva.

La política PPO mediana acumula 478 episodios de posición cerrados, mientras que los híbridos DeepSeek y Azure acumulan 1.102 y 880. Estas cifras muestran que abstenerse en más barras no implica operar menos: el veto puede fragmentar posiciones. La explicación contable de mayor fricción es verificable dentro del simulador, mientras que identificar por qué el modelo acepta determinadas direcciones requiere un contraste adicional del contexto o del prompt, predefinido antes de una nueva evaluación.

## 4.7. Acciones, representación y límites de los regímenes

FIGURE: figures/05_acciones_regimen.png | Figura 4.9. SHORT, FLAT y LONG del híbrido DeepSeek por estado descriptivo R0–R4. Quinta coordenada recuperada por residuo; categorías no utilizadas para alterar decisiones o retornos.

La distribución por regímenes es informativa sobre comportamiento, pero no prueba que la política se haya adaptado causalmente al mercado. Los estados tienen diferente número de sesiones y la quinta coordenada fue reconstruida con fines descriptivos. Presentar la figura con rótulos “calmo, tendencia y shock” por semejanza con la propuesta ocultaría tanto el número de estados como la incertidumbre de su interpretación.

FIGURE: figures/08_calidad_datos.png | Figura 4.10. Porcentaje de barras con O = H = L = C en el histórico preservado. La variación de representación es una amenaza a la validez de features basadas en rangos, no evidencia de una mejora económica en 2026.

La calidad de la representación limita el alcance de los resultados positivos y negativos. No puede afirmarse que una política gana capturando microestructura negociable sin observar bid/ask y ejecución; tampoco puede afirmarse que perdería menos si las velas fueran distintas. El resultado de esta auditoría es localizar qué observaciones y comparaciones deben instrumentarse en el siguiente ciclo.

## 4.8. Contrastes y límites de interpretación

La comparación siempre corto contra no operar tiene p centrado aproximado 0,1794: no comparte el suelo Monte Carlo de las otras comparaciones. Para los restantes brazos de la familia principal, el archivo conserva cero excedencias entre 10.000 réplicas del contraste centrado, con corrección de una unidad: p = 1/10.001, aproximadamente 0,0001. Tras Holm dentro de esa familia, el valor es aproximadamente 0,0010. El p de colas percentiles, cercano a 0,0002 en esos casos, es otro cálculo y se etiqueta como tal. No se usa p = 0 ni se trata la resolución Monte Carlo como certeza absoluta.

Estos contrastes describen pérdidas respecto de no operar bajo un modelo de dependencia y un contrato de costos. No convierten 2023 en un conjunto confirmatorio ni prueban ausencia de otras políticas rentables. La incertidumbre de selección, disponibilidad histórica y ejecución permanece aunque la diferencia económica observada sea amplia. La conclusión de ausencia de ventaja certificada es más limitada y más defendible que una afirmación sobre imposibilidad estructural del mercado.

El DSR no se publica como cero definitivo con un N supuestamente actualizado. El paquete marca explícitamente pendiente la reconciliación de trials. PBO se omite por falta de reconstrucción del proceso de selección pertinente. Del mismo modo, no se añaden Diebold–Mariano, SPA o Reality Check sólo porque estuvieran enumerados en la propuesta: su elección exige hipótesis, pérdidas y universo comparables. Una tabla más extensa de pruebas no sustituye ese diseño.

{{TABLE_STATISTICS}}

Tabla 4.6. Diez comparaciones de retorno medio diario contra no operar, nueve en el suelo de resolución del contraste centrado y siempre corto fuera de ese suelo. Se excluyen el propio flat y el pasivo sin costos, que conserva una sola tenencia. Los intervalos son percentiles exploratorios y los p centrados corresponden al remuestreo bajo la nula: son construcciones distintas. Holm corrige esta familia, no todos los intentos históricos del activo.

## 4.9. Balance del capítulo

El experimento produjo hallazgos diferenciados. PPO mediano y LogReg tuvieron bruto histórico positivo, pero no rentabilidad neta bajo las fricciones aplicadas. Los LLM y los híbridos perdieron ya en bruto. El veto aumentó episodios de posición y no mejoró el componente PPO. Los contrastes jerárquicos no establecieron un beneficio robusto del régimen y la incertidumbre bruta del PPO impide declarar alfa demostrado.

El resultado favorable del trabajo no es una curva seleccionada, sino una explicación verificable de estas diferencias. La ingeniería permite reproducir liquidaciones y separar convenciones; la evidencia económica cuantifica las fricciones; la revisión identifica qué partes del diseño original no se ejecutaron y qué información impide una conclusión más fuerte. Sobre esa base se formula el capítulo de discusión, conservando abiertas únicamente las preguntas que requieren nuevos datos o experimentos.

# Anexo técnico de los capítulos 3 y 4

La fuente principal es el manifiesto THESIS-RETROSPECTIVE-BUNDLE-1 del paquete research_grade_20260912_v5, SHA-256 00fc276e22b62ffac144bca1abb6c468f15e272225c46743304c67fe120ae2da. Los archivos results.json, daily_series.json, metrics.csv, actions_regime.json, cost_stress.json y regime_provenance.json conservan resultados y procedencia. La entrega incluye copias de esos archivos, CSV derivados, el protocolo de diagnósticos y un manifiesto de salida. No se modifica el paquete fuente.

La reproducción documental usa el comando: python scripts/presentation/build_thesis_manuscript.py --output outputs/thesis-delivery/entrega_nueva. Requiere las bibliotecas listadas con versión en delivery_manifest.json. El directorio de salida debe ser nuevo. El código no carga claves ni llama a modelos, no entrena y no vuelve a evaluar el hold-out. La matriz claim_evidence_matrix.csv relaciona afirmaciones centrales con campos de los artefactos.

El cruce diario adicional se encuentra en usdcop_daily_cross_source_v2.json. Su etiqueta de acuerdo se interpreta como coherencia descriptiva entre esas fuentes, no como certificación de ejecuciones M5. La preparación de 38 variables se documenta por separado en regime5_20260914_preparation_post_review.json: entrenamiento y piloto siguen sin ejecutarse en ese ciclo. Las restricciones de fuentes, disponibilidad histórica, identidad de solicitudes, registro de intentos y fills no desaparecen por generar un PDF.
