---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-14
supersedes: []
code_anchors:
  - scripts/presentation/build_research_grade_thesis.py
---

# Resumen

Esta memoria estudia un sistema de trading algorítmico intradía para el par dólar estadounidense/peso colombiano mediante aprendizaje supervisado, aprendizaje por refuerzo y modelos de lenguaje. Su propósito es comparar decisiones económicas bajo una contabilidad común, identificar qué parte del rendimiento procede de las posiciones y cuantificar el efecto de los costos. La evaluación central comprende 226 sesiones de selección entre el 3 de enero y el 29 de diciembre de 2023. Se presentan resultados retrospectivos reproducibles; el período no constituye una prueba confirmatoria de las modificaciones motivadas por su inspección.

Se comparan referencias simples, regresión logística, dos configuraciones de Proximal Policy Optimization (PPO), decisiones históricas de DeepSeek y Azure y combinaciones de acuerdo entre PPO y cada proveedor. Todas las estrategias activas evaluadas tuvieron retorno neto negativo bajo el contrato de costos supuesto. Sin embargo, la política definida por la mediana de los pesos de PPO produjo un retorno bruto compuesto de +10,08 %, frente a un retorno neto compuesto de −36,67 %. La suma de sus retornos brutos diarios fue +9,90 % y su intervalo bootstrap publicado del 95 % abarcó −3,51 % a +24,14 %. El valor central positivo es un hallazgo descriptivo, pero ese intervalo no establece habilidad y no es un intervalo del retorno compuesto. Los brazos LLM e híbridos también presentaron pérdidas antes de costos en el período analizado.

La contribución del trabajo consiste en una comparación trazable, una descomposición económica que distingue ganancias brutas de viabilidad neta y una auditoría que identifica límites de datos, disponibilidad macroeconómica, agregación de semillas y ejecución. No se demuestra alfa ajustado por riesgo ni rentabilidad ejecutable. Se establece, en cambio, qué resultados pueden reproducirse, qué afirmaciones excederían la evidencia y qué observaciones futuras permitirían resolverlas.

Palabras clave: USD/COP; trading intradía; aprendizaje por refuerzo; PPO; modelos de lenguaje; costos de transacción; reproducibilidad.

# Abstract

This thesis studies an intraday algorithmic trading system for the USD/COP exchange rate using supervised learning, reinforcement learning, and language models. The objective is to compare economic decisions under common accounting rules, distinguish position returns from trading frictions, and assess the strength of the resulting evidence. The main evaluation covers 226 selection sessions from January 3 to December 29, 2023. Results are reproducible retrospective findings, not confirmatory evidence for changes informed by inspection of that period.

The study compares simple references, logistic regression, two Proximal Policy Optimization configurations, historical DeepSeek and Azure decisions, and agreement-based combinations of PPO with each provider. Every evaluated active strategy produced negative net returns under the assumed cost contract. Nevertheless, the policy obtained from median PPO weights achieved a compounded gross return of +10.08%, compared with a compounded net return of −36.67%. Its summed daily gross returns were +9.90%, with a published 95% bootstrap interval spanning −3.51% to +24.14%. The positive point estimate is a descriptive finding; the interval does not establish skill and must not be interpreted as an interval for compounded returns. The language-model and hybrid arms also lost money before costs over the evaluated period.

The contribution is a traceable comparison, an economic decomposition separating gross performance from net viability, and an audit of limitations involving data quality, historical macroeconomic availability, seed aggregation, and execution assumptions. Risk-adjusted alpha and executable profitability are not established. The study instead identifies reproducible findings and the future observations required to distinguish promising historical performance from a generalizable trading advantage.

# 1. Introducción general

## 1.1. Problema de investigación

Un sistema de trading no se valida únicamente porque prediga correctamente algunos movimientos del precio. Debe convertir información disponible en una secuencia de posiciones, asumir riesgo y pagar por modificar esa exposición. La diferencia entre una predicción y una estrategia resulta especialmente importante cuando las decisiones se actualizan cada cinco minutos: una ventaja pequeña por movimiento puede coexistir con una pérdida considerable después de operar repetidamente. Por ello, esta memoria aborda la inteligencia artificial como parte de un proceso económico completo, desde la observación hasta la liquidación.

El objeto de estudio es USD/COP dentro de una ventana intradía delimitada. La restricción de cerrar posiciones al finalizar la sesión define tanto las oportunidades como las obligaciones del agente. No puede interpretarse su comportamiento como el de un inversor que mantiene una posición durante todo el año, porque los momentos de entrada, salida y pago de costos son diferentes. Una comparación rigurosa necesita explicitar esas diferencias, incluso cuando ambos resultados se expresan como porcentajes.

La pregunta central es qué rendimiento obtienen los enfoques supervisado, PPO, LLM e híbrido bajo los datos y reglas efectivamente implementados, y cómo intervienen la exposición, la información y las fricciones en ese rendimiento. La pregunta no presupone que una familia de modelos deba ganar. Tampoco permite concluir que una familia sea universalmente inadecuada a partir de una configuración histórica. Su alcance es evaluar un diseño identificable y establecer qué inferencias admite.

## 1.2. Contexto, continuidad y motivación

La memoria original sitúa el proyecto como continuación de un sistema de operaciones para series temporales desarrollado por el autor. La ampliación propone pasar de la producción de pronósticos a la comparación de agentes capaces de decidir posiciones, incorporar contexto macroeconómico y estudiar mecanismos de combinación. El interés de esa evolución no reside simplemente en agregar componentes, sino en averiguar si cada componente aporta información o modifica de manera útil la conducta económica del sistema.

La versión inicial del documento contemplaba un corpus financiero en español y modulación por sentimiento. Los resultados disponibles no ejecutan íntegramente ese diseño: los prompts históricos evaluados no contenían noticias y el híbrido observado aplica acuerdo o veto entre decisiones. Esta actualización conserva esa distinción. El procesamiento textual especializado permanece como una propuesta no contrastada en el experimento central, mientras que DeepSeek y Azure se analizan por las decisiones que efectivamente registraron. Una arquitectura prevista no equivale a un resultado experimental.

También se conserva el componente supervisado del título mediante la regresión logística evaluada. Su función es aportar un comparador adicional y mostrar que una salida predictiva debe juzgarse después de transformarla en exposición. No se presenta como representación exhaustiva del aprendizaje supervisado ni se atribuyen a XGBoost, LightGBM u otros modelos cifras que pertenecen a ese comparador concreto.

## 1.3. Objetivos y preguntas operativas

El objetivo general es construir y evaluar un sistema reproducible para comparar decisiones intradía sobre USD/COP, con trazabilidad entre datos, políticas, retornos y conclusiones. Este objetivo se desarrolla en cinco dimensiones: documentar identidad y procesamiento de datos; implementar una contabilidad compartida; comparar los brazos realmente ejecutados; cuantificar rendimiento e incertidumbre; y determinar qué evidencia falta para una validación prospectiva.

Las preguntas operativas distinguen niveles de exigencia. Primero, si las liquidaciones se reproducen a partir de los registros disponibles. Segundo, si alguna política produce rendimiento bruto positivo y si ese resultado se conserva tras costos. Tercero, si el contexto de régimen o el acuerdo con un LLM mejora la política de referencia. Finalmente, si los datos y el procedimiento de selección permiten atribuir los resultados a habilidad y esperar su generalización. Estas preguntas organizan la discusión; no se presentan como hipótesis nuevas registradas antes de observar 2023.

La publicación de resultados negativos forma parte del objetivo de evaluación. Evita que la documentación conserve únicamente una semilla, una métrica o una combinación favorable. A la vez, publicar pérdidas no exime de explicar los resultados positivos parciales: un bruto favorable, una liquidación reproducible o un mecanismo de rotación identificado constituyen hallazgos distintos que deben describirse en sus propios términos.

## 1.4. Alcance y contribuciones

La evidencia económica central corresponde a 226 sesiones de selección de 2023. Los experimentos históricos sobre otros bloques y la preparación de un contrato de observación posterior se mantienen separados. Una modificación realizada después de inspeccionar un período no puede recibir confirmación independiente de ese mismo período. Esta delimitación sustituye la ambigüedad de presentar indistintamente selección, hold-out y forward como si fueran sinónimos.

El aporte empírico es una comparación con datos y decisiones reales registrados, aunque las ejecuciones sean simuladas bajo costos supuestos. El aporte técnico es una cadena verificable entre posiciones y contabilidad. El aporte metodológico es la identificación explícita de amenazas a la validez y de la diferencia entre lo reproducible y lo causalmente certificado. Ninguno requiere ocultar que la rentabilidad neta fue negativa.

El trabajo no certifica un sistema operativo con dinero real ni caracteriza todos los mercados cambiarios. Tampoco utiliza resultados de oro para compensar el desempeño de USD/COP. Una eventual extensión a otro instrumento deberá definir sus propios datos, unidad económica, calendario, costos y protocolo de selección.

# 2. Fundamentos y criterios de evaluación

## 2.1. De la información a la decisión económica

Conviene separar cuatro objetos: observación, pronóstico, acción y ejecución. La observación representa lo que el sistema conoce; el pronóstico resume una expectativa; la acción prescribe una exposición; la ejecución establece cuándo y a qué precio puede materializarse. Dos modelos con pronósticos similares pueden obtener resultados diferentes si cambian la frecuencia de ajuste o la regla de abstención. Del mismo modo, una buena decisión aparente puede no ser ejecutable al precio utilizado por el simulador.

En una notación simplificada por barra, el rendimiento bruto depende del producto entre la exposición vigente y el retorno del activo. El neto descuenta los costos asociados a las transiciones y al cierre. La identidad contable debe verificarse en la granularidad implementada antes de agregar sesiones. La capitalización diaria utiliza el producto de uno más cada retorno; no es intercambiable con su suma. Este detalle determina qué significa un porcentaje publicado y evita comparaciones entre magnitudes distintas.

El control always-flat representa ausencia de exposición dentro del experimento y produce retorno de trading cero. No representa una estimación completa de rentabilidad de tesorería: no remunera el efectivo ni modela todas las alternativas de inversión. Su utilidad consiste en preguntar si operar mejora el resultado contable de no asumir esas posiciones, bajo una convención explícita.

## 2.2. Aprendizaje supervisado, PPO y modelos de lenguaje

Un modelo supervisado ajusta una relación entre entradas y una variable objetivo observada durante el entrenamiento. En esta memoria, la regresión logística constituye un comparador de esa familia. La calidad predictiva no sustituye la evaluación económica de la regla que convierte su salida en posiciones. Por esa razón, su resultado bruto y su resultado neto se publican separadamente.

El aprendizaje por refuerzo ajusta una política a partir de recompensas asociadas a la interacción con un entorno. PPO pertenece a los métodos de gradiente de política y utiliza un objetivo sustituto que limita cambios excesivos respecto de la política que generó las observaciones. Schulman y colaboradores lo proponen como un método de optimización práctica; su artículo no garantiza rentabilidad en series financieras. Aquí su conveniencia se evalúa mediante el entorno y las decisiones concretas, no mediante la reputación general del algoritmo. [Schulman et al., 2017](https://arxiv.org/abs/1707.06347).

Stable-Baselines3 ofrece implementaciones con interfaces consistentes y pruebas de software. Adoptar una biblioteca de ese tipo ayuda a reducir errores de implementación del algoritmo, pero no valida automáticamente la recompensa, las features ni el simulador financiero construido alrededor. La separación es esencial: el algoritmo puede estar correctamente implementado y optimizar un entorno que represente de manera incompleta el problema económico. [Raffin et al., 2021](https://www.jmlr.org/papers/v22/20-1364.html).

El agente LLM evaluado transforma un contexto serializado en una respuesta que debe validarse antes de convertirse en posición. Su identidad experimental comprende proveedor, modelo recuperable, prompt, parámetros de inferencia, parser y manejo de errores. Una respuesta sintácticamente válida no demuestra comprensión financiera ni información predictiva. Además, dos proveedores que reciben el mismo contexto comparten restricciones del diseño y no constituyen dos muestras independientes del mercado.

Un híbrido introduce otra función entre la política base y la posición final. En el caso evaluado, esa función permite o veta exposición según el acuerdo entre PPO y el LLM. Aunque pueda parecer un mecanismo de reducción de riesgo, interrumpir una posición y reabrirla puede aumentar el número de transiciones. Por ello, la combinación debe liquidarse como política propia: no basta promediar las métricas de sus componentes.

## 2.3. Regímenes y causalidad temporal

Un modelo de estados latentes describe regularidades estadísticas mediante categorías no observadas directamente. En un HMM, las probabilidades de estado dependen del modelo de transición y de las observaciones disponibles. Para usar esas probabilidades en una decisión histórica, interesa el filtrado con información acumulada hasta el momento; una reconstrucción que utiliza observaciones posteriores tiene un significado retrospectivo diferente.

Las etiquetas de régimen no son explicaciones económicas por sí mismas. Un estado puede recibir un nombre descriptivo después de examinar sus propiedades, pero dicho nombre no demuestra que represente un mecanismo estable. También debe preservarse el número de estados efectivamente ajustado: reducir una figura a tres categorías por razones estéticas podría alterar el objeto observado. En este trabajo, cualquier recuperación descriptiva de una coordenada ausente se identifica como tal y no modifica decisiones ya emitidas.

La causalidad de los datos requiere más que ordenar fechas. Para una variable macroeconómica deben distinguirse el período observado, su publicación, su primera recepción y sus revisiones. Un desplazamiento T−1 evita ciertas incorporaciones del mismo día, pero no demuestra qué versión histórica conocía el operador. Asimismo, reproducir una descarga desde una fuente confirma consistencia con esa fuente, no independencia ni ausencia de revisiones.

## 2.4. Retorno, alfa y habilidad

El rendimiento bruto positivo es una propiedad de la trayectoria observada antes de costos. Alfa es un concepto relativo a un modelo de referencia. En la formulación clásica de evaluación de carteras, la comparación ajusta por exposición al riesgo y pregunta si queda un rendimiento adicional no explicado por esa referencia. Jensen desarrolló una medida de desempeño en ese contexto; trasladarla a una estrategia intradía exige especificar los riesgos pertinentes, no importar automáticamente un factor accionario. [Jensen, 1968](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1968.tb00815.x).

En consecuencia, la memoria no utiliza alfa como sinónimo de cualquier cifra positiva. Para hablar de habilidad se necesita distinguir exposición favorable, decisiones de timing, ruido muestral y selección entre intentos. Una referencia de exposición emparejada puede ayudar a esa descomposición, pero su construcción retrospectiva no convierte por sí sola el residual en una señal conocida antes de operar. Tampoco se ha estimado aquí un alfa multifactorial certificado.

El +10,08 % bruto compuesto de la política PPO de pesos medianos significa que su trayectoria simulada antes de costos incrementó el capital por ese porcentaje durante las sesiones evaluadas. No significa un porcentaje de aciertos, una fracción del dataset ni una tasa anualizada generalizable. Que otras agregaciones de PPO produzcan cifras cercanas no autoriza a intercambiarlas: promediar retornos entre políticas y tomar la mediana de sus posiciones son operaciones diferentes.

## 2.5. Incertidumbre y múltiples intentos

La incertidumbre temporal y la del entrenamiento deben considerarse separadamente. Varias semillas sobre las mismas sesiones no añaden mercados independientes; varias sesiones de una sola semilla tampoco describen la variabilidad completa del procedimiento de aprendizaje. El trabajo de Agarwal y colaboradores muestra la fragilidad de evaluaciones de aprendizaje por refuerzo basadas en pocas ejecuciones y estimadores puntuales. Su discusión motiva informar distribuciones e intervalos; las propiedades de benchmarks con muchas tareas no se transfieren automáticamente a un único par cambiario. [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264).

El Sharpe resume media y dispersión, pero no identifica el origen económico del resultado. Un Sharpe agregado puede diferir del de cada semilla porque la agregación modifica la variabilidad. Si la varianza es cero, como en always-flat, el cociente no está definido y se informa como no aplicable. El drawdown, por su parte, describe caída desde máximos anteriores y complementa, sin sustituir, las medidas de retorno.

La exploración de muchas variantes aumenta las oportunidades de encontrar un resultado favorable por selección. El Deflated Sharpe Ratio busca ajustar parte de esa inflación considerando múltiples intentos y no normalidad. Su interpretación depende de las entradas y de la reconstrucción del proceso de búsqueda; no convierte un conteo incompleto en evidencia válida. Por ello, la reconciliación pendiente de trials impide emplearlo como certificado concluyente de ventaja. [Bailey y López de Prado, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

## 2.6. Pruebas de ingeniería y evidencia económica

Las pruebas sintéticas utilizan entornos construidos para verificar propiedades conocidas, por ejemplo que operar un precio constante con costos no mejore la abstención. Su finalidad es comprobar que el software puede representar un problema y que el aprendizaje responde a controles definidos. No reemplazan precios observados ni deben aparecer mezcladas con curvas económicas de la tesis.

Un control sólo respalda la versión que se probó. Cambiar dimensiones, recompensa o construcción de features requiere verificar nuevamente su compatibilidad. Por ello, la preparación de una observación de 38 entradas no hereda automáticamente resultados de sanidad o rentabilidad de modelos históricos con 37. La trazabilidad entre versión, prueba y conclusión constituye una condición de interpretación, no un detalle administrativo.

# 5. Discusión y conclusiones

## 5.1. Resultado principal y significado económico

La evidencia permite concluir que ninguna estrategia activa de la comparación retrospectiva obtuvo rentabilidad neta positiva en selección 2023 bajo el contrato de costos utilizado. Always-flat conservó un retorno de trading cero, mientras que las políticas activas cerraron el período con pérdidas. Este resultado describe configuraciones y condiciones identificables. No demuestra que USD/COP carezca de oportunidades ni que una clase de algoritmos sea incapaz de encontrarlas.

La evaluación también produjo un resultado bruto positivo concreto. La política PPO de pesos medianos alcanzó +10,08 % compuesto antes de costos y −36,67 % compuesto después de ellos. La suma bruta de +9,90 % tiene un intervalo publicado del 95 % entre −3,51 % y +24,14 %. El intervalo incluye resultados no positivos y corresponde a la suma, no al compuesto. Por tanto, la evidencia respalda la existencia del rendimiento histórico observado, pero no establece que su esperanza sea positiva o que provenga de habilidad.

No sería correcto afirmar que sólo PPO consiguió un bruto favorable. La regresión logística presenta +91,99 % bruto compuesto en el mismo paquete de resultados, junto con −55,08 % neto. Publicar esa cifra evita seleccionar la única historia positiva que convenga al argumento. Además, refuerza una conclusión económica: incluso un bruto considerable puede no sobrevivir a la frecuencia y magnitud de los costos de la política que lo genera. Su origen y estabilidad tampoco quedan certificados por el valor puntual.

## 5.2. Qué se logró medir

El trabajo identifica una separación útil entre producir ganancias antes de fricciones y transformarlas en viabilidad económica. La reconstrucción de posiciones, retornos y cargos permite localizar esa diferencia dentro del simulador, sin atribuirla vagamente a la complejidad del mercado. Se trata de una explicación contable comprobable. La explicación causal del origen del bruto requiere controles adicionales sobre información, exposición y generalización.

También se obtuvo una comparación que no depende exclusivamente de la curva final. Las trayectorias de drawdown, la variabilidad de semillas, las acciones y la sensibilidad a costos muestran facetas distintas del mismo experimento. La presentación conjunta evita que un resultado favorable de una métrica o de una agregación sustituya al balance completo. El valor de la instrumentación es hacer observables esas tensiones, incluso cuando no produce un candidato operable.

Los brazos DeepSeek y Azure perdieron antes y después de costos en el período examinado. Los híbridos basados en acuerdo también empeoraron el neto de la política PPO de pesos medianos. Esto permite rechazar una interpretación favorable de esas combinaciones observadas, pero no demostrar que la señal LLM esté universalmente invertida. Los proveedores compartieron contexto y prompt; la selección retrospectiva de posiciones aceptadas y rechazadas está condicionada por la exposición de PPO. Convertir el complemento rechazado en una nueva estrategia requeriría otro estudio.

## 5.3. Regímenes, semillas y alcance estadístico

La media de cinco políticas con régimen tuvo mejor Sharpe neto que la correspondiente configuración backbone, aunque su retorno compuesto fue más negativo. No existe contradicción: las dos métricas responden a propiedades diferentes de las trayectorias. El intervalo jerárquico de la diferencia de Sharpe, aproximadamente −2,08 a +4,46, incluye cero. Una mejora puntual no alcanza para establecer un aporte robusto del régimen cuando se incorpora variabilidad temporal y entre semillas.

Las cinco semillas por configuración tampoco equivalen a diez réplicas de la misma estrategia ni a diez muestras independientes. Preservar esa distinción evita exagerar el tamaño efectivo de la evidencia. La figura de semillas debe mostrar los puntos reales; el gráfico de proveedores debe identificar variantes de modelo o repeticiones observadas, sin asignarles artificialmente la semántica de una semilla de entrenamiento PPO.

La muestra de selección fue examinada y sus resultados motivaron correcciones. En consecuencia, las estimaciones corregidas tienen carácter retrospectivo. Una revisión puede mejorar la exactitud de la medición, pero no restaurar la independencia perdida para seleccionar nuevos diseños. Las comparaciones estadísticas ayudan a describir incertidumbre bajo sus supuestos; no eliminan ese límite del procedimiento de investigación.

## 5.4. Amenazas a la validez y dirección de sus efectos

La primera amenaza corresponde a la información disponible. La identidad de una serie y su coincidencia numérica con una descarga son controles necesarios, pero distintos de certificar publicaciones y versiones históricas. La reconciliación macro no debe presentarse como cerrada si faltan esas evidencias. Es posible que corregir disponibilidad reduzca un bruto aparente; no corresponde asumir que mejores datos elevarían los retornos.

La segunda amenaza es la representación del precio. Las velas con apertura, máximo, mínimo y cierre iguales y los cambios de calidad entre períodos limitan la interpretación de indicadores basados en rangos. Un cruce diario con otra fuente aporta una comprobación de coherencia a esa frecuencia, pero no valida el precio ejecutable de cada barra intradía. La dependencia de esos artefactos constituye una hipótesis que necesita medición específica, no una excusa suficiente para atribuir todas las pérdidas a los datos.

La tercera amenaza es el costo. Las liquidaciones utilizan un contrato supuesto, no un registro de bid/ask y fills alcanzables. Por ello, el neto cuantifica el comportamiento bajo ese contrato. No se sabe a partir de esta evaluación si un operador concreto habría soportado fricciones menores o mayores. El costo de equilibrio conserva utilidad como contrafactual contable de posiciones congeladas, pero no demuestra que exista un proveedor capaz de ofrecerlo.

La cuarta amenaza es el alcance del agente de lenguaje. Los prompts históricos no incorporaban noticias ni explicitaban adecuadamente escalas, posición y costos. El hallazgo concierne a esa interfaz y sus decisiones; no contrasta la capacidad de FinMA-ES, un corpus financiero en español o una arquitectura textual diferente. El ensayo permite localizar un diseño no satisfactorio, pero no separar causalmente cuál de esas omisiones explica su resultado.

Finalmente, el registro de intentos y la procedencia original de solicitudes requieren cierre. Los hashes protegen identidad de archivos, no prueban por sí solos la historia completa de decisiones de investigación o de inferencia. Estas reservas no borran la contabilidad observada; delimitan qué tipo de certificación no puede deducirse de ella.

## 5.5. Contribuciones, continuidad y conclusión final

La contribución técnica es un marco que vincula datos, decisiones y resultados con controles reproducibles. La contribución empírica es una comparación completa que publica bruto y neto, configuraciones individuales y agregaciones, resultados favorables y desfavorables. La contribución analítica es distinguir pérdida por fricciones dentro del simulador de pérdida que ya aparece antes de descontarlas, sin transformar esa descomposición en una afirmación causal excesiva.

La continuidad del programa debe resolver primero las restricciones de información y ejecución. Para un nuevo ciclo será necesario congelar el contrato temporal de cada serie, comprobar la nueva representación y repetir los controles de aprendizaje sobre esa versión. La preparación técnica de 38 entradas y cinco posiciones disponibles para probabilidades de régimen no constituye un reentrenamiento concluido ni una mejora económica demostrada. Sus resultados deberán conservar un identificador y una evaluación propios.

Una evaluación prospectiva posterior al congelamiento permitiría observar decisiones selladas antes del retorno, junto con tiempos de recepción y fricciones pertinentes. Esa observación futura deberá separar monitoreo de integridad y miradas de rendimiento, y establecer tamaño y criterio de análisis antes de evaluar. No bastará acumular algunas sesiones para declarar confirmación; será necesario considerar precisión, dependencia y número de operaciones efectivamente observadas.

En síntesis, la tesis no demuestra una estrategia rentable ni alfa certificado. Sí documenta un rendimiento bruto histórico positivo en PPO, muestra que no sobrevivió al contrato de costos y explica por qué ese resultado no debe confundirse con ausencia total de aprendizaje ni con habilidad ya probada. El valor del trabajo reside en hacer verificable esa frontera: qué se observó, qué se pudo explicar y qué experimento permitiría pasar de una expectativa prometedora a evidencia de ventaja económica.

# Bibliografía

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. y Bellemare, M. G. (2021). *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. Advances in Neural Information Processing Systems, 34. [Versión de los autores](https://arxiv.org/abs/2108.13264).

Bailey, D. H. y López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality*. The Journal of Portfolio Management, 40(5). [Manuscrito de los autores](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

Jensen, M. C. (1968). *The Performance of Mutual Funds in the Period 1945–1964*. The Journal of Finance, 23(2), 389–416. [Artículo original](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1968.tb00815.x).

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. y Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. Journal of Machine Learning Research, 22(268), 1–8. [Artículo original](https://www.jmlr.org/papers/v22/20-1364.html).

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. y Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. [Trabajo original](https://arxiv.org/abs/1707.06347).
