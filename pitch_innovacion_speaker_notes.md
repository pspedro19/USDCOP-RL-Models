# Speaker Notes — Pitch de Innovación · GlobalMinds USD/COP

> **Deck**: `pitch_innovacion.html` (12 slides) · Maestría en IA — UBA/FIUBA-CEIA · *Gestión de la Tecnología y la Innovación*
> **Autor**: Esp. Ing. Pedro Pérez Salazar · linkedin.com/in/pedro-perez · pedro@bitlink.dev
> **Marca**: GlobalMinds (motor de IA) · SignalBridge (OMS de ejecución)
> **Duración total objetivo**: ~12–14 min (1–1.5 min/slide) + Q&A
> **Fuente de verdad de todas las cifras**: `PROJECT_DEFINITION.md` (2026-05-09) y los JSON verificables en `usdcop-trading-dashboard/public/data/production/`.
>
> **Tres frases que debes poder defender de memoria:**
> 1. *"El valor de la IA no es predecir el mercado — es saber CUÁNDO el mercado no es predecible y abstenerse."*
> 2. *"+25.63% OOS 2025, Sharpe 3.35, p=0.006 — y el modelo direccional tiene R² < 0. El alpha es el regime gate, no la predicción."*
> 3. *"El sistema ya está en producción; la tesis lo eleva a un marco comparativo reproducible con robustez estadística (CPCV / Deflated Sharpe)."*

---

## Nota de honestidad metodológica (léela antes de presentar)

Para sostener cualquier número bajo escrutinio académico, ten presente estas distinciones que el deck respeta:

- **Producción real ≠ track de research.** Lo que está en producción es el **pipeline H5 semanal (Smart Simple v2.0)**: ML supervisado (Ridge + BayesianRidge + XGBoost) + **regime gate**. El **agente RL (PPO) intradía** y el **LLM-as-trader** son el **objeto de la tesis comparativa** — el RL **aún no es estadísticamente significativo** (p=0.272). Esto no es una debilidad: es un hallazgo honesto que refuerza el mensaje (el alpha está en la gestión de régimen/riesgo, no en la arquitectura del agente).
- **El backtest es Smart Simple v2.0** (no v1.1.0). v1.1.0 daba +20.03%; v2.0 da +25.63% al añadir regime gate, effective hard stop, dynamic leverage y XGBoost.
- **Sharpe 3.35** es con rf=0; **3.00** es vs US 3M T-Bill (convención institucional). Ambos son defendibles; usa 3.35 como titular y 3.00 si te cuestionan el risk-free.
- **N=34 trades, 12 meses.** Cuando alguien diga "muestra chica", respondes con el **p-value (0.006, bootstrap 10K)** y el **CI del Sharpe [2.49, 4.21]** — no escondas la limitación, encuádrala.

---

## Slide 01 · Portada

**Objetivo**: posicionar tesis + venture en una frase. Tono sobrio, sin vender humo.

**Guion (≈40 s):**
> "Buenas. Soy Pedro Pérez Salazar. Lo que les presento es mi trabajo de la maestría en IA de la UBA-CEIA, en Gestión de la Tecnología y la Innovación: un **agente híbrido RL + LLM para trading en mercados emergentes**, con el caso **USD/COP**. Detrás hay un emprendimiento de base tecnológica, **GlobalMinds**, con su capa de ejecución **SignalBridge**. No es un ejercicio teórico: hay un sistema operando en producción que es el antecedente del marco que voy a comparar."

**Transición**: "Veamos primero por qué este problema importa."

---

## Slide 02 · Agenda

**Objetivo**: dar el mapa. No te detengas.

**Guion (≈15 s):**
> "Siete bloques: el problema y el contexto, el proyecto, los atributos de la innovación, la vigilancia tecnológica, los socios, la propiedad intelectual y el cierre."

**Transición**: "Empecemos por el problema."

---

## Slide 03 · El problema (Contexto)

**Objetivo**: establecer la doble naturaleza — **vacío de conocimiento académico** + **oportunidad práctica**.

**Guion (≈60 s):**
> "Quien opera el dólar en mercados emergentes decide de memoria, y la investigación lo ignora. Casi toda la evidencia sobre agentes RL y LLM en trading está en acciones de EE.UU. El **USD/COP intradía no tiene baselines ni evaluación rigurosa**. Cuatro datos: hay operadores moviendo entre **50 y 200 mil dólares al día** sin un modelo riguroso; cerca del **35% de las exportaciones colombianas son petróleo**, lo que ata el peso al WTI — es un mercado con estructura macro fuerte y predecible en parte; **casi cero baselines** publicados para este par intradía; y en NLP financiero en español **FinMA-ES está casi en solitario**. Es a la vez un hueco de conocimiento y una oportunidad práctica."

**Datos de respaldo:**
- 50–200k USD/día: rango de merchants P2P (PROJECT_DEFINITION §6.1).
- Correlación COP–petróleo: el pipeline usa `oil_close_lag1` como feature macro (§4 features compartidos).
- Fuentes públicas en `presentation/.../SOURCES.md`.

**Posible pregunta**: *"¿35% es dato actual?"* → "Es el orden de magnitud histórico de la canasta exportadora; lo uso para ilustrar la dependencia estructural, no como cifra de tesis."

**Transición**: "Frente a ese vacío, ¿qué propongo?"

---

## Slide 04 · El proyecto

**Objetivo**: el proyecto es un **marco comparativo y reproducible**, no un solo modelo.

**Guion (≈60 s):**
> "El proyecto compara **tres familias de agentes de decisión** sobre el mismo par: un **agente RL con PPO**, un **LLM actuando como trader**, y un **híbrido** estilo FinRL-DeepSeek. La pregunta es cuál ofrece la mejor relación retorno/riesgo — pero medida bien, con **CPCV y Deflated Sharpe**, para no caer en las métricas infladas de un backtest ingenuo. El estado: investigación preliminar hecha, diseño de alto nivel propuesto, y un **pipeline MLOps ya operativo** — Airflow con 27 DAGs, MLflow, dashboard de 8 páginas — con validación experimental en curso."

**Datos de respaldo:**
- 27 DAGs Airflow, MLflow server, dashboard 8 páginas / 47 API routes (§3).
- CPCV/Deflated Sharpe = la contribución metodológica de la tesis (control de sobreajuste).

**Posible pregunta**: *"¿Ya sabes qué familia gana?"* → "El antecedente en producción es ML supervisado + gestión de régimen; el RL puro aún **no es significativo** (p=0.272). El hallazgo provisional es que el alpha vive en el control de régimen y riesgo, no en la sofisticación del agente. Eso es justo lo que el marco busca cuantificar con rigor."

**Transición**: "Y no parto de cero — esto ya corre."

---

## Slide 05 · Antecedente operativo (slide ancla — la prueba)

**Objetivo**: credibilidad dura. Aquí están los números reales. **Es el slide más importante.**

**Guion (≈75 s):**
> "No es una idea en papel. Hay un sistema en producción, **Smart Simple v2.0**. En backtest auditado **fuera de muestra de 2025**: **+25.63% de retorno**, Sharpe **3.35** — **3.00** si descuento el T-Bill a 3 meses —, Calmar **4.19**, profit factor **2.76**, máxima caída de solo **6.12%**, win rate **82.4%** sobre 34 trades, y **p-value de 0.006** en bootstrap de 10 mil iteraciones. Contra comprar y mantener, que perdió 14.5% ese año, son **más de 40 puntos de alpha**. Pero aquí está lo contraintuitivo, y es el corazón de la tesis: **el alpha no viene de predecir**. El modelo direccional tiene **R² negativo**. El alpha viene del **regime gate** — un detector de régimen con el exponente de Hurst que **se abstiene de operar cuando el mercado es mean-reverting**. En el primer trimestre de 2026 bloqueó 13 de 14 semanas y convirtió lo que habría sido −5.17% en +0.61%."

**Datos de respaldo (verificables en repo):**
- `public/data/production/summary_2025.json` — todas las métricas.
- `trades/smart_simple_v11_2025.json` — los 34 trades.
- 2026 YTD: +0.61%, 1 trade ganador, gate bloqueó 13/14 semanas (§5.7).
- Alpha del gate: +5.78 pp (contrafactual v1.1.0 sin gate = −5.17%).

**Posible pregunta**: *"¿R² negativo y aun así gana? ¿No es contradictorio?"* → "No. El modelo no acierta la dirección, pero el sistema gana porque (1) el gate evita los regímenes donde perdería, (2) los stops adaptativos y el effective hard stop capan la pérdida al 3.5% del portafolio, y (3) el dynamic leverage reduce tamaño en drawdown. Es alpha de gestión de riesgo y selectividad, no de adivinación. Eso lo descubrió un audit de 10 agentes en marzo 2026."

**Posible pregunta**: *"¿N=34 no es poco?"* → "Sí, es una limitación: 12 meses, 34 trades, CI del Sharpe [2.49, 4.21]. Por eso reporto el p-value, que mide directamente la probabilidad de que sea suerte: 0.6%. Y por eso la tesis usa CPCV en vez de un solo split."

**Transición**: "¿Sobre qué y cómo se innova exactamente?"

---

## Slide 06 · Atributos de la innovación (1) — ¿Sobre qué y cómo?

**Objetivo**: clasificar la innovación en el marco del curso.

**Guion (≈55 s):**
> "¿Sobre qué? Es innovación de **producto-servicio** y de **método**: un marco metodológico y un sistema. ¿Qué grado de cambio? **Incremental con componentes radicales** — reutilizo PPO, CPPO y agentes LLM que ya existen, pero el aporte radical es **operativo**: el regime gate que decide cuándo NO operar, que es la fuente real del alpha cuando el modelo tiene R² negativo. ¿Cómo se hace? Como emprendimiento de base tecnológica, GlobalMinds, con dirección académica de la UBA-CEIA. Investigación sobre stack abierto — Stable-Baselines3, FinRL, MLflow, vLLM — y producción sobre scikit-learn, XGBoost, Airflow y Azure OpenAI. Financiación propia, con créditos de Azure vía Microsoft for Startups en evaluación."

**Datos de respaldo:**
- Stack producción: §3 (scikit-learn, XGBoost, LightGBM, CatBoost; Azure OpenAI GPT-4o + Anthropic Claude).
- Azure créditos: hasta USD 150k Founders Hub (§10.1).

**Transición**: "¿Y para quién genera valor?"

---

## Slide 07 · Atributos de la innovación (2) — ¿Por qué y para quién?

**Objetivo**: stakeholders + naturaleza de la transformación.

**Guion (≈55 s):**
> "Tres beneficiarios. Para los **operadores en COP**: mejor decisión y gestión del riesgo cambiario en un mercado desatendido. Para **GlobalMinds y SignalBridge**: una capacidad analítica diferencial — el recurso clave que transforma la propuesta de valor. Para la **academia**: un caso de estudio sobre mercado emergente y un aporte al NLP financiero en español. ¿Qué tiene de nuevo? Arquitecturas de frontera sobre el peso colombiano, español financiero, un regime gate que prioriza saber cuándo no operar, y robustez estadística contra el sobreajuste. La transformación es sobre todo **económica**, con un componente de **conocimiento** — cerrar un vacío de investigación."

**Transición**: "Esto lo confirma la vigilancia tecnológica."

---

## Slide 08 · Vigilancia tecnológica

**Objetivo**: demostrar que hiciste el estado del arte y que confirma el nicho.

**Guion (≈55 s):**
> "Busqué en arXiv, Google Patents, Scopus y MINCyT. Los trabajos relevantes son recientes: **FinRL-DeepSeek** (2025), baseline híbrido directo; **FLAG-Trader** (2025), agente-LLM con RL; **LM-Guided RL** (2025), donde el LLM guía al RL; **FinMA-ES** (2024), el FinLLM bilingüe; y **DRL-FX** (2019) como ancla en divisas. La lectura: campo joven, acelerado desde 2024 por la ola de LLMs abiertos; predominan equities e índices de EE.UU., con IBM dominando las patentes de ML-trading. Dos vacíos: **USD/COP y emergentes intradía** apenas cubiertos — mi diferenciador central — y el **español financiero**, incipiente. Y en IP: sin barreras de patente para una tesis comparativa sobre datos propios."

**Posible pregunta**: *"¿IBM bloquea?"* → "IBM domina patentes en ML-trading genérico, pero una tesis comparativa sobre datos propios y sin explotación comercial no infringe; el riesgo de IP se traslada a la fase de comercialización, donde se evalúa licencia con el socio."

**Transición**: "Para cerrar las brechas necesito aliados."

---

## Slide 09 · Selección de socios

**Objetivo**: mostrar que sabes qué te falta y a quién pedírselo.

**Guion (≈50 s):**
> "Tres brechas: **datos intradía USD/COP**, **noticias financieras en español** y **cómputo GPU** para entrenar. Tres tipos de socio: un proveedor de datos FX como **SET-ICAP / SET-FX**; nube — **Microsoft Azure**, con el que ya uso Azure OpenAI y al que apunto vía Founders Hub; y academia — **UBA-CEIA y AI4Finance**. La elección la formalizo con una **matriz multicriterio**: 5 grupos, 11 criterios — acceso a datos, costo, alineación estratégica, soporte técnico y condiciones de IP."

**Datos de respaldo**: Azure OpenAI ya en uso; migración a Azure mapeada servicio por servicio (§3 Cloud Migration Path).

**Transición**: "Una palabra sobre propiedad intelectual."

---

## Slide 10 · Propiedad intelectual

**Objetivo**: claridad sobre el presente abierto y el riesgo futuro.

**Guion (≈45 s):**
> "Hoy: documentación en repositorio privado; modelos y frameworks open-source sin restricciones de uso; y sin barreras de patente que afecten una tesis comparativa sobre datos propios y sin explotación comercial. A futuro, el riesgo de IP **se traslada a la fase de comercialización**: ahí evalúo licencia y estrategia de protección junto al socio, vigilando a IBM como titular dominante en ML-trading."

**Transición**: "Cierro con cuatro reflexiones."

---

## Slide 11 · Reflexiones finales

**Objetivo**: síntesis memorable. Cuatro ideas.

**Guion (≈50 s):**
> "Cuatro. Uno: la vigilancia confirma un vacío real — **USD/COP intradía más español** es el diferenciador. Dos: las **alianzas** en datos y cómputo son críticas. Tres: la **robustez estadística** — CPCV, Deflated Sharpe — es lo que separa un resultado real de un backtest ingenuo. Y cuatro: el **antecedente operativo** — Smart Simple v2.0 en producción, +25.63% auditado con p=0.006 y un regime gate ya validado en 2026 — respalda que esto es viable, no aspiracional. Y como dijimos en clase: comunicar bien una idea es tan importante como la idea misma."

**Transición**: "Gracias."

---

## Slide 12 · Cierre

**Objetivo**: frase-ancla + contacto. Pausa antes de la última línea.

**Guion (≈25 s):**
> "La decisión cuantitativa que hoy está reservada a los grandes, al alcance de cualquier operador de la región. Eso es GlobalMinds. Gracias — quedo para preguntas."

**Contacto en pantalla**: pedro@bitlink.dev · linkedin.com/in/pedro-perez

---

## Banco de preguntas difíciles (Q&A)

| Pregunta | Respuesta corta |
|----------|-----------------|
| *"¿Por qué RL si no es significativo?"* | "Es el objeto de comparación de la tesis. El hallazgo —que el RL puro no supera al gate de régimen— es un resultado válido y publicable. La producción ya migró a ML + gate." |
| *"Sharpe 3.35 suena demasiado bueno."* | "Por eso reporto p=0.006, Calmar 4.19 (no necesita risk-free) y el alpha vs B&H (+40 pp), que es la métrica más robusta porque no requiere annualización ni conversión de moneda." |
| *"¿El 25% se replica en 2026?"* | "2026 va +0.61% YTD: el peso siguió fortaleciéndose, el peor escenario, y el gate bloqueó 13 de 14 semanas. Que en el peor trimestre la estrategia no pierda valida el control de riesgo, no solo el año bueno." |
| *"¿Y el riesgo cambiario para un inversor en COP?"* | "Honesto: en COP 2025 el Sharpe baja a ~0.43 porque el peso se apreció 14% (año atípico). En escenario FX-flat el Calmar es ~3.4. Lo reporto en las 4 perspectivas, sin esconderlo." |
| *"¿Cómo evitas look-ahead bias?"* | "Macro con shift(1) + merge_asof backward, expanding window 2020→último viernes, StandardScaler solo sobre train, y walk-forward. La tesis suma CPCV." |
| *"¿Es asesoría financiera?"* | "No. Es una herramienta de decisión; resultados pasados no garantizan futuros. Disclaimer formal en PROJECT_DEFINITION §11 Apéndice D." |

---

*Documento de apoyo · Confidencial · Alineado a `PROJECT_DEFINITION.md` (2026-05-09). Todas las cifras son verificables en `usdcop-trading-dashboard/public/data/production/`.*
