---
kind: roadmap
status: PARTIAL
version: 2.0.0
last_verified: 2026-08-25
supersedes: []
code_anchors:
  - config/research/partition.yaml
  - config/research/evaluation_mask.json
  - config/research/feature_schema.json
  - src/research/evaluation_mask.py
  - src/research/regime_hmm.py
  - src/research/cost_model.py
  - src/research/session_env.py
  - src/research/session_gym.py
  - src/research/features.py
  - src/research/dataset.py
  - src/research/inference.py
  - scripts/analysis/thesis_baselines.py
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/thesis_statistics.py
  - scripts/presentation/generar_resultados_y_figuras.py
  - airflow/dags/research_thesis_ppo_training.py
  - services/common/metrics.py
---

> **Ubicacion y nombre (2026-08-24).** Este documento llego como
> `.claude/plan-definitivo-usdcop (1).md` en la raiz de `.claude/` — el `(1)` era un
> duplicado de descarga, y al no tener front matter ni enlaces entrantes rompia
> `test_knowledge_frontmatter.py` y `test_knowledge_graph.py` (nota huerfana). Vive
> ahora bajo `specs/planes/`, gobernado por `specs-gate.yml`.
>
> Plan de ejecucion e integracion con el repo (limpieza pre-F0, huecos de F0 §5,
> reutilizacion de DAGs y estrategias nuevas): [`README.md`](README.md) de esta
> carpeta y [`02-usdcop-doble-rol-y-trials.md`](02-usdcop-doble-rol-y-trials.md)
> para el contador de trials unico.

# Propuesta definitiva — Sistema de trading algorítmico USD/COP (RL / LLM / Híbrido)

**Documento único de especificación.** Sustituye a todas las versiones anteriores.

**Reglas de ejecución (dos, no una — la anterior era circular):**

```text
REGLA A — no se inicia el HPO (F5) hasta haber superado F0–F4
          y congelado: datos, calendario, features, contrato de costos,
          régimen (incluido K) y cadencia.

REGLA B — no se abre el hold-out hasta haber completado F8
          y firmado el manifiesto pre-hold-out.
```

**Qué corrige esta versión** (detalle en el Anexo A): circularidad de la regla de ejecución · costo del cierre
terminal ausente en el retorno diario · mezcla de log-retornos con costos en retornos simples · sesiones
inválidas contadas como retorno cero · sesgo de selección en la serie OOF · uso de 2024+2025 como evidencia
confirmatoria · la ablación presentada como escalera cuando no lo es · desigualdad de información en H1 ·
universo exacto de White RC/SPA/PBO · regla de fallo del PBO · fórmula que convierte posteriores del HMM en
spread · tolerancias del fallback entre proveedores · grid híbrido con tres columnas idénticas · condición del
shuffle test · justificación de `rf = 0` · compuerta de benchmark de cómputo al final de F0.

---

## 1. La decisión estratégica, ya CERRADA (Opción C, 2026-08-24)

> **RESUELTA.** SSOT: `config/research/partition.yaml` (CTR-RESEARCH-PARTITION-001).
> Guard: `tests/regression/test_research_partition_is_separate.py`.

Este documento proponía elegir entre A y B. **Ninguna de las dos era ejecutable**: ambas
arrancan el desarrollo en 2019 y **2019 no existe** en la serie de 5 minutos, que empieza el
2020-01-02 (hay 9 sesiones sueltas de dic-2019, insuficientes para formar bloque).

Se adopta la **Opción C**: la B desplazada por la realidad del dato, más el tramo de 2026.

| Bloque | Rango | Sesiones | **Efectivas** tras la máscara | Propósito |
|---|---|---|---|---|
| Desarrollo | 2020-01-02 → 2022-12-31 | 777 | **558** | Entrenamiento (y HPO, si lo hubiera) |
| Selección | 2023-01-01 → 2023-12-31 | 260 | **235** | Elección entre configuraciones |
| Hold-out | 2024-01-01 → 2026-08-24 | 677 | **584** | Juicio final, una sola apertura |

Las **efectivas** son las que cuentan: la máscara de evaluación (§9.5) excluye 95 festivos
colombianos y 245 sesiones incompletas. Reportar la potencia sobre 677 en vez de 584 sería
inflar la muestra.

### El análisis de poder, ahora MEDIDO

§11.2 estimaba en abstracto que harían falta ~990 sesiones para un ΔSharpe de 0,20 con
ρ = 0,99. Con el entorno ya construido esa cifra se puede **medir** en vez de suponer.
Simulación sobre n=584 con la correlación realmente observada entre brazos (**ρ ≈ 0,92**, que
es lo que hace válido el bootstrap pareado de §11.4):

| ΔSharpe verdadero | Potencia |
|---|---|
| 0,41 | 45% |
| 0,83 | 90% |
| ≥ 1,55 | 100% |

**Mínimo detectable al 80% ≈ ΔSharpe 0,7.** Por debajo de eso el contraste sale INDECIDIBLE, y
así se reportará — no como "sin diferencias" (§11.1).

La estimación original era pesimista porque asumía ρ = 0,98-0,99, una correlación mayor que la
real: cuanto más correlacionados van dos brazos, más pequeña es la diferencia que el pareo
puede resolver, pero también más pequeña tiende a ser la diferencia. Con ρ = 0,92 el contraste
resuelve diferencias moderadas, no finas.

### Advertencia declarada ex-ante sobre el hold-out

**2025 está dentro del hold-out y ya fue mirado** por el track de producción H5 (grid de 42
celdas, "#8 de 42", documentado en `CLAUDE.md`). Se mantiene ahí a conciencia, para no bajar de
las 500 sesiones que §11.2 llama el límite, con el N heredado declarado
(`partition.yaml::trials.inherited_n: 111`) y el DSR deflactado con él. **No se presentará como
hold-out virgen** (`holdout_partially_looked_at: true`).

---

## 2. Decisiones congeladas

| # | Decisión | Resolución | Estado |
|---|---|---|---|
| 1 | Partición temporal | Opción A o B de §1 | **Abierta — única pendiente** |
| 2 | Cadencia | 5 min en todos los brazos; fallback a 15 min **para todos** solo si falla la compuerta del LLM antes del HPO | Congelada con compuerta |
| 3 | Brazo 1 principal | `PPO+régimen`; `PPO backbone` es ablación | Congelada |
| 4 | Espacio de acción | `{−1, −0.5, 0, +0.5, +1}` | Congelada |
| 5 | Cierre de sesión | El entorno fuerza `w=0` tras el último retorno **y cobra su costo**, que entra en el retorno diario | Congelada |
| 6 | Convención textual | `u_d > 0` favorece al COP; señal operativa `s_d = −u_d` | Congelada |
| 7 | Híbrido | Política = misma instancia congelada de `PPO+régimen`; `s_d` solo en wrapper externo; con `β=0`, posiciones idénticas | Congelada |
| 8 | Costos | Mid + débito explícito; sin doble conteo del spread; spread esperado por posteriores del HMM | Congelada |
| 9 | Selección | HPO solo en desarrollo; top-3 en selección; refit único antes del hold-out | Congelada |
| 10 | Inferencia | H2 y H3 confirmatorias; **H1 se reporta con precisión declarada, no como contraste decidible**; H4 exploratoria | Congelada |
| 11 | Hold-out | Universo cerrado, una apertura, sin Optuna, grids ni cambios de prompt | Congelada |
| 12 | Infraestructura | Git, entorno bloqueado, Parquet/DuckDB, DVC, MLflow, CLI, pytest; plataforma distribuida opcional | Congelada |
| 13 | Presupuesto | CPU ≈ 200 h · API USD 150 · 50 GB; todo recorte se decide antes de la fase afectada | Congelada |
| 14 | Narrativa | Capítulos 4 y 5 rederivados de resultados reales | Congelada |
| 15 | Contabilidad | Retornos simples, equity compuesta, costo terminal incluido | Congelada |
| 16 | Subperíodos | Solo predicciones OOF, etiquetadas; nunca in-sample junto al hold-out | Congelada |
| 17 | Salida inferencial | IC de la diferencia pareada como salida primaria; no-rechazo ≠ equivalencia | Congelada |
| 18 | DSR | Cantidad reportada con regla de interpretación, no compuerta binaria | Congelada |
| 19 | Máscara de evaluación | Todas las estrategias sobre el mismo conjunto de sesiones válidas; las inválidas no pertenecen a `n` | Congelada |
| 20 | `K` del HMM | Se elige una sola vez dentro del bloque de desarrollo y se congela antes del HPO; igual en todos los folds | Congelada |
| 21 | Regla de fallo del PBO | Predeclarada en §11.7; `PBO` alto **no** autoriza retunear | Congelada |

---

## 3. Principios

1. **Causalidad estricta.**
2. **Hold-out único**, con manifiesto firmado.
3. **Todo experimento cuenta** para `N_trials`.
4. **Variabilidad de semillas e incertidumbre temporal se reportan por separado.**
5. **La contabilidad económica es independiente del reward.**
6. **Los resultados negativos son válidos.**
7. **La narrativa sigue a los datos.**
8. **La precisión se declara antes que el resultado.** Si un contraste no tiene poder para detectar el efecto declarado, se dice antes de ejecutarlo.
9. **Un contraste indecidible se reporta como indecidible**, no como "sin diferencias significativas".

---

## 4. Reconciliación de nomenclatura con la memoria

Al fijar **Brazo 1 = `PPO+régimen`**, varias tablas del docx quedan con filas duplicadas.

| Nombre en el docx | Nombre aquí | Acción sobre el capítulo 4 |
|---|---|---|
| "PPO base" (baseline, tabla 4.2) | `PPO backbone` | Pasa de baseline a **ablación** (tabla 4.5) |
| "PPO con régimen" (baseline, tabla 4.2) | `PPO+régimen` | Es el **Brazo 1**; sale de la tabla de baselines |
| "Brazo 1: PPO" (tabla 4.3) | `PPO+régimen` | Fila única; se fusiona con la anterior |
| "PPO + sentiment" (tabla 4.5) | `PPO+sentimiento` | Sin cambio |

Consecuencias: la tabla 4.2 pierde dos filas y gana los baselines externos nuevos; la 4.3 pierde una fila
duplicada; la 4.5 pasa a tener una sola base, lo que elimina el salto superaditivo del escenario sintético;
y §3.6 del capítulo 3 debe decir que el Brazo 1 incluye el régimen en el estado.

---

## 5. Infraestructura y reproducibilidad

**Obligatorio:** Git con commit por corrida · `uv.lock` con versiones de Python, CUDA y librerías · imagen
Docker con digest · Parquet + DuckDB · DVC · MLflow · `make data | features | train | evaluate | report` ·
`pytest` para los 16 tests de §13.

**Opcional:** Airflow, Grafana, TimescaleDB, PostgreSQL operativo, MinIO. Son ~100 000 barras; F0 no puede
consumir semanas.

**Artefactos por corrida:** `run_id · experiment_type · config_hash · git_commit · data_hash ·
feature_schema_hash · calendar_hash · cost_contract_hash · seed · train_dates · validation_dates · model_hash ·
scaler_hash · vecnormalize_hash · hmm_hash · prompt_hash + model_id · N_trials_acumulado ·
status: completed | pruned | failed | invalid`.

---

## 6. Dataset numérico

### 6.1 Especificación

| Parámetro | Valor |
|---|---|
| Activo · frecuencia | USD/COP spot · 5 minutos |
| Período objetivo | 2019-01-01 → 2025-12-31 |
| Sesión | 08:00–13:00 COT, 60 intervalos semiabiertos `[t, t+5min)` |
| Almacenamiento | UTC; conversión a COT en la capa de calendario |
| Posición inicial / final | `w = 0` / `w = 0` forzado con costo |
| Calendario | Feriados Colombia ∪ EE. UU., versionado y hasheado |

### 6.2 Fuentes y tolerancias de sustitución (congeladas en `data.yaml`)

| Fuente | Uso |
|---|---|
| Dukascopy | Primaria (bid/ask → mid) |
| TwelveData | Validación secundaria y contingencia |
| HistData | Solo si se confirma cobertura; si no, se elimina de la memoria |
| TRM Banco de la República | Control diario; nunca señal intradía |

Regla: **descartar una sesión incompleta antes que mezclar proveedores.** La sustitución solo se permite si:

- solape ≥ 60 sesiones entre proveedores en el tramo afectado;
- diferencia absoluta mediana de precio ≤ 2 pips;
- correlación de retornos de 5 min entre proveedores ≥ 0.95 en el solape;
- ≤ 5 % de barras sustituidas por sesión y ≤ 2 % del dataset total;
- cada barra sustituida marcada con procedencia.

Si cualquiera falla, la sesión se descarta.

### 6.3 Contrato de barras y calidad

| Ítem | Regla |
|---|---|
| Cobertura de auditoría | ≥ 48/60 barras crudas |
| `ffill` | ≤ 2 barras consecutivas y ≤ 4 imputadas por sesión |
| OHLC inválido | `H < max(O,C)`, `L > min(O,C)` o precio ≤ 0 → barra inválida |
| Sesión con barra inválida | **Sesión inválida**: sale de la máscara de evaluación (§8.5), no entra como retorno 0 |
| `H = L` o desvío cero | Divisiones afectadas devuelven 0, no NaN ni inf |
| Outlier | `abs(logret) > 8·σ_rolling`, ventana de 1 560 barras intra-sesión → barra `suspect`, sesión fuera de train, sin revisión discrecional |
| Warm-up | 78 retornos intra-sesión previos disponibles |
| Validación TRM | Correlación diaria agregada ≥ 0.98 bajo la alineación documentada de antemano; si falla, F1 se detiene |

### 6.4 Fórmulas congeladas

`logret_1 = log(C_t/C_{t−1})` sin retornos overnight · **ATR(14)** y **RSI(14)** de Wilder · volatilidad
realizada `sqrt(Σ logret²)` sin anualizar · **Parkinson** `sqrt((1/(4·ln2·n))·Σ ln(H/L)²)` ·
**Garman–Klass** `sqrt((1/n)·Σ[0.5·ln(H/L)² − (2ln2−1)·ln(C/O)²])` · ventanas de 12 y 78 barras concatenan solo
retornos intra-sesión · todas validadas contra fixtures numéricos.

### 6.5 Features

**Precio:** `logret_1/3/6/12`, `ret_sesion_acum`, `close_pos_rango`.
**Volatilidad y rango:** `rv_12`, `rv_78`, `rv_ratio`, `atr_14`, `atr_norm`, `parkinson_12`, `garman_klass_12`.
**Tendencia:** `(C−EMA_k)/ATR` con `k ∈ {12,26,72}`, `macd_norm`, `macd_signal_norm`, `slope_20`, `rsi_14`, `zscore_60`.
**Volumen: ELIMINADO** (medido 2026-08-25). La condición era «barras con volumen cero ≤ 5 %».
Medición sobre la serie reparada: **99.714 de 99.714 barras tienen volumen 0**, y la columna
entera contiene un solo valor distinto. USD/COP spot es un mercado OTC y el proveedor no
publica volumen. Lo elimina el criterio que este mismo documento fijó, no una preferencia.
**Temporales:** minutos desde apertura, seno/coseno de hora, primeros/últimos 30 min, día de semana, evento macro.
**Estado de posición (endógeno):** `w_prev`, PnL no realizado normalizado, barras en posición, drawdown de
sesión, número de cambios. **Solo disponible para agentes secuenciales** (§10.6).
**Macro as-of:** retornos previos de Brent y DXY, y el diferencial BanRep–Fed
(`FINC_RATE_IBR_OVERNIGHT_COL_D_IBR` − `FINC_BOND_YIELD2Y_USA_D_DGS2`), todos con
`merge_asof(backward)`. **Sorpresa macro: ELIMINADA** — la condición era «consenso histórico
verificable» y el repo no tiene serie de consenso; fabricarla sería inventar exactamente el
dato que la feature mide.

> **Incidencia de calidad detectada al construir este grupo (2026-08-25).** La serie de Brent
> tenía **59 días corruptos dentro del hold-out** (2025-09-25 → 2025-12-19: valores de 21-23
> USD cuando Brent real estaba en 60-70). Dos criterios independientes —nivel < 30, y ratio
> contra WTI de 0,366 frente al 1,062 normal— señalaban exactamente las mismas 59 filas.
> Reparado desde FRED `DCOILBRENTEU` (`scripts/ops/fix_brent_corrupt_block.py`).
>
> **Lo relevante no es el arreglo sino el hallazgo**: `config/l0_macro_sources.yaml:509` ya
> declaraba el rango `[30, 150]` con `validation.enabled: true`, y el dato entró igual.
>
> **CORRECCIÓN (2026-08-25, posterior).** La primera versión de este párrafo decía que
> `RangeValidator` *"solo se instancia en tests"*. **Es falso, y el diagnóstico equivocado es
> peor que no tenerlo**: manda a la siguiente persona a cablear algo que ya está cableado.
> El validador `RangeValidator` **sí está cableado** — `data_validators.py:604` lo incluye en la lista por defecto de `ValidationPipeline` y `l0_macro_backfill.py:835` la construye. Lo que falla son **cuatro huecos distintos**: (A) el DAG de ingesta DIARIA (`l0_macro_update.py`) no valida nada — es por ahí por donde entró el Brent malo; (B) `validate_data` del backfill nunca lanza (`fail_fast=False`, solo `logger.warning`); (C) la rama de restore desde seeds salta la validación entera; y (D) si el validador no encuentra el config devuelve `{}` **en silencio** y valida cero variables.
**Texto:** `s_d`, `n_docs`, dispersión de scores, horas desde el último documento, según §10.7.
**Régimen:** posteriores filtradas y etiqueta descriptiva.

**Total: 39 features** en 7 grupos, tras eliminar volumen, sorpresa macro y texto
(este último por el recorte a 2 brazos). Orden, dtype, grupo, disponibilidad temporal y hash en
`config/research/feature_schema.json` (`sha256` = `5427409c451bc46b`), implementadas en
`src/research/features.py`. Causalidad verificada por perturbación, no por lectura del código:
`tests/regression/test_research_features_are_causal.py` altera el futuro y comprueba que ninguna
feature del pasado se mueve.

### 6.6 Escalado

`StandardScaler` para observaciones, ajustado solo en el train de cada fold · `VecNormalize` **solo** para el
reward (`norm_obs=False`, `norm_reward=True`) · en evaluación `training=False`, `norm_reward=False`, objeto
serializado · clipping a ±5 · para el refit final, scaler y HMM se ajustan una vez sobre el bloque
desarrollo+selección y se congelan.

### 6.7 Particiones

CV purgada cronológica `K=6` con embargo de una sesión por lado, **solo dentro del bloque de desarrollo**
(`K=4` si el período efectivo arranca en 2021 o después). Walk-forward expansivo mensual como sensibilidad
pre-hold-out; no selecciona nada.

---

## 7. Corpus textual y señal de sentimiento

**Almacenamiento:** `url · titulo · cuerpo · fuente · published_at_utc · retrieved_at_utc · first_seen_at_utc ·
hash_contenido · hash_normalizado`.

**Filtro de relevancia:** palabras clave sobre USD/COP y sus determinantes → filtro semántico contra 50
documentos semilla → umbral calibrado solo en el bloque de desarrollo y congelado → deduplicación exacta y
aproximada → reporte de recolectados, retenidos, descartados y % de sesiones con `s_d = 0`. El rango de
5 000–8 000 documentos del capítulo 3 es descriptivo; si el filtro causal da otra cifra, **se corrige la memoria**.

**Etiquetado:** FinMA-ES (o checkpoint equivalente), cuantización 4-bit con offload a CPU, inferencia offline
por lotes; contingencia con endpoint remoto de `model_id` fijo, y si se activa **todo** el corpus se reetiqueta
con la misma versión. One-shot, cacheado por hash. Salida: score continuo `u_i ∈ [−1,1]` y clase, desde la
perspectiva del COP.

**Convención y agregación:**

```text
u_d = clip( Σ_i exp(−Δt_i/12h)·u_i / Σ_i exp(−Δt_i/12h), −1, 1 )
s_d = −u_d
```

Solo documentos publicados en las 72 h anteriores a las 08:00 COT con `published_at_utc < apertura_d` · fines de
semana y feriados pasan a la siguiente sesión válida dentro de esas 72 h · si solo hay timestamp de
actualización, el documento se atrasa una sesión · sin documentos válidos, `s_d = 0` · `s_d` constante durante la sesión.

**Validación humana:** 200 documentos estratificados por clase, fuente y año; dos anotadores con adjudicación;
se reporta acuerdo humano, precisión/recall/F1 **por clase**, macro-F1 y matriz de confusión. Con una sola
persona, segunda anotación ciega tras ≥ 2 semanas, reportada como **acuerdo intra-anotador**.

**Sanitización:** sin HTML, enlaces ni código; cada documento encapsulado como datos no ejecutables;
presupuesto de prompt ≤ 2 000 tokens congelado tras el piloto; máximo 8 documentos por relevancia y luego
recencia, máximo 3 por fuente; toda salida fuera de esquema se registra como fallo o posible inyección.

---

## 8. Regímenes

### 8.1 Datos y temporalidad

Features diarias al cierre de `d−1`: volatilidad realizada diaria y log-volatilidad, ATR diario normalizado,
retorno diario y absoluto, rango/ATR, autocorrelación intradía, retornos as-of de DXY y Brent, dummy de evento
macro. Para decidir en `d` se usa el posterior filtrado tras procesar `d−1`, constante durante `d`.

### 8.2 Modelo y selección de `K`

`GaussianHMM`, covarianza `full` (fallback `diag`), 500 iteraciones, 20 inicializaciones, se conserva la de
mayor log-likelihood. **`K` se elige una sola vez** comparando BIC para `K ∈ {2,3,4,5}` dentro del bloque de
desarrollo, y se congela antes del HPO: **el mismo `K` en todos los folds**. `K=3` se mantiene salvo que otro
mejore el BIC en más de 10 puntos; si gana otro, se adopta y se corrige la memoria. Los `K` no elegidos quedan
como sensibilidad descriptiva. En validación y test, **recursión forward filtrada**; nunca Viterbi sobre la
secuencia completa ni posterior suavizada.

### 8.3 Etiquetado de estados — y el hallazgo que obliga a matizar el nombre «régimen»

Orden por volatilidad media: bajo → `calmo`, alto → `shock`. El estado medio se llama `tendencial` **solo si**
sus métricas de dirección y persistencia lo respaldan; si no, `intermedio`. La persistencia de la diagonal de
transición desempata. Se reportan matriz de transición y duración media por estado.

**RESULTADO (2026-08-25, ajuste congelado sobre desarrollo 2020-01-08 → 2022-12-29):**

| K | BIC |
|---|---|
| 2 | 10.726,7 |
| 3 | 6.255,6 |
| **4** | **5.765,1** |
| 5 | 5.873,6 |

Gana **K = 4**, y por un margen de 490 puntos sobre K=3 — muy por encima de la histéresis de 10
puntos que §8.2 exige para desplazar al K=3 por defecto. Covarianza `full`, sin necesidad del
fallback `diag`. Etiquetas resultantes: `calmo · intermedio · intermedio · shock`. Ninguno de
los dos estados medios alcanza la persistencia de 0,8 que §8.3 exige para llamarse
`tendencial`, así que ambos quedan como `intermedio` — el criterio se aplicó, no se relajó.

**Lo que hay que decir en la memoria, y no es cosmético.** Las persistencias diagonales son
0,405 · 0,390 · 0,454 · 0,484, o sea **duraciones medias de 1,6 a 1,9 días**:

> El HMM no está encontrando *regímenes* en el sentido macro del término —tramos de semanas o
> meses con un comportamiento común—, sino **tipos de día**. Un estado que dura menos de dos
> sesiones no es un régimen: es una clasificación de la jornada.

Esto **no invalida** su uso. El propósito operativo declarado en §8.4 es alimentar el spread
esperado del día siguiente, y para eso un clasificador de tipo de día es exactamente lo
apropiado. Lo que invalida es la *narrativa*: no se puede escribir que el agente «aprende a
reconocer regímenes de mercado» cuando lo que recibe es una etiqueta que cambia cada día y
medio. La contribución de esa información se mide en H2, y ahí el número manda sobre el
relato.

### 8.4 Del posterior al spread (fórmula, antes ausente)

El HMM produce posteriores, no una etiqueta. El costo usa el **spread esperado**, que es continuo y evita
saltos arbitrarios de turnover:

```text
spread_d = Σ_k p_{d,k} · spread_k        con spread = {2, 3, 6} pips por nivel de volatilidad
```

El `argmax(p_{d,k})` se usa **solo** para el desglose descriptivo por régimen en las tablas, nunca para costear.

### 8.5 Refit y alternativas

En cada fold, HMM y scaler solo en train. Tras la selección, el HMM final se ajusta sobre desarrollo+selección
y se congela; durante el hold-out solo evoluciona el posterior filtrado diario. Terciles rolling de volatilidad
y GMM son sensibilidades pre-hold-out: no seleccionan el modelo principal ni amplían el universo del hold-out.

---

## 9. Entorno de trading, contabilidad y costos

### 9.1 Cronología

Barras `b = 0 … 59`.

```text
cierre de barra b, b = 0 … 58
  → observar x_b con información ≤ cierre_b
  → decidir w_b
  → pagar costo por Δw_b = w_b − w_{b−1}
  → mantener w_b durante la barra b+1
  → recibir r_{b+1}

cierre de barra 59  (paso terminal, sin retorno)
  → Δw_close = 0 − w_58
  → cobrar costo terminal
  → fin del episodio
```

`w_{−1} = 0`. Hay **59 retornos operables**. El cierre terminal es una regla del entorno, no una acción evitable.
Posición positiva = largo USD/COP.

### 9.2 Contabilidad (corregida: retornos simples y equity compuesta)

La versión anterior mezclaba un log-retorno con un costo expresado como retorno simple. Se adopta una
contabilidad homogénea:

```text
r_{b+1}     = C_{b+1}/C_b − 1                                  # retorno simple
pnl_{b+1}   = w_b · r_{b+1} − cost_ret(Δw_b)
equity_{b+1} = equity_b · (1 + pnl_{b+1})

paso terminal:  equity_final = equity_59 · (1 − cost_ret(−w_58))

retorno_diario_d = equity_final_d / equity_inicial_d − 1
```

- El **costo del cierre terminal entra en el retorno diario**. Antes no estaba y se perdía sistemáticamente un
  costo por sesión.
- La métrica primaria se calcula sobre `retorno_diario_d`, **nunca sobre barras de 5 minutos**.
- Retorno anualizado por composición de `retorno_diario_d`; Sharpe con `√252`.
- **`rf = 0` por materialidad:** el capital está expuesto cinco horas al día y la remuneración de efectivo
  correspondiente es pequeña frente al error de estimación. Se excluye de forma consistente en todos los
  sistemas y se declara así en el capítulo 2. (La justificación no es "no hay carry": el carry de la posición FX
  sí desaparece por no mantener overnight, pero el efectivo no invertido podría rendir.)

### 9.3 Contrato de costos

| Componente | Regla |
|---|---|
| Pip | 1 pip = 1.00 COP por USD |
| Precio base | Mid de Dukascopy |
| Spread | `spread_d` esperado por posteriores (§8.4) |
| Comisión | 0.5 pips por lado |
| Volatilidad para slippage | `σ12_pips = C_b · rv_12` (queda en pips porque 1 pip = 1 COP) |
| Slippage | `0.1 · abs(Δw_b) · σ12_pips` |
| Costo por cambio | `cost_pips = abs(Δw_b)·(spread_d/2 + 0.5) + 0.1·abs(Δw_b)·σ12_pips` |
| Conversión | `cost_ret = cost_pips / C_b` |
| Flip `+1 → −1` | `abs(Δw) = 2`, cobra dos lados automáticamente |
| Cierre terminal | Cobra como cualquier otro cambio |

**Regla contra doble conteo:** se usan retornos de mid y se resta `cost_ret`. **No** se mueve además el precio
de ejecución. El mismo `spread_d` causal y congelado aplica a todos los brazos en una sesión.

Round-trip determinista mínimo sin slippage: **3 / 4 / 7 pips** para vol baja/media/alta.

### 9.4 Sensibilidad de costos

Sobre los **mismos vectores de posiciones**, sin reentrenar:
`base · 3 · 5 · 10 · 30 pips round-trip fijos`, con `cost_pips_fixed = abs(Δw_b)·round_trip/2`.
El costo fijo **reemplaza** spread, comisión y slippage. La columna `base` es la de la tabla de desempeño principal.

### 9.5 Máscara de evaluación (corrección)

```text
Sesión inválida (datos incompletos, barra inválida, outlier)
    → NO pertenece al calendario de evaluación
    → NO entra en n
    → NO se cuenta como retorno 0

Sesión válida en la que el sistema queda flat
    → retorno_diario = 0, sí entra en n

Sesión válida sin noticias (s_d = 0)
    → el sistema puede operar normalmente
```

**Todas las estrategias, brazos, ablaciones, baselines y controles se evalúan sobre exactamente la misma
máscara de sesiones válidas.** La máscara se hashea y entra en el manifiesto. Contar sesiones inválidas como
cero deprimía artificialmente la volatilidad e inflaba el Sharpe.

### 9.6 Reward de entrenamiento

```text
reward_{b+1} = pnl_{b+1} − κ_turn·cost_ret(Δw_b) − λ_dd·max(0, DD_{b+1} − DD_b)
```

`κ_turn` y `λ_dd` son regularizadores y no tocan las métricas económicas. La penalización de drawdown usa el
**incremento**. El Sharpe diferencial (Moody y Saffell) es ablación de reward pre-hold-out y cuenta en `N_trials`.

---

## 10. Modelos y comparadores

### 10.1 Brazo 1 — `PPO+régimen`

> **LO QUE SE EJECUTÓ (2026-08-25)** difiere de esta tabla, y las tres diferencias están
> decididas y justificadas en otra parte del documento:
>
> | Parámetro | Planificado aquí | **Ejecutado** | Por qué |
> |---|---|---|---|
> | Hiperparámetros | valores iniciales + espacio HPO de 13 ejes | **congelados de `v215b_baseline.yaml`**: lr 3e-4, `n_steps` 4096, batch 128, γ 0.98, gae_λ 0.95, clip 0.2, `ent_coef` 0.01, red `[256,256]` Tanh | sin HPO (§15): prior ex-ante ⇒ 0 trials, pero universo de 2 ⇒ sin White/SPA (§11.6) |
> | Timesteps | 1,5 M | **300.000** | 1,5 M sobre 29.441 pasos de desarrollo serían 51 pasadas; se hereda la receta, no un conteo atado a otro tamaño de muestra |
> | Semillas | 10 | **5** — `[42,123,456,789,1337]` | `experiment-protocol.md` regla 2 fija estas cinco |
>
> **Ensamble primario: NO se construyó.** Las hipótesis se probaron sobre la **media entre las
> 5 semillas**, no sobre un ensamble por voto mayoritario. Es una simplificación real y se
> declara: un ensamble podría dar un número algo distinto, aunque con 0/10 semillas positivas
> y un rechazo a p<0,0001 no hay margen para que cambie el signo.
>
> **Diagnóstico de trial degenerado**: ninguna corrida colapsó a «siempre neutral» — al
> contrario, todas mantuvieron |exposición| ≈ 0,6-0,93 y 433-813 cambios en 234 sesiones. El
> problema no fue falta de exploración sino **exceso de operación** frente al costo.


| Parámetro | Valor inicial |
|---|---|
| Algoritmo · acción | PPO (SB3) · 5 niveles discretos |
| Learning rate | 2.5e-4 |
| `n_steps` / `batch_size` | 2048 / 256 |
| Épocas · `γ` · GAE `λ` · clip | 10 · 0.995 · 0.95 · 0.20 |
| Red | `[128,128]`, Tanh |
| Timesteps de confirmación · semillas | 1.5 M · 10 |

**Espacio HPO:** `learning_rate` 1e-5…1e-3 log · `n_steps` {512,1024,2048,4096} · `batch_size`
{64,128,256,512} · `n_epochs` 3…15 · `γ` {0.97,0.99,0.995,0.999} · `gae_lambda` 0.90…0.99 · `clip_range`
{0.1,0.2,0.3} · `ent_coef` 1e-5…1e-2 log · `vf_coef` 0.25…1.0 · `max_grad_norm` 0.3…1.0 · `net_arch`
{[64,64],[128,128],[256,256]} · activación {Tanh, ReLU} · `κ_turn` 0.5…3.0 · `λ_dd` 0…0.5.

`ent_coef` es el crítico: con costos penalizados el agente colapsa a "siempre neutral" si la exploración es baja.

**Trial degenerado:** sin cambios de posición · una acción > 95 % de las barras · turnover diario medio > 20
cambios · NaN/inf · incumplimiento del contrato del entorno.

**Ensamble primario:** voto mayoritario del signo entre las 10 semillas → empate da `0` → tamaño = mediana de
`abs(w)` entre las semillas del signo ganador → acción final dentro de los cinco niveles. Las hipótesis se
prueban sobre la serie diaria del ensamble.

### 10.2 Arquitectura de selección

**Desarrollo:** 60 trials de Optuna · 3 semillas · 300 k timesteps · CV purgada K=6 con embargo · objetivo =
IQM del Sharpe OOF diario neto con penalización por degeneración/turnover · cada trial produce una serie OOF
agregada entre semillas.

**Selección:** top-3 configuraciones · 10 semillas · 1.5 M timesteps · selección por Sharpe diario neto del
ensamble · desempates: menor MaxDD, menor turnover, menor complejidad · **no se amplía el espacio de
hiperparámetros después de mirar el bloque de selección**.

**Refit final:** la ganadora se reentrena una vez sobre desarrollo+selección con las 10 semillas; scaler,
normalizador de reward y HMM se reajustan solo sobre ese tramo; todo se congela antes del hold-out; **no hay
reentrenamiento durante el hold-out**.

### 10.3 Ablaciones: diseño factorial, no escalera (corrección)

La presentación anterior sugería `backbone → +régimen → +sentimiento → híbrido`, pero `PPO+sentimiento` no
contiene el régimen: no es acumulativa. Se presenta como diseño, sin flechas:

| Sistema | Régimen | Sentimiento | Tipo de integración |
|---|---|---|---|
| `PPO backbone` | No | No | — |
| `PPO+régimen` | Sí | No | Feature en la red |
| `PPO+sentimiento` | No | Sí | Feature en la red |
| `Híbrido` | Sí | Sí | Régimen en la red + sentimiento en wrapper externo |

**Doble reporte obligatorio**, porque responden preguntas distintas:

- **Ablación controlada** — las cuatro variantes con los **mismos hiperparámetros congelados** (los de la
  configuración ganadora). Es la que sustenta H2 y H3, porque aísla el efecto de la variable.
- **Mejor sistema** — cada variante con HPO independiente, contando todos sus trials en `N_trials`. Es una
  comparación entre sistemas optimizados, no una estimación del efecto del componente. Se reporta como secundaria.

### 10.4 Brazo 2 — LLM como decisor directo

> **DESCOPADO (2026-08-25).** El brazo LLM no se construyó. Con él caen §10.5 (híbrido), §7
> (corpus textual), H3 y H4, los tests 5/11/12, y las entregas 4.5b/4.8/4.11 de §14.
>
> Lo que sigue escrito **no** se ejecutó y se conserva como diseño para un trabajo posterior.
> Se deja en su sitio en vez de borrarlo porque el recorte de alcance es una decisión del
> operador con fecha, no un olvido, y el lector tiene que poder ver qué se planificó y qué se
> hizo. `06-RESULTADOS.md` lista esto entre las limitaciones declaradas.


**Compuerta de viabilidad (antes del HPO):** 200 llamadas miden latencia p50/p95, tokens, tasa de error y de
JSON inválido, costo real y estabilidad del `model_id`. Si la corrida completa a 5 min cabe en presupuesto y
latencia, todos siguen a 5 min; si no, **todos** pasan a 15 min antes de F5. Nunca cadencias mixtas.

**Configuración:** DeepSeek V3 con `model_id` exacto congelado en el piloto · modelo local cuantizado como
sensibilidad opcional · tres réplicas con caches separados, ensamble = mediana de las tres acciones firmadas ·
en el bloque de desarrollo, solo llamadas piloto y muestra estratificada para construir el prompt; la
evaluación secuencial completa se reserva al bloque de selección y, tras el congelamiento, al hold-out.

**Entrada causal:** últimas 24 barras y resumen de indicadores · posición y PnL no realizado · macro as-of ·
posteriores filtradas de régimen · `s_d` · hasta 8 documentos preapertura sanitizados.

**Salida:** `{ "direccion": "short|flat|long", "tamano": 0.0, "confianza": 0.0, "justificacion": "…" }` con
`tamano ∈ {0, 0.5, 1}` compatible con `direccion`; el parser convierte a los cinco niveles; un reintento por
JSON inválido; agotados reintentos o timeouts se **mantiene `w_prev`** y se marca `unavailable` — pasar a flat
sería una decisión de trading disfrazada de fallback.

**Reproducibilidad:** se registra prompt, respuesta cruda, acción parseada, tokens, latencia, timestamp,
proveedor, `model_id`, intento y error. Canario de 20 prompts fijos comparando acuerdo de acción y estructura.
Si el proveedor cambia materialmente durante una corrida, la corrida se invalida.

**Contaminación y alcance inferencial:** el modelo pudo ver el período evaluado. El test de anonimización no lo
elimina. Además, **el LLM no se evalúa secuencialmente en el bloque de desarrollo, así que no entra en la matriz
OOF** y queda fuera del universo de White RC, SPA y PBO. Por ambas razones, H4 es exploratoria y el brazo se
describe como **simulación retrospectiva**, no como out-of-sample puro.

### 10.5 Brazo 3 — Híbrido

Política = **misma instancia congelada de `PPO+régimen`**. `s_d` no entra en su red ni altera su entrenamiento.

```text
s_d = −u_d

si sign(s_d) = sign(w_hat)  y  abs(s_d) > τ:
        w = clip( w_hat · (1 + β·abs(s_d)), −1, 1 )
si no:
        w = w_hat
```

- Configuración principal: `β = 0.35`, `τ = 0.25`.
- **Grid efectivo: 16 estrategias distintas**, no 18. Con `β = 0`, `τ` no tiene efecto, así que los tres puntos
  `(0, 0.10)`, `(0, 0.25)`, `(0, 0.40)` son la misma estrategia: se registra **un solo punto `β = 0`** más
  `5 valores de β > 0 × 3 de τ`. White RC, SPA y PBO no reciben columnas idénticas. `N_trials` adopta la
  convención conservadora (cuenta los 18 registros) y se documenta.
- `β = 0` debe reproducir **bit a bit** posiciones y curva de capital de `PPO+régimen`.
- Sensibilidad alternativa: modulación del **umbral de entrada** en vez del tamaño.

**Diagnósticos obligatorios:** % de decisiones donde se cumple la condición · % donde la exposición cambia tras
el clip · distribución de `abs(w − w_hat)` · **y `ρ`, la correlación entre las series diarias del híbrido y de
`PPO+régimen`**, porque de ella depende directamente el poder de H3 (§11.2). Si el cambio efectivo ocurre en
menos del 10 % de las decisiones, la modulación se declara débil o cosmética.

### 10.6 Baselines supervisados (especificación completa, antes ausente)

| Ítem | Definición |
|---|---|
| Target | Signo del retorno simple de la barra siguiente, tres clases con banda neutral |
| Banda neutral | `abs(r_{b+1}) < 0.25·σ12`; la clase neutral se predice explícitamente, no se descarta |
| Horizonte | Una barra (5 min), coherente con la cadencia congelada |
| Pérdida | Log-loss multiclase con pesos inversos a la frecuencia de clase |
| Probabilidad → posición | Cuatro umbrales sobre `p(alcista) − p(bajista)`, calibrados **solo en train**, que mapean a `{−1,−0.5,0,+0.5,+1}` |
| Estado endógeno | **Excluido** de las features de entrenamiento. `w_prev` solo aparece durante su propio backtest secuencial, generado por sus propias decisiones previas |
| HPO | Reducido y contabilizado en `N_trials` |

La exclusión del estado endógeno es necesaria: `w_prev`, PnL no realizado y barras en posición dependen de las
decisiones previas de la propia estrategia y no existen como columnas históricas estáticas.

### 10.7 Igualdad de información en H1 (corrección)

Se congelan **dos versiones** de cada baseline supervisado:

- **`baseline_matched`** — precio, volatilidad, tendencia, tiempo, macro y **régimen**, *sin* sentimiento. Es el
  comparador de **H1 confirmatoria**, porque tiene exactamente la misma información que `PPO+régimen`.
- **`baseline_all_features`** — añade `s_d`. Se reporta como **comparación exploratoria de sistemas**.

Sin esto, un baseline podía ganar por disponer de una variable extra, no por ser mejor método.

### 10.8 Controles y baselines externos

**Controles:** `always_flat` (Sharpe se reporta `NA`, no se fuerza a 0) y `random` (100 corridas uniformes sobre
los cinco niveles).

**Baselines externos:** buy-and-hold intradía · momentum SMA(10,30) · momentum con volatilidad objetivo ·
regresión logística (matched y all-features) · LightGBM (matched y all-features) · regla solo-régimen · regla
solo-`s_d` · mean reversion intradía por z-score.

### 10.9 Matriz de información

| Sistema | Precio/indic. | Posición | Macro | Régimen | `s_d` | Docs | Acción |
|---|---|---|---|---|---|---|---|
| `PPO backbone` | Sí | Sí | Sí | No | No | No | 5 niveles |
| `PPO+régimen` (Brazo 1) | Sí | Sí | Sí | Sí | No | No | 5 niveles |
| `PPO+sentimiento` | Sí | Sí | Sí | No | Sí | No | 5 niveles |
| `Híbrido` (Brazo 3) | Sí | Sí | Sí | Sí | Wrapper | No | Continua tras modulación |
| `LLM directo` (Brazo 2) | Resumen 24 barras | Sí | Sí | Sí | Sí | ≤ 8 | JSON → 5 niveles |
| `baseline_matched` | Sí | Solo `w_prev` propio | Sí | Sí | **No** | No | 5 niveles calibrados |
| `baseline_all_features` | Sí | Solo `w_prev` propio | Sí | Sí | Sí | No | 5 niveles calibrados |
| Reglas técnicas | Sí | Según regla | No | No | No | No | −1/0/+1 o escalada |

---

## 11. Plan de análisis estadístico

### 11.1 El resultado que condiciona todo lo demás

Con `SE(ΔSR_anual) ≈ √252 · sqrt(2(1−ρ)/n)`, el número de sesiones necesarias para detectar una mejora de
Sharpe de **0.20** con 80 % de potencia a α = 0.05 es:

| ρ entre las dos series diarias | Sesiones necesarias |
|---|---|
| 0.50 | ≈ 49 400 (≈ 196 años) |
| 0.90 | ≈ 9 900 (≈ 39 años) |
| 0.98 | ≈ 1 980 (≈ 8 años) |
| 0.99 | ≈ 990 (≈ 4 años) |
| 0.995 | ≈ 494 (≈ 2 años) |

**Conclusión, que se escribe en el capítulo 4 antes de cualquier tabla:** comparar dos estrategias poco
correlacionadas por diferencia de Sharpe es **indecidible a cualquier tamaño de muestra realista**. Solo los
contrastes entre sistemas que comparten backbone tienen alguna posibilidad.

Esto reordena las hipótesis y, bien argumentado, **es una contribución del trabajo**: buena parte de la
literatura que afirma que un agente RL supera a un baseline sobre uno o dos años de datos está reportando ruido.

### 11.2 Precisión por ventana

`SE(SR_anual) ≈ √252 · sqrt((1 + SR_d²/2)/n)` para el nivel absoluto:

| Ventana | n | IC 95 % del Sharpe absoluto |
|---|---|---|
| Hold-out de 1 año (opción A) | 250 | ± 1.97 |
| Hold-out de 2 años (opción B) | 500 | ± 1.39 |
| OOF del bloque de desarrollo | ~1 000–1 250 | ± 0.88 a ± 0.98 |

IC 95 % de la **diferencia pareada**, por `ρ`:

| ρ | n = 250 | n = 500 | n = 1 000 |
|---|---|---|---|
| 0.50 | ± 1.97 | ± 1.39 | ± 0.98 |
| 0.80 | ± 1.24 | ± 0.88 | ± 0.62 |
| 0.90 | ± 0.88 | ± 0.62 | ± 0.44 |
| 0.95 | ± 0.62 | ± 0.44 | ± 0.31 |
| 0.98 | ± 0.39 | ± 0.28 | ± 0.20 |
| 0.99 | ± 0.28 | ± 0.20 | ± 0.14 |

**Un año de hold-out no distingue un Sharpe de 1.24 de uno de 0.62.** Esta es la razón de la recomendación de §1.

> **CIFRAS REALES (2026-08-25).** Las tablas de arriba son analíticas y su fila relevante ya no
> es ninguna de las listadas: el hold-out efectivo tiene **n = 584** (no 500 ni 677 — la máscara
> excluye 95 festivos y 245 sesiones incompletas), y la correlación medida entre los dos brazos
> es **ρ ≈ 0,92**, no los 0,98-0,99 que se supusieron.
>
> Con el entorno construido, la potencia se **simula** en vez de interpolarse en la tabla.
> Bootstrap estacionario pareado, 2.000 réplicas, bloques 5-20, 20 repeticiones por celda:
>
> | ΔSharpe verdadero | Potencia (IC 95% excluye 0) |
> |---|---|
> | 0,41 | 45% |
> | 0,83 | 90% |
> | 1,55 | 100% |
>
> **Mínimo detectable al 80% ≈ ΔSharpe 0,7** *a ρ = 0,92*. La interpolación analítica para
> ρ=0,90 y n=500 daba ±0,62, así que el orden de magnitud coincide; lo que cambia es que ahora
> es una medición.
>
> **Y una advertencia que el trabajo terminó necesitando**: esa cifra depende críticamente de
> la correlación, que **no es constante entre bloques**. En selección los dos brazos van a
> ρ = 0,97; en el hold-out, a ρ = 0,58. Con esa segunda correlación la potencia se desploma:
>
> | ρ | ΔSharpe 0,4 | 0,8 | 1,2 | 2,0 |
> |---|---|---|---|---|
> | 0,92 | 25% | 75% | 95% | 100% |
> | **0,58** | 20% | 20% | **35%** | **60%** |
>
> Es decir: un «mínimo detectable» calculado con la ρ de un bloque **no se traslada** al otro.
> Reportar un único número de potencia para todo el trabajo habría sido engañoso.
>
> **Y una cifra que acota lo que este trabajo puede prometer**: en el hold-out el IC del Sharpe
> del propio buy&hold pasivo es **[−1,83, +0,45]** — cruza el cero. Con 584 sesiones ni siquiera
> el benchmark pasivo es distinguible de cero. Cualquier conclusión sobre diferencias finas está
> fuera del alcance de estos datos, y decirlo es parte del resultado.

### 11.3 Hipótesis, reordenadas según el poder disponible

**Confirmatorias (familia Holm–Bonferroni, FWER 0.05) — solo las que tienen `ρ` alta por construcción:**

- **H2:** `PPO+régimen` supera al `PPO backbone` (ablación controlada, mismos hiperparámetros).
- **H3:** el híbrido supera al mismo `PPO+régimen` del que parte.

**Reportada con precisión declarada, no como contraste decidible:**

- **H1:** `PPO+régimen` frente a `baseline_matched`. Se reporta la diferencia pareada, su IC, `ρ` y `n`, junto
  con la afirmación predeclarada de que **el diseño no puede resolver este contraste** y de que la ausencia de
  significancia no es evidencia de equivalencia.

**Exploratoria:**

- **H4:** LLM directo frente a `PPO+régimen`, con las salvedades de §10.4.

### 11.4 Estadístico primario y ventanas de evidencia

- Métrica: Sharpe anualizado sobre `retorno_diario_d`.
- Estadístico: **diferencia pareada de Sharpe** sobre los mismos días de la máscara común.
- p-values: bootstrap estacionario pareado, 10 000 réplicas, longitud de bloque por regla automática truncada a 5–20 sesiones.
- Holm–Bonferroni solo sobre H2 y H3.

| Ventana | Estatus inferencial |
|---|---|
| **Hold-out** | **Confirmatoria estricta.** Única fuente de p-values confirmatorios |
| Bloque de selección | Desempeño de selección/confirmación, presentado **por separado**, sin lenguaje de validación independiente |
| Selección + hold-out combinados | **Sensibilidad descriptiva únicamente**: sin p-value confirmatorio. Declararlo de antemano no elimina que un año se usó para seleccionar |
| OOF del bloque de desarrollo | Evidencia pre-hold-out **temporalmente causal pero sujeta a sesgo de selección**; se interpreta junto con White RC, SPA, PBO y DSR |

**Corrección importante:** la serie OOF es válida temporalmente (cada bloque lo predijo un modelo que no lo vio),
pero el desempeño OOF de la configuración *ganadora* sigue siendo optimista porque fue elegida por ese mismo
desempeño. `OOF sin leakage ≠ OOF sin sesgo de selección`. En el documento no vuelve a aparecer la expresión
"sin contaminación de selección" aplicada a la serie OOF.

### 11.5 Umbrales de relevancia

| Criterio | Umbral |
|---|---|
| Mejora de Sharpe | `ΔSharpe ≥ 0.20` |
| Rentabilidad exigente | Retorno anual neto > 0 bajo 10 pips round-trip fijo |
| PBO | ≤ 0.20, con la regla de fallo de §11.7 |
| DSR | Reportado con la regla de §11.8 |
| Mínimo por régimen | 30 sesiones |

> **RESULTADO (2026-08-25, descomposición CTR-RESEARCH-DECOMP-001, 0 trials).** El umbral
> «rentabilidad exigente bajo 10 pips round-trip fijo» se queda muy corto como criterio: el
> trabajo no falla por 10 pips, falla por **cualquier** costo.
>
> El **spread de break-even es negativo** (`s* = −0,29` pips en `ppo_regime`): el alfa —0,67
> pips por operación— no cubre **ni la comisión de 0,5 pips por lado**, así que la estrategia
> no es rentable ni con un round-trip de **cero**.
>
> Y sin embargo **hay señal**: bruto +27,95% en hold-out, Sharpe bruto +2,23 con IC [+1,21,
> +3,29] que excluye el cero, 10/10 corridas positivas. El bruto es una **cota superior
> inalcanzable** (exige costo cero) y no es un claim de edge, pero descarta que el agente sea
> ruido.
>
> **La relevancia económica que este trabajo mide, entonces, no es «cuánto gana» sino «cuánto
> costo aguantaría»** — y la respuesta es: menos que la comisión. Ese es el umbral que un
> trabajo posterior tiene que mover, y no se mueve con el modelo.
>
> Detalle en [`06-RESULTADOS.md`](06-RESULTADOS.md) §5b.

### 11.6 Universo exacto de White RC, SPA y PBO (congelado tras F7, antes de F8)

`N_trials` y la matriz OOF **no son lo mismo**:

```text
N_trials      = todos los experimentos intentados, incluidos podados,
                fallidos y abandonados.

Matriz OOF    = solo candidatos COMPLETADOS, con serie de retornos
                disponible sobre la misma máscara de fechas.
```

Parámetros congelados: benchmark = `baseline_matched` seleccionado en el bloque de selección · 5 000
remuestreos para White RC y SPA · `S = 16` particiones para CSCV · candidatos con folds faltantes **excluidos de
la matriz y contabilizados aparte** · el grid híbrido entra deduplicado (§10.5) · los umbrales de los baselines
predictivos entran como candidatos.

> **RESUELTO (2026-08-25): White RC y SPA NO se computan, y esto es una decisión, no un olvido.**
>
> Ambos contrastan un ganador contra el universo de candidatos que compitió por serlo. Al
> renunciar al HPO (§15) y al recortar el alcance a **dos brazos**, ese universo tiene
> exactamente **dos miembros**: `ppo_regime` y `ppo_backbone`. Un SPA sobre dos candidatos
> produce un número con aspecto de rigor y contenido nulo — no hay distribución de máximos
> que deflactar cuando el máximo se toma sobre dos.
>
> Se declara la omisión con su razón en vez de publicar el número vacío
> (`src/research/inference.py::WHITE_SPA_OMISSION`, y en el JSON de cada corrida).
>
> **PBO y DSR sí se computan**, porque ninguno depende del tamaño del universo: el PBO mide si
> el ganador in-sample sobrevive fuera, y el DSR deflacta por el CONTEO DE TRIALS del activo,
> que es otra cosa (§11.12).

### 11.7 Regla de fallo del PBO (predeclarada)

```text
Si PBO > 0.20:
  - NO se modifica el universo candidato ni se retunea nada;
  - se suspende la pretensión confirmatoria del trabajo;
  - el hold-out puede abrirse únicamente como evaluación de un sistema
    con riesgo alto de selección, etiquetado así en todas las tablas,
    o el estudio se cierra sin apertura;
  - la elección entre esas dos opciones se registra por escrito
    ANTES de calcular el PBO.
```

Sin esta regla, la compuerta se convertiría en otro objetivo de tuning.

### 11.8 DSR: cantidad reportada, no compuerta

Con `N_trials ≈ 80–100`:

| Serie | n | Sharpe observado | Sharpe umbral por azar | DSR |
|---|---|---|---|---|
| Hold-out de 1 año | 250 | 1.24 | ≈ 2.45 | ≈ 0.11 |
| OOF de desarrollo | 1 250 | 1.24 | ≈ 1.10 | ≈ 0.63 |
| Desarrollo+selección | 1 500 | 1.24 | ≈ 1.04 | ≈ 0.69 |
| OOF de desarrollo | 1 250 | 1.60 | ≈ 1.10 | ≈ 0.87 |

*(skew 0 y kurtosis normal; se recalcula con los momentos reales.)*

Exigir `DSR ≥ 0.95` equivalía a declarar el trabajo fallido por construcción. **Decisión:** el DSR se calcula
sobre la serie OOF del bloque de desarrollo y se reporta de nuevo, sin reselección, sobre el hold-out, siempre
junto a `n`, `N_trials` y el umbral por azar. Interpretación: `≥ 0.90` evidencia fuerte contra el sobreajuste de
selección · `0.60–0.90` evidencia parcial · `< 0.60` no se puede descartar que la ventaja venga de la búsqueda,
y así se escribe.

### 11.9 Subperíodos

Los subperíodos anteriores al hold-out (COVID 2020, ciclo macro 2022) se reportan **únicamente con predicciones
OOF** de la CV purgada y etiquetados `OOF`; el hold-out se etiqueta `OOS estricto`. Nunca se ponen números
in-sample y de hold-out en la misma columna sin etiqueta. Si un subperíodo carece de cobertura OOF suficiente,
**la celda queda vacía**.

### 11.10 Incertidumbre

Entre semillas/réplicas: IQM, IC bootstrap y performance profiles (`rliable`) · temporal/económica: bootstrap
estacionario sobre retornos diarios · combinada: bootstrap jerárquico (bloques de días y, dentro de cada
réplica temporal, agregación entre semillas). Nunca `día × semilla` como observaciones independientes.

### 11.11 Métricas secundarias

Retorno anualizado, Sortino, Calmar, Profit Factor, Win Rate, Information Ratio contra buy-and-hold intradía,
MaxDD, VaR 95 %, CVaR 95 %, volatilidad, turnover, número de operaciones, tiempo en mercado, costo total y `ρ`
frente al comparador correspondiente. El desglose por régimen y subperíodo es descriptivo.

### 11.12 Conteo de `N_trials`

Un trial = combinación única de `feature_set · reward · architecture · hyperparameters · β · τ · cadence ·
prompt_version · model_id`. Las semillas son réplicas y no multiplican el conteo. Podados, fallidos y
abandonados cuentan. Umbrales de los baselines predictivos y prompts probados cuentan.

> **CORRECCIÓN (2026-08-25): el contador NO empieza en cero, y no sale de MLflow.**
>
> Este apartado decía «contador automático desde MLflow», lo que implicaba arrancar de 0 con
> los experimentos de esta tesis. Es incorrecto y **sub-deflactaría el DSR**, que es la
> violación concreta que la constitución §2 prohíbe.
>
> La autoridad es `.claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md`, cuyo front-matter
> declara `n_trials_total` para el ACTIVO, y el ADR-0023 fija que **una partición nueva jamás
> resetea N**. Estado al abrir este trabajo: **111 heredados**. La apertura
> `H-TESIS-RL-01` suma **+2 AT** (las dos configuraciones evaluadas sobre selección) → **113**.
>
> No se cobran trials por: los hiperparámetros (congelados de `v215b_baseline.yaml`, prior
> ex-ante, §15), ni las 5 semillas (réplicas, como dice este mismo apartado), ni la selección
> de `K` del HMM por BIC — que ocurrió íntegramente dentro de desarrollo con un criterio
> declarado antes de mirar nada. Ese último caso se declara explícitamente en el registro
> porque es discutible: un revisor que lo considere trial debe sumar **4 FT** y releer el DSR
> con N=117.

### 11.13 Regla de interpretación final

Soporte fuerte para una hipótesis confirmatoria solo si: p ajustado por Holm < 0.05 · `ΔSharpe ≥ 0.20` ·
retorno anual neto positivo bajo 10 pips fijos · `PBO ≤ 0.20` · el IC de la diferencia pareada excluye 0 **y**
el diseño tenía poder declarado para ese tamaño de efecto. Si solo se cumple una parte, el resultado se
describe como evidencia estadística sin relevancia económica, relevancia económica sin precisión suficiente, o
**contraste indecidible**. El criterio no se modifica después de abrir el hold-out.

---

## 12. Protocolo de hold-out

> **IMPLEMENTADO EN CÓDIGO (2026-08-25), no solo escrito.**
>
> La Regla B vivía en dos documentos. Una regla que solo vive en un documento se salta sin
> querer un martes por la tarde, y su violación no deja rastro. Ahora hay **dos gates
> ejecutables**, y los dos salen con código 2 sin leer un byte del hold-out mientras el
> pre-registro no esté en `IMPLEMENTED`:
>
> | Gate | Qué protege |
> |---|---|
> | `scripts/analysis/thesis_statistics.py --block holdout` | que nadie calcule estadística sobre el hold-out |
> | `scripts/analysis/thesis_open_holdout.py` | que nadie evalúe un modelo sobre el hold-out |
>
> El segundo es además el **único** script del repositorio que evalúa modelos sobre ese
> bloque, y deja constancia de la apertura en `outputs/thesis/holdout_opening.json` (fecha,
> hash de la máscara, modelos evaluados, `n_trials` vigente). Si ese fichero ya existe, avisa:
> una segunda apertura no es una repetición inocente, es convertir el juez final en un segundo
> bloque de selección.
>
> Guard de no-regresión: `tests/regression/test_research_inference.py::
> test_holdout_is_blocked_while_the_preregistration_is_unsigned` ejecuta el script y comprueba
> el código de salida — comportamiento, no intención.
>
> **Manifiesto**: no se materializó como `configs/frozen_YYYYMMDD.yaml`. Sus campos viven
> repartidos en artefactos ya hasheados y versionados (`partition.yaml`,
> `evaluation_mask.json`, `feature_schema.json`, `holdout_opening.json`, la ficha del HMM en
> el dataset portable) y la firma está en §9 del pre-registro. Es una desviación del plan y se
> declara como tal; lo que el manifiesto garantizaba —que los hashes estén fijados antes de
> abrir— sí se cumple.


**Universo permitido:** `PPO backbone`, `PPO+régimen`, `PPO+sentimiento` e híbrido principal (ablación
controlada) · LLM directo con prompt, `model_id` y tres réplicas congeladas · baselines externos en su versión
`matched` y `all_features` y los dos controles · escenario de costos base y 3/5/10/30 pips fijos sobre las
mismas posiciones · desglose predefinido por régimen y subperíodo · métricas, pruebas H2–H3, reporte de H1 con
precisión declarada y análisis exploratorio H4.

**Prohibido:** Optuna o cualquier HPO · grid de β/τ · nuevas features · nuevos baselines · cambio de prompt,
modelo, cadencia o fallbacks · selección de reward · reentrenamiento · inspección iterativa para corregir el
sistema según resultados.

**Manifiesto pre-apertura** (`configs/frozen_YYYYMMDD.yaml`, firmado, commiteado y copiado fuera del equipo):

```text
git_commit · dependency_lock_hash · environment_image_digest
calendar_hash · evaluation_mask_hash · data_hash · corpus_hash · corpus_filter_hash
feature_schema_hash · cost_contract_hash · model_config_hash · model_weight_hashes
seed_list · scaler_hash · vecnormalize_hash · hmm_hash · hmm_K
prompt_hash · API_model_id · cadence
N_trials_acumulado · oof_matrix_hash
primary_hypotheses · primary_metric · statistical_analysis_plan
power_analysis (rho esperado, n e IC previsto por contraste)
pbo_failure_decision   # escrita antes de calcular el PBO
allowed_holdout_outputs
holdout_open_count: 0
```

**Apertura y errores:** el pipeline se ejecuta completo y genera artefactos inmutables antes de reescribir la
narrativa. Ante un bug material, la primera corrida se marca `invalid` y no se borra; se corrige, se reejecuta
todo el bloque pre-hold-out pertinente, y la segunda apertura se declara incrementando `holdout_open_count` y
`N_trials`. No se presenta una reapertura como si fuera la primera.

---

## 13. Tests automatizados obligatorios

| # | Test | Fase |
|---|---|---|
| 1 | **Sentinel de futuro:** una feature construida a propósito con el retorno siguiente debe mejorar de forma marcada; confirma que el detector funciona | F1 |
| 2 | **Shuffle:** con retornos aleatorizados **no existe ventaja positiva estable**; el desempeño es compatible con cero o con always-flat y no hay relación predictiva reproducible. *(No se exige que se parezca al control random: un PPO bien regularizado puede aprender a quedarse flat.)* | F5 |
| 3 | **HMM filtrado:** serie completa vs truncada en `t` producen el mismo posterior en `t` | F3 |
| 4 | **Scaler:** parámetros invariantes al contenido de validación/test | F1 |
| 5 | **Texto:** ningún documento posterior al corte de 08:00 entra en `s_d` ni en el prompt | F2 |
| 6 | **Macro as-of:** una publicación no aparece antes de su timestamp real | F1 |
| 7 | **Paridad de backtest:** entorno propio vs vectorbt, tolerancia 1e-6 | F4 |
| 8 | **VecNormalize congelado** durante la evaluación | F5 |
| 9 | **Costos:** apertura, reducción, flip y cierre cobran los lados correctos sin doble conteo | F4 |
| 10 | **Cronología:** una acción al cierre de `b` solo captura el retorno `b+1` | F4 |
| 11 | **Signo híbrido:** casos manuales validan largo/corto desde la perspectiva USD/COP | F7 |
| 12 | **Identidad híbrida:** `β=0` reproduce posiciones y PnL de `PPO+régimen` | F7 |
| 13 | **Contabilidad:** la serie diaria coincide con la variación de equity **e incluye el costo del cierre terminal**; el Sharpe no se calcula sobre barras | F4 |
| 14 | **Etiqueta de subperíodo:** toda celda anterior al hold-out procede de OOF y viene marcada | F8b |
| 15 | **Máscara común:** todos los sistemas se evalúan sobre el mismo conjunto de sesiones válidas; ninguna sesión inválida entra como retorno 0 ni cuenta en `n` | F4 |
| 16 | **Spread esperado:** `spread_d` se calcula con los posteriores y no con `argmax`; el `argmax` solo aparece en el desglose descriptivo | F4 |

Los 16 tests son irrenunciables.

---

## 14. Entregables

> **AJUSTADO al alcance real (2026-08-25).** El listado original suponía tres brazos —PPO, LLM
> e híbrido— y HPO con Optuna. Con **2 brazos y sin HPO**, las entregas ligadas a lo descopado
> dejan de existir: **4.5b** (sensibilidad β/τ), **4.8** (métricas LLM), **4.11** (activación
> del híbrido) y la figura de **sensibilidad de β** no se producen porque no hay β, ni τ, ni
> LLM. Tampoco hay logs de Optuna. Lo que se elimina es lo que mide algo que no se construyó;
> nada de lo que sí se construyó se recorta.

**Ocho tablas:** 4.1–4.2 datos y particiones · 4.3 desempeño principal bajo costo base **con IC de
Sharpe** · 4.4 desempeño por régimen **con la regla de N<20 aplicada** · 4.5 ablación
`ppo_regime` vs `ppo_backbone` · 4.6 sensibilidad de costos ×1/×2/×3 · 4.7 DSR y PBO · 4.9
comparaciones **pareadas, con `ρ`, `n` e IC** · 4.10 robustez entre semillas.

**Nueve figuras:** curvas de capital · underwater/drawdown · Sharpe móvil de 60 sesiones ·
sensibilidad a costos · distribución de acciones por régimen · perfiles por semilla · esquema de
particiones · Sharpe por régimen · sesión representativa con posiciones.

**Artefactos:** `config/research/feature_schema.json` (hasheado) · `config/research/evaluation_mask.json`
(hasheado) · `config/research/partition.yaml` · retornos diarios por sistema y por semilla
(`data/thesis/ppo/*.json`) · el dataset portable con la ficha del HMM · logs por tarea en Airflow
(`research_thesis_ppo_training`) · reporte automático de coherencia.

`scripts/presentation/generar_resultados_y_figuras.py` produce las 8 tablas y 9 figuras desde
artefactos, **sin edición manual de números**, y emite el reporte de coherencia de §19.4 con seis
comprobaciones.

---

## 15. Fases, DoD y compuertas

> **ESTADO REAL (2026-08-25).** Esta tabla se ejecutó **fuera de orden y con menos alcance** del
> planificado, y conviene decirlo aquí antes de que alguien la lea como un plan cumplido.
>
> | Fase | Estado | Nota |
> |---|---|---|
> | F0 | **PARCIAL** | Sin benchmark formal de cómputo. Medido de hecho: 300k pasos ≈ 25 min por corrida en CPU. |
> | F1 | **HECHA** | Máscara hasheada, partición congelada, features y su schema. Corregidos por el camino: timezone, escala macro FX, y 59 días de Brent corrupto dentro del hold-out. |
> | F2 | **DESCOPADA** | No hay brazo LLM. |
> | F3 | **HECHA** | K=4 por BIC en desarrollo, congelado. Hallazgo de «tipo de día» en §8.3. |
> | F4 | **HECHA** | Entorno, costos y baselines, con los tests 3/4/8/9/10/13/15/16. |
> | **F5** | **SIN HPO — decisión explícita** | ver abajo |
> | F6, F7 | **DESCOPADAS** | Sin LLM no hay híbrido. |
> | F8 | **PARCIAL** | Estadística construida y verificada; White/SPA declarados no computables (§11.6). |
> | F8b | **BLOQUEADA por la Regla B** | El pre-registro sigue en `PARTIAL`. El gate está en código: `thesis_statistics.py --block holdout` sale con código 2 y no lee un solo dato. |
>
> ### F5 sin HPO: qué se gana y qué se paga
>
> Los hiperparámetros se toman **congelados** de `config/experiments/v215b_baseline.yaml`. Son un
> prior declarado ex-ante, no una elección hecha mirando estos datos, así que **no suman trials**
> (§11.12) y el DSR se queda cerca del N heredado en vez de saltar a ≥171.
>
> Lo que se paga, y hay que escribirlo: **sin universo de candidatos, White RC y SPA pierden su
> sentido** (§11.6). Y no se puede afirmar que el resultado sea el mejor alcanzable con PPO —
> solo que es el que dan unos hiperparámetros razonables y declarados de antemano. Un resultado
> negativo bajo esta configuración **no cierra** la pregunta «¿puede PPO?»; cierra la pregunta
> «¿puede PPO con esta receta, sobre estos datos, contra estos costos?».
>
> **El número de pasos sí se decidió aquí**: 300.000, no los 2.000.000 del baseline. Ese baseline
> entrena sobre 70.000 barras continuas; desarrollo son 499 × 59 = 29.441 pasos, así que 2M serían
> 68 pasadas sobre los mismos datos. Lo que se hereda es la RECETA, no un conteo atado a otro
> tamaño de muestra. 300k ≈ 10 pasadas. La decisión es **común a las dos configuraciones**, así
> que no puede sesgar la ablación de H2.


| Fase | Contenido | DoD | Compuerta |
|---|---|---|---|
| **F0** | Repo, entorno, DVC, MLflow, DuckDB, CLI, Docker | Experimento dummy con commit, config y hash · **benchmark de cómputo: 100 k timesteps → segundos por corrida, memoria máxima, throughput con 4 procesos, proyección del HPO completo** | Si la proyección supera el presupuesto, se activa la ruta mínima viable **antes** de Optuna |
| **F1** | Auditoría e ingesta numérica | Cobertura por año, contrato de datos, tolerancias de proveedor, features base, máscara de evaluación, tests 1/4/6 | Si el período no es viable, recortar y activar `K=4`; decidir opción A o B de §1 |
| **F2** | Corpus y sentimiento | Filtro congelado, señal completa, 200 documentos anotados, test 5 | Si `s_d = 0` en > 40 % de sesiones, ampliar fuentes o revisar ventana usando solo pre-hold-out |
| **F3** | Régimen | HMM causal, `K` elegido y congelado, alineación de estados, test 3 | `K` no vuelve a discutirse después de esta fase |
| **F4** | Entorno, costos, baselines y piloto LLM | Baselines matched y all-features, 2 controles, tests 7/9/10/13/15/16, piloto de 200 llamadas | Congelar 5 o 15 min **para todos** antes del HPO (**Regla A**) |
| **F5** | PPO y HPO | 60 trials en desarrollo, matriz OOF, top-3 evaluadas en selección, tests 2/8 | No ampliar el espacio tras mirar el bloque de selección |
| **F6** | LLM en el bloque de selección | Prompt y `model_id` congelados, tres réplicas, JSON inválido < 5 %, canario y anonimización | Si cambia el proveedor, invalidar y congelar una versión estable |
| **F7** | Híbrido y ablaciones | Grid deduplicado, ablación controlada y optimizada, tests 11/12, diagnóstico de activación y `ρ` | — |
| **F8** | Selección final, estadística y refit | Matriz OOF congelada, White/SPA/PBO/DSR, decisión de fallo del PBO escrita **antes** de calcularlo, baseline principal, refit, análisis de poder redactado, manifiesto firmado | **Regla B**: sin manifiesto no se abre el hold-out |
| **F8b** | Hold-out | Universo cerrado ejecutado una vez, 12 tablas y 10 figuras, test 14, conteo de apertura registrado | — |
| **F9** | Escritura | Capítulos 4 y 5 rederivados; limitaciones y resultados negativos explícitos | — |

---

## 16. Calendario y ruta mínima viable

Quedan unos cuatro meses hasta la defensa prevista para diciembre de 2026. F2 y F5 son las fases más agresivas:
F2 abarca siete años de corpus con deduplicación, filtro semántico y 200 anotaciones; F5 son cientos de
entrenamientos.

| Semana | Fechas | Fase |
|---|---|---|
| 1–2 | 1–14 sep | F0 + F1 (la auditoría de cobertura y la decisión de §1 mandan) |
| 3–4 | 15–28 sep | F2 |
| 5 | 29 sep – 5 oct | F3 |
| 6–7 | 6–19 oct | F4 |
| 8–10 | 20 oct – 9 nov | F5 |
| 11 | 10–16 nov | F6 |
| 12 | 17–23 nov | F7 |
| 13 | 24–30 nov | F8 |
| 14 | 1–7 dic | F8b |
| 15–16 | 8–19 dic | F9 |

**Camino crítico:** F1 → F5 → F8 → F8b.

**Ruta mínima viable si se atrasa**, decidida **antes** de entrar en la fase afectada: (1) Optuna de 60 a 30
trials y CV de 6 a 4 folds · (2) semillas de 10 a 5 · (3) eliminar walk-forward de sensibilidad, RecurrentPPO,
baseline mean reversion y LLM local · (4) ejecutar el Brazo 2 sobre una muestra estratificada de sesiones del
hold-out, declarándolo y ajustando `n` en el análisis de poder.

**No se recorta jamás:** los 16 tests, el manifiesto, el contrato de costos, el hold-out único, la contabilidad
de `N_trials` y el análisis de poder.

---

## 17. Presupuesto y cumplimiento

CPU ≈ 200 h · API LLM USD 150 · 50 GB · defensa en diciembre de 2026.

Revisar términos de uso, `robots.txt` y límites de las fuentes · no redistribuir cuerpos completos de artículos:
publicar URLs, hashes, metadatos y scores cuando sea permitido · claves API fuera del repositorio con escaneo de
secretos en pre-commit · backup externo del corpus etiquetado, respuestas LLM, manifiestos y resultados de
hold-out · toda sustitución o pérdida de fuente se documenta y versiona.

---

## 18. Riesgos

| Riesgo | Mitigación congelada |
|---|---|
| Cobertura USD/COP insuficiente pre-2021 | Auditoría F1; recorte transparente y `K=4` |
| Mezcla inconsistente de proveedores | Tolerancias numéricas de §6.2; preferir descarte |
| Error de signo textual | Convención `s_d = −u_d` y test 11 |
| `β` inoperante | Cinco niveles, wrapper continuo y diagnóstico de cambio efectivo |
| Colapso a flat | `ent_coef`, criterios de degeneración y control always-flat |
| Doble conteo de spread | Mid + costo explícito y test 9 |
| Costo terminal perdido | Paso terminal explícito y test 13 |
| Sesiones inválidas como cero | Máscara común y test 15 |
| Leakage del HMM | Posterior filtrado y test 3 |
| Leakage macro/textual | Timestamps as-of y tests 5 y 6 |
| Sobreajuste por HPO | Matriz OOF congelada, White/SPA/PBO, DSR y `N_trials` automático |
| PBO alto tratado como objetivo de tuning | Regla de fallo predeclarada §11.7 |
| Contaminación del LLM | H4 exploratoria, ausencia de la matriz OOF declarada |
| Cambio silencioso de la API | Canario, `model_id`, invalidez si cambia durante la corrida |
| **Contrastes indecidibles presentados como concluyentes** | §11.1 escrita antes de las tablas; H1 reportada con precisión declarada |
| **DSR inalcanzable por construcción** | Regla de interpretación §11.8 |
| **Subperíodos in-sample** | Solo OOF, con etiqueta, y test 14 |
| Presupuesto insuficiente | Compuerta de benchmark en F0 y orden de recorte |

### Riesgos MATERIALIZADOS durante la ejecución (2026-08-25)

> **El riesgo que NO se materializó, y hay que decirlo también**: se temía que el supuesto de
> spread de §8.4 —una constante declarada, nunca medida, sobre un seed que solo tiene OHLCV—
> dejara la conclusión colgando de un número inventado. La descomposición lo descartó: el
> break-even es **negativo**, así que el veredicto se sostiene incluso a spread cero. El
> supuesto sigue sin medirse, pero **ya no es una amenaza a la validez**.


Los de la tabla anterior son los que se anticiparon. Estos ocurrieron de verdad, y se listan
porque un plan que solo enumera los riesgos previstos oculta cómo falla el trabajo real.

| Riesgo materializado | Qué pasó | Cómo se detectó | Estado |
|---|---|---|---|
| **Corrupción de una serie macro dentro del hold-out** | 59 días de Brent en 21-23 USD cuando el real estaba en 60-70 (2025-09-25 → 2025-12-19) | Log-retornos de ±108% en la feature; confirmado por ratio Brent/WTI de 0,366 frente a 1,062 | Reparado desde FRED (`fix_brent_corrupt_block.py`) |
| **Un guard declarado que no impidió nada** | `config/l0_macro_sources.yaml` declaraba el rango correcto `[30,150]` con `validation.enabled: true` y el dato entró igual. El validador **sí está cableado** (`data_validators.py:604`); los huecos son otros cuatro: ingesta diaria sin validar, `validate_data` que nunca lanza, restore que la salta, y config ausente que desactiva el guard en silencio | Al buscar por qué no había saltado; la primera hipótesis ("solo se instancia en tests") era **incorrecta** | CERRADO 2026-08-25 — ver §1 del plan de infraestructura |
| **Volumen inexistente asumido como feature** | El grupo de volumen de §6.5 tenía un criterio de admisión; medido, el 100% de las barras tiene volumen 0 | Al implementar el grupo | Grupo eliminado por su propio criterio |
| **`enable_deterministic_png` rompía el import de `pyplot`** | El wrapper no preservaba `__qualname__`; `pyplot._add_pyplot_note` lo rechazaba al importarse después del parche | 8 errores de colección en `tests/unit`, dos de ellos por esto | Corregido con `functools.wraps` + guard |
| **«Régimen» que dura día y medio** | Las persistencias del HMM dan duraciones de 1,6-1,9 días: son tipos de día, no regímenes | Al inspeccionar la matriz de transición | Declarado en §8.3; cambia la narrativa, no el uso |
| **El agente tenía señal y nadie lo había mirado** | Bruto +27,95% en hold-out (Sharpe +2,23, IC excluye cero, 10/10 corridas). El rechazo se reportó sin descomponerlo, y «PPO no funciona» era una lectura incorrecta de «ejecutar lo que PPO aprende no funciona» | Al reconstruir `bruto = neto + costo` de los artefactos guardados | **CERRADO 2026-08-25** — §5b de `06-RESULTADOS.md` |
| **La intuición sobre la frecuencia era errónea** | Se asumió que 59 decisiones/sesión era el problema; medido, el alfa vive en la alta frecuencia igual que el costo, y a 1 decisión/sesión el bruto ya es negativo | Re-scoring de las mismas decisiones cada `k` barras | **CERRADO 2026-08-25** — matiza la línea 1 de `BL-48` |
| **Potencia sobreestimada en la tabla analítica** | §11.2 asumía ρ = 0,98-0,99; el real es 0,92 en selección | Simulación directa | §11.2 corregida con la potencia medida |
| **Una violación de rango solo era error por encima del 10%** | `data_validators.py`: `if out_of_range_pct > 10: errors else: warnings`. Los 59 días de Brent nunca llegaron a ese umbral en un lote, así que un validador perfectamente cableado **y bloqueante** los habría dejado pasar como aviso | Al hacer verdes dos tests unitarios que llevaban tiempo en rojo esperando exactamente esa semántica | **CERRADO 2026-08-25** — un valor imposible es un error haya o no otros 999 correctos al lado |
| **La ρ entre brazos NO es constante entre bloques** | 0,97 en selección, **0,58** en hold-out; con la segunda, la potencia frente a ΔSharpe 1,2 cae del 95% al 35% | Al leer la tabla 4.5 del hold-out | §11.2 avisa de que un «mínimo detectable» no se traslada entre bloques |
| **Postgres muere por OOM al recorrer `asset_daily_ohlcv`** | 2.430 chunks de 7 días para 60.542 filas (25 filas/chunk): el hypertable se creó con el intervalo por defecto y el backfill 1979-2026 escribió décadas con él | 4 tests de `test_data_quality_floor.py` fallando con `server closed the connection unexpectedly` | **CERRADO 2026-08-25** — `fix_daily_hypertable_chunks.py`: **2.431 chunks → 6**, con respaldo CSV verificado de las 60.542 filas y las 3 vistas dependientes recreadas. La consulta pasa de matar al servidor a 0,7 s; los 7 tests del fichero pasan |
| Retail OHLCV vs SET-FX | Limitación explícita; no extrapolar a microestructura interbancaria |

---

## 19. Sustitución de resultados sintéticos

1. Marcar todas las celdas sintéticas del capítulo 4 antes de reemplazarlas.
2. Generar los resultados reales en archivos separados y versionados.
3. Sustituir tablas únicamente desde el script de reportes.
4. Verificar automáticamente. **IMPLEMENTADO** en
   `scripts/presentation/generar_resultados_y_figuras.py::coherence_report`, que escribe
   `outputs/thesis/coherencia_<bloque>.json` y devuelve código de salida distinto de cero si
   alguna comprobación falla. Las comprobaciones ligadas a `β` desaparecen con el híbrido; las
   seis que quedan son:

   | # | Comprobación | Qué rompería si fallara |
   |---|---|---|
   | 1 | `n` idéntico en partición, tabla y todas las series | una estrategia evaluada sobre menos sesiones que otra |
   | 2 | el Sharpe de la tabla 4.3 **es el mismo** que el del contraste pareado de la 4.9 | dos números distintos para la misma cosa en dos tablas |
   | 3 | `always_flat` tiene retorno y costo exactamente cero | el motor cobraría por no operar |
   | 4 | las 10 corridas (2 configuraciones × 5 semillas) existen | una tabla de semillas incompleta presentada como completa |
   | 5 | el DSR se deflacta con el N **heredado** (111), no con cero | sub-deflactar el DSR — la violación concreta de la constitución §2 |
   | 6 | la omisión de White/SPA está declarada en el JSON | una omisión silenciosa que se lee como olvido |
5. Aplicar la reconciliación de nomenclatura de §4 a las tablas 4.2, 4.3 y 4.5.
6. Reescribir discusión y conclusión desde cero con los resultados reales.
7. Si híbrido, PPO o LLM no superan controles y baselines, se reporta sin modificar el protocolo.

**No se afirma que una señal textual preapertura anticipa shocks intradía.** Cualquier ventaja en estados de
alta volatilidad se separa con la ablación `PPO+régimen` y se formula como asociación, no como anticipación causal.

---

## 20. Criterio final de terminado

> **Resultados y conclusiones**: [`06-RESULTADOS.md`](06-RESULTADOS.md).

Los 16 tests en verde · dataset, corpus y máscara versionados con hashes · partición de §1 decidida antes de F1 ·
cadencia y `K` congelados antes del HPO · matriz OOF y `N_trials` reproducibles · refit final ejecutado ·
manifiesto firmado · hold-out abierto una sola vez o la reapertura declarada · 12 tablas y 10 figuras generadas
automáticamente · análisis de poder escrito **antes** de las tablas de resultados · H2 y H3 respondidas con
significancia, incertidumbre, precisión declarada y relevancia económica · H1 reportada como contraste
indecidible con su IC · H4 presentada como exploratoria · capítulos 4 y 5 reflejando los resultados reales,
incluidos los negativos.

### Estado del criterio (2026-08-25)

| Criterio | Estado |
|---|---|
| Tests obligatorios en verde | **PARCIAL** — ver desglose |

**Desglose de los 16 tests obligatorios de §13** (81 tests en verde en los ficheros del carril):

| # | Test | Estado | Fichero |
|---|---|---|---|
| 1 | Contrato de datos | PARCIAL | cubierto de hecho por `test_seed_ohlcv_integrity.py` |
| 2 | Shuffle test | **PENDIENTE** | — |
| 3 | HMM causal (posterior filtrado) | **VERDE** | `test_regime_hmm_is_causal.py` |
| 4 | Escalador ciego a la evaluación | **VERDE** | `test_research_scaling_is_frozen.py` |
| 5 | Anti-leakage textual | N/A | sin brazo LLM |
| 6 | Macro as-of | PARCIAL | `test_research_features_are_causal.py` lo cubre por perturbación |
| 7 | Paridad del motor | **VERDE** | `test_engine_parity_independent.py` |
| 8 | `VecNormalize` congelado | **VERDE** | `test_research_scaling_is_frozen.py` |
| 9 | Costos sin doble conteo | **VERDE** | `test_session_env_and_costs.py` |
| 10 | Cronología (`b` captura `r_{b+1}`) | **VERDE** | `test_session_env_and_costs.py` |
| 11 | Signo híbrido | N/A | sin híbrido |
| 12 | Identidad híbrida `β=0` | N/A | sin híbrido |
| 13 | Contabilidad + costo terminal | **VERDE** | `test_session_env_and_costs.py` |
| 14 | Etiqueta de subperíodo OOF/OOS | **PENDIENTE** | — |
| 15 | Máscara común | **VERDE** | `test_evaluation_mask.py` |
| 16 | Spread esperado, no `argmax` | **VERDE** | `test_regime_hmm_is_causal.py` |

> **Sobre el test 7**: §13 nombraba `vectorbt` como segundo motor. No está instalado, y añadir
> una dependencia pesada para una comprobación de aritmética empeora la reproducibilidad sin
> mejorar lo que el test mide. Se implementó con una **referencia independiente de formulación
> distinta** —contabilidad de caja y unidades frente a pesos y retornos— que coincide a 1e-6
> sobre 60 sendas aleatorias. Incluye contraprueba: la referencia **discrepa** de un motor al
> que se le quita el costo terminal, de modo que la paridad no es decorativa.

Añadidos fuera de la lista de §13, porque los fallos que cazan ocurrieron de verdad:
paridad `gym.Env` ↔ `run_session` (`test_session_gym_parity.py`), causalidad de features por
perturbación y RSI de Wilder (`test_research_features_are_causal.py`), Regla B ejecutable y
bootstrap pareado (`test_research_inference.py`), determinismo de PNG sin romper `pyplot`
(`test_png_determinism.py`).
| Dataset y máscara hasheados | HECHO (`f6958be1b59e9767` / `5427409c451bc46b`) |
| Corpus versionado | N/A — sin brazo LLM |
| Partición decidida antes de F1 | HECHO (Opción C, `partition.yaml`) |
| `K` congelado antes del HPO | HECHO (K=4 por BIC en desarrollo; no hubo HPO) |
| Matriz OOF y `N_trials` reproducibles | HECHO (10 corridas + ledger encadenado, 113) |
| Refit final ejecutado | HECHO (`tesis_ppo_refit_001`) |
| Manifiesto firmado | HECHO como §9 del pre-registro; **no** como `configs/frozen_*.yaml` (ver §12) |
| Hold-out abierto una vez | HECHO, con constancia en `holdout_opening.json` |
| Tablas y figuras automáticas | HECHO — 8 tablas y 9 figuras, sin números a mano |
| Análisis de poder antes de las tablas | HECHO (§11.2, medido) |
| H2 respondida | HECHO — **decidible en selección** (+2,120, p=0,0276) e **INDECIDIBLE en hold-out** (+0,432); se reporta como NO CONFIRMADA |
| H3 | N/A — era del híbrido |
| H1 reportada con su IC | HECHO — **decidible EN CONTRA**, ΔSharpe −4.640 (p<0.0001) |
| H4 exploratoria | N/A — era del LLM |
| Resultados negativos publicados | HECHO — son *el* resultado |

---

## Anexo A — Cambios respecto de la versión anterior

| # | Corrección | Sección |
|---|---|---|
| 1 | Regla de ejecución circular dividida en Regla A y Regla B | Cabecera |
| 2 | Costo del cierre terminal incorporado al retorno diario | §9.1, §9.2, test 13 |
| 3 | Contabilidad homogénea: retornos simples y equity compuesta, en vez de mezclar log-retornos con costos simples | §9.2 |
| 4 | Sesiones inválidas fuera de `n`; máscara de evaluación común y hasheada | §9.5, test 15 |
| 5 | La serie OOF es causal pero **sujeta a sesgo de selección**; se elimina la expresión "sin contaminación de selección" | §11.4 |
| 6 | Selección + hold-out combinados pasan a sensibilidad descriptiva, sin p-value confirmatorio | §11.4 |
| 7 | Ablaciones como diseño factorial, con doble reporte controlado/optimizado | §10.3 |
| 8 | Baselines supervisados especificados por completo y con versión `matched` para H1 | §10.6, §10.7 |
| 9 | Universo de White RC/SPA/PBO congelado y distinguido de `N_trials` | §11.6 |
| 10 | Regla de fallo del PBO predeclarada | §11.7 |
| 11 | Spread esperado por posteriores del HMM en vez de `argmax` | §8.4, test 16 |
| 12 | `K` del HMM elegido una vez y congelado antes del HPO | §8.2 |
| 13 | Tolerancias numéricas del fallback entre proveedores | §6.2 |
| 14 | Grid híbrido con 16 estrategias distintas, sin columnas duplicadas | §10.5 |
| 15 | Condición del shuffle test reformulada | Test 2 |
| 16 | `rf = 0` justificado por materialidad, no por ausencia de carry | §9.2 |

### Anexo A.2 — Cambios de la ejecución (2026-08-24 / 2026-08-25)

Los de arriba son correcciones al PLAN. Estos son los cambios que impuso ejecutarlo.

| # | Cambio | Motivo | Sección |
|---|---|---|---|
| 17 | Partición **Opción C**, no A ni B | 2019 no existe en la serie de 5 min | §1 |
| 18 | Potencia **medida** (ρ=0,92, n=584, mínimo detectable ΔSharpe≈0,7) en vez de interpolada | el entorno ya permite simularla | §11.2 |
| 19 | Alcance recortado a **2 brazos**: sin LLM, sin híbrido | decisión del operador | §14, §15 |
| 20 | **Sin HPO**: hiperparámetros congelados de `v215b_baseline.yaml` | prior ex-ante ⇒ no suma trials | §15, §11.12 |
| 21 | **White RC y SPA no se computan**, con la razón declarada | universo de 2 candidatos | §11.6 |
| 22 | Grupo de **volumen eliminado** por su propio criterio | 100% de las barras con volumen 0 | §6.5 |
| 23 | **Sorpresa macro eliminada**; entra el diferencial IBR−DGS2 | no hay consenso histórico verificable | §6.5 |
| 24 | El HMM encuentra **tipos de día**, no regímenes (1,6-1,9 días) | persistencias medidas | §8.3 |
| 25 | `N_trials` **no arranca en cero**: 111 heredados + 2 AT = 113 | ADR-0023; MLflow no es la autoridad | §11.12 |
| 26 | El **listón es `always_flat`**, no buy&hold | 4,81 pips/sesión ⇒ 71,4% de costos sobre 584 sesiones | §9.3, §10 |
| 27 | 300k pasos, no 2M | 2M serían 68 pasadas sobre 29.441 pasos de desarrollo | §15 |
| 28 | Regla B **implementada en código**, no solo escrita | `thesis_statistics.py --block holdout` sale con 2 sin firma | §12 |
| 29 | Reporte de coherencia con **6 comprobaciones** ejecutables | las de `β` desaparecen con el híbrido | §19.4 |
| 30 | Sección de **riesgos materializados** | los previstos no fueron los que ocurrieron | §18 |
| 17 | Compuerta de benchmark de cómputo al final de F0 | §15 |
| 18 | H1 reclasificada: se reporta con precisión declarada porque es indecidible al tamaño de muestra disponible | §11.1, §11.3 |
| 19 | Recomendación explícita de partición con hold-out de dos años | §1 |
