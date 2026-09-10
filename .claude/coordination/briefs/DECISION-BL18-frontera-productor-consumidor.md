---
kind: decision
status: OPEN
version: 2.3.0
last_verified: 2026-08-04
supersedes: []
code_anchors:
  - src/monitoring/system_health.py
  - src/monitoring/system_health_contract.py
  - airflow/dags/control_system_health.py
  - src/metrics/engine.py
  - src/metrics/persistence.py
  - database/migrations/070_fabric_control_plane.sql
---

# Decisión BL-18 — ¿quién escribe en `control.metric_event`?

> Brief conjunto pedido por CXD-359. **Ningún código escrito.** Decisión abierta para el operador.
> Levantado por CLAUDE (CLD-364/365) y CODEX (CXD-359/363). v2.0.0 incorporó la objeción técnica
> de CODEX (Opción A′) y retiró los agregados no gobernados; v2.1.0 corrige la clasificación de
> `withdrawal_protocol_triggered`, añade los `kind` dinámicos `data_*` y fija que la frontera es
> semántica (CXD-365); **v2.2.0 corrige la fuente de la serie y registra el veredicto
> `BLOCKED/PARTIAL` de A′** (CLD-399/CXD-404).

---

## 1. Por qué esto está bloqueado (y no es plomería)

Tras aplicar `fabric-v1` medí qué código productivo escribe en las tablas que crea. El resultado
cualitativo: **la mayoría no tiene ni escritor ni lector**. Esquemas completos (`exec`, `lineage`,
`forecast`, `quality`) no se mencionan en ningún archivo fuera de migraciones, tests y specs.

De las pocas que sí aparecían, varias eran falsos positivos de mi propia búsqueda —rutas de módulo
Python como `from src.portfolio.snapshot import …`, y comentarios— y `scripts/data/backfill_catalog_facts.py`
sólo lo importan tests.

> Método, para que sea reproducible y no un número congelado: barrido sobre archivos productivos
> (`.py/.ts/.yaml/.sh/.sql`) excluyendo `database/migrations/`, `tests/`, `.claude/` y `docs/`;
> más dos comprobaciones de falso negativo — **ningún** `search_path` apunta a un esquema fabric, y
> los candidatos de `INSERT/UPDATE` sin cualificar resultaron ser prosa inglesa en comentarios.

**Aplicar el esquema no encendió nada por sí solo.** El cuello de botella sigue siendo el cableado.

## 2. La única costura donde la demanda YA existe

```
airflow/dags/control_system_health.py:70-72     ← DAG productivo, corriendo hoy
    SystemHealthEngine(event_sink=JsonlMetricEventSink(EVENTS_PATH))

src/monitoring/system_health.py:121-128
    class MetricEventSink(Protocol): ...        ← el contrato del sink ya está definido
    class JsonlMetricEventSink:  "default hasta que exista control.metric_event"
```

`data/health/metric_events.jsonl` recibe eventos reales hoy. Hay un **productor vivo**, y su
condición literal de espera —"hasta que exista `control.metric_event`"— se cumplió al aplicar
`fabric-v1`. No hay que inventar demanda: sólo existe ésta.

## 3. Hechos medidos que estrechan la decisión

| # | Hecho | Cómo se verifica |
|---|---|---|
| 1 | El scheduler de Airflow **no tiene `asyncpg`** (sí `psycopg2` y `sqlalchemy`) | `docker exec usdcop-airflow-scheduler python -c "import asyncpg"` |
| 2 | `trading-api`, `analytics-api` y `signalbridge` **sí tienen `asyncpg`** | idem por contenedor |
| 3 | `metric_event_id` es **UUIDv5 determinista** sobre la identidad semántica canónica | `src/metrics/engine.py:445` |

El hecho 3 importa: releer el mismo JSONL produce el **mismo** UUID, así que
`ON CONFLICT (metric_event_id) DO NOTHING` hace la reingesta idempotente por construcción. El
índice `uq_metric_event_semantic_identity` es la red que la respalda — **siempre que el id se
derive y nunca se regenere con `uuid4()`**.

## 4. El bloqueo real: `HealthEvent` no puede ser un `MetricEvent`

`HealthEvent` (`src/monitoring/system_health_contract.py:177-191`) expone: `kind`, `clock`,
`signal`, `action`, `message`, `metric_value`, `threshold`, `timestamp`.

`control.metric_event` exige `NOT NULL` en columnas que **`HealthEvent` no tiene ni puede derivar**:
`catalog_version`, `formula_version`, `entity_type`, `entity_id`, `metric_namespace`. Y trae un solo
`threshold` donde la tabla distingue `threshold_warning` de `threshold_critical`.

> **Hallazgo histórico, ya corregido.** `system_health_contract.py` prometía que *"cuando BL-18
> materialice `control.metric_event`, estos eventos se insertan tal cual"*. Era falso —faltan cinco
> `NOT NULL`— y rellenar esas columnas con constantes habría sido **fabricar linaje**. El docstring
> quedó corregido en `b3f2ff58`, y `a9abfc7e` le puso candados. Se conserva aquí como registro de
> por qué la frontera está donde está, **no como estado actual del código**.

## 5. `HealthEvent` es un envelope de naturalezas heterogéneas (objeción de CODEX, CXD-363/365)

Éste es el matiz que invalida la recomendación original de mandar *todo* por `MetricEngine`.
Los `kind` emitidos por `src/monitoring/system_health.py` no se parten por tener o no un número,
sino por **qué son contractualmente** — y no son dos grupos, sino los que enumera la tabla:

| Naturaleza contractual | `kind` emitidos | ¿Métrica declarada en catálogo? |
|---|---|---|
| **Observación métrica** | `model_drift_psi`, `prediction_drift`, `sharpe_decay`, `slippage_excess` | Candidatas a declararse |
| **Ausencia de métrica** | `metric_missing` | No — es la *negación* de una observación |
| **Incidente / diagnóstico** | `parity_broken`, `data_{probe.status}` (dinámico, `system_health.py:176,189`) | No |
| **Acción de protocolo** | `withdrawal_protocol_triggered` | No — aunque **sí porta** `metric_value=te_z` y `threshold=TRACKING_ERROR_SIGMA` (`:349`) como evidencia causal |

**La frontera es semántica, no la nulabilidad de `metric_value`.** Es el punto de CXD-365 y es el
que hay que implementar con cuidado:

- `metric_missing` es la **ausencia** de una métrica. Persistirla como `metric_event` con un
  `metric_name` inventado sería linaje fabricado.
- `withdrawal_protocol_triggered` **sí** trae una medición, pero su evento principal es un **acto
  de gobierno** — el retiro se dispara por protocolo (`.claude/rules/quant-constitution.md` §5).
  Registrarlo como una observación de catálogo lo disfrazaría de medición rutinaria.

> **Modo de fallo concreto a evitar:** implementar el router como
> `if event.metric_value is not None: → control.metric_event`. Con esa regla,
> `withdrawal_protocol_triggered` acabaría silenciosamente en la tabla de métricas. El router debe
> preguntar *"¿este `kind` está declarado y gobernado como métrica en el catálogo?"*, no *"¿trae un
> número?"*.

Esta sección corrige la v1.0.0 (donde recomendé la Opción A indiscriminada) y la v2.0.0 (donde
clasifiqué mal `withdrawal_protocol_triggered` y omití los `data_*`).

## 6. Las opciones

### Opción A′ — híbrida (recomendada por ambos agentes)

1. Sólo las **observaciones declaradas y gobernadas como métricas en el catálogo**, calculadas o
   normalizadas por `MetricEngine`, llegan a `control.metric_event`. Heredan identidad, versiones
   y el UUIDv5 sin inventar nada. **El router es el catálogo, nunca `metric_value != null`**
   (ver el modo de fallo de la sección 5).
2. **Ausencias, incidentes/diagnósticos y actos de protocolo** siguen siendo `HealthEvent` en
   JSONL y necesitan un contrato operativo propio. `control.incident` es el candidato natural,
   **pero no se decide aquí.** Un incidente puede adjuntar la medición que lo causó sin convertirse
   por ello en una observación de catálogo.
3. El servicio consumidor ingiere un envelope tipado y persiste **sólo** `MetricEvent` ya
   gobernados; el JSONL queda como buffer durable y reintentable.

- **Coste real**: declarar las métricas de salud en el catálogo (`config/metrics/catalog.yaml`).
- **A favor**: no inventa linaje, no colapsa métricas con incidentes, un solo gobierno de métricas.

### Opción B — `HealthEvent` crece con los campos que le faltan

- **A favor**: cambio local, no toca el catálogo.
- **En contra**: duplica gobierno — dos sitios decidirían qué es la identidad de una métrica. Es
  precisamente la deriva que el fabric existe para impedir. Y no resuelve el problema de la
  sección 5: seguiría metiendo incidentes en una tabla de métricas.

## 7. Dónde corre el sink (independiente de A′/B)

- El DAG **sigue escribiendo JSONL** — pasa de apaño a buffer durable.
- Un **servicio consumidor** con `asyncpg` lo ingiere usando el sink gobernado ya verificado.
- Sin rebuild de Airflow, sin SQL duplicado, un solo sink.

**Condición no negociable acordada por ambos agentes**: el JSONL **no se retira**. Un monitor de
salud que pierde eventos porque la base está caída es peor que uno que escribe a fichero. Aquí el
**fail-open es lo correcto**, y queda declarado como excepción consciente al fail-closed por defecto.

## 8. Corrección de fuente y estado contractual de A′ (2026-08-05)

Al derivar el mapping de A′ apareció que **la fuente propuesta era la equivocada**, y eso corrige
la premisa de las secciones 4 y 5:

| | `HealthEvent` | `ClockStatus.metrics` |
|---|---|---|
| Cuándo existe | **sólo si se cruza el umbral** (`system_health.py:248,269,363,386`) | **en cada evaluación** |
| Identidad | enterrada en el `message` como prosa | **estructurada** (`psi_by_feature` es `{feature: valor}`, `:283`) |
| Naturaleza | alerta condicionada | **observación** |

Catalogar desde `HealthEvent` produciría una serie **que sólo existe cuando algo va mal** — un
histórico de alertas, no una métrica— y obligaría a parsear la entidad de un string.
**La serie gobernada debe alimentarse de `ClockStatus.metrics`.** Ambos agentes lo conceden
(CLD-399 / CXD-404). `HealthEvent` se queda siendo lo que la §5 ya describía.

### Resuelto bilateralmente

- **`source` = `system_health_engine`**, anclado al productor real. Describe procedencia del
  cálculo; no finge una tabla.
- **Los priors NO se re-declaran.** `CTR-SYSTEM-HEALTH-001` sigue siendo la fuente normativa. Se
  acepta un **espejo numérico en el catálogo sólo como mirror gobernado**: comentario de procedencia,
  bump de `catalog_version` y **prueba de paridad que falla si diverge cualquiera de los dos lados**.
  Cambiar ambos sigue exigiendo **ADR** — el espejo no convierte esto en tuning permitido.

### BLOQUEADO — y por eso no se cataloga ninguna entrada todavía

1. **Identidad.** `psi_by_feature` da un candidato a `entity_id`, pero **no identifica modelo,
   estrategia ni baseline de referencia**. El mismo feature evaluado en contextos distintos
   produciría una identidad global ambigua. Las otras tres métricas son **escalares sin dueño**.
   Y `entity_type`/`entity_id` son `NOT NULL` en `control.metric_event`.
2. **`formula_version`.** No existe versionado de estas fórmulas. Se propuso derivarlo de
   `CONTRACT_VERSION` (`system_health_contract.py:30`) en vez de inventar un literal; **sin acuerdo
   todavía**.
3. **`warning` vs `critical`.** Se propuso mapear el prior a `critical` y dejar `warning` en `null`,
   sustentado en que `MetricEngine._status` (`engine.py:588-603`) evalúa `critical` primero y en que
   cruzar el prior **siempre dispara una acción automática**, nunca un aviso pasivo. **Sin acuerdo
   todavía.**
4. **`unit: sigma`** sería vocabulario nuevo (hoy: `ratio`, `decimal`, `probability`). Un z-score no
   es honestamente ninguno de los tres.

> **Veredicto conjunto: A′ queda `BLOCKED/PARTIAL` por identidad y versionado.** No se cataloga
> ninguna entrada, no se toca `HealthEvent`, y **no es permiso para inventar contexto**. El
> siguiente paso es un contrato separado sobre **cómo transportar identidad explícita desde el
> caller** — propuesta, no implementación.

## 9. Estado

**BL-18 permanece PARTIAL.** Lo verificado hoy contra PostgreSQL real es el sink
(`persist_metric_event`: insert, replay idempotente, colisión bloqueada, offsets equivalentes del
mismo instante). Lo que falta es productor y consumidor productivos, y eso depende de la decisión
de abajo.

## 10. Qué se pide al operador

**A′ ya no está en discusión**: es la dirección conjunta de ambos agentes (§6) y B queda descartada
—no la reabras—. Lo que bloquea hoy es una capa por debajo, y eso es lo que se decide:

1. **El contrato de identidad explícita.** La identidad no falta en el sistema: el caller la tiene
   (`control_system_health.py` importa `H5_PRODUCTION_STRATEGY_ID` y filtra por `strategy_id`) y la
   firma de `evaluate_*_clock()` **la descarta**. Decidir el shape que la transporta desbloquea tres
   de las cuatro métricas. La cuarta —`prediction_drift_z`— necesita identidad de **modelo**, que es
   otra dimensión y **no existe**; no se inventa.
2. **`formula_version`.** Derivarlo de `CONTRACT_VERSION` (`system_health_contract.py:30`) o fijar
   otro criterio. Es acuerdo, no relleno.

**Y hay un orden que no es opcional**: los productores H5 ya usan `strategy_id` contra un esquema
que **no lo tiene** (migración `064`, fuera de todo plan — CXD-405). Apoyar la identidad de las
métricas en una columna ausente sería construir sobre esa misma deuda. **Primero `064` en un plan,
después identidad, después catálogo.**

Queda además una pregunta menor, que **no bloquea** lo anterior: dónde persisten los incidentes
(`control.incident` u otro contrato). Se aborda después, no aquí.
