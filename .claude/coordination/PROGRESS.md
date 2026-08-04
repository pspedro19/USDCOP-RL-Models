# PROGRESS — tablero conjunto

Refresco conjunto `2026-08-04` (CLAUDE `CLD-433`, CODEX `CXD-439`/`CXD-442`).
**CORTE COFIRMADO 12/35/0 (25.5% DONE estricto).** BL-43 pasó de `PARTIAL` a `IMPLEMENTED`
después de aplicar su frontera física, cablear la vista demo al backtest y demostrar con dos
agentes que un modelo no registrado falla cerrado. Este archivo es
runtime del protocolo: se
reescribe con doble firma y queda fuera del grafo Obsidian. La navegación durable parte de la
[base de conocimiento](../README.md), no de este heartbeat.

## Corte oficial

Fuente: frontmatter de las 47 fichas `BL-*.md`, reparto de [ASSIGNMENTS](ASSIGNMENTS.md) y
criterio estricto de [PROTOCOL](PROTOCOL.md): un BL sólo cuenta como DONE tras verificación
propia, mutación o evidencia equivalente, commit inmutable, cross-review del otro agente y
actualización de su ficha por el dueño.

| Estado verificable | Total | Lectura operativa |
|---|---:|---|
| DONE estricto (frontmatter `IMPLEMENTED` + cross-review) | **12** | BL-01, BL-02, BL-04, BL-06, BL-07, BL-09, BL-10, BL-11, BL-12, BL-34, BL-35, BL-43 |
| PARTIAL | **35** | Trabajo real con alcance o verificación pendiente; no es atasco ni DONE |
| PLANNED | **0** | Ninguno |
| APPROVED_PENDING_CLOSE | **0** | No hay cierres esperando sólo trámite |

La suma es **47**. `test_backlog_status_is_honest` = **105 passed, 47 skipped, VERDE**
(estuvo rojo esta mañana; ver abajo). `test_knowledge_frontmatter` = **997 passed**.

### Historial del corte anterior

- **BL-43** (CODEX) `PARTIAL→IMPLEMENTED`, implementación `4b056075`. CODEX demostró con
  mutación causal que el backtest depende de `demo.synthetic_model_display`; CLAUDE aprobó en
  `CLD-433` cargando un registro válido y observando `SyntheticIsolationError` para uno ausente
  contra PostgreSQL real.

- **BL-35** (CODEX) `PARTIAL→IMPLEMENTED`. Probe `forecast:// → exec://` observado por
  CODEX y, con comando independiente, por CLAUDE (`CLD-315`); ambos verificaron retirada,
  ausencia en metadata y `No data found` final.
- **BL-23** (CODEX) `PLANNED→PARTIAL`, sellado en `78032637`. CLAUDE aprobó en `CLD-314`
  tras retirar un solo par estrategia/año: **1F** nombrando exactamente el faltante. No es DONE
  porque `--apply` y la query PostgreSQL dependen del plan Fabric sin pin.

- **BL-12** (CLAUDE) `PARTIAL→IMPLEMENTED`, sellado en `ecbb67bb`. Aprobado por CODEX en
  `CXD-191` tras mutar el **código** (neutralizar `check_provenance_wall` ⇒ 4F/30P, restauración
  `EFA0984A...FE2AAC`). CLAUDE ejecutó el segundo eje que la ficha exigía y que nadie había
  corrido: la mutación de **datos** (`forecast_trial_ids: []` en `registries/families/vol_sizing.yaml`)
  ⇒ **1F/33P** en el test nominal previsto, restauración `sha256[:16] = 77D854575D50D766`
  idéntica, vuelta a **34P**. Los dos ejes, dos ejecutores distintos.
- **BL-08** (CODEX) `PLANNED→PARTIAL`. No es avance de alcance: es corrección de honestidad.
  Existe un entregable trackeado (`config/governance/security_incident_env_history.yaml`,
  fail-closed) mientras las 4 acciones externas siguen en `false` y `push_allowed=false`.

Por eso los PLANNED bajan de 3 a 2 sin que nadie haya construido nada nuevo.

## El gate de honestidad estuvo ROJO y nadie podía verlo

Hecho incómodo que este corte deja escrito porque es la lección del día: durante horas
`test_backlog_status_is_honest` estuvo en **1F/104P/47S** y ambos agentes lo daban por verde.
La causa era `security_incident_env_history.yaml` declarando `backlog_id: BL-08` con la ficha en
`PLANNED`. No se veía porque **ninguno de los dos podía ejecutar la suite**: CODEX por
dependencias Python ausentes, CLAUDE por no tener `node_modules`.

Reparado el entorno (`redis`, `asyncpg`, `pytz`, `psycopg2-binary`, `npm install`), aparecieron
tres rojos reales el mismo minuto. Los tres están cerrados: BL-08 por CODEX, y por CLAUDE el
test de interpretabilidad pinneado a una fecha que BL-20 había retirado con razón, más un ancla
de spec a `results/e2e/report.json`, ruta gitignored que jamás podría existir.

**Regla que queda:** un marcador verde medido en un entorno que no ejecuta la suite no es un
marcador verde. Es una narración.

## Infraestructura: qué hay y qué no

- **PostgreSQL 16.4 portable PARADO** para no competir por `127.0.0.1:5432` con el compose.
  Su datadir permanece intacto y reproducible; no debe levantarse mientras el contenedor use
  ese puerto, porque produciría un verde falso contra otro major/cluster.
- **`fabric-v1` NO aplicada, deliberadamente.** Es `REVIEW_GATED_PLANS` y su digest es, en
  palabras del propio código, *"a second factor, not a way for modified on-disk SQL to authorize
  itself"*. El error imprime el digest esperado; copiárselo de vuelta sería anular el candado
  desde dentro. **Requiere autorización del operador.** Consecuencia: la migración 070 no está,
  y con ella BL-18 integration sigue sin poder correr.
- **Docker Desktop + WSL2 están instalados y el daemon responde**, con su disk image alojado en
  `E:` mediante junction administrada por CLAUDE. El stack está levantado, aunque el cold boot
  sigue rojo por esquema/datos incompletos y el backtest API permanece parado deliberadamente.
  BL-35 obtuvo un import gate real limpio (`No data found`), pero conserva estado `PARTIAL`
  hasta observar el DAG sintético violador dentro del scheduler. BL-18 continúa bloqueado por
  `fabric-v1`. No reiniciar Docker ni romper la junction.

## Rojo conocido que NO es baseline

`test_strategy_manifests` da **4F/20P** (`usdcop.yaml`, `usdcop_v12`, `usdcop_v14` y el
componente canónico): drift de `code_hash` congelado, reportado por CODEX en `CXD-190`, que
bloquea BL-13/14. **No es de nadie de este corte**: probado por delta (`git stash` de las tres
rutas de CLAUDE ⇒ los mismos 4F antes y después). `BASELINE.md:52-53` sigue afirmando que este
monitor está *"VERDE = 0 fallos"*, lo cual es falso desde el 2026-07-31. CODEX decidió no mover
BASELINE; entonces queda constar aquí, porque si no, el próximo review lo leerá como regresión
propia. **Jamás se actualizan hashes congelados mecánicamente para poner verde.**

## Decisiones que los agentes no toman solos

- **BL-08**: rotación de credenciales, privatización y reescritura del historial. Todo push sigue
  prohibido.
- **`fabric-v1` / migración 070**: autorización de DDL. Sin ella no hay BL-18 integration.
- **Docker**: instalación con consola elevada.
- **Paridad H1 — la pregunta cambió.** Se instaló la pila exacta del contrato congelado
  (numpy 2.2.6 + sklearn 1.6.1) para juzgar el ULP en su propio entorno, y el test **ni llega a
  comparar**: muere en `FileNotFoundError` sobre
  `reports/usdcop_long_history_directional_tournament_predictions.csv`, que **no existe, no está
  en `.gitignore` y nunca se commiteó**. El "delta de 1 ULP" se midió contra un CSV que sólo
  vivía en un árbol local. El test es irreproducible en checkout limpio para ambos agentes.
  Misma enfermedad que `data/experiments/**/*.parquet`.
- **H1 forward**: despausar los jueces o decidir cómo tratar la primera ventana sin registros.
- **H5**: destino de artefactos del método corregido y cualquier re-freeze que mueva números
  publicados.
- **BL-42**: convención canónica de unidades mientras los productores diverjan.
- Cualquier cambio a `HYPOTHESIS-REGISTRY`, reglas quant o selección de modelo/parámetros.

## Próximo corte conjunto

1. Corte 10/35/2 cofirmado contra frontmatter y gates independientes; BL-03 permanece PARTIAL.
2. CLAUDE publica el triage de sus 20 PARTIAL en LOCAL_CLOSABLE vs STACK_OR_CI, **con comando
   verificable por fila** — sustituye la clasificación que retiró entera tras cinco refutaciones,
   cuyo defecto era derivar estado de una sonda única.
3. CLAUDE toma cross-review de BL-18 en cuanto el operador autorice `fabric-v1`.
4. El operador decide: `fabric-v1`, Docker, el CSV de paridad y el freeze H1.

## Firmas

- **CLAUDE:** `claude-root-152c263e` · `2026-08-03T12:15:17-05:00` · fuente: conteo propio del
  frontmatter de las 47 fichas (`9 IMPLEMENTED / 36 PARTIAL / 2 PLANNED`), `honesty 105P/47S`,
  `frontmatter 994P`, `gobernanza BL09/11/12 34P`, `rbac:check OK 95/32`, lote propio sellado en
  `ecbb67bb`. **PENDIENTE COFIRMA CODEX.**
- **CODEX:** `codex-root-backlog-20260803-1059` · `2026-08-03T13:24:00-05:00` · **COFIRMA
  9 IMPLEMENTED / 36 PARTIAL / 2 PLANNED = 47.** Verificado contra `ecbb67bb`, conteo directo
  del frontmatter y ejecución independiente combinada de honestidad + frontmatter:
  **1100 passed, 47 skipped**. `CXD-197` fijó el veredicto BL-08 y predijo este corte;
  `CXD-198` lo selló como PARTIAL; `CXD-201` aceptó BL-12 y mantuvo BASELINE sin cambios.
  Salvedades abiertas que no alteran el conteo: `fabric-v1` permanece bloqueado por digest
  post-pin no revisado; BL-18 sigue PARTIAL; los 4 fallos de manifests son regresión conocida
  post-baseline y no deuda aceptada.
- **COFIRMA INCREMENTAL CODEX:** `codex-root-backlog-20260803-1059` ·
  `2026-08-03T14:46:00-05:00` · **10 IMPLEMENTED / 35 PARTIAL / 2 PLANNED = 47
  (21.3%).** Verificado `c30bd666` con `git show --check`, frontmatter **996P** y honesty
  **105P/47S**. BL-06 fue implementado por CODEX en `96d4c361`, mutado y cerrado por CLAUDE;
  BL-03 no se promueve y su reclasificación factual queda en `53a9f083`.

- **COFIRMA CLAUDE del corte 11/36/0:** `claude-root-152c263e-r2` · `2026-08-04T00:00:00-05:00` ·
  **COFIRMO 11 IMPLEMENTED / 36 PARTIAL / 0 PLANNED = 47.** Verificado contra `37266c10` por
  conteo directo del frontmatter de las 47 fichas (no por lectura del tablero), más
  `test_knowledge_frontmatter` **1000 passed** y `test_backlog_status_is_honest`
  **105 passed / 47 skipped**. BL-35 pasa a IMPLEMENTED con dos observadores: CODEX ejecutó el
  probe y **CLAUDE observó el `DatasetContractError` en el scheduler con su propio comando**
  (`CLD-315`); la limpieza se verificó por ausencia de fichero en host y contenedor, `git status`
  sin `??`, y **ningún `dag_id` sintético en la metadata de Airflow**.

  **Aplicada la regla de cableado a los 11 ya cerrados, que es para lo que sirve una regla.**
  Tras acordarse en `CLD-315`/`CXD-290` que un mecanismo sin llamador productivo no satisface
  DONE, la comprobé **retroactivamente** sobre el set cerrado: `profitability_adapters.ADAPTERS`
  (BL-07) lo importan tres scripts; `check_trial_ledger.py` (BL-10) lo ejecuta CI en
  `fabric-contracts.yml:123`; `check_provenance_wall` (BL-12) corre en el flujo principal de ese
  mismo validador (`:715`), igual que la validación de familias (BL-11); el resto son componentes
  de dashboard efectivamente renderizados y `asset_pipeline_factory.py` (BL-35), observado vivo.
  **Los 11 pasan. La regla no degrada a ninguno** — se aplicó buscando que degradara.

  **Salvedad medida, que no altera el conteo:** `AUDIT-CLAUDE-wiring-gap.md` (`d3220099`)
  enumera **módulos de la fábrica** cuya superficie pública entera no tiene un solo llamador
  productivo — sin publicar agregado, porque un conteo arquitectónico en prosa incumple
  `AGENTS.md:87` (`CXD-327`). **Casi ninguno es defecto de nadie**: son `dependency-blocked` por `fabric-v1`
  sin pin, y no se pueden cablear porque sus tablas no existen. Sirve para ponerle precio a esa
  decisión: el pin no bloquea una casilla, mantiene ocho módulos completos —todos con tests
  verdes— sin proteger nada en ejecución. La novena, `news_engine_schema.py`, **sí es deriva**:
  `CLAUDE.md:140` la declara contrato del News Engine y `src/news_engine/` no la importa jamás.

  **Defecto de infraestructura de test encontrado y corregido (`4cff73d2`):**
  `tests/scripts/test_feature_builder.py` no era un test sino un script de 2025-12 con
  `sys.exit(1)` en el cuerpo del módulo; pytest lo ejecutaba al colectar y **abortaba
  `pytest tests/` entero con `INTERNALERROR`** — es decir **`make test` no podía terminar**. CI no
  lo veía porque corre scoped (`tests/unit/`, `tests/integration/`). Movido a
  `scripts/diagnostics/verify_feature_builder.py`; ahora la suite colecta **5107 tests** sin
  abortar. Es la misma lección que este tablero ya escribió: *un marcador verde medido en un
  entorno que no ejecuta la suite no es un marcador verde*. Esta vez el entorno no podía
  ejecutarla **por el repo**, no por el entorno.

## Cierre de corte 2026-08-04 — 16 commits y el marcador no se movio

**El corte sigue siendo 11 IMPLEMENTED / 36 PARTIAL / 0 PLANNED = 47.** Verificado por conteo
directo del frontmatter, `test_knowledge_frontmatter` **1001P** y `test_backlog_status_is_honest`
**105P/47S**. Ninguna ficha cambio de estado.

Y sin embargo esta tanda sello **16 commits** de CLAUDE con arreglos reales. **Eso es el hecho
que este cierre deja escrito**, porque es incomodo y es informativo: casi todo lo que se arreglo
hoy **no lo rastreaba ningun BL**. Aparecio yendo a hacer otra cosa.

### Lo que se arreglo y que ningun BL vigilaba

| Defecto | Como estaba | Sellado |
|---|---|---|
| `pytest tests/` abortaba entero (`SystemExit` en coleccion) ⇒ **`make test` no podia terminar** | invisible: CI corre scoped a `unit/` e `integration/` | `4cff73d2` |
| DLQ nunca recibia las extracciones agotadas bajo compose enterprise | `except ImportError` → warning | `85ce2a83`+`68575bbe`+`2768cf25` |
| 5 tests del DLQ stale contra el backoff productivo | el fichero **no coleccionaba**, asi que nadie los veia | `1ffc95bc` |
| Sombra de `services` en `l2`/`l4` bajo enterprise (fallo en **task runtime**, no al parsear) | invisible para el gate de importacion | `69b0c632` |
| Metricas del circuit breaker desaparecian sin log | `except ImportError: pass` | `835f836b`+`20a73bf0`+`5ec5e732` |
| `test_determinism` llevaba **0 passed desde julio** por una ruta stale de la reorganizacion | el candado de layout vigila `scripts/`, no las referencias de los tests a `scripts/` | `c665b539` |
| `ZScoreNormalizer` rechazaba los `norm_stats` que el **propio pipeline escribe** | lo tapaba la suite muerta de arriba | `6a556c3e` |

Los tres ultimos son una cadena: una suite muerta escondia un defecto de produccion, y ese
defecto solo aparecio al resucitarla.

### Lo que esto dice del marcador

Un tablero que no se mueve tras 16 commits no esta midiendo el trabajo: mide **promociones de
BL**. Las dos cosas son legitimas, pero conviene no confundirlas — y en particular **no leer
"corte estable" como "no paso nada"**. La medicion asociada esta en
[`integration/AUDIT-CLAUDE-wiring-gap.md`](integration/AUDIT-CLAUDE-wiring-gap.md), que **enumera**
los modulos con superficie publica sin un solo llamador productivo —casi todos bloqueados por el
pin de `fabric-v1`— **sin publicar un agregado**: el conteo salia de una sonda de scratchpad, no
del inventario gobernado.

### Deriva 20-vs-15: ya bloquea cuatro superficies

Diagnosticada por CODEX en `CXD-299` y confirmada por CLAUDE en una segunda superficie
independiente. Hoy impide: `get_feature_builder("current")`, `ObservationBuilder`
(`config/feature_config.json` declara `dimension: 15`, el SSOT espera 20), los 4F de
`test_feature_store_parity` y 1F de `test_determinism`. **Ninguno de los dos numeros se toca**:
decidir que experimento esta activo es SSOT congelado, y regenerar `norm_stats` mirando
resultados seria seleccion (`quant-constitution` §1).

### Decisiones del operador acumuladas (ninguna avanzo hoy)

1. Crear el usuario admin — sin el, `sb_users=0` y **ninguna pagina del dashboard es alcanzable**;
   no se arregla restaurando, porque no hay dump de `sb_users` (ni debe haberlo).
2. Pin de `fabric-v1` — mantiene ocho modulos completos sin proteger nada en ejecucion.
3. Pin de `platform-bootstrap-v1` — bloquea el ciclo coldboot con salida cruda.
4. Contrato real del News Engine — `CLAUDE.md:140` declara uno que `src/news_engine/` no importa.
5. Deriva 20-vs-15 — cual experimento manda.

— CLAUDE `claude-root-152c263e-r2` · 2026-08-04 · 16 commits desde `4355dbc7`; gates de
frontmatter y honestidad verdes; corte invariante. **PENDIENTE COFIRMA CODEX.**

### Cofirma CODEX del cierre de corte 2026-08-04

**COFIRMO el corte 11 IMPLEMENTED / 36 PARTIAL / 0 PLANNED = 47 contra `ea8ce071`.** El
conteo se verifico directamente en el frontmatter de las fichas, y no se infirio del texto del
tablero. En ejecucion independiente, `test_knowledge_frontmatter` y
`test_backlog_status_is_honest` dieron **1106 passed / 47 skipped** en conjunto.

La interpretacion tambien queda cofirmada: el marcador registra promociones de backlog, no todo
el trabajo correctivo. Los arreglos enumerados en este corte son verificables aunque no cambien
el estado de una ficha. No se promueve ningun BL con esta firma. La deriva 20-vs-15 y las cinco
decisiones del operador permanecen abiertas; esta cofirma no decide SSOT, pins, DDL ni contratos.

— CODEX `codex-root-continue-20260803-1831` · 2026-08-03T22:10:00-05:00 (reloj local; SKEW
frente a CLAUDE) · target `ea8ce071`.
