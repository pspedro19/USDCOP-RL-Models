# PROGRESS — tablero conjunto

Refresco CLAUDE `2026-08-03T12:15:17-05:00` (`claude-root-152c263e`).
**CORTE NUEVO 9/36/2, PENDIENTE DE COFIRMA CODEX.** El corte anterior 5/47 llevaba desde el
2026-07-31 sin moverse y ya no describía el árbol. Este archivo es runtime del protocolo: se
reescribe con doble firma y queda fuera del grafo Obsidian. La navegación durable parte de la
[base de conocimiento](../README.md), no de este heartbeat.

## Corte oficial

Fuente: frontmatter de las 47 fichas `BL-*.md`, reparto de [ASSIGNMENTS](ASSIGNMENTS.md) y
criterio estricto de [PROTOCOL](PROTOCOL.md): un BL sólo cuenta como DONE tras verificación
propia, mutación o evidencia equivalente, commit inmutable, cross-review del otro agente y
actualización de su ficha por el dueño.

| Estado verificable | Total | Lectura operativa |
|---|---:|---|
| DONE estricto (frontmatter `IMPLEMENTED` + cross-review) | **9** | BL-01, BL-02, BL-04, BL-07, BL-09, BL-10, BL-11, BL-12, BL-34 |
| PARTIAL | **36** | Trabajo real con alcance o verificación pendiente; no es atasco ni DONE |
| PLANNED | **2** | BL-23, BL-28 |
| APPROVED_PENDING_CLOSE | **0** | No hay cierres esperando sólo trámite |

La suma es **47**. `test_backlog_status_is_honest` = **105 passed, 47 skipped, VERDE**
(estuvo rojo esta mañana; ver abajo). `test_knowledge_frontmatter` = **994 passed**.

### Los dos movimientos de este corte

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

- **PostgreSQL 16.4 portable VIVO** en `127.0.0.1:5432`, base `usdcop_trading`, sin admin y sin
  servicio de Windows. `--plan legacy-init` = **10 succeeded / 4 failed** (uno es de orden:
  `20-signalbridge-schema` pide `users` que crea el `21-`).
- **`fabric-v1` NO aplicada, deliberadamente.** Es `REVIEW_GATED_PLANS` y su digest es, en
  palabras del propio código, *"a second factor, not a way for modified on-disk SQL to authorize
  itself"*. El error imprime el digest esperado; copiárselo de vuelta sería anular el candado
  desde dentro. **Requiere autorización del operador.** Consecuencia: la migración 070 no está,
  y con ella BL-18 integration sigue sin poder correr.
- **Docker NO está instalado** y no se puede instalar desde esta sesión: es no-admin y Docker
  Desktop exige elevación más WSL2 (ausente). Airflow, Redis, MinIO, Grafana y todo E2E servido
  siguen fuera de alcance. Cualquier cierre que los exija es **STACK_OR_CI**, no LOCAL_CLOSABLE.

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

1. CODEX cofirma este 9/36/2 contra frontmatter, no contra narración.
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
