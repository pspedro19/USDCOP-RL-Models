---
kind: spec
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - .claude/specs/planes/backlog/README.md
  - .claude/rules/quant-constitution.md
  - scripts/validation/check_trial_ledger.py
  - scripts/validation/report_ledger_dsr.py
  - src/policy_engine/runner.py
  - src/contracts/passport.py
  - tests/regression/test_trial_ledger.py
  - .github/workflows/ci.yml
---

# BDD-MATRIX — matriz maestra de verificación de los 47 BLs

> Escrita por `claude-root-9c3f1e42` (rol ARQUITECTO DE PRUEBAS) el 2026-07-28.
> **PROPUESTA**: queda ACTIVE cuando CODEX cofirme al final.
> No arrancó Docker ni servicios. Nada se commiteó. Este documento es
> documentación + esqueletos; los tests se escriben en la fase siguiente
> (`TDD-GAPS.md` es la cola de trabajo).

---

## 0. Cómo leer esta matriz

**Orden por RIESGO, no por número de BL.** Los clusters van de mayor a menor daño
si el escenario falla en producción:

| Cluster | Qué protege | BLs |
|---|---|---|
| **R1 · DINERO** | ejecución, sizing, kill switch, fallbacks | 21 · 26 · 27 · 30 · 42 · 46 · 47 |
| **R2 · HONESTIDAD** | claims de edge, DSR, N<20, copy de UI | 01-04 · 07 · 09-12 · 14 · 32 · 33 |
| **R3 · SEGURIDAD** | RBAC, credenciales, artefactos monetizables | 05 · 06 · 08 · 20 · 34 · 41 · 43 |
| **R4 · FUGA TEMPORAL** | look-ahead en 3 capas, cutoff, vintage | 13 · 24 · 26 · 29 · 35 · 38 · 39 |
| **R5 · IDENTIDAD / DETERMINISMO** | hashes, factories, paridad, strangler | 16 · 17 · 28 · 31 · 45 |
| **R6 · DATOS & DB** | calidad, cuarentena, esquemas, inventario | 19 · 36 · 37 · 40 · 44 |
| **R7 · OBSERVABILIDAD & HECHOS** | métricas, facts, relojes, anti-supervivencia | 18 · 22 · 23 · 25 |

**Vocabulario de la columna “estado real hoy”** (verificado el 2026-07-28 contra
`git log`, `git ls-files` y lectura de código — NO contra el estado declarado en
`PROGRESS.md`):

| Código | Significado |
|---|---|
| `DONE` | cross-review del otro APROBADO (solo BL-06 y BL-07) |
| `IMPL-C@<hash>` | implementado y **commiteado** por CLAUDE, sin cross-review |
| `IMPL-X-UNTRACKED` | implementado por CODEX pero **sin commitear** (`git status ?? `) |
| `PARCIAL` | hay código pero el BL declara puntos abiertos |
| `NO-ARRANCADO` | ni código ni spec de implementación |
| `NO VERIFICADO` | no lo comprobé en esta sesión — no lo trates como conocido |

**Vocabulario de la columna “test”**:

| Código | Significado |
|---|---|
| `EXISTE:<ruta>` | hay test y muerde (mutación demostrada por su autor) |
| `INSUF:<ruta>` | existe pero no muerde: no cubre el escenario, o cubre el camino feliz |
| `FALTA` | no hay nada. Va a `TDD-GAPS.md` |
| `FALTA-CI` | el test existe pero **ningún workflow lo ejecuta** |

**Niveles**: `unit` · `contrato` (paridad Py↔TS sobre fixture compartida) ·
`integración` (dos módulos reales, sin red) · `E2E` (Playwright / pipeline completo).

---

## 1. Advertencia de método (léela antes de repartir trabajo)

Tres hechos condicionan TODA la matriz. Están verificados y cambian qué significa
“verde”:

1. **`vitest` no corre en ningún workflow.** `grep -rn "vitest\|npm test" .github/workflows/`
   ⇒ 0 resultados. Las tres fixtures compartidas Py↔TS
   (`policy_contract_cases.v1.json`, `policy_backend_cases.v1.json`,
   `forecast_output_cases.v1.json`) **solo se verifican del lado Python en CI**.
   Todo escenario marcado `contrato` es hoy *media* garantía.
2. **La gobernanza del ledger no está cableada a CI.** `ci.yml` corre `tests/unit/`
   e `tests/integration/`; `specs-gate.yml` corre `tests/regression/` con una
   allowlist de seis ficheros que **no incluye** `test_trial_ledger.py`,
   `test_bl09_bl11_bl12_governance.py` ni `test_bl10_legacy_estimate_contract.py`.
   51 tests de gobernanza + 2 CLIs fail-closed pueden romperse con CI en verde.
3. **El 100 % del ledger es `env: legacy_backfill`** (239/239). Todas las ramas
   fail-closed *prospectivas* (checks 8-10, `check_cutoff_policy`) son **código
   muerto frente a datos reales**: solo se ejercitan en fixtures de mutación. Y las
   10 familias declaran `claims_edge: false`, así que **el gate DSR hoy no gatea
   nada**. Verde ≠ probado.

---

## 2. R1 · DINERO — ejecución, sizing, kill switch, fallbacks

> Si algo de este cluster falla, se pierde dinero real. Es el único cluster donde
> un `FALTA` es BLOQUEANTE por defecto.

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-46/47 | CLAUDE | `IMPL-C@1f2c0da` / `IMPL-C@cdd6494` | R1-01..R1-05 |
| BL-21 | CODEX | `IMPL-X-UNTRACKED` (`074_exec_event_sourcing.sql`, `src/execution/events.py`) | R1-06..R1-08 |
| BL-26/27 | CODEX | `IMPL-X-UNTRACKED` (`077_portfolio_control.sql`, `src/portfolio/`) | R1-09..R1-12 |
| BL-30 | CODEX | `IMPL-X-UNTRACKED` (`src/execution/service.py`, runbook) | R1-13..R1-15 |
| BL-42 | CLAUDE | `IMPL-C@e5c72b5` (declarado; **NO VERIFICADO** en esta sesión) | R1-16 |

---

### BDD-R1-01 · Un solo motor de evaluación · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: la misma política evaluada en backtest y en live produce la misma decisión
  Dado una política congelada con policy_hash H y un snapshot S con available_at fijo
  Cuando la evalúo por la ruta de backtest
  Y la evalúo por la ruta de producción
  Entonces ambas rutas pasan por el MISMO punto de entrada
  Y las dos decisiones son idénticas campo a campo, incluido el fingerprint
  Y ninguna ruta llama a Policy.evaluate() saltándose el runner
```

- **Invariante**: `strategy-engines.md` invariante 4 (“un solo motor de evaluación;
  dos implementaciones = el backtest miente”).
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA**: `src/policy_engine/runner.py::evaluate_policy` (la
  entrada única declarada) **no tiene ningún consumidor de producción** — solo
  `tests/unit/test_policy_backend_contract.py`. El único código que evalúa
  políticas de verdad, `scripts/validation/check_policy_parity.py:58`, llama
  `policy.evaluate(snap, PolicyContext(...))` **directamente**, saltándose los
  fallbacks. La invariante está construida pero no cableada.

### BDD-R1-02 · Dos `build_policy` con semánticas distintas · **nivel: unit** · BLOQUEANTE

```gherkin
Escenario: un spec SPEC_ONLY no se puede construir por ninguna puerta
  Dado config/policies/smart_simple_v11.yaml con migration.status = SPEC_ONLY
  Cuando pido construir la política por cualquier constructor público del repo
  Entonces todos fallan cerrado con el mismo tipo de error
  Y ninguno devuelve un objeto Policy ejecutable
```

- **Invariante**: fail-closed (canal §4) + `strategy-engines.md` invariante 2
  (congelar la receta ES congelar la estrategia).
- **Estado del test**: `INSUF: tests/unit/test_policy_specs.py` — cubre el loader,
  no el runner.
- **DIVERGENCIA VERIFICADA (dentro del propio lote de CLAUDE)**: hay **dos**
  `build_policy` que no son la misma función:
  `src/policy_engine/runner.py:66` (2 roots de allowlist, **ignora
  `migration.status`**, no comprueba `policy_hash` congelado, no aplica las reglas
  §11) vs `src/strategies/policies/loader.py:256` (1 root, fail-closed ante
  `SPEC_ONLY` en `loader.py:260-264`, verifica hash declarado == derivado en
  `loader.py:194-200`). **La puerta más laxa es la que se exporta como superficie
  pública del motor** (`src/policy_engine/__init__.py:20-27`). Un spec `SPEC_ONLY`
  es rechazado por el loader y **aceptado por el runner**.

### BDD-R1-03 · El fallback declarado en el YAML se honra · **nivel: unit** · BLOQUEANTE

```gherkin
Escenario: una feature ausente bloquea, no produce un default silencioso
  Dado una política cuyo spec declara missing_input_policy = FAIL_CLOSED
  Cuando evalúo con un snapshot al que le falta una feature requerida
  Entonces no se emite ninguna decisión
  Y el error nombra la feature ausente

Escenario: staleness declarada en el spec, no en el llamador
  Dado una política cuyo spec declara stale_input_policy = FLAT
  Y un snapshot cuya feature más reciente supera max_age
  Cuando la evalúo SIN pasar ningún kwarg de fallback
  Entonces la decisión es FLAT con reason_code de staleness
  Y el motor NO usa su default de compilación
```

- **Invariante**: `strategy-engines.md` invariante 9 (“fallbacks declarados… sin
  default explícito no hay freeze”) + fail-closed del canal.
- **Estado del test**: `INSUF: tests/unit/test_policy_backend_contract.py` — prueba
  los kwargs, no la lectura del spec.
- **DIVERGENCIA VERIFICADA — TRES semánticas de fallback conviviendo**:
  (a) `runner.py:172-215` toma el fallback como **kwarg del llamador** y **nunca lee
  `spec.policy.missing_input_policy`**;
  (b) `src/strategies/policies/base.py:104-119` (CodedPolicy) sí lee el spec;
  (c) `src/contracts/policy_dsl.py:511-515` (DeclarativePolicy, el motor de spx500)
  tiene el `raise` **hardcoded** e ignora lo declarado.
  Los tres specs con `stale_input_policy: FLAT`
  (`btc_hodl_b1.yaml:55`, `gold_trend_simple.yaml:64`, `spx500_daily_ma200_v1.yaml:55`)
  dependen de un llamador que **no existe**.

### BDD-R1-04 · Vocabulario de fallback cerrado y único · **nivel: unit** · GRAVE

```gherkin
Escenario: un valor de fallback aceptado por el validador no explota en el motor
  Dado un spec con stale_input_policy = HOLD
  Cuando lo valido y luego lo evalúo
  Entonces o ambos lo aceptan, o ambos lo rechazan con el mismo error
  Pero nunca "validador verde, motor ValueError"
```

- **Invariante**: contract-first + espejo (canal §4); DRY del vocabulario.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA**: `loader.py:44` acepta
  `("FAIL_CLOSED","FLAT","HOLD")`; `runner.py:51` solo `("FAIL_CLOSED","FLAT")`.
  Un spec con `HOLD` pasa la validación y **revienta con `ValueError` en
  `runner.py:186-187`**. Nadie usa `HOLD` hoy: bomba latente, no incendio.
  Y hay un **cuarto** vocabulario en el lote de CODEX:
  `src/portfolio/snapshot.py:16-20` define
  `MissingPolicy = FLAT | KEEP_POSITION_UNTIL_EXPIRY | EXIT_ONLY | USE_LAST_VALID_WITH_MAX_AGE`.

### BDD-R1-05 · El factory nunca ramifica por `strategy_id` · **nivel: unit**

```gherkin
Escenario: añadir un motor nuevo no toca el núcleo
  Dado un engine.type nuevo registrado con su panel
  Cuando construyo y evalúo una política de ese tipo
  Entonces no hizo falta modificar el evaluador ni el factory
  Y ninguna rama del factory menciona un strategy_id literal
```

- **Invariante**: `strategy-engines.md` DO NOT (“no ramificar el factory por
  `strategy_id`”) + SOLID/OCP.
- **Estado del test**: `INSUF: tests/unit/test_policy_specs.py` (17 checks §11,
  anti-eval sobre texto crudo) — no tiene un test de extensión (motor nuevo ⇒ cero
  cambios en el core).
- **Verificado**: el factory ramifica por `implementation.mode`
  (`loader.py:265-268`, `runner.py:80-112`). El único dict keyed por id es
  `scripts/validation/check_policy_parity.py:155-159` (`CHECKS = {...}`), que es un
  arnés de paridad contra implementaciones legacy congeladas, no el factory. **No
  viola el DO-NOT, pero es el sitio a vigilar.**

### BDD-R1-06 · Idempotencia de órdenes · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: reintentar la emisión de una orden no duplica exposición
  Dado una decisión con execution_fingerprint F
  Cuando el emisor la envía dos veces (reintento tras timeout)
  Entonces existe exactamente una fila en exec.order_header con esa idempotency_key
  Y la segunda llamada devuelve la orden existente, no una nueva
```

- **Invariante**: fail-closed del canal; `risk-management.md` (kill switch/OMS).
- **Estado del test**: `FALTA` (y `src/execution/events.py` está **untracked**).

### BDD-R1-07 · El estado de una orden es proyección, no columna · **nivel: contrato** · GRAVE

```gherkin
Escenario: el consumidor lee el estado desde la proyección canónica
  Dado una secuencia de exec.order_status_event para una orden
  Cuando consulto el estado de esa orden
  Entonces lo obtengo de exec.v_order_state
  Y el literal de estado pertenece a un vocabulario CERRADO
```

- **Invariante**: fail-closed; SSOT (un concepto, una fuente).
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D6)**: `exec.order_status_event.status` es
  `TEXT NOT NULL` **sin CHECK** (`074:37`). Nadie garantiza el literal. El Passport
  de CLAUDE consulta `o.state='OPEN'` — columna que **no existe**.

### BDD-R1-08 · Los `reference.*` son precondición dura · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: no se puede insertar un hecho de un instrumento desconocido
  Dado un instrument_id no presente en reference.instrument
  Cuando intento insertar en exec.order_header o fact.position
  Entonces la FK lo rechaza
  Y existe un poblador de reference.* que corre ANTES en el orden de arranque
```

- **Invariante**: `data-governance.md` (identidades canónicas) + fail-closed.
- **Estado del test**: `FALTA`.
- **RIESGO OPERATIVO VERIFICADO**: `reference.asset/instrument/provider/
  provider_symbol/instrument_alias/calendar` son FK obligatorias de `market.*`,
  `fact.*`, `exec.*` y `portfolio.target`, y **ningún código las puebla**. Además
  `scripts/ops/db_migrate.py:120` hace `sorted(MIGRATIONS_DIR.glob("*.sql"))` **sin
  allowlist**: las nueve migraciones 070-078 se aplicarían en el próximo arranque
  aunque nadie las haya cross-revisado.

### BDD-R1-09 · La barrera temporal del libro es dura · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: una señal posterior al cutoff no entra en el snapshot
  Dado un portfolio_snapshot con cutoff_time T
  Y una señal cuyo available_at es posterior a T
  Cuando construyo el snapshot
  Entonces esa señal aparece en stale_signals, no en las aceptadas
  Y el allocator no la puede ver

Escenario: falta una sleeve requerida
  Dado required_sleeves que incluye una sleeve sin señal fresca
  Cuando construyo el snapshot
  Entonces la sleeve figura en missing_sleeves
  Y se aplica su MissingPolicy declarada, nunca un 0 silencioso
```

- **Invariante**: `quant-constitution.md` §4 capa Datos (anti-look-ahead) +
  fail-closed.
- **Estado del test**: `INSUF: tests/unit/test_codex_fabric_contracts.py` — toca el
  módulo, no la barrera con datos reales.
- **DIVERGENCIA VERIFICADA (D16)**: la columna SQL es `missing_sleeves`
  (`077:11`), el dataclass Python la llama `missing_signals`
  (`src/portfolio/snapshot.py:39,104`). Un writer que mapee por nombre falla.

### BDD-R1-10 · El allocator respeta caps y no inventa vol · **nivel: unit** · BLOQUEANTE

```gherkin
Escenario: sin previsión de volatilidad no se asigna capital
  Dado una sleeve sin volatility en el input del allocator
  Cuando ejecuto la asignación
  Entonces esa sleeve recibe budget cero y una razón explícita
  Y NO se usa una vol por defecto

Escenario: los multiplicadores están acotados
  Dado cualquier combinación de multiplicadores
  Entonces final_risk_budget nunca supera base_risk_budget
  Y cada multiplicador cae en [0,1]
```

- **Invariante**: fail-closed; `risk-management.md`.
- **Estado del test**: `INSUF: tests/unit/test_codex_fabric_contracts.py`.
- **Nota verificada**: `src/portfolio/allocator.py:35` recibe
  `volatility: Mapping[str,float]` como **input** y **nunca lo devuelve ni lo
  persiste**; `portfolio.allocation` no guarda vol. El Passport lee
  `vol_target_pct`/`vol_forecast_pct` en 4 sitios (`compose.ts:553-554,720-721`) —
  no hay de dónde.

### BDD-R1-11 · Gate de novedad del allocator · **nivel: unit**

```gherkin
Escenario: una sleeve sin historial suficiente no entra a tamaño completo
  Dado una sleeve cuya evidencia forward es menor que el mínimo declarado
  Cuando corre el allocator
  Entonces su multiplier_forward la mantiene en tier reducido
  Y el motivo queda registrado
```

- **Invariante**: `quant-constitution.md` §5 (retiro pre-firmado, umbrales no se
  relajan) + §2.
- **Estado del test**: `FALTA`.

### BDD-R1-12 · Escritura del allocator a su propia tabla · **nivel: integración** · GRAVE

```gherkin
Escenario: la asignación decidida es la asignación persistida
  Cuando el allocator produce una asignación
  Entonces existe una fila en portfolio.allocation con ese allocation_id
  Y los pesos leídos de la tabla reproducen los decididos
```

- **Invariante**: SSOT; “decisiones sobre números publicados”
  (`quant-constitution.md` §7).
- **Estado del test**: `FALTA`.
- **VERIFICADO**: `src/portfolio/allocator.py` **no escribe a su propia tabla**.
  `portfolio.allocation`, `portfolio.snapshot_signal` y `portfolio.pretrade_decision`
  no tienen ni un writer ni un lector fuera de la migración.

### BDD-R1-13 · Kill switch independiente del orquestador · **nivel: E2E** · BLOQUEANTE

```gherkin
Escenario: el kill switch bloquea aunque Airflow esté caído
  Dado el servicio de ejecución corriendo y el scheduler detenido
  Cuando se acciona el kill switch a nivel EXIT_ALL
  Entonces ninguna orden nueva se acepta
  Y las abiertas se cancelan según el nivel
  Y queda asiento append-only en audit_log
```

- **Invariante**: `rbac.md` §4 (kill global solo admin, siempre a `audit_log`) +
  `risk-management.md`.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D10)**: el Passport declara
  `kill_switch_engaged: Sourced<boolean>` (`compose.ts:545`); lo real es
  `portfolio.kill_switch_event.level` con enum de **cinco** niveles
  (`CLEAR|BLOCK_NEW|CANCEL_OPEN|EXIT_ALL|ACCOUNT_FREEZE`, `077:85`) y llave por
  `account_id`, **sin llave de estrategia**. Un booleano no puede representar
  cinco niveles sin perder información.

### BDD-R1-14 · Paper-first sigue siendo el último gate · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: el servicio nuevo no crea un camino que evite PreTradeGate
  Dado el servicio de ejecución externo
  Cuando emite una orden por cualquier ruta
  Entonces la orden pasó por PreTradeGate
  Y con trading_mode = PAPER simula y NO envía al exchange
```

- **Invariante**: `rbac.md` §6 (paper-first, fail-safe: error ⇒ BLOCK) + DO NOT
  “saltarte PreTradeGate en ningún path de orden nuevo”.
- **Estado del test**: `FALTA`.
- **RIESGO VERIFICADO**: hay **dos** clases `ExecutionService` —
  `src/execution/service.py` (nueva, untracked, CODEX) y
  `services/signalbridge_api/app/services/execution.py` (la de producción, usada
  por 8 rutas). Colisión de nombre + posible segundo camino al exchange.

### BDD-R1-15 · Reconciliación contra fills · **nivel: integración**

```gherkin
Escenario: una posición no reconciliada no se publica como live
  Dado fills del exchange que no cuadran con fact.position
  Cuando se calcula el estado de la sleeve
  Entonces su reconciliation_status es MISMATCH
  Y ninguna superficie la etiqueta como "live reconciliado"
```

- **Invariante**: honestidad del canal (§4); `quant-constitution.md` §6.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D9)**: el Passport hace `bool_and(r.reconciled)`;
  `exec.reconciliation_event` usa `status TEXT CHECK IN
  ('RECONCILED','MISMATCH','QUARANTINED')` (`078:15`) y está keyed por
  `account_id`+`instrument_id` (`078:7-8`), no por estrategia. **No hay columna
  booleana que leer.**

### BDD-R1-16 · Unidades decimales en la señal · **nivel: contrato**

```gherkin
Escenario: una exposición nunca se interpreta en dos unidades
  Dado una señal con target_exposure
  Cuando la leen backend y frontend
  Entonces ambos la interpretan en la MISMA unidad declarada
  Y un valor fuera de rango se rechaza en los DOS lenguajes
```

- **Invariante**: `strategy-contract.md` invariante 5 (métricas por activo) +
  contract-first espejo.
- **Estado del test**: `NO VERIFICADO` — BL-42 se declara integrado en `e5c72b5`;
  no lo comprobé en esta sesión.

---

## 3. R2 · HONESTIDAD — claims de edge, DSR, N<20, copy de UI

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-01..04 | CLAUDE | `IMPL-C@b86083e/f2350db` (BL-01 MD dice `PLANNED`; el test **sí existe**) | R2-01..R2-04 |
| BL-07 | CODEX | `DONE` | R2-05 |
| BL-09/11/12 | CLAUDE | `IMPL-C@1bc41ee` | R2-06..R2-10 |
| BL-10 | CODEX | contenido íntegro, **cierre administrativo pendiente** (incidente CXD-045) | R2-11 |
| BL-14 | CLAUDE | `IMPL-C@5a2cf5d+ecbfca5` | R2-12 |
| BL-32 | CLAUDE | `IMPL-C@1bc41ee` | R2-13..R2-15 |
| BL-33 | CODEX | `NO-ARRANCADO` | R2-16 |

---

### BDD-R2-01 · El caveat es incondicional en las tres superficies · **nivel: unit**

```gherkin
Escenario: ninguna superficie de forecasting se renderiza sin caveat
  Dado cualquiera de las tres superficies (GM, legacy dashboard, weekly Gold/BTC)
  Cuando se renderiza con datos válidos o vacíos o en error
  Entonces el disclaimer aparece siempre
  Y su texto viene del SSOT compartido, no de un literal local
```

- **Invariante**: `quant-constitution.md` §6 (desconfianza de la magia) + honestidad.
- **Estado del test**: `EXISTE: usdcop-trading-dashboard/tests/unit/components/forecasting-caveat-surfaces.test.tsx`
  + `tests/regression/test_forecasting_caveat_present.py` (candado estático,
  20 passed, 8 mutaciones demostradas). **`FALTA-CI` del lado TS** (vitest no corre).

### BDD-R2-02 · Las predicciones no se colorean como recomendación · **nivel: unit**

```gherkin
Escenario: una predicción no usa el vocabulario ni la paleta de una orden
  Dado la tabla de predicciones de cualquier activo
  Entonces no aparece "LONG" ni "SHORT" como etiqueta de la predicción
  Y el signo no se colorea con verde/rojo de ejecución
  Y la columna de convicción se rotula como proxy, no como probabilidad
```

- **Invariante**: honestidad del canal §4; `rbac.md` §8 (subscribers ven outputs).
- **Estado del test**: `EXISTE` (12 render tests, mutaciones
  quitar/ocultar/rama/css/recolorear/re-etiquetar ⇒ rojo). `FALTA-CI`.

### BDD-R2-03 · El banner de Gold weekly no depende del modo · **nivel: unit**

```gherkin
Escenario: modo rule-based también avisa
  Dado un activo con forecast_mode rule-based (Gold)
  Cuando se renderiza su inferencia semanal
  Entonces el banner fuerte aparece igual que en el modo model-zoo
```

- **Invariante**: honestidad; `strategy-contract.md` invariante 5.
- **Estado del test**: `EXISTE` (cubierto por R2-01). `FALTA-CI`.

### BDD-R2-04 · Un solo caveat, un solo SSOT de texto · **nivel: unit** · DRY

```gherkin
Escenario: cambiar el texto del caveat lo cambia en todas partes
  Dado que edito el SSOT del disclaimer
  Cuando renderizo las tres superficies
  Entonces las tres muestran el texto nuevo
  Y no queda ninguna copia literal del texto viejo en el árbol
```

- **Invariante**: DRY/SSOT del canal §4.
- **Estado del test**: `INSUF` — el candado estático generalizado detecta ausencia,
  no detecta una **segunda copia** del literal.

### BDD-R2-05 · `timing_ratio` no se reporta como edge · **nivel: unit**

```gherkin
Escenario: un timing_ratio calculado no habilita un claim
  Dado el timing_ratio one-off de las 4 campeonas
  Entonces se publica con su N y su fuente
  Y no aparece junto a un Sharpe con N<20
```

- **Invariante**: `quant-constitution.md` §6; `strategy-contract.md` invariante 6.
- **Estado del test**: `EXISTE` (BL-07 es `DONE`, cross-review `CLD-118`).

### BDD-R2-06 · Ningún claim de edge sin DSR trial-aware · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: declarar edge sin DSR es una violación dura
  Dado una familia con claims_edge = true
  Cuando corre el gate de gobernanza
  Entonces exige un DSR recomputado con el N actual del linaje
  Y si DSR <= 0.95 el gate falla con exit != 0

Escenario: el N usado es el N vigente, no uno congelado
  Dado que se añade un trial al ledger
  Cuando recomputo el DSR
  Entonces el DSR baja
  Y cualquier claim previo que dependiera del N viejo queda invalidado
```

- **Invariante**: `quant-constitution.md` §2 (bar DSR > 0.95 sin ADR).
- **Estado del test**: `EXISTE: tests/regression/test_bl09_bl11_bl12_governance.py`
  (34 tests, 22 mutaciones) — pero **`FALTA-CI`** y hoy el gate **no gatea nada**:
  las 10 familias declaran `claims_edge: false`.
- **DIVERGENCIA VERIFICADA — dos gates DSR con tres N distintos**:
  `scripts/pipeline/train_and_export_smart_simple.py:1235` calcula el gate de
  Vote 1 con `n_trials = fm["n_trials_total"]` = **N=111 (total del activo)**;
  `scripts/validation/report_ledger_dsr.py` gatea la misma estrategia con
  **N=60 / 138 / 239** (familia/cluster/global). Sin cross-referencia entre ambos.

### BDD-R2-07 · El DSR se calcula en UNA sola implementación · **nivel: unit** · GRAVE

```gherkin
Escenario: nadie reimplementa el SSOT constitucional
  Dado cualquier módulo del repo que produzca un DSR
  Entonces obtiene el valor de services/common/metrics.py
  Y no existe una segunda fórmula de deflated/probabilistic Sharpe en el árbol
```

- **Invariante**: DRY/SSOT del canal §4; `quant-constitution.md` §2.
- **Estado del test**: `INSUF: tests/regression/test_quant_library_gate.py::test_promoted_skills_do_not_shadow_the_metrics_ssot`.
- **DIVERGENCIA VERIFICADA — duplicación REAL del SSOT constitucional**:
  `src/strategies/spx500_regime_gated_v1/deflated_sharpe.py` (194 líneas,
  **git-tracked**) es una segunda implementación completa
  (`expected_max_sharpe`, `probabilistic_sharpe`, `deflated_sharpe`,
  `min_track_record_length`), usa `scipy.stats.norm` donde el SSOT usa un
  `_norm_ppf` propio, y devuelve `float` donde el SSOT devuelve
  `{sr0, dsr, significant}`. Está **viva**: `kernels.py:18` la importa. El gate que
  existe para evitar exactamente esto **solo itera `.claude/skills/**` y no puede
  ver `src/`**. El duplicado está en el punto ciego del test escrito para
  detectarlo.

### BDD-R2-08 · El ledger es append-only y la cadena lo prueba · **nivel: unit**

```gherkin
Escenario: editar un asiento pasado rompe la cadena
  Dado registries/ledger.jsonl con su hash-chain
  Cuando modifico un carácter de un asiento intermedio
  Entonces el validador falla identificando la línea
  Y el exit code es distinto de 0
```

- **Invariante**: `quant-constitution.md` §2; append-only.
- **Estado del test**: `EXISTE: tests/regression/test_trial_ledger.py` (check 3).
  **`FALTA-CI`**.
- **Verificado por mí**: 239 líneas · FT=55 · AT=184 · usdcop=111 · xauusd=77 ·
  btcusdt=34 · spx500=17. Génesis `FT-0001` con `prev_hash` = 64 ceros. Los
  números que CLAUDE declaró son **exactos**.

### BDD-R2-09 · La prosa no puede contradecir al ledger · **nivel: unit**

```gherkin
Escenario: una cabecera con conteos falsos falla
  Dado registries/README.md declarando 237 trials
  Y un ledger con 239
  Cuando corre el validador
  Entonces falla campo a campo (n_global, n_ft, n_at, per_asset, per_family)
```

- **Invariante**: “ningún número vive en prosa” (sistema de conocimiento) +
  `quant-constitution.md` §2.
- **Estado del test**: `EXISTE` (check 12, mutación demostrada). `FALTA-CI`.

### BDD-R2-10 · La muralla FT→AT exige provenance explícita · **nivel: unit**

```gherkin
Escenario: convertir un forecast en señal económica cobra su trial
  Dado una familia con provenance.crosses_wall = true
  Entonces declara inherits_from_family existente de kind forecast del mismo cluster
  O declara forecast_trial_ids existentes de kind forecast
  Y omitir ambos hace fallar el validador
```

- **Invariante**: `quant-constitution.md` §2 (doble linaje FT/AT, ADR-0022).
- **Estado del test**: `EXISTE` (check 11). `FALTA-CI`.
- **Hueco declarado**: la **completitud** es indemostrable. Los 13 checks verifican
  consistencia interna (ledger↔YAML↔README↔front-matter). Un trial ejecutado y
  nunca cargado es invisible para todos ellos. **El ledger prueba que nadie lo
  editó, no que alguien lo registró todo.** Esto es una limitación estructural, no
  un bug: debe quedar declarado, no “arreglado” con un test que mienta.

### BDD-R2-11 · `legacy_estimate` no se puede blanquear · **nivel: unit**

```gherkin
Escenario: quitar la nota de estimación a un bloque legacy falla
  Dado una celda backfilled con label legacy_estimate
  Cuando le quito la nota que declara la estimación
  Entonces el validador la rechaza
```

- **Invariante**: honestidad; `quant-constitution.md` §2.
- **Estado del test**: `EXISTE: tests/regression/test_bl10_legacy_estimate_contract.py`.
  `FALTA-CI`.

### BDD-R2-12 · La receta del predictor está congelada · **nivel: contrato**

```gherkin
Escenario: cambiar la receta cambia el hash
  Dado el bloque components del passport de v11
  Cuando altero un hiperparámetro de la receta
  Entonces feature_set_hash o params_hash cambian
  Y la versión congelada deja de validar
```

- **Invariante**: `ssot-versioning.md` invariante 2 (config congelado);
  `strategy-engines.md` invariante 2.
- **Estado del test**: `NO VERIFICADO`.

### BDD-R2-13 · No existe el estado “estimated” en el Passport · **nivel: contrato**

```gherkin
Escenario: un campo sin fuente publicada se declara unavailable
  Dado un campo del Passport sin artefacto publicado
  Entonces su source.status es unavailable
  Y su value es null
  Y se nombra quién lo debe
  Y NUNCA se rellena con un proxy calculado
```

- **Invariante**: honestidad del canal §4; `quant-constitution.md` §7.
- **Estado del test**: `EXISTE: tests/unit/test_passport_contract.py` +
  `usdcop-trading-dashboard/tests/unit/contracts/passport-contract.test.ts` —
  pero **`INSUF`**: son dos definiciones paralelas **sin fixture compartida**, así
  que Py y TS pueden divergir sin que nada lo detecte. `FALTA-CI` en el lado TS.

### BDD-R2-14 · Sharpe publicado con N<20 se rechaza · **nivel: contrato**

```gherkin
Escenario: N pequeño ⇒ solo conteo y PnL
  Dado una sleeve con 12 trades y un Sharpe publicado
  Cuando valido el Passport
  Entonces ambos lenguajes rechazan el documento con el mismo error
```

- **Invariante**: `quant-constitution.md` §6; `strategy-contract.md` invariante 6.
- **Estado del test**: `INSUF` (misma causa que R2-13: sin fixture compartida).

### BDD-R2-15 · El semáforo de retiro por sleeve no miente · **nivel: contrato**

```gherkin
Escenario: sin dato por sleeve, el semáforo es unknown
  Dado que system_health solo publica banderas GLOBALES
  Cuando compongo el Passport de una sleeve
  Entonces su semáforo de retiro es unknown, no verde
  Y los tiers sin comprobar cuentan null, no 0
```

- **Invariante**: honestidad; `quant-constitution.md` §5.
- **Estado del test**: `EXISTE: tests/unit/test_passport_contract.py` (declarado
  por diseño). **`INSUF` en el frontera**: ver R7-04, el reloj que sí existe se
  pierde por un nombre.

### BDD-R2-16 · La Readiness Matrix vive de evidencias, no de opiniones · **nivel: integración**

```gherkin
Escenario: una fila sin evidencia no puede estar en verde
  Dado un criterio de la matriz institucional
  Cuando no tiene artefacto publicado que lo respalde
  Entonces su estado es "sin evidencia", no "cumple"
```

- **Invariante**: honestidad; `quant-constitution.md` §7.
- **Estado del test**: `FALTA` (BL-33 `NO-ARRANCADO`).

---

## 4. R3 · SEGURIDAD — RBAC, credenciales, artefactos monetizables

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-05 | CLAUDE | `IMPL-C@6f76934` (remedio a11y) | R3-01 |
| BL-06 | CLAUDE | **`DONE`** | R3-02 |
| BL-08 | CODEX | `NO-ARRANCADO` — **bloquea el push de TODO** | R3-03 |
| BL-20 | CLAUDE | `IMPL-C@57c3e1c` (TreeSHAP declarado PARTIAL) | R3-04..R3-05 |
| BL-34 | CLAUDE | `IMPL-C@a18be01` | R3-06 |
| BL-41 | CODEX | `NO-ARRANCADO` (contrato+TDD sí, **DDL prohibida** hasta Vault real) | R3-07 |
| BL-43 | CODEX | `NO-ARRANCADO` | R3-08 |

---

### BDD-R3-01 · El paper ledger es legible sin ser accionable · **nivel: E2E**

```gherkin
Escenario: un rol no-admin ve el A/B pero no puede promover
  Dado un usuario con research:read en /production
  Cuando abre el panel de candidatas paper (v11/v12/v14)
  Entonces ve los números publicados
  Y no existe ningún control de aprobación en la página
```

- **Invariante**: `approval-gates.md` invariante 3 (“`/production` es read-only”);
  `rbac.md` §1.
- **Estado del test**: `EXISTE: usdcop-trading-dashboard/tests/e2e/paper-candidates-a11y.spec.ts`
  + `tests/unit/components/PaperCandidatesPanel.test.tsx`. **`FALTA-CI`** (ni
  vitest ni Playwright corren en workflows).

### BDD-R3-02 · La muralla frontend: forecasting no aprueba ni ejecuta · **nivel: unit**

```gherkin
Escenario: ninguna superficie de forecasting expone acción de dinero
  Dado cualquier componente bajo forecasting
  Entonces no importa ni renderiza ApprovalPanel, Deploy ni Promote
```

- **Invariante**: `approval-gates.md` invariantes 3 y 4; `rbac.md` §1.
- **Estado del test**: `EXISTE` (candado estático, `20 passed`, cross-review CODEX
  APROBADO). **Este es el patrón de referencia**: es el único BL cuya verificación
  sobrevivió a un cross-review adversarial completo.

### BDD-R3-03 · Ningún secreto en el historial ni en el árbol · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: el escaneo del historial no encuentra credenciales
  Dado el historial completo del repositorio
  Cuando corre el escáner de secretos sobre TODOS los commits
  Entonces no aparece ningún .env con valores reales
  Y las credenciales expuestas figuran como rotadas con fecha
```

- **Invariante**: `rbac.md` §5 + DO NOT “hardcodear API keys”.
- **Estado del test**: `INSUF: .github/workflows/security-scan.yml` (Gitleaks) —
  **escanea el árbol, no demuestra que el historial esté purgado**.
- **Estado real**: `.env` real en `ee91273` sigue siendo PÚBLICO (memoria del
  operador). `git push` prohibido hasta cerrar esto. **DECISIÓN DEL OPERADOR**:
  rotación + purga + privatización no las puede ejecutar un ingeniero LLM.

### BDD-R3-04 · Los artefactos SHAP no son alcanzables sin permiso · **nivel: integración**

```gherkin
Escenario: 401/403 ANTES de tocar el filesystem
  Dado un usuario anónimo o sin admin:all
  Cuando pide un artefacto de interpretabilidad
  Entonces recibe 401/403
  Y el servidor no llegó a resolver ninguna ruta de fichero

Escenario: traversal y symlink no escapan del directorio
  Dado una ruta con ../ o un symlink que apunta fuera
  Entonces se rechaza con un error genérico
  Y el mensaje no revela la estructura del filesystem
```

- **Invariante**: `rbac.md` §1 (deny-by-default) y DO NOT “artefactos
  monetizables nuevos en `public/` sin gate”.
- **Estado del test**: `EXISTE` (25 tests adversariales + 8 pytest de schema;
  artefactos movidos con `git mv` fuera de `public/` a `data/interpretability/`).
  `FALTA-CI` del lado TS.

### BDD-R3-05 · TreeSHAP declarado PARTIAL no se presenta como completo · **nivel: unit**

```gherkin
Escenario: la UI no promete atribución que no calcula
  Dado un modelo cuya atribución es PARTIAL
  Entonces la vista lo declara explícitamente
  Y no muestra un ranking como si fuera TreeSHAP exacto
```

- **Invariante**: honestidad del canal §4.
- **Estado del test**: `EXISTE: usdcop-trading-dashboard/tests/unit/components/InterpretabilitySection.test.tsx`
  + `usdcop-trading-dashboard/tests/test_interpretability_schema.py`. `FALTA-CI` (TS).

### BDD-R3-06 · `/replay` es READ-ONLY de verdad · **nivel: E2E**

```gherkin
Escenario: /replay no ofrece Voto 2 a nadie
  Dado un admin autenticado en /replay
  Entonces no aparece ningún botón de aprobar ni de desplegar
  Y una nota indica que el Voto 2 vive en /dashboard

Escenario: el middleware decide, no el componente
  Dado un anónimo pidiendo /replay
  Entonces recibe 401 desde el middleware
  Y un free recibe 403
```

- **Invariante**: `approval-gates.md` invariantes 3 y 5; `rbac.md` §1 y §4.
- **Estado del test**: `EXISTE` (17/17 middleware server-side + 27/27 contrato RBAC
  + spec Playwright con rojo reproducido: `'Aprobar count=1 expected 0'`).
  `FALTA-CI` para Playwright.

### BDD-R3-07 · Seguridad DB P0 sin DDL prematura · **nivel: contrato**

```gherkin
Escenario: el contrato de secretos existe antes que el cutover
  Dado el contrato de secret.* y credenciales consolidadas
  Entonces sus tests existen y pasan
  Y NINGUNA migración con DDL de cutover está aplicada
  Y ningún rol de la migración es superusuario
```

- **Invariante**: `rbac.md` §5; guardarraíl del protocolo (“BL-41 sin DDL ni
  cutover hasta Vault real y roles no-super”).
- **Estado del test**: `FALTA`.
- **RIESGO OPERATIVO**: como `db_migrate.py:120` aplica por glob sin allowlist,
  **cualquier `.sql` dejado en `database/migrations/` se aplica en el próximo
  arranque**. Esto convierte “escribí la migración pero no la apliqué” en una
  afirmación que el sistema no respalda. Ver TDD-GAPS G-07.

### BDD-R3-08 · El modelo sintético demo no sale de demo · **nivel: unit**

```gherkin
Escenario: CI bloquea el sintético fuera de su perímetro
  Dado el modelo sintético de demostración
  Cuando se importa desde una ruta que no es de demo
  Entonces el gate de CI falla nombrando el importador
```

- **Invariante**: honestidad (un número sintético no puede parecer real).
- **Estado del test**: `INSUF: usdcop-trading-dashboard/tests/unit/synthetic-metrics.test.ts`
  — existe algo del lado TS; no hay gate de CI. `FALTA-CI`.

---

## 5. R4 · FUGA TEMPORAL — anti-look-ahead en tres capas

> `quant-constitution.md` §4 define tres capas (Datos / Modelos / Clasificadores).
> Esta sección exige una prueba por capa, no una declaración.

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-13 | CLAUDE | `IMPL-C@3861568` | R4-01 |
| BL-24 | CODEX | `IMPL-X-UNTRACKED` (`076_lineage_graph.sql`, `src/lineage/`) | R4-02..R4-03 |
| BL-26 | CODEX | ver R1-09 | — |
| BL-29 | CODEX | `IMPL-X-UNTRACKED` (`scripts/analysis/qlab.py`) | R4-04 |
| BL-35 | CODEX | `IMPL-X-UNTRACKED` (`src/orchestration/dataset_uri.py`) | R4-05 |
| BL-38 | CODEX | `IMPL-X-UNTRACKED` (`073_market_quality.sql`, `src/market/`) | R4-06 |
| BL-39 | CLAUDE | `IMPL-C@3861568` | R4-07 |

---

### BDD-R4-01 · `surface` es explícito y `normalize` lo respeta · **nivel: unit**

```gherkin
Escenario: un manifiesto sin surface no se normaliza en silencio
  Dado un manifiesto sin campo surface
  Cuando lo normalizo
  Entonces falla cerrado
  Y no se le asigna una surface por defecto
```

- **Invariante**: fail-closed; `strategy-contract.md` invariante 1.
- **Estado del test**: `EXISTE: tests/regression/test_strategy_manifests.py`
  (44 passed, hash canónico LF reproducible en cualquier OS, 5 walls rojas
  demostradas en checkout limpio).

### BDD-R4-02 · `availability_quality` bloquea la promoción · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: un PIT sin vintage real no promueve
  Dado que solo el 9.1 % del PIT es vintage real
  Y publication_date es 100 % NULL en monthly y quarterly
  Cuando se evalúa la promoción de una estrategia que consume ese macro
  Entonces la promoción se BLOQUEA
  Y el motivo cita la calidad de disponibilidad, no una métrica de performance
```

- **Invariante**: `quant-constitution.md` §4 capa Datos; `data-governance.md`
  (macro T-1, `merge_asof(backward)`).
- **Estado del test**: `FALTA`. `lineage.node.availability_quality` existe como
  columna (`076:7-23`); **nadie la lee**.

### BDD-R4-03 · Revisión de vintage tipificada · **nivel: unit**

```gherkin
Escenario: una corrección del proveedor no se confunde con un error de pipeline
  Dado una revisión de un dato ya publicado
  Entonces revision_type pertenece al vocabulario cerrado
  Y la rama as_released sigue consultable después de la revisión
```

- **Invariante**: `quant-constitution.md` §4; anti-look-ahead capa Datos.
- **Estado del test**: `FALTA`. La materia prima existe
  (`lineage.revision_event`, `076:41-49`, con `branch as_released|latest_revised`);
  falta la vista `last_vintage_revision` que el Passport pretende leer.

### BDD-R4-04 · El cutoff lo impone la capa de LECTURA · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: no se puede leer el futuro ni pidiéndolo
  Dado un cutoff T declarado en la consulta
  Cuando pido features con available_at posterior a T
  Entonces la capa de lectura no las devuelve
  Y no hay forma de saltarse el filtro desde la política
```

- **Invariante**: `strategy-engines.md` invariante 5 (“prohibido
  `SELECT ... ORDER BY time DESC LIMIT n` desde una política”);
  `quant-constitution.md` §4.
- **Estado del test**: `FALTA`.

### BDD-R4-05 · La arista `forecast → allocator` está prohibida en parseo · **nivel: unit**

```gherkin
Escenario: un forecast no puede alimentar directamente al libro
  Dado una URI de dataset con esquema forecast://
  Cuando se declara como input de strategy, action, portfolio o exec
  Entonces el parseo la rechaza
```

- **Invariante**: `quant-constitution.md` §2 (la muralla FT→AT es física, no
  documental).
- **Estado del test**: `INSUF: tests/unit/test_codex_fabric_contracts.py` —
  la regla existe en `src/orchestration/dataset_uri.py:60-69`; falta el test que
  demuestre el rechazo en el grafo real de DAGs, no solo en el parser aislado.

### BDD-R4-06 · Mercado canónico: el resampleo no adelanta información · **nivel: unit**

```gherkin
Escenario: una barra agregada no usa datos de después de su cierre
  Dado raw_bar de 5 minutos
  Cuando se agrega a 1h/4h/1d
  Entonces cada barra agregada solo contiene observaciones anteriores a su cierre
  Y el timestamp se ancla en UTC antes de aplicar el offset de cierre
```

- **Invariante**: `data-governance.md` invariante 1 (bug “Sunday pile-up”, ancla
  UTC) + `quant-constitution.md` §4.
- **Estado del test**: `INSUF` — existen
  `tests/regression/test_ohlcv_timestamps_are_instants.py` y
  `test_multi_symbol_marking.py`, pero no cubren los caggs nuevos.

### BDD-R4-07 · Feature contract por estrategia-versión, con hash reproducible · **nivel: unit**

```gherkin
Escenario: el hash del feature set no depende del sistema operativo
  Dado un checkout limpio en Windows y otro en Linux
  Cuando recomputo el hash canónico desde el blob de git
  Entonces obtengo el mismo valor
  Y una diferencia de CRLF no cambia el hash
```

- **Invariante**: determinismo/identidad; `ssot-versioning.md` invariante 1.
- **Estado del test**: `EXISTE: tests/regression/test_feature_contracts.py`
  (hash canónico LF, 5 walls rojas demostradas). Es el otro patrón de referencia.

---

## 6. R5 · IDENTIDAD Y DETERMINISMO

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-16 | CODEX | `NO-ARRANCADO` como workflow | R5-01 |
| BL-17 | CODEX | `IMPL-X-UNTRACKED` (`src/identity/`) | R5-02..R5-04 |
| BL-28 | CODEX | `IMPL-X-UNTRACKED` (`src/orchestration/semantic_diff.py`) | R5-05 |
| BL-31 | CLAUDE | `IMPL-C@6f76934` | R5-06 |
| BL-45 | CLAUDE | **R1 solo**; R2/R3 `NO-ARRANCADO` | R5-07 |

---

### BDD-R5-01 · CI constitucional Etapa 0 · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: la legalidad y la serialización canónica se verifican en CI
  Dado un cambio que rompe la serialización canónica
  Cuando corre CI
  Entonces un workflow falla y nombra el fichero
```

- **Invariante**: determinismo; canal §4 “tests que muerden”.
- **Estado del test**: `FALTA-CI` estructural. Hoy **ninguno** de estos corre en un
  workflow: `check_trial_ledger.py`, `report_ledger_dsr.py`,
  `validate_policy_specs.py`, `check_policy_parity.py`, ni `vitest`. Este es el
  escenario con mayor apalancamiento de toda la matriz: sin él, todo lo demás es
  reversible por accidente.

### BDD-R5-02 · Un objeto lógico ⇒ un hash · **nivel: unit** · GRAVE

```gherkin
Escenario: el mismo objeto lógico produce el mismo hash por cualquier ruta
  Dado el objeto {"a": "café", "b": 1, "f": 0.1+0.2}
  Cuando lo hasheo por la vía canónica de identidad
  Y lo hasheo por la vía de políticas
  Entonces obtengo el MISMO valor
```

- **Invariante**: DRY/SSOT del canal §4; “dos implementaciones del mismo cálculo =
  el backtest miente en algún sitio”.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA — el test de arriba está HOY EN ROJO**. Hay **una** sola
  `semantic_hash` (`src/identity/canonical.py:121`) y `src/strangler/parity.py:20`
  la **consume** (no la duplica — la sospecha del brief se descarta). Pero existe
  una familia paralela de 4 copias del mismo idiom en
  `src/contracts/policy.py:374`, `policy_dsl.py:131`, `policy_version.py:210`,
  `src/strategies/policies/loader.py:98`, que **diverge del canónico** en cuatro
  puntos que cambian el hash: `ensure_ascii` (canonical usa `False`, la familia
  policy hereda `True`), floats (`canonical.py:77` cuantiza a `Decimal` con
  `ROUND_HALF_EVEN`; la familia hashea el float crudo), `unicodedata.normalize("NFC")`
  (`canonical.py:67`, ausente en la familia) y `datetime` (canonical serializa a
  `_utc_z`, la familia lanza `TypeError`). Comprobación empírica:

  ```
  identity bytes: b'{"a":"caf\xc3\xa9","b":1,"f":"0.3"}'
  policy   bytes: b'{"a":"caf\\u00e9","b":1,"f":0.30000000000000004}'
  EQUAL? False
  ```

  Y **la divergencia es silenciosa**: `policy.py:72`
  `HASH_PATTERN = ^sha256:[0-9a-f]{8,64}$` acepta ambas longitudes, así que ninguna
  validación la delata.

### BDD-R5-03 · El código commiteado importa solo código commiteado · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: un clone limpio del HEAD importa
  Dado un checkout limpio del commit HEAD sin ficheros untracked
  Cuando importo todos los módulos de src/
  Entonces ningún import falla
```

- **Invariante**: reproducibilidad; canal §4 “ningún 'verificado' que no se
  ejecutó”.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA POR MÍ (la más peligrosa del lote)**:
  `git ls-files src/identity/` ⇒ **0 ficheros**. `src/identity/` está untracked.
  Pero `src/strangler/parity.py:20` — **commiteado** en `6f76934` — hace
  `from src.identity.canonical import CanonicalizationError, canonical_json_bytes, semantic_hash`.
  **El commit de CLAUDE no compila en un clone limpio.** Los 38 tests verdes de
  `tests/regression/test_strangler_cop.py` son **falso-verde**: pasan solo porque
  el directorio untracked existe en esta copia de trabajo. Misma exposición para
  todo consumidor de `src/execution/`, `src/orchestration/`, `src/portfolio/`,
  `src/metrics/`, `src/lineage/`, `src/market/`, `src/governance/`,
  `src/validation/*_harness.py` y `tests/unit/test_codex_fabric_contracts.py`.

### BDD-R5-04 · Separación de dominio en los fingerprints · **nivel: unit**

```gherkin
Escenario: un spec y una decisión con payload idéntico no colisionan
  Dado el mismo payload
  Cuando computo spec_fingerprint y decision_fingerprint
  Entonces los dos valores son distintos
```

- **Invariante**: identidad; SOLID (una responsabilidad por namespace).
- **Estado del test**: `INSUF: tests/unit/test_codex_fabric_contracts.py`.
  El diseño es correcto (`src/identity/fingerprints.py:13` envuelve en
  `{"namespace", "version":1, "payload"}`); falta el test de colisión.

### BDD-R5-05 · Diff semántico: campos volátiles no rompen la paridad · **nivel: unit**

```gherkin
Escenario: re-ejecutar un bundle produce diff vacío
  Dado dos ejecuciones del mismo bundle congelado
  Cuando comparo semánticamente ignorando los campos volátiles declarados
  Entonces equal es true
  Y cambiar un número económico lo pone en false nombrando el primer campo distinto
```

- **Invariante**: determinismo; `strategy-engines.md` invariante 8 (re-ejecutar una
  política congelada = 0 trials).
- **Estado del test**: `INSUF: tests/unit/test_codex_fabric_contracts.py:86` — es
  el **único** caller de `semantic_diff.compare()` en todo el repo. La lógica
  existe; **nada persiste ni publica un `SemanticDiff`**, y el Passport lee
  `replay_parity` como artefacto publicado (`compose.ts:790`).
- **Nota de honestidad**: un subagente reportó a media auditoría un `ImportError`
  vivo en `src/orchestration/semantic_diff.py:10`
  (`canonical_bytes`/`canonical_hash` inexistentes). Al re-verificarlo yo mismo,
  la línea ya importaba nombres correctos. **El árbol de trabajo se movió durante
  la auditoría**: trata cualquier hallazgo sobre ficheros untracked como fechado,
  no como permanente.

### BDD-R5-06 · Strangler: el camino legacy sigue vivo · **nivel: integración**

```gherkin
Escenario: migrar no apaga nada
  Dado una estrategia en estado PARITY_PENDING
  Cuando corre el pipeline
  Entonces el productor legacy sigue siendo la fuente de verdad publicada
  Y la política nueva solo se compara, no sustituye

Escenario: la paridad es exacta o el cutover no ocurre
  Dado una divergencia de exposición fuera de la ventana de calentamiento
  Entonces el estado NO avanza a CUTOVER
```

- **Invariante**: `strategy-engines.md` invariante 4; fail-closed.
- **Estado del test**: `EXISTE: tests/regression/test_strangler_cop.py` (38 tests)
  — pero **`INSUF`**: son falso-verdes por R5-03 (dependen de `src/identity/`
  untracked).
- **DECISIÓN DEL OPERADOR ya registrada por BL-47, no la resuelvo aquí**: Gold
  tiene **dos productores del mismo `strategy_id` que no coinciden**
  (`scripts/analysis/gold_trend_simple.py::simulate`, el del bundle publicado, SIN
  multiplicador de régimen, vs `src/gold_rl/strategies.py::build_positions`, el
  registrado en `STRATEGIES` y ejecutado por el DAG semanal, CON
  `REGIME_RISK_MULT`). Elegir cuál es la SSOT es modelado.

### BDD-R5-07 · BL-45 R2/R3 no existen — el índice no lo produce nadie · **nivel: observación**

```gherkin
Escenario: el registry de políticas tiene productor
  Dado el índice de policy_version
  Entonces existe al menos un config/strategies/*.yaml con engine.type
  Y una factory que lo consume
```

- **Invariante**: `strategy-engines.md` invariante 1 (un solo registro con
  discriminador `engine.type`).
- **Estado del test**: `FALTA`. Declarado por el propio BL-46: “BL-45 R2/R3 no
  existen en el repo — solo R1”.

---

## 7. R6 · DATOS Y DB

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-19 | CODEX | `IMPL-X-UNTRACKED` (`071_forecast_schema_roles.sql`) | R6-01 |
| BL-36 | CLAUDE | `IMPL-C@14687cd` (matriz desde código, cero DDL) | R6-02 |
| BL-37 | CODEX | `IMPL-X-UNTRACKED` (`072_reference_identity.sql`) | R6-03 |
| BL-40 | CODEX | `IMPL-X-UNTRACKED` (`073_market_quality.sql`, `src/data_quality/rules.py`) | R6-04 |
| BL-44 | CODEX | `NO-ARRANCADO` | R6-05 |

### BDD-R6-01 · `forecast_writer` no puede escribir hechos · **nivel: integración**

```gherkin
Escenario: el rol de escritura de forecast está acotado
  Dado una conexión con el rol forecast_writer
  Cuando intenta escribir en fact.* o exec.*
  Entonces la BD lo rechaza por permisos
```

- **Invariante**: `rbac.md` §5 (scoping estricto); mínimo privilegio.
- **Estado del test**: `FALTA`.

### BDD-R6-02 · La matriz de verdad DB no se contradice con el inventario · **nivel: unit**

```gherkin
Escenario: cada tabla del inventario tiene destino declarado
  Dado el inventario de tablas generado desde el código
  Entonces cada una está clasificada (viva / a migrar / a retirar)
  Y ninguna clasificación se apoya en prosa sin ancla
```

- **Invariante**: “ningún número vive en prosa”; SSOT.
- **Estado del test**: `INSUF: tests/regression/test_knowledge_inventory.py`
  (cubre el inventario de conocimiento, no la matriz DB).
- **Hallazgo verificado que BL-36 debe absorber**: **9 grupos de tablas nuevas sin
  ningún consumidor** — `exec.*` (5 objetos), `portfolio.*` (4), `fact.*` (2),
  `control.artifact_identity`, `control.strategy_declaration`, `market.raw_bar`,
  `forecast.*` (4), `reference.*` (6), `quality.*` (2). Crear esquema sin lector
  ni escritor es deuda, no progreso.

### BDD-R6-03 · Identidades canónicas resuelven alias sin ambigüedad · **nivel: unit**

```gherkin
Escenario: dos símbolos de proveedor apuntan a un instrumento
  Dado dos provider_symbol distintos del mismo instrumento
  Cuando resuelvo cada uno
  Entonces obtengo el mismo instrument_id
  Y un símbolo desconocido falla cerrado
```

- **Invariante**: `data-governance.md` invariante 5 (multi-par por columna
  `symbol`); SSOT de identidad.
- **Estado del test**: `FALTA`.

### BDD-R6-04 · Cuarentena: una anomalía no entra al entrenamiento · **nivel: integración**

```gherkin
Escenario: una barra anómala queda fuera del set de entrenamiento
  Dado una barra marcada en cuarentena
  Cuando construyo el dataset de entrenamiento
  Entonces esa barra no aparece
  Y la exclusión queda registrada con su motivo
```

- **Invariante**: `data-governance.md` invariante 2 (“barra en día no-sesión =
  ERROR duro”); `data-freshness.md` invariante 2.
- **Estado del test**: `INSUF: tests/regression/test_data_quality_floor.py`.

### BDD-R6-05 · TimescaleDB ops no cambia semántica · **nivel: integración**

```gherkin
Escenario: comprimir o retener no altera los valores leídos
  Dado una política de compresión/retención aplicada
  Cuando consulto un rango histórico
  Entonces los valores son idénticos a los previos
```

- **Invariante**: `data-governance.md` DO NOT (“no borrar los 9 MASTER”).
- **Estado del test**: `FALTA`.

---

## 8. R7 · OBSERVABILIDAD Y HECHOS

| BL | Dueño | Estado real hoy | Escenarios |
|---|---|---|---|
| BL-18 | CODEX | `IMPL-X-UNTRACKED` (`src/metrics/`, `config/metrics/catalog.yaml`, `070_*.sql`) | R7-01..R7-02 |
| BL-22 | CODEX | `IMPL-X-UNTRACKED` (`075_fact_position_pnl.sql`) | R7-03 |
| BL-23 | CODEX | **`AUSENTE`** — no hay productor de `held_out` | R7-05 |
| BL-25 | CODEX/CLAUDE | `IMPL@254ce8f` + DAG `control_system_health.py` | R7-04 |

### BDD-R7-01 · Motor único de métricas, sin bypass · **nivel: integración** · GRAVE

```gherkin
Escenario: toda métrica persistida pasó por el catálogo
  Dado cualquier escritor de control.metric_event
  Cuando inserta una métrica
  Entonces su metric id existe en el catálogo
  Y el catálogo rechaza ids desconocidos
```

- **Invariante**: SSOT del canal §4; `strategy-engines.md` invariante 1 (un solo
  motor).
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D4) — CODEX se contradice consigo mismo**:
  `scripts/data/backfill_catalog_facts.py:46-53,105-115` inserta en
  `control.metric_event` **sin pasar por `MetricEngine` ni por el catálogo**, con
  `metric_name ∈ {sharpe, sortino, calmar, max_drawdown_pct, total_return_pct,
  n_trades, p_value, dsr}`. **Ninguno de esos ids está en
  `config/metrics/catalog.yaml`** (que define exactamente 5: `strategy.sharpe`,
  `strategy.calmar`, `strategy.max_drawdown`, `strategy.timing_ratio`,
  `research.dsr`) y `MetricCatalog.get()` los rechazaría
  (`src/metrics/engine.py:119-123`). El “motor único” tiene un bypass en el camino
  de producción. Además `MetricEngine` **no tiene ni un solo caller**, ni en tests.

### BDD-R7-02 · La MV del Passport compila contra la tabla real · **nivel: integración** · BLOQUEANTE

```gherkin
Escenario: la DDL publicada del Passport se puede crear
  Dado el esquema real de control.metric_event
  Cuando ejecuto la DDL de v_strategy_passport_live
  Entonces la vista se crea sin error de columna inexistente
```

- **Invariante**: canal §4 “ningún 'verificado' que no se ejecutó”.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D1/D2/D3)**: la DDL de referencia hace
  `FILTER (WHERE m.metric_id = 'return_pct')`
  (`.claude/specs/platform/passport-control-tower.md:90`) y `control.metric_event`
  **no tiene `metric_id`**: tiene `metric_namespace` + `metric_name`
  (`070:145-146`). Además `env`→`environment` (`070:144`), `as_of`→`event_time`
  (`070:136`), `metric_engine_version` y `n_trades` **no existen como columnas**.
  Y los tres ids que la MV pide (`return_pct`, `max_dd_pct`, `dsr_family`) **no
  están en el catálogo**. La MV publicada **no compila**.

### BDD-R7-03 · `timing_ratio` publicado tiene de dónde leerse · **nivel: integración**

```gherkin
Escenario: o hay fuente publicada, o el campo es unavailable
  Dado el Passport de una sleeve
  Cuando pide timing_ratio
  Entonces o lo lee de un artefacto publicado
  O lo declara unavailable nombrando a su deudor
  Y NUNCA lo deriva de pnl_timing como si fuera un ratio
```

- **Invariante**: honestidad (`quant-constitution.md` §7); fail-closed.
- **Estado del test**: `INSUF: tests/unit/test_passport_contract.py` (el Passport
  ya lo declara `unavailable` — correcto).
- **DIVERGENCIA VERIFICADA (D12)**: `fact.pnl` solo tiene
  `pnl_component IN (...,'pnl_timing',...)` con `amount NUMERIC` (`075:31-37`) —
  `pnl_timing` es un **monto**, no un ratio. El ratio existe únicamente como
  fórmula no persistida (`config/metrics/catalog.yaml:34`,
  `src/metrics/engine.py:185-197`). El Passport lo lee en 4 sitios
  (`compose.ts:337,360,388,659`). **No hay de dónde.**

### BDD-R7-04 · Los tres relojes se llaman igual en productor y consumidor · **nivel: contrato** · GRAVE

```gherkin
Escenario: ningún reloj publicado se descarta por el nombre
  Dado que system_health publica sus relojes
  Cuando el Passport los consume
  Entonces consume TODOS los publicados
  Y ninguna nota al usuario afirma que se publican menos de los que hay
```

- **Invariante**: contract-first + espejo (canal §4); honestidad.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D14) — la única que YA está perdiendo dato hoy**:
  el productor define `Clock.DATA/MODEL/**PNL**`
  (`src/monitoring/system_health_contract.py:52-57`); el consumidor itera
  `HEALTH_CLOCKS = ['data','model','**exec**']` (`src/contracts/passport.py:68`,
  `passport.contract.ts:55`) y busca `health.clocks['exec']`
  (`compose.ts:752-753`). **Descarta silenciosamente el reloj `pnl` que sí se
  publica.** Y la nota que se renderiza al usuario —*"system_health publica data y
  model"* (`compose.ts:757`)— **ya es falsa**: publica tres.

### BDD-R7-05 · `held_out` es almacenable · **nivel: contrato** · BLOQUEANTE

```gherkin
Escenario: el vocabulario de entorno es único en todo el sistema
  Dado el conjunto de entornos que el Passport declara
  Cuando persisto un hecho de cada uno
  Entonces la BD los acepta todos
```

- **Invariante**: contract-first + espejo; SSOT del vocabulario.
- **Estado del test**: `FALTA`.
- **DIVERGENCIA VERIFICADA (D11) — la más estructural de las tres**:
  `PASSPORT_ENVS = ("backtest","held_out","paper","canary","live")`
  (`src/contracts/passport.py:58`, espejado en TS) contra el CHECK de CODEX
  `env IN ('replay','paper','canary','live')` en **cuatro** sitios —
  `exec.order_header` (`074:11`), `fact.position` (`075:10`), `fact.pnl`
  (`075:30`), `control.artifact_identity` (`070:83`). Es decir: `backtest` se
  llama `replay`, y **`held_out` es literalmente inalmacenable**. La columna
  `held_out` del Passport no puede llenarse **ni cuando BL-23 exista**, sin
  cambiar un CHECK. Y BL-23 no tiene productor: lo más cercano,
  `scripts/data/backfill_catalog_facts.py:114`, inserta con `'replay'`
  **hardcodeado**.

---

## 9. Resumen

| Métrica | Valor |
|---|---|
| Escenarios BDD | **62** (R1 16 · R2 16 · R3 8 · R4 7 · R5 7 · R6 5 · R7 5, más 2 sub-escenarios contados en su padre) |
| BLOQUEANTES | 14 |
| Escenarios con test que EXISTE y muerde | 12 |
| Escenarios `INSUF` | 16 |
| Escenarios `FALTA` | 33 |
| Escenarios cuyo test existe pero **ningún workflow lo corre** (`FALTA-CI`) | 11 |
| BLs con estado real `DONE` | 2/47 (BL-06, BL-07) |
| BLs de CODEX **sin commitear** | ~20 (todo `src/{identity,metrics,lineage,portfolio,market,governance,orchestration}` y las 9 migraciones) |

**Lo que esta matriz NO cubre y debe declararse**: BL-08 (decisión del operador),
BL-42 (`NO VERIFICADO` en esta sesión), BL-29/BL-30/BL-33/BL-44 tienen un solo
escenario cada uno porque su código está untracked y no pude leerlo completo sin
riesgo de reportar un árbol en movimiento.

---

## FIRMAS

- `claude-root-9c3f1e42` · 2026-07-28 · **PROPUESTO**. Toda evidencia de línea es
  lectura directa o ejecución de esta sesión; lo que no verifiqué está marcado
  `NO VERIFICADO`. No arranqué Docker, no commiteé, no tomé ninguna decisión de
  modelado (las que aparecieron van marcadas DECISIÓN DEL OPERADOR).
- `codex-root-880ff498` · _pendiente de cofirma / objeción_
