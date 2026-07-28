---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/portfolio/allocator.py
  - src/portfolio/snapshot.py
  - src/portfolio/target.py
  - src/metrics/engine.py
  - src/metrics/formulas.py
  - src/governance/declaration.py
  - config/book/allocator_v1.yaml
  - config/metrics/catalog.yaml
  - src/identity/canonical.py
  - src/execution/service.py
  - tests/unit/test_codex_fabric_contracts.py
  - tests/unit/test_codex_safety_contracts.py
---

# AUDIT — Claude (adversarial) sobre CODEX IA-R001 · portfolio / metrics / governance

- **Auditor**: Claude, revisor independiente, rol adversarial (intentar refutar).
- **Objeto sellado**: `b18720d10a2d84c5217c3800fe421ea23607f921` (CXD-061). **Toda** la
  evidencia se ejecutó sobre un `git archive` de ESE SHA extraído a scratchpad, nunca
  sobre el working tree.
- **Alcance**: `src/portfolio/{allocator,snapshot,target}.py`, `config/book/allocator_v1.yaml`,
  `src/metrics/{engine,formulas}.py`, `config/metrics/catalog.yaml`, `src/governance/declaration.py`.
- **Método**: lectura estática + **4 sondas ejecutadas** (`p1_allocator.py`, `p2_identity.py`,
  `p3*_metrics.py`, `p4_gov_and_ids.py`) en scratchpad. Cero Docker, cero DB, cero servicios.
  Cero ediciones al código auditado, cero commits.
- **Veredicto global**: **RECHAZADO para sellar como VERIFIED** — 3 BLOQUEANTES, 8 GRAVES,
  5 MEDIOS, 6 informativos. El diseño es serio y varias de mis objeciones previas están
  genuinamente cerradas; pero tres de las cuatro afirmaciones de CXD-061 que pediste que
  atacara no se sostienen tal como están redactadas.

---

## 0. Contexto honesto (leer antes de los hallazgos)

**0.1 — Radio de explosión hoy.** De los módulos auditados, **solo `PortfolioTarget` está
cableado a código de aplicación** (`src/execution/service.py:17,175,353`). `AllocatorV1`,
`SnapshotBuilder` y `MetricEngine` no los importa ningún DAG, servicio ni script en el SHA
sellado — únicamente los tests. Por tanto **ningún hallazgo de abajo mueve capital hoy**; son
defectos de contrato, y los contratos son precisamente lo que se está sellando.

**0.2 — Lo que SÍ está bien hecho** (verificado, no concedido por cortesía):

| Afirmación de CODEX | Veredicto | Evidencia |
|---|---|---|
| "target recomputa hash+UUID" | **CONFIRMADO** | B1: constructor directo rechaza hash forjado y UUID forjado |
| "validación independiente" del solver | **CONFIRMADO, no es tautología K-041** | `_validated_solution` re-deriva **cada** restricción desde las primitivas del `OptimizationRequest` con numpy/`math.fsum`; no consulta residuales ni status del solver. A5: NaN, inf, negativo, clave faltante y violación de cap → los 5 rechazados |
| `prohibited: normalize_after_clip` | **CONFIRMADO** | A6: tras clipear a `asset_cap=0.7` la suma queda en 0.70 y **no** se re-normaliza a `gross_cap=1.0` |
| governance "96 → 26" | **CONFIRMADO exacto** | D1: 96 combinaciones enumeradas → 26 aceptadas / 70 rechazadas, y `__post_init__` valida (ya no hay puerta trasera por constructor directo) |
| Bar DSR = 0.95 (antes 0.90) | **CORREGIDO** | `config/metrics/catalog.yaml:59` `critical: 0.95` |
| "Sharpe 8.14 con N=5" | **CORREGIDO** | C1: serie N=5 con Sharpe crudo 49.99 → `value=None status=INSUFFICIENT_SAMPLE` |
| "Sharpe 4.087e16 sobre varianza cero" | **CORREGIDO** | C1: serie constante → `value=None status=N_A`; DSR sobre serie constante → bloqueado |
| Guardas fail-closed del engine | **CONFIRMADO** | C11: `as_of` naive, `window_end != as_of`, etiqueta `26w` vs rango 52w, métrica no catalogada, entorno inválido, ventana no catalogada, `returns` vacío, `n_trades` no-int, activo sin anualización → **8/8 bloqueados** |
| Barrera de cutoff del snapshot | **CONFIRMADO** | B6: señal futura → excluida y registrada como `missing` con payload FLAT; `USE_LAST_VALID` sin último válido → bloqueado |
| Covarianza | **CONFIRMADO** | validada finita/simétrica/PSD **antes** de llamar al solver (`allocator.py:573-592`) |

---

## 1. BLOQUEANTES

### P-01 · `PortfolioSnapshot` sigue teniendo la puerta trasera que ya cerraste dos veces
**Severidad**: BLOQUEANTE · **Fichero**: `src/portfolio/snapshot.py:134-146`

**Qué está mal.** `PortfolioTarget` (target.py:116) y `GovernanceDeclaration`
(declaration.py:98) recibieron `__post_init__`. `PortfolioSnapshot` **no tiene ninguno**: es un
`@dataclass(frozen=True, slots=True)` puro. Todas sus invariantes (hash canónico,
`snapshot_id = uuid5(hash)`, `cutoff_time` tz-aware, coherencia
`required_sleeves`↔`materialized_inputs`) viven **solo** en `SnapshotBuilder.build`. El
snapshot es el ancla de causalidad de todo el libro; `PortfolioTarget.snapshot_id` no es más
que la cadena que alguien le pase.

**Escenario de fallo.** Un adaptador de repositorio que rehidrata un snapshot desde la DB (o
un test, o un fixture, o un `dataclasses.replace`) construye `PortfolioSnapshot(...)`
directamente. Nada comprueba que el `semantic_hash` corresponda al contenido ni que el
`snapshot_id` derive de él. Un snapshot con cutoff **naive** —es decir, sin barrera temporal
verificable— entra al libro sin un solo error.

**Evidencia ejecutada** (`p2_identity.py`, B3):
```
has __post_init__ ? False
forged snapshot_id    : whatever-i-want
forged semantic_hash  : not-even-a-sha256
naive cutoff accepted : datetime.datetime(2026, 1, 5, 0, 0)  hash=''
```

**Remedio.** Portar a `PortfolioSnapshot` el mismo `__post_init__` de `PortfolioTarget`:
recomputar `semantic_hash` sobre el payload canónico, exigir
`snapshot_id == uuid5(NAMESPACE_URL, hash)`, `cutoff_time` tz-aware, y
`set(materialized_inputs) == set(required_sleeves)`. Y hacer que `PortfolioTarget.snapshot_id`
sea un UUID validado, no texto libre (ver P-16).

**Test/mutación que lo cierra.** Rojo: `PortfolioSnapshot(**asdict_del_builder, semantic_hash="sha256:"+"0"*64)`
debe lanzar `SnapshotError`. Mutación: borrar una línea del `__post_init__` nuevo y comprobar
que el rojo vuelve.

---

### P-02 · `trials_sharpe_std = 0` desactiva silenciosamente la deflación por trials
**Severidad**: BLOQUEANTE · **Ficheros**: `src/metrics/engine.py:227-234`,
`src/metrics/formulas.py:103-113`, `config/metrics/catalog.yaml:52-63`

**Qué está mal.** `_dsr` acepta `trials_sharpe_std >= 0`. `expected_max_sharpe_null` devuelve
`0.0` en cuanto `trials_sharpe_std <= 0`, así que el benchmark E[max SR] colapsa y el DSR se
convierte en un PSR sin deflactar **con cualquier número de trials declarado**. El catálogo no
declara `plausible_min` ni cota inferior para ese campo, y nada lo ata al `trial_ledger` que
la propia definición declara como `source`.

**Escenario de fallo.** Un job que publica el gate de graduación calcula `trials_sharpe_std`
sobre una familia con **un solo** trial registrado → dispersión muestral 0 → lo pasa tal cual.
El gate constitucional (`quant-constitution.md` §2, bar 0.95) se vuelve un adorno.

**Evidencia ejecutada** (`p3c_metrics.py`, C5c — misma serie de 90 semanas, mismo `n_trials=42`):
```
n_trials=42  trials_sharpe_std=0.5  -> DSR=0.000165  status=CRITICAL  REJECT
n_trials=42  trials_sharpe_std=0.0  -> DSR=1.000000  status=OK        PASS (> 0.95)
```
y la insensibilidad total a `n_trials` cuando la dispersión es 0 (C5b):
```
n_trials=     1  trials_sharpe_std=0.5    -> DSR=0.947113
n_trials=   500  trials_sharpe_std=0.5    -> DSR=0.000000
n_trials=   500  trials_sharpe_std=0.0    -> DSR=0.947113
n_trials= 10000  trials_sharpe_std=0.0    -> DSR=0.947113
n_trials= 10000  trials_sharpe_std=1e-12  -> DSR=0.947113
```

**Remedio.** (a) Exigir `trials_sharpe_std > 0` en `_dsr` (fail-closed, `MetricContractError`),
con un piso declarado en el catálogo. (b) Cuando `n_trials > 1` y la dispersión no sea
estimable, usar un prior conservador declarado en el catálogo, **nunca** 0. (c) Registrar
`n_trials` y `trials_sharpe_std` en `dimensions` para que el número sea auditable a posteriori.

**Test/mutación.** Rojo paramétrico: `trials_sharpe_std in {0.0, -0.0, 1e-18}` ⇒
`MetricContractError`. Segundo rojo: DSR debe ser **monótono decreciente** en `n_trials` para
`trials_sharpe_std` fijo > 0 (hoy es constante cuando vale 0).

---

### P-03 · `n_trades` nunca se contrasta con la muestra que dice gobernar
**Severidad**: BLOQUEANTE · **Fichero**: `src/metrics/engine.py:334-342`

**Qué está mal.** El gate `min_trades` —el que corrige "Sharpe 8.14 con N=5"— compara
`definition.min_trades` contra `context["n_trades"]`, un entero que **suministra el llamante**
y que jamás se coteja con `len(returns)`, `len(equity)` ni con la ventana declarada. El gate
es de honor.

**Escenario de fallo.** Cualquier exportador que rellene `n_trades` desde un contador distinto
al de la serie (o desde un `summary.json` viejo, o con un off-by-one) republica exactamente el
defecto que se acaba de corregir, **con status OK**.

**Evidencia ejecutada** (`p3b_metrics.py`, C2b):
```
N=4 weekly returns [0.02, 0.005, 0.03, -0.01]
raw annualised Sharpe = 4.6357  (dentro de la banda plausible +/-5)
  n_trades=4  -> value=None    status=INSUFFICIENT_SAMPLE
  n_trades=25 -> value=4.6357  status=OK
```
Un solo entero convierte "muestra insuficiente" en un Sharpe publicable calculado con cuatro
observaciones. (Con la serie N=5 del defecto original, C2, sale `value=49.99 status=CRITICAL`:
la banda de plausibilidad lo marca, pero el número se emite igual.)

**Remedio.** Añadir al catálogo `min_obs` por métrica y validar en el engine
`len(serie_reducida) >= min_obs`, **independiente** de `n_trades`; y exigir
`n_trades <= len(returns)` cuando ambos existan (un trade no puede ocupar menos de una barra).
`n_trades` puede seguir siendo un dimension informativo, pero no puede ser el único gate.

**Test/mutación.** Rojo: `returns` de longitud 4 con `n_trades=25` ⇒ `INSUFFICIENT_SAMPLE`
(hoy `OK`). Mutación: subir `min_obs` a `len(returns)+1` y comprobar que el verde cae.

---

## 2. GRAVES

### P-04 · Los "cuatro fallbacks" son dos: los niveles 2 y 3 comparten región factible con el 1
**Severidad**: GRAVE · **Fichero**: `src/portfolio/allocator.py:428-527` ·
**Config**: `config/book/allocator_v1.yaml:15-19`

**Qué está mal.** `request_for(candidate, allowed_turnover)` coloca `candidate` **solo** en
`provisional`, y `CvxpyBudgetOptimizer` usa `provisional` **únicamente en el objetivo**
(`cp.sum_squares(budgets - provisional)`, línea 116) — nunca en una restricción. Los intentos
2..7 (fallback 1, los cuatro shrinkages del fallback 2, y el fallback 3 inverse-vol) pasan
**todos** `relaxation_limit`. Por tanto los siete intentos usan **dos** conjuntos factibles y
los seis últimos son **el mismo**. La factibilidad es propiedad de las restricciones, no del
objetivo: si el fallback 1 es infactible, los fallbacks 2 y 3 son matemáticamente infactibles
también. Solo pueden "rescatar" por ruido numérico del solver (un punto OPTIMAL a tolerancia
del solver que caiga fuera de la tolerancia 1e-7 del validador).

**Escenario de fallo.** El operador lee `allocator_v1.yaml` y cree que tiene cuatro grados de
degradación graciosa; en la práctica el libro salta de "turnover relajado" a "target cero
CRITICAL" sin escalones intermedios, y el incidente `ALLOCATOR_FALLBACK_2_SHRINK_TO_ZERO` es
código muerto.

**Evidencia ejecutada** (`p1_allocator.py`, A1 — se firman los 12 campos que definen la región
factible de cada `OptimizationRequest`):
```
attempts issued      : 7
distinct feasible sets among attempts 2..7 : 1  (1 == es el mismo conjunto)
objective vectors differ : 5 distinct provisionals
final fallback_level : 4
```
Y contra un **solver fiel** (implementación exacta del programa declarado, no un mock), A2:
```
solver calls   : 7
fallback_level : 4
incidents      : ['ALLOCATOR_PRIMARY_INFEASIBLE', 'ALLOCATOR_FALLBACK_1_INFEASIBLE',
                  'ALLOCATOR_FALLBACK_2_INFEASIBLE', 'ALLOCATOR_FALLBACK_4_TARGET_ZERO']
```
Los tests actuales no pueden detectarlo porque `_ScriptedBudgetOptimizer`
(`tests/unit/test_codex_fabric_contracts.py:367-381`) decide la factibilidad por guion: prueban
la fontanería, no la economía.

**Remedio.** O bien los fallbacks 2 y 3 **relajan una restricción distinta** (p.ej. fallback 2
= reducir `target_vol` objetivo; fallback 3 = soltar bounds de factor no vinculantes), o se
documentan honestamente como "reintentos de objetivo ante fallo numérico del solver" y la
lista de `fallbacks:` del YAML se corrige a dos escalones económicos.

**Test/mutación.** Rojo: con un solver fiel y un conjunto factible vacío, aserta
`fallback_level == 4` y `ALLOCATOR_FALLBACK_2_SHRINK_TO_ZERO not in [i.code for i in incidents]`
— y además un test que construya un caso donde el nivel 2 SÍ deba dispararse; si no se puede
construir, el nivel no existe.

---

### P-05 · El fallback 4 (target-cero) se salta `_validated_solution` y viola lo que dice respetar
**Severidad**: GRAVE · **Fichero**: `src/portfolio/allocator.py:519-527`

**Qué está mal.** Los niveles 0-3 pasan por `_validated_solution`. El nivel 4 construye
`zero = {sleeve: 0.0}` y lo devuelve por `_result` **sin validar nada**. El vector cero
satisface caps y vol, pero **no** satisface el presupuesto de turnover cuando había posición
previa, ni los **límites inferiores** de exposición a factor.

**Escenario de fallo.** Libro con `previous_budgets = 0.30`, `turnover_relaxation_limit = 0.10`
y un factor con banda `(0.6, 1.0)`. El allocator emite un target que exige liquidar 0.30 de
riesgo en un ciclo —3× el techo de turnover pre-registrado— y sitúa la exposición al factor en
0.0, fuera de su banda inferior. El incidente dice "all registered feasible-target attempts
failed"; **no dice** que el propio remedio incumple dos restricciones. Ir a plano puede ser la
decisión correcta, pero debe ser una **excepción declarada y registrada**, no un silencio.

**Evidencia ejecutada** (`p1_allocator.py`, A3):
```
risk_budgets returned               : {'a1': 0.0}
declared turnover_relaxation_limit  : 0.10
turnover actually implied by result : 0.3
turnover budget respected?          : False
factor 'beta' bound                 : (0.6, 1.0)  -> exposure 0.0 inside bound? False
```

**Remedio.** Emitir en el nivel 4 un incidente adicional explícito por cada restricción que el
target cero incumple (`ALLOCATOR_FALLBACK_4_TURNOVER_OVERRIDE`,
`ALLOCATOR_FALLBACK_4_FACTOR_BOUND_OVERRIDE`) y propagarlo al `PortfolioTarget`. Alternativa
más fuerte: liquidación escalonada dentro del turnover, con el residual como incidente abierto.

**Test/mutación.** Rojo: el caso de A3 debe producir incidentes de override; hoy la lista de
incidentes no menciona ni turnover ni factor.

---

### P-06 · Un sleeve `side == 0` (FLAT) puede recibir presupuesto de riesgo positivo
**Severidad**: GRAVE · **Ficheros**: `src/portfolio/allocator.py:86-118` (falta restricción),
`allocator.py:644-693` (`_validated_solution` nunca mira `side`)

**Qué está mal.** `allocate()` pone 0.0 a los sleeves con `side==0`, pero el programa convexo
solo impone `budgets >= 0`: nada fija a cero los sleeves planos. Con penalización L1 de
turnover, el óptimo de `min (b-0)² + λ·|b - prev|` para un sleeve plano con posición previa es
`b = λ/2 > 0`. `_validated_solution` **no valida `side` en absoluto**, así que ese presupuesto
positivo pasa la "validación independiente" y consume gross cap, presupuesto de volatilidad y
asset cap sin producir exposición.

**Escenario de fallo.** Un sleeve entra en FLAT por el gate de régimen tras haber tenido 0.50
de riesgo. El allocator le reserva presupuesto, y el sleeve que sí tiene señal recibe **menos**
riesgo del que le corresponde. Además `sum(risk_budgets) != sum(|signed_weights|)`, así que
cualquier reconciliación entre presupuesto y exposición no cuadra.

**Evidencia ejecutada** (`p1_allocator.py`, A4; `previous_budgets={'b1': 0.5}`, `λ=0.05`):
```
provisional (pre-solver) para el FLAT b1 : {'a1': 1.0, 'b1': 0.0}
risk_budgets tras el solve               : {'a1': 0.975, 'b1': 0.025}
signed_weights tras el solve             : {'a1': 0.975, 'b1': 0.0}
fallback_level                           : 0
gross consumido por risk_budgets         : 1.0000 de gross_cap 1.0
gross realmente expuesto (|signed|)      : 0.9750
```
`b1 = 0.025 = λ/2`, exactamente el óptimo predicho.

**Remedio.** (a) Añadir `budgets[i] == 0` para todo sleeve con `side == 0` en
`CvxpyBudgetOptimizer`. (b) Añadir a `_validated_solution` la comprobación
`side[s] == 0 ⇒ budget[s] == 0` (la política, no el solver, es quien debe decidirlo) —
requiere pasar `side` dentro del `OptimizationRequest`.

**Test/mutación.** Rojo: el caso A4 debe dar `risk_budgets['b1'] == 0.0`. Mutación: quitar la
restricción del solver y comprobar que la validación independiente lo caza igual (defensa en
profundidad).

---

### P-07 · `strategy.calmar` no tiene `min_trades` — y Calmar es la métrica **primaria** de graduación
**Severidad**: GRAVE · **Fichero**: `config/metrics/catalog.yaml:16-26`

**Qué está mal.** `quant-constitution.md` §2 dice literal: "Métrica primaria de graduación:
**Calmar** (y Sortino). Sharpe es secundario". Y §6 / `strategy-contract.md` §6: con N<20 solo
conteo y PnL. `strategy.sharpe` y `strategy.timing_ratio` tienen `min_trades: 20`;
`strategy.calmar` **no tiene ninguno** y ni siquiera pide `n_trades` en `required_inputs`. El
gate se puso en la métrica secundaria y se olvidó en la primaria.

**Escenario de fallo.** El comité de promoción mira la fila Calmar. La misma muestra de 10
semanas que el engine **suprime** para Sharpe se publica como **Calmar 19.86 con status OK**.
Peor: la anualización `equity[-1]**(52/n)` amplifica muestras cortas, así que un Calmar de
seis cifras es trivial.

**Evidencia ejecutada** (`p3b_metrics.py`, C3b y `p3_metrics.py`, C3):
```
catalog min_trades for strategy.calmar : None
returns (10 obs semanales) [0.005 x9, -0.010]
  -> Calmar=19.8601  status=OK  n_trades=None
  misma serie, sharpe -> value=None status=INSUFFICIENT_SAMPLE

returns [0.40, -0.02]            -> Calmar=186234.577  status=CRITICAL
returns [0.30, -0.01, 0.25]      -> Calmar=379332.947  status=CRITICAL
```
La banda `plausible_max: 20` atrapa los casos absurdos (justo es decirlo), pero (i) devuelve
`CRITICAL` —"estrategia mala"— y no `INSUFFICIENT_SAMPLE` —"sin evidencia"—, que son cosas
distintas para quien lee, y (ii) el valor 186 234.58 **se emite igual** en `metric_value`.

**Remedio.** `min_trades: 20` + `min_obs` en `strategy.calmar` y `strategy.max_drawdown`,
y `n_trades` en sus `required_inputs`. Además: cuando el status sea `INSUFFICIENT_SAMPLE` o
`CRITICAL` por plausibilidad, `metric_value` debería ir a `null` (hoy se publica el número).

**Test/mutación.** Rojo: Calmar con 10 observaciones ⇒ `status == "INSUFFICIENT_SAMPLE"` y
`metric_value is None`.

---

### P-08 · `strategy.max_drawdown` publica "0.0, OK" a partir de UN punto de equity
**Severidad**: GRAVE · **Ficheros**: `src/metrics/formulas.py:23-27`,
`config/metrics/catalog.yaml:28-39`

**Qué está mal.** `max_drawdown` devuelve `0.0` si `len(equity) < 2`, `_finite_array` acepta
arrays de tamaño 1, y el catálogo no impone `min_trades`. Resultado: `DD = 0.00 %`, status
`OK` (por debajo de warning 0.10 y critical 0.20) fabricado de una sola observación.

**Escenario de fallo.** Exactamente el patrón que la constitución §6 nombra
("DD < 1% ⇒ look-ahead o costos ignorados hasta demostrar lo contrario"), pero generado por el
propio engine de gobierno, que es el que debería impedirlo. Un panel de riesgo mostraría verde.

**Evidencia ejecutada** (`p3_metrics.py`, C4):
```
catalog min_trades : None
equity=[1.0]       -> value=0.0 status=OK
equity=[1.0,1.02]  -> value=0.0 status=OK
```

**Remedio.** `max_drawdown` debe devolver `None` con `len < 2` (como hacen `sharpe_ratio` y
`calmar_ratio`), y el catálogo debe declarar `min_obs` para la serie de equity.

**Test/mutación.** Rojo: `equity=[1.0]` ⇒ `metric_value is None` y status ≠ `OK`.

---

### P-09 · `metric_event_id` colisiona entre activos y relojes de anualización
**Severidad**: GRAVE · **Fichero**: `src/metrics/engine.py:368-383`

**Qué está mal.** El payload de identidad del evento incluye catálogo, fórmula, entidad,
métrica, ventana, entorno, `as_of`, `run_id` y `context_hash`. **No incluye `asset_id`, ni
`strategy_id`, ni la anualización resuelta.** Como la anualización es per-activo
(`from_asset_registry`), dos métricas con el **mismo contexto** pero distinto reloj producen
**valores distintos con el mismo UUID**.

**Escenario de fallo.** `strategy-contract.md` §5: "métricas anualizadas por activo; nunca
compararlos en una misma tabla de ranking". Aquí no solo se pueden comparar: **se fusionan**.
Un `INSERT ... ON CONFLICT (metric_event_id) DO UPDATE` conserva el que llegue último y borra
el otro; un dedup por id descarta silenciosamente una de las dos filas.

**Evidencia ejecutada** (`p4_gov_and_ids.py`, D2 — mismo contexto, `asset_id` distinto):
```
usdcop  (reloj 52)  value=5.6588   id=2fdc996d-7a6f-51e4-887a-4fdd1a7d27d0
btcusdt (reloj 365) value=14.9924  id=2fdc996d-7a6f-51e4-887a-4fdd1a7d27d0
same metric_event_id for different assets/values? True
```
Y en C7c, el mismo `asset_id` con dos registries de anualización distintos también colisiona:
`'annualization' recorded? False`, `same metric_event_id for the two clocks? True`.

**Remedio.** Incluir `asset_id`, `strategy_id` y la **anualización efectiva** en
`event_identity`, y además emitir la anualización en `dimensions` para que el reloj de cada
número sea auditable a posteriori (hoy es invisible en el evento).

**Test/mutación.** Rojo: dos `compute` idénticos salvo `asset_id` ⇒ `metric_event_id`
distintos. Segundo rojo: `"annualization" in event.dimensions`.

---

### P-10 · `lineage` (y `dimensions`) escapan a la prohibición de NaN/Infinity
**Severidad**: GRAVE · **Fichero**: `src/metrics/engine.py:400-407`

**Qué está mal.** `metric_value` está protegido por `math.isfinite` (engine.py:343) y el
`context` está protegido indirectamente porque se canonicaliza para el `context_hash`
(bien: un NaN en context revienta). Pero `lineage` se copia crudo al evento
(`lineage=dict(lineage or {})`) sin canonicalizar ni validar, y `to_record()` lo devuelve tal
cual.

**Escenario de fallo.** `strategy-contract.md` §2 / `quant-constitution`: "Ningún JSON
exportado puede contener `Infinity`, `NaN` ni `undefined`". `json.dumps` de Python emite
`Infinity`/`NaN` **sin error** y produce JSON inválido que revienta en el parser del dashboard.

**Evidencia ejecutada** (`p3c_metrics.py`, C6c):
```
NaN en context -> bloqueado por CanonicalizationError   [bien]
lineage en el evento emitido : {'cost_multiplier': inf, 'pnl_check': nan}
json.dumps(...) -> {"cost_multiplier": Infinity, "pnl_check": NaN}
is that valid JSON? False
```

**Remedio.** Pasar `lineage` (y `dimensions`) por `canonicalize()` en el constructor del
evento, o al menos rechazar no-finitos con `MetricContractError`.

**Test/mutación.** Rojo: `lineage={"x": float("inf")}` ⇒ `MetricContractError`.

---

### P-11 · `config/book/allocator_v1.yaml` es configuración muerta: nadie la lee
**Severidad**: GRAVE (SSOT/DRY) · **Ficheros**: `config/book/allocator_v1.yaml:1-32`,
`src/portfolio/allocator.py:136-183, 260-279, 716-720`

**Qué está mal.** Ningún módulo del SHA sellado carga `config/book/allocator_v1.yaml`
(`git grep` sobre `src/ services/ scripts/ airflow/ tests/` → 0 hits). Los mismos números
viven **hardcodeados** en Python:

| Número | YAML | Python |
|---|---|---|
| diversification 0.70–1.10 | `allocator_v1.yaml:23` | `allocator.py:268-272` |
| forward/liquidity/drawdown 0.0–1.0 | `allocator_v1.yaml:21,22,25` | `allocator.py:268-272` |
| operations ∈ {0,1} | `allocator_v1.yaml:24` | `allocator.py:277-278` |
| novelty gate 0.60 / 0.15 | `allocator_v1.yaml:26-28` | `allocator.py:720` |
| `target_vol 0.10`, `turnover 0.20`, `relaxation 0.30` | `allocator_v1.yaml:11-14` | **no existen en el código**: son argumentos libres de `allocate_constrained` |

Dos fuentes para el mismo número: una miente en cuanto alguien toque una. Y el
"pre-registro" de `target_vol` / `turnover_budget` no es tal: el llamante puede pasar lo que
quiera. Además `shadow: {method: HRP, ...}` (línea 5) no tiene implementación alguna en `src/`,
y `prohibited: [normalize_after_clip, result_selected_thresholds]` (líneas 29-31) es una
declaración sin mecanismo de enforcement (el comportamiento es correcto — ver A6 — pero por
convención del autor, no por contrato verificado).

**Remedio.** `AllocatorV1.from_config(path)` que cargue el YAML como SSOT único, con los
bounds de multiplicador y el novelty gate leídos de ahí, y un test de contrato que falle si un
literal del código diverge del YAML. Marcar `shadow.HRP` como `status: not_implemented` o
retirarlo.

**Test/mutación.** Rojo: cambiar `diversification.max` a 1.05 en el YAML debe hacer fallar un
test de paridad config↔código (hoy no cambia nada).

---

## 3. MEDIOS

### P-12 · El orden canónico de exposures se impone solo en la fábrica, no en la invariante
**Severidad**: MEDIA · **Fichero**: `src/portfolio/target.py:277-286` vs `target.py:147-184`

`TargetBuilder.build` ordena por `(sleeve_id, instrument_id, allocation_id)` **antes** de
hashear; `__post_init__` **no reordena**, solo comprueba que el hash cuadre con el orden que
recibe. Un llamante que construya directamente con las exposures al revés y calcule el hash de
forma coherente obtiene un target válido con **otro `target_id`** para el mismo contenido
económico. La canonicidad se pierde justo donde importa (idempotencia / dedup / reconciliación).

**Evidencia ejecutada** (`p2_identity.py`, B2):
```
builder  target_id : 439b7d67-5524-50b1-a1f9-034ee0513794
shadow   target_id : ac1848b8-bb3a-59db-b1d5-0a4c7829c975
identical economic content? True
same target_id?    : False
```
**Remedio**: reordenar dentro de `__post_init__` antes de calcular `identity_payload`, o
rechazar exposures no ordenadas. **Rojo**: construir con orden inverso ⇒ mismo `target_id` que
el builder (o `TargetError`).

### P-13 · `USE_LAST_VALID_WITH_MAX_AGE` reproduce una señal **expirada**
**Severidad**: MEDIA · **Fichero**: `src/portfolio/snapshot.py:344-363`

Una señal se clasifica stale si `valid_until < cutoff` **o** si excede `max_age`. En el
fallback solo se re-verifica `max_age` y el cutoff; **`valid_until` se ignora**. Una señal que
declaró explícitamente su propia caducidad vuelve a ser el input vivo del allocator.

**Evidencia ejecutada** (`p2_identity.py`, B5):
```
signal valid_until      : 2026-01-04T23:59:00+00:00
cutoff                  : 2026-01-05T00:00:00+00:00
classified stale        : ('sig-1',)
resolution              : USE_LAST_VALID_WITH_MAX_AGE
materialized payload    : {'side': 1, 'target_weight': Decimal('0.4')}
materialized valid_until: 2026-01-04T23:59:00+00:00 (< cutoff: True)
```
**Remedio**: en el fallback, exigir `fallback_signal.valid_until >= cutoff`, o declarar
explícitamente en el contrato que `max_age` **anula** `valid_until` y por qué. **Rojo**: el
caso B5 debe lanzar `SnapshotError`.

### P-14 · Desajuste de contrato `AllocationResult` ↔ `PortfolioTarget`
**Severidad**: MEDIA · **Ficheros**: `src/portfolio/allocator.py:202,336`,
`src/portfolio/target.py:154-163`

Dos incoherencias, ambas ejecutadas (`p4_gov_and_ids.py`, D3):
```
libro vacío        -> fallback_level=4 incident='NO_SLEEVES_TARGET_ZERO' incidents=()
libro gateado a 0  -> fallback_level=0 incident='RISK_GATES_TARGET_ZERO'
```
(a) Nivel 4 (CRITICAL) con `incidents` vacío: cualquier adaptador que haga
`result.incidents[-1].code` para rellenar `fallback_incident_id` lanza `IndexError`. Además un
libro **vacío por configuración** no es lo mismo que una infactibilidad crítica y hoy comparten
nivel. (b) `PortfolioTarget` exige `fallback_incident_id is None` cuando
`infeasibility_fallback is None`, y el nivel 0 mapea a `None`: por tanto
`RISK_GATES_TARGET_ZERO` **no es representable** en el target canónico que consume execution —
el motivo de un libro plano se pierde.
**Remedio**: `_result` para todos los caminos (incluidos los dos returns tempranos de
`allocate`), y un campo de incidente independiente del nivel de fallback en `PortfolioTarget`.

### P-15 · El tamaño de muestra nunca se reconcilia con la ventana declarada
**Severidad**: MEDIA · **Fichero**: `src/metrics/engine.py:431-460`

`_validate_window` valida la etiqueta (`26w`, `52w`) contra `window_start`/`window_end` al día
—bien— pero nunca contra el array reducido.
**Evidencia** (`p3c_metrics.py`, C8c): `window='52w'` validado al día con **3** returns →
`Sharpe=14.4222 status=CRITICAL`. **Remedio**: para ventanas con etiqueta temporal, exigir
`len(returns)` coherente con la ventana y la anualización (± tolerancia declarada).

### P-16 · `PortfolioTarget.snapshot_id` es texto libre
**Severidad**: MEDIA · **Fichero**: `src/portfolio/target.py:117-126`

`decision_fingerprint` y `semantic_hash` se validan con regex sha256; `snapshot_id` solo debe
ser "string no vacío". **Evidencia** (`p2_identity.py`, B4): se acepta
`'snapshot-that-never-existed'`. Combinado con P-01, la cadena causal
snapshot → target → orden no está anclada por ningún lado. **Remedio**: validar formato UUID y
resolver contra el repositorio de snapshots antes de admitir el target en execution.

---

## 4. INFORMATIVOS / MENORES

- **P-17** — `src/metrics/engine.py:462-472`: `_identity_context` normaliza solo numpy de
  **primer nivel**. Una `list[np.float64]` (el resultado más común de `list(series)`) atraviesa
  el engine y revienta con `CanonicalizationError`, un tipo ajeno al contrato del módulo.
  Evidencia (C10c): `src.identity.canonical.CanonicalizationError: numpy scalars must be
  converted to Python primitives`. Es fail-closed —correcto— pero el llamante no puede
  capturarlo con `except MetricContractError`. Lo mismo aplica al NaN en context (C6c). Remedio:
  normalizar recursivamente y envolver en `MetricContractError`.
- **P-18** — `src/metrics/engine.py:236-243`: mezcla de `ddof`. El Sharpe del numerador usa
  `ddof=1` y los momentos (skew/kurtosis) del denominador usan `ddof=0`. Cosmético con n=60,
  material en el suelo de `n_obs >= 3` que el propio engine permite.
- **P-19** — `src/metrics/engine.py:199-208`: `int(annualization)` trunca. Un reloj `365.25`
  se convierte silenciosamente en 365.
- **P-20 — la evidencia "54/54 verde" de CXD-061 no es reproducible en el SHA sellado.**
  Ejecutado sobre `git archive b18720d`:
  `3 failed, 51 passed in 2.09s`. Dos fallos son
  `ModuleNotFoundError: No module named 'src.research'`
  (`test_codex_fabric_contracts.py:534,717` importan `src.research.qlab`), y **`src/research/`
  no existe en el commit** — qlab vive en `scripts/analysis/qlab.py`. El tercero
  (`test_catalog_backfill_inventory...`) depende de `basetemp` y es ambiental, no lo cuento.
  Esto no invalida el código auditado, pero sí la afirmación de evidencia: **el corte no es
  autocontenido respecto de su propia suite**.
- **P-21** — `config/book/allocator_v1.yaml:5`: `shadow: {method: HRP, ...}` declarado sin
  implementación en `src/` (0 hits para `HRP`).
- **P-22** — `cvxpy` es un extra opcional (`pyproject.toml:81-84`) **no instalado** en este
  entorno, y los únicos tests del camino restringido usan `_ScriptedBudgetOptimizer`. Por tanto
  `CvxpyBudgetOptimizer.solve` (allocator.py:68-133) tiene **cobertura ejecutada cero**: la
  construcción de restricciones CVXPY nunca se ha ejecutado en ningún CI. Recomiendo un test
  marcado `@pytest.mark.skipif(no cvxpy)` que resuelva un problema con solución analítica
  conocida y verifique el óptimo, no solo que "no lanza".

---

## 5. Respuesta directa a las cuatro preguntas de CXD-061

1. **"¿Los cuatro fallbacks producen carteras válidas?"** — Los niveles 0-3 sí (los valida
   `_validated_solution`); el **nivel 4 no** (P-05: viola turnover y bandas inferiores de
   factor). Y **no son cuatro**: los niveles 2 y 3 son inalcanzables salvo por ruido numérico
   (P-04).
2. **"¿La validación independiente es tautológica (K-041)?"** — **No.** `_validated_solution`
   re-deriva cada restricción desde las primitivas del request con numpy/`math.fsum`, sin
   consultar al solver. Los cinco ataques de solver deshonesto (NaN, inf, negativo, cap
   excedido, clave faltante) fueron **todos** rechazados. Su punto ciego no es la tautología:
   es que **no valida `side`** (P-06) y **no se aplica al camino cero** (P-05).
3. **"¿Turnover antes o después del clip de caps?"** — En `allocate_constrained` es
   **simultáneo** (restricción del programa convexo) y se re-valida después: correcto, la
   objeción no procede ahí. En `allocate()` (público, `fallback_level=0`) **no existe turnover
   en absoluto**; si alguien alimenta un target con su resultado, el presupuesto de turnover
   no se aplicó nunca.
4. **"¿El constructor directo de `target` es una puerta trasera?"** — **No**, esa está cerrada
   (B1). La puerta trasera **se mudó a `PortfolioSnapshot`** (P-01), y la canonicidad de orden
   sigue viviendo solo en la fábrica (P-12).

---

## 6. Condiciones de aceptación

**No apruebo el sello como VERIFIED.** Para levantar el RECHAZO:

| Bloqueo | Cierra con |
|---|---|
| P-01 | `__post_init__` en `PortfolioSnapshot` + rojo de hash forjado |
| P-02 | `trials_sharpe_std > 0` obligatorio + rojo de monotonía en `n_trials` |
| P-03 | `min_obs` por métrica + `n_trades <= len(returns)` + rojo N=4/n_trades=25 |

Los 8 GRAVES requieren ACK explícito con plan; P-04, P-06, P-07, P-08, P-09 y P-11 los
considero pre-condición para cablear estos módulos a cualquier DAG o servicio.

**Sondas reproducibles** (scratchpad, fuera del repo):
`p1_allocator.py` (A1-A6) · `p2_identity.py` (B1-B6) · `p3_metrics.py` / `p3b_metrics.py` /
`p3c_metrics.py` (C1-C11) · `p4_gov_and_ids.py` (D1-D3).
Todas se ejecutan con `PYTHONPATH=<archive de b18720d>` desde la raíz del archive.
