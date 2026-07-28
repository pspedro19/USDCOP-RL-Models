---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/strangler/gates.py
  - config/migration/strangler_usdcop.yaml
  - tests/regression/test_strangler_cop.py
  - airflow/dags/forecast_h5_l3_weekly_training.py
  - airflow/dags/forecast_h5_l7_multiday_executor.py
---

# BL-31 — Migración strangler de USD/COP (L7 al final)

**Fuente**: FABRIC §29 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
COP artesanal L0→L7 en producción con A/B vivo (v11/v12/v14) — intocable durante la migración; sus ledgers SON el patrón de paridad.

## Qué falta exactamente
Orden ingest→…→execute; paralelo ≥2 semanas por capa con paridad semantic_hash; sensores→Assets al migrar cada capa; L7 al FINAL (resto ≥1 mes verde + ejecución externa probada en canary de otro flujo); rollback por capa declarado.

## Impacto frontend
Ninguno (paridad invisible si sale bien).

## Dependencias
BL-28, BL-30, BL-17.

## Verificación
Tabla de paridad por capa en evidencia; ninguna capa avanza sin la anterior verde sostenida.

## Notas constitución
La cadena con dinero real migra última y con red doble.

---

## IMPLEMENTADO 2026-07-28 (parcial, `CTR-STRANGLER-COP-001` v1.0.0)

Se entrega el **plano de control de la migración**: el orden, la evidencia y los frenos.
NO se entrega el camino nuevo (es BL-28) ni el servicio de ejecución externo (es BL-30);
ambos quedan como **interfaz declarada y fail-closed**.

### Qué se construyó

| Archivo | Rol |
|---|---|
| `src/strangler/contracts.py` | Tipos: 9 capas §29 en orden, estados, `HashKind`, veredictos, `RollbackPlan` (imposible declarar uno con pérdida de datos), `SensorMigration`, los 15 criterios §30, `ExecutionReadiness` (interfaz a BL-30, todos los flags `False` por defecto). |
| `src/strangler/parity.py` | Harness de paridad + ledger JSONL **append-only** canónico. JSON → `semantic_hash` canónico (orden de claves/espacios no rompen paridad); no-JSON → solo `bytes_hash`, jamás promovido a paridad semántica. `Infinity`/`NaN` ⇒ observación `INVALID`, nunca hash. |
| `src/strangler/gates.py` | Gate por capa: predecesora MIGRATED (§29.2), racha verde ≥ `min_parallel_days` (§29.1), swap sensor→Asset declarado (§29.3), rollback declarado (§29.6); L7 añade cadena verde ≥30d, atestación BL-30 y 15/15 criterios §30 (§29.4). Devuelve **todos** los bloqueadores, no el primero. |
| `src/strangler/plan.py` | Loader estricto del YAML (clave desconocida ⇒ error; capas fuera de orden ⇒ error). |
| `config/migration/strangler_usdcop.yaml` | El plan: anclas reales por capa, artefactos, rollback, caveats honestos, cohorte A/B intocable (§29.5), los 15 criterios §30 con su BL dueño. |
| `scripts/validation/check_strangler_parity.py` | CLI `status` / `observe` / `gate` / `transition`. Exit 0 ok, 1 bloqueado, 2 error. `transition --state MIGRATED` **consulta el gate y se niega** si bloquea. |
| `tests/regression/test_strangler_cop.py` | 38 tests. |
| `.claude/evidence/strangler_cop/parity_table.{md,json}` | La tabla de paridad por capa que pide la Verificación. |

### Decisiones de diseño (todas ingeniería, 0 trials)

- **El estado vive en el ledger, no en el YAML**: un rollback es un *evento*, no una edición.
  `derive_states()` reproyecta; un `ROLLED_BACK` vuelve a bloquear la capa siguiente (test).
- **Paridad sostenida = ininterrumpida**: cualquier `MISMATCH` o `INVALID` pone la racha a 0.
  "Casi verde" no es verde (§29.2).
- **Ninguna capa acepta paridad por bytes sobre formatos externos** (§31): las nueve exigen
  `canonical_json` sobre un *parity manifest*. Comparar bytes de Parquet está rechazado por
  el propio plan y hay test que lo impide.
- **Los tres números (14d, 30d, 15 criterios) son citas de §29.1/§29.4/§30**, no parámetros
  ajustados. El contrato rechaza un plan con `min_parallel_days < 14`.

### Verificación ejecutada (2026-07-28)

```
python -m pytest tests/regression/test_strangler_cop.py -q   ->  38 passed
python scripts/validation/check_strangler_parity.py status --write-evidence  ->  exit 1 (rojo honesto)
python -m pytest tests/regression/test_knowledge_frontmatter.py -q -> 47 failed, 762 passed (BASELINE: 47 failed, 759 passed; DELTA=0)
python -m pytest tests/regression/test_strategy_manifests.py -q   -> 21 passed
python -m pytest tests/regression/test_scripts_layout.py -q       -> 20 passed
```

La tabla generada dice la verdad incómoda: **9/9 capas en `NOT_STARTED`, 0 pueden avanzar,
15/15 criterios §30 en `PENDING`, readiness de ejecución `NOT ATTESTED`**. La migración no
ha empezado y el gate lo demuestra en vez de afirmarlo.

### Pendiente de otros BL (interfaz declarada, NO implementada aquí)

- **BL-28** — `candidate_generator` es `null` en las nueve capas. Sin generador no hay segundo
  camino que comparar: el gate bloquea por construcción. Cuando BL-28 entregue, se rellena el
  campo y empiezan a registrarse observaciones.
- **BL-30** — `ExecutionReadiness` (servicio fuera de Airflow, idempotencia, pre-trade,
  kill switch con Airflow caído, reconciliación) se **consume**, no se produce. Sin atestación
  L7 está bloqueada.
- **BL-17/21/22/24/25/16/18/23/26/32/35** — dueños de los 15 criterios §30 (columna `owner`
  del YAML). El campo es enrutamiento de evidencia, no afirmación de que exista.
- **Emisor del parity manifest**: ni el camino artesanal ni el nuevo lo emiten hoy. Es parte
  del generador de BL-28 más un sidecar en el lado legacy.

### Nota de contrato

No se tocó `src/contracts/` ni `lib/contracts/`: el plano de control es backend-only y BL-31
declara impacto frontend nulo, así que no hay espejo TS que sincronizar. **Si BL-32 (Control
Tower) llega a renderizar esta tabla, el espejo pasa a ser obligatorio** y debe proponerse
como cambio de contrato.
