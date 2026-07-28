---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - scripts/analysis/generate_interpretability.py
  - tests/unit/test_interpretability_artifacts.py
  - data/interpretability/zoo/usdcop/ridge/2026-07-27/summary.json
---

# Migración de identidad de los seis artefactos de interpretabilidad

> Deuda declarada por `claude-root-9c3f1e42` en `CLD-167` y saldada aquí.
> Contexto: `CXD-058` (hallazgo de CODEX) → remedio en `39bc3e1` → `VERIFIED`
> funcional por CODEX en `CXD-060` tras re-ejecutar su propia sonda.

---

## Por qué hubo que migrar

El hallazgo `CXD-058` demostró tres huecos en el escritor de artefactos. El
segundo obligó a cambiar el cálculo de identidad: **`supersedes` estaba excluido
del hash**, así que dos cadenas de sustitución distintas colapsaban al mismo
`artifact_id` (`sha256:49c987…`). El campo que documenta qué artefacto reemplaza a
cuál no estaba ligado criptográficamente a nada.

El arreglo —incluir `supersedes` en la identidad— tiene una consecuencia
inevitable: **los seis artefactos ya publicados llevaban ese campo y su id se
había computado excluyéndolo**. Sin migrar, la verificación nueva los habría
marcado a los seis como manipulados de forma permanente, dejando el generador
inutilizable contra la evidencia real.

Es decir: la migración no fue una decisión de conveniencia, fue la única salida
que no dejaba el sistema en falso positivo eterno.

## Cómo se hizo, y por qué es auditable

Se añadió `--migrate-identity [--apply]`, **fail-closed por construcción**: solo
toca un fichero cuyo `artifact_id` almacenado coincide **exactamente** con su
identidad de contenido bajo el esquema viejo. Esa condición es una **prueba de
no-manipulación**: si alguien hubiera alterado el contenido de un artefacto antes
de la migración, su id no cuadraría bajo el esquema viejo y la migración lo
habría rechazado en vez de re-sellarlo.

Los seis cumplían la condición.

| Artefacto | SHA-256 antes | SHA-256 después |
|---|---|---|
| `rule_based/spx500/spx500_regime_gated_v1` | `d35030f0…` | `249cc208…` |
| `zoo/usdcop/bayesian_ridge` | `600ec3b3…` | `98e3a2ca…` |
| `zoo/usdcop/catboost` | `4f8dca71…` | `6a07b7f3…` |
| `zoo/usdcop/lightgbm` | `194d07a4…` | `bad8a2af…` |
| `zoo/usdcop/ridge` | `62a21861…` | `b95689bb…` |
| `zoo/usdcop/xgboost` | `d8363ceb…` | `8a464342…` |

## Lo que NO cambió — el punto que importa

- **`git diff` = UNA línea por fichero**, y esa línea es `artifact_id`.
- **`content_identity` es idéntico antes y después** en los seis, y se publica en
  el reporte de migración. Esa es la prueba de que **la ciencia no se tocó**: las
  atribuciones, los folds, los cortes por régimen y las kill-rules son byte a byte
  los mismos.
- **0 trials**: no se re-ejecutó ningún backtest, no se observó ningún número de
  performance, no se tomó ninguna decisión de modelado. Es un re-sellado de
  identidad, no una regeneración.
- La migración es **idempotente**: la segunda corrida reporta `already_current 6/6`.

## Residuo declarado

Los seis publicados conservan el `code_fingerprint` del generador **anterior al
arreglo**. La próxima regeneración real exigirá `--supersede` explícito
(verificado: falla cerrado, no corrompe). **No se regeneraron aquí a propósito**:
hacerlo habría reescrito la ciencia y no solo la identidad, que es exactamente lo
que esta migración evita.

## Verificación

- Sonda de CODEX (`integration/probes/interpretability_writer_integrity_probe.py`):
  `EXIT=1` antes → `EXIT=0` después, con `stored_payload_tamper detected:true`,
  `supersedes_bound:true` (`85d4b7a8…` ≠ `046a4a56…`) y carrera
  `["conflict","success"]`.
- `27 passed` en las suites de artefactos y schema (baseline 23, +4 tests nuevos;
  los 3 primeros estaban rojos antes del fix).
- Re-corrida real del generador contra la evidencia publicada: **falla cerrado**
  con los bytes intactos y cero ficheros de staging huérfanos.
