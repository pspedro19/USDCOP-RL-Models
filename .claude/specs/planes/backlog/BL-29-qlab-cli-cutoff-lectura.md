---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - scripts/analysis/qlab.py
  - src/research/point_in_time.py
  - src/research/qlab.py
  - tests/unit/test_qlab_point_in_time.py
  - .claude/rules/quant-constitution.md
---

# BL-29 — CLI qlab + cutoff impuesto por la capa de lectura

**Fuente**: FABRIC §13.5 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
La investigación corre como scripts sueltos con disciplina humana del cutoff; los trials se cobran editando registries a mano.

## Qué falta exactamente
`qlab` (family declare/screen --charge-trial/freeze/promote/close) FUERA de Airflow ('un trial no es idempotente'); capa read() con assert available_at<=cutoff cuando env=screening — el look-ahead falla el JOB, no al humano. Tabla de entornos §13.5.

## Impacto frontend
Ninguno.

## Dependencias
BL-09, BL-11, BL-19 (o vista equivalente con available_at).

## Verificación
Screening intentando leer >cutoff ⇒ excepción; retry de qlab no duplica cobro (ledger idempotente por trial_id).

## Estado real tras remediación 2026-07-28

Implementado y listo para revisión cruzada:

- `src/research/` ya forma parte del checkout limpio; el patrón genérico
  `research/` de `.gitignore` tiene una excepción acotada a sus módulos Python.
- `qlab screen` exige una fuente finita real (`jsonl/json/csv/parquet`) y llama
  a `read_point_in_time(..., environment=SCREENING)` **antes** de cobrar el
  trial o mutar la familia.
- El cutoff llega al reader y se vuelve a comprobar sobre todas las filas
  materializadas. Una fila sin `available_at` o posterior al cutoff aborta el
  job; el ledger y la familia permanecen intactos.
- Los cutoffs `YYYY-MM-DD` se normalizan al final de ese día en UTC y el
  `data_hash` se deriva del contenido cuando el llamante no entrega uno.

```bash
python -m pytest tests/unit/test_qlab_point_in_time.py \
  tests/unit/test_codex_fabric_contracts.py::test_qlab_idempotency_rejects_same_trial_id_with_changed_result \
  tests/unit/test_codex_fabric_contracts.py::test_family_projection_repairs_after_ledger_only_retry -q
# 5 passed
```

Mutación ejecutada: sustituir el `assert_available_at()` posterior a la lectura
por `return rows` produjo **2 failed / 1 passed**: el test directo y el CLI
aceptaron una fila `available_at=2025-06-01` con cutoff `2025-01-01`. Restaurado
por SHA256 exacto y suite **5 passed**.

Pendiente para cierre integral: adaptador de lectura DB/BL-19 con pushdown
server-side; hoy el camino gobernado soporta evidencia finita en fichero. El
CLI gobierna provenance/cobro/transiciones, pero no ejecuta por sí mismo el
modelo o estadístico de screening.

## Notas constitución
El control de mayor apalancamiento de todo el sistema.
