---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-07-29
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

## Estado real tras remediación R2 2026-07-29

R2 listo para revisión cruzada, todavía **PARTIAL** por el límite explícito al
final de esta sección:

- `qlab screen` lee una fuente finita real (`jsonl/json/csv/parquet`) por
  `read_point_in_time(..., environment=SCREENING)` antes de cobrar o mutar la
  familia. Tanto el lector síncrono como el asíncrono revalidan todas las filas
  materializadas.
- El identificador se valida con `^(FT|AT)-\d{4}$`, incluido el tipo FT/AT,
  antes de crear el directorio, tomar el lock, buscar un retry o escribir el
  ledger append-only.
- `bounded_select_sql` ya no acepta `additional_where`: no queda un predicado
  de texto crudo capaz de neutralizar `available_at <= :pit_cutoff`.
- El CLI fija el campo causal a `available_at`, deriva siempre el hash de los
  bytes y normaliza LF/CRLF para fuentes de texto. Ya no existen los overrides
  `--available-at-field` ni `--data-hash`.
- Cada asiento con fuente persiste y encadena `available_at_field`, `n_rows` y
  `max_available_at`; el ledger exige los tres, UTC canónico, hash SHA-256
  canónico y `max_available_at <= cutoff`.
- Una fecha `YYYY-MM-DD` termina en la zona `session.timezone` del
  `AssetProfile` declarado por la familia y se almacena normalizada en UTC. El
  CLI rechaza que `--asset` difiera del activo de la familia, evitando elegir
  otra zona horaria para ampliar el corte.

Rojo inicial de R2 contra el código anterior: **7 failed / 2 passed**. Verde
final focal:

```bash
python -m pytest tests/unit/test_qlab_point_in_time.py \
  tests/unit/test_codex_fabric_contracts.py::test_qlab_idempotency_rejects_same_trial_id_with_changed_result \
  tests/unit/test_codex_fabric_contracts.py::test_family_projection_repairs_after_ledger_only_retry -q
# 13 passed
```

Mutaciones causales ejecutadas y restauradas:

1. aceptar cualquier número de dígitos en `trial_id`: **1F / 1P**;
2. reintroducir `additional_where`: **1F**;
3. devolver el resultado async sin `assert_available_at`: **1F**;
4. hashear bytes CRLF sin normalización: **1F**;
5. omitir los tres campos auditables en el cobro: **1F**;
6. volver a fin-de-día UTC fijo: **2F**;
7. reabrir los dos overrides del parser: **1F**;
8. retirar la identidad familia↔activo: **1F**.

Restauración final por SHA-256:

- `scripts/analysis/qlab.py`:
  `4C1F00A31146352D45C4CC54C6548631882E6C7AEB7E949518033A1FAE8C07DB`
- `src/research/point_in_time.py`:
  `247F5ABAA9B0B2FB9E6619F13F837447689B020D1493B566D2E73C11923DB634`
- `src/research/qlab.py`:
  `D541AA3CC23724DAC9EB5D0A3143A968FE3B4FEA39465C47AC7E8ECEDC9DBC77`
- `tests/unit/test_qlab_point_in_time.py`:
  `5BDD48BEF5AA9BF3B95A919788BCD74408EE846B8790A9EE795F14CB9582FA0B`

Límite honesto para cierre integral: no existe todavía un adaptador DB/BL-19
que invoque el pushdown en producción. El camino gobernado actual soporta
evidencia finita en fichero y el CLI gobierna provenance, cobro y
transiciones, pero no ejecuta por sí mismo el modelo o estadístico de
screening.

## Notas constitución
El control de mayor apalancamiento de todo el sistema.
