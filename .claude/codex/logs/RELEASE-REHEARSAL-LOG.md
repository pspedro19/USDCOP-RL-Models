---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Release / production unblock rehearsal

Fecha: 2026-07-20

Se incorporaron verificaciones fail-closed en `src/validation/production_harness.py`:

- `verify_artifact_manifest`: exige `model_version` y `dataset_hash`; valida hash SHA-256 declarado cuando existe.
- `rehearse_rollback`: simula candidato fallido y confirma que el activo vuelve al champion previo sin mutar despliegue.
- `evaluate_observability`: exige PSI, latencia p95, error-rate y uptime antes de evaluar SLO.

`src/validation/release_harness.py` aporta `MockProvider` determinista para probar aprobación,
replay idempotente y refund sin credenciales externas.

Evidencia: `python -m pytest tests/unit/test_production_harness.py -q` => 4 passed.

Esto resuelve bloqueos internos de release/rollback/telemetría. Permanecen externos y correctamente
bloqueados: datasets PIT reales, evidencia OOS auténtica y sandbox real del PSP.
