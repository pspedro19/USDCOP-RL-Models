---
kind: roadmap
status: PARTIAL
version: 1.1.1
last_verified: 2026-07-31
supersedes: []
code_anchors:
  - .claude/specs/planes/03-institutional-readiness.md
  - .claude/specs/planes/04b-readiness-matrix.md
  - tests/regression/test_readiness_matrix.py
---

# BL-33 — Institutional Readiness Matrix con evidencias

**Fuente:** plan 03 §6-§7 · **Ola:** T · **Esfuerzo:** M · **Trials:** 0

## Estado actual

**PARTIAL verificado 2026-07-31.** La
[readiness matrix](../04b-readiness-matrix.md) dejó de ser una tabla decorativa: cada control tiene
ID estable, dominio, evidencia esperada, evidencia observada enlazada, estado fail-closed, dueño y
fecha. Las primeras filas conservan los dos bloqueos que motivaron el BL: incidente histórico de
secretos y segregación de identidades.

El registro distingue explícitamente:

- control probado sólo dentro del repositorio;
- implementación o evidencia parcial;
- precondición externa que el agente no puede fabricar;
- capacidad sin evidencia.

No autoriza capital, no afirma readiness institucional y no confunde una spec con un simulacro.

## Evidencia rojo→verde

El gate nuevo `tests/regression/test_readiness_matrix.py` falló `3/3` contra el stub anterior:
faltaban las columnas auditables, los estados no tenían semántica y no existían links de evidencia.
Después de reconstruir el registro pasó `3/3`.

El corte factual adicional produjo:

- `199 passed, 3 failed, 1 skipped` en la batería amplia de contratos, safety, kill switch, ledger,
  restore y approvals; el agregado con los `3 passed` del gate propio es `202 passed, 3 failed,
  1 skipped`;
- focal exacto pre-trade/fencing: `7 passed`;
- frontmatter: `992 passed`;
- links relativos: `664 internal links resolve`;
- contrato RBAC y cobertura RBAC: verdes.

Los tres fallos amplios se preservan como evidencia adversa, no se maquillan como fallo de BL-33:

1. dos consumidores de `MetricEngine` siguen pasando `annualization_by_asset` a un constructor que
   ya no acepta ese argumento;
2. el plan `fabric-v1` ya no coincide con su digest pinneado.

La matriz los registra como `RISK-06` y `TECH-06`, ambos `PARTIAL`.

## Qué falta para cierre

1. Cross-review de CLAUDE contra un hash inmutable; working tree sólo admite review preliminar.
2. Resolver o asignar formalmente los dos gaps nuevos sin cambiar digests ni APIs por conveniencia.
3. Incorporar evidencia operativa real: simulacros, sign-off humano independiente, Vault/roles,
   RTO/RPO, reconciliación firmada y controles del Caso B cuando correspondan.
4. Mantener el registro actualizado por evidencia; una fila no sube porque exista el archivo que
   describe la intención.

Por estas brechas el BL avanza de `PLANNED` a `PARTIAL`, no a `IMPLEMENTED` ni DONE.

## Verificación

```powershell
python -m pytest tests/regression/test_readiness_matrix.py -q
python -m pytest tests/regression/test_contract_mirrors.py tests/unit/test_codex_safety_contracts.py tests/regression/test_trading_flags.py tests/unit/test_command_pattern.py tests/regression/test_trial_ledger.py tests/regression/test_bl09_bl11_bl12_governance.py tests/regression/test_restore_resyncs_sequences.py tests/regression/test_approval_store_private.py -q
node usdcop-trading-dashboard/scripts/test-rbac-contract.mjs
node usdcop-trading-dashboard/scripts/check-rbac-coverage.mjs
python -m pytest tests/regression/test_knowledge_frontmatter.py -q
python scripts/validation/check_knowledge_links.py
```

## Cross-references

- [Evaluación institucional](../03-institutional-readiness.md)
- [Readiness matrix](../04b-readiness-matrix.md)
- [BL-08 — incidente de secretos](BL-08-incidente-env-historial.md)
- [BL-41 — seguridad DB](BL-41-seguridad-db-p0.md)
