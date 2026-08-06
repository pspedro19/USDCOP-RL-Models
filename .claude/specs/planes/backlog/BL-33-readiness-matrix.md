---
kind: roadmap
status: PARTIAL
version: 1.4.0
last_verified: 2026-08-06
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

La revisión adversarial `CLD-267` encontró que 34 filas aceptaban cualquier archivo existente como
evidencia. El R2 pinnea el conjunto de targets revisados por `Control ID`; por tanto un enlace
resoluble pero irrelevante ya no satisface correspondencia.

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

En cross-review, sustituir la evidencia de `INV-04` por `[licencia](../../../LICENSE)` mantuvo
`3 passed`: el gate sólo comprobaba presencia. Tras el R2, el gate ampliado da `5 passed`; una
sonda en memoria con la misma sustitución produce exactamente un error atribuido a `INV-04`.

El corte factual adicional produjo:

- `199 passed, 3 failed, 1 skipped` en la batería amplia de contratos, safety, kill switch, ledger,
  restore y approvals; el agregado con los `3 passed` del gate propio es `202 passed, 3 failed,
  1 skipped`;
- focal exacto pre-trade/fencing: `7 passed`;
- frontmatter: `992 passed`;
- links relativos: `664 internal links resolve`;
- contrato RBAC y cobertura RBAC: verdes.

Los tres fallos amplios pertenecen al corte original y no se maquillan como fallos de BL-33.
Dos mostraban deriva del constructor de `MetricEngine`; la secuencia `bf1e02f8`, `89a7732d` y
`2fea6f7e` migró los fixtures afectados a `MetricEngine.from_asset_registry`, y el gate de seguridad
fue revalidado en **35 passed** el 2026-08-06. `RISK-06` sigue `PARTIAL` por los huecos vigentes de BL-18: falta un productor y
consumidor productivos del evento persistido, el allowlist heredado conserva entradas y queda por
decidir la colisión de identidad semántica. El tercer fallo histórico —el digest divergente de
`fabric-v1`— permanece visible en `TECH-06`.

## Qué falta para cierre

El cross-review R2 ya se ejecutó en `CLD-271`: sustituir la evidencia de `INV-04` por `LICENSE`
produjo **1F/4P**, la restauración fue byte-exacta y Claude cerró su objeción. La garantía
resultante es deliberadamente de inmutabilidad de targets revisados, no de verdad material; toda
actualización legítima exige revisar y mover el pin, nunca relajar el test por conveniencia.

1. Resolver o asignar formalmente los dos gaps nuevos sin cambiar digests ni APIs por conveniencia.
2. Incorporar evidencia operativa real: simulacros, sign-off humano independiente, Vault/roles,
   RTO/RPO, reconciliación firmada y controles del Caso B cuando correspondan.
3. Mantener el registro actualizado por evidencia; una fila no sube porque exista el archivo que
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
