---
kind: rule
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Rules Index (auto-loaded)

> `rules/*.md` entra en cada sesión: aquí sólo viven invariantes y prohibiciones. La
> explicación densa se carga desde `specs/` cuando la tarea la necesita.

| Regla | Referencia on-demand |
|---|---|
| [`data-governance.md`](data-governance.md) | [`l0-data-reference.md`](../specs/pipelines/l0-data-reference.md) |
| [`data-freshness.md`](data-freshness.md) | [`freshness-recovery.md`](../specs/operations/freshness-recovery.md) |
| [`strategy-contract.md`](strategy-contract.md) | [`strategy-schemas.md`](../specs/platform/strategy-schemas.md) |
| [`approval-gates.md`](approval-gates.md) | [`approval-lifecycle.md`](../specs/platform/approval-lifecycle.md) |
| [`experiment-protocol.md`](experiment-protocol.md) | [`experiment-config-template.md`](../templates/experiment-config-template.md) |
| [`quant-constitution.md`](quant-constitution.md) | [`constitution-modeling.md`](../specs/assets/btcusdt/design/constitution-modeling.md) |
| [`rbac.md`](rbac.md) | [`rbac-monetization.md`](../specs/platform/rbac-monetization.md) |
| [`strategy-engines.md`](strategy-engines.md) | [`05-rule-based-strategies.md`](../specs/planes/05-rule-based-strategies.md) |
| [`ssot-versioning.md`](ssot-versioning.md) | [`experiment-protocol.md`](experiment-protocol.md) |

Excepciones de ownership: calendario DAG → [`elite-operations.md`](../specs/operations/elite-operations.md);
sesión/tz/annualization por activo →
[`_asbuilt-implementation.md`](../specs/assets/_asbuilt-implementation.md). Mapa completo:
[`../README.md`](../README.md).
