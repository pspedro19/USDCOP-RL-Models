# Review pack BL-24(A) / C033 (inmutable)

commit: `23dce48f`

scope: primer ladrillo acordado de BL-24; writer real de revisiones macro + reloj de verificación.
No implementa todavía el camino paper→snapshot→L0 de (C), ni modifica el ledger/dashboard de (B).

paths:

- `database/migrations/086_lineage_last_verified_at.sql`
- `src/lineage/macro_revision.py`
- `airflow/dags/services/upsert_service.py`
- `airflow/dags/l0_macro_update.py`
- `scripts/ops/db_migrate.py`
- `tests/unit/test_lineage_last_verified_migration.py`
- `tests/unit/test_macro_revision_writer.py`
- `.claude/generated/inventory.json` (generador oficial)

contract: C033 ACK CLD-516; commit cita C033. Plan `lineage-verification-v1` está review-gated y
deliberadamente UNPINNED. Digest independiente requerido antes de apply:
`sha256:90ee1aa036e9f57fb1b227583579a73fa08076c032882cf30c8e624c7b6f67c0`.

red-first:

- writer ausente: error de colección `ModuleNotFoundError`.
- C033 antes de implementación: **4 failed, 9 passed** (086 ausente + writer sin sello).
- plan 086 antes de registrar: **1 failed**, unknown migration plan.

verde:

- safety + migration + writer + path: **55 passed**.
- focal lineage completo: **20 passed**.
- strategy manifests **24 passed**; scripts layout **20 passed**; mirrors **18 passed**.
- knowledge frontmatter **1006 passed**; inventory **6 passed**; autoload **13 passed**.
- inventory/doc-index/links, `py_compile`, `git diff --check`: verdes.
- knowledge graph: único rojo basal `.claude/coordination/HANDOFF-CODEX.md` huérfano.

limitaciones honestas:

- 086 no está pinneada ni aplicada; no hay claim PostgreSQL verde.
- suite legacy `tests/unit/test_l0_macro_update.py` no carga en host porque el paquete local
  `airflow/` oculta la distribución y `from airflow import DAG` falla; el nuevo dataflow se cubre
  por AST + servicio aislado. El import real del DAG debe probarse en scheduler cuando exista.
- ninguna fila del paper ledger ganó IDs en este incremento; BL-24 sigue PARTIAL.

ataques pedidos:

1. misma observación/valor debe avanzar `last_verified_at` sin `revision_event`;
2. cambio histórico default debe emitir `PROVIDER_CORRECTION` y no inferirse de rows_affected;
3. fallo al emitir lineage debe rollbackear el upsert del valor;
4. mutar un byte de 086 debe cambiar digest y mantener plan no autorizado;
5. confirmar que el backfill legacy está declarado como inferido, no observado.
