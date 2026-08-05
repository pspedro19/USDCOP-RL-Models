# Review pack BL-24 — export publico de lineage (inmutable)

commit: `b96172c7`

alcance: arreglo aislado solicitado por Claude en CLD-509; no implementa todavía BL-24(A),
BL-24(C), ni cambia contratos compartidos, DB, ledger o dashboard.

paths:

- `src/lineage/__init__.py`
- `tests/unit/test_lineage_path.py`

C-EXEMPT: API interna Python; restaura los seis símbolos ya importados por el paquete. No toca
`src/contracts/`, `lib/contracts/`, esquema SQL ni artefactos servidos.

red-first:

- `python -m pytest tests/unit/test_lineage_path.py -q` → **1 failed, 5 passed**.
- fallo exacto: faltaban `LineageEdge`, `LineagePathError` y `resolve_unique_path` en `__all__`.

verde final:

- focal → **6 passed**.
- knowledge frontmatter → **1005 passed**.
- strategy manifests → **24 passed**.
- scripts layout → **20 passed**.
- `py_compile` → verde.
- knowledge graph → único rojo basal ya registrado: `.claude/coordination/HANDOFF-CODEX.md`
  huérfano; no tocado por este commit.

ataque pedido: comprobar que `from src.lineage import *` expone los seis nombres declarados y
que no se introdujo un export extraño. Revisar contra `b96172c7`, no contra el working tree vivo.
