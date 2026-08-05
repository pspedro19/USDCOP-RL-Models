# Review pack BL-24(B) / paper path (inmutable)

commit: `4edd4d0e`

remediación R2: `c9b6002c`

scope: camino persistido de una señal paper real hacia el snapshot consumido y el bar L0 de
entrada. Implementa el contrato acordado en CLD-531. No reclama todavía el cierre global de BL-24
ni modifica la migración 086 o el writer de revisiones macro de BL-24(A).

paths:

- `src/forecasting/dataset_loader.py`
- `scripts/pipeline/train_and_export_smart_simple.py`
- `scripts/pipeline/candidates_paper_ledger.py`
- `src/lineage/paper_writer.py`
- `src/lineage/paper_path.py`
- `tests/unit/test_forecasting_dataset_provenance.py`
- `tests/unit/test_paper_lineage_writer.py`
- `tests/unit/test_paper_lineage_verifier.py`
- `tests/unit/test_candidate_ledger_identity.py`
- `usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json`

contrato acordado:

- declaración usa `timestamp` y RESOLVED exige exactamente una fila coincidente en `trades[]`;
- `paper_signal` se identifica por `{strategy_id,timestamp,side}`, sin PnL/equity posteriores;
- `data_snapshot` usa la procedencia real elegida por el loader y hash del frame consumido;
- `bar_l0` usa hash canónico del contenido OHLC diario;
- añadir lineage cambia `semantic_hash`, pero no `decision_fingerprint`.

verde:

- focal + regresiones loader: **31 passed, 1 skipped ambiental**;
- `py_compile`: PASS;
- `git diff --check`: PASS;
- ledger identity: PASS (`semantic_hash=sha256:04d4153ddea8b4f6827a37339a3d3e3738c67b02c3a150a4b7c306e10e95f6bd`);
- PostgreSQL real: `RESOLVED`, `coverage=1`, `verified=true`, camino único
  `paper_signal -> data_snapshot -> bar_l0`;
- knowledge gate: **1072 passed, 1 failed** únicamente por el baseline conocido
  `.claude/coordination/HANDOFF-CODEX.md` huérfano; inventario, índices y links verdes.

limitaciones honestas:

- Ruff no está instalado ni en host ni en el contenedor (`No module named ruff`); no se afirma lint verde.
- Un primer intento anterior al commit final alcanzó el commit DB antes de fallar el sello NumPy y dejó
  tres nodos content-addressed idempotentes; el orden fue corregido y blindado antes de `4edd4d0e`.
- Este pack no marca DONE: requiere veredicto causal de Claude contra el hash inmutable.

R2 tras cross-review CLD-532:

- `load_data_with_provenance()` salió del trainer congelado; el trainer volvió sin delta a los
  bytes anteriores a `4edd4d0e` y `test_strategy_manifests.py` quedó **24 passed** sin re-freeze;
- el preflight JSON valida ausencia, estructura y unicidad antes de conectar;
- PostgreSQL inalcanzable es `UNAVAILABLE`, exit 3, distinto de RESOLVED=0, BROKEN=1 y ABSENT=2;
- batería R2: **58 passed, 1 skipped ambiental**; focal+manifiestos intermedio **52 passed**;
- PostgreSQL real volvió a `RESOLVED`, coverage 1; identidad local/contenedor verde con
  `derivation_id=sha256:741d99d1172064add59bc879d96660a5b1d3e2b1c35708ccda1a3c827018796e`;
- durante R2 un primer run detectó `ROOT` indefinido antes de conectar/publicar; se corrigió a
  `REPO` y quedó un candado explícito. No hubo ledger publicado por ese intento.

ataques pedidos:

1. cero y dos filas con el mismo `timestamp` deben dar BROKEN;
2. alterar `side` debe invalidar la identidad de `paper_signal`;
3. alterar un byte del snapshot o del bar debe cambiar/rechazar su hash;
4. fallar antes del commit DB debe impedir la publicación por `os.replace` y ejecutar rollback;
5. añadir solo `lineage` debe mover `semantic_hash` sin mover `decision_fingerprint`.
