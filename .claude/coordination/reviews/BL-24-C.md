# Review pack BL-24(C) — verificador fail-closed (inmutable)

commit: `bc6d2170`

scope: verificador persistente y CLI diagnóstica cofirmados en CLD-525. Distingue tres resultados:
`RESOLVED`, `BROKEN`, `ABSENT`. `ABSENT` tiene cobertura cero, salida no-cero y no permite cerrar
BL-24. No modifica el ledger servido ni afirma que el camino real ya exista.

paths:

- `src/lineage/paper_path.py`
- `scripts/diagnostics/verify_paper_lineage.py`
- `tests/unit/test_paper_lineage_verifier.py`

contrato de entrada provisional para BL-24(B): cada estrategia podrá declarar bajo `lineage` los
IDs persistidos `signal_node_id`, `snapshot_node_id`, `bar_l0_node_id`. Hasta que (B) los produzca,
el ledger real debe quedar `ABSENT`; una declaración parcial pasa a `BROKEN`.

evidencia verde:

- `python -m pytest tests/unit/test_paper_lineage_verifier.py tests/unit/test_lineage_path.py -q`
  → **16 passed**.
- `python -m pytest tests/regression/test_scripts_layout.py -q` → **20 passed**.
- CLI contra `smart_simple_v11` del ledger servido → `status=ABSENT`, `coverage=0`,
  `verified=false`; el subprocess prueba exit code **2**.
- PostgreSQL vivo, probe rollback-only: insertar `LEGITIMATE_RELEASE` dejó historia y descendiente
  `VALID`; `rollback_clean=True`.

mutación:

- cambiar el resultado de ledger sin IDs de `ABSENT` a `RESOLVED` → **2 failed**: el test de
  librería y el subprocess CLI; restaurado y 16P.

ataques pedidos:

1. hacer que `ABSENT` devuelva exit 0 o `verified=true` debe caer;
2. quitar una arista intermedia debe resultar `BROKEN`;
3. añadir señal→L0 directa además de señal→snapshot→L0 debe resultar ambiguo/`BROKEN`;
4. reemplazar un tipo persistido esperado debe resultar `BROKEN`;
5. revisar si los tres nombres de IDs son el contrato correcto para (B) antes de publicarlos.

limitación honesta: el ledger real está `ABSENT`; por tanto (C) está implementado pero el criterio
de verificación end-to-end de BL-24 sigue incumplido y BL-24 permanece `PARTIAL`.

