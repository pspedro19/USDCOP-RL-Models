# Handoff — «continúa con Claude»

Actualizado por CODEX: 2026-08-03T22:36:00-05:00 (SKEW respecto al reloj Claude).

Cuando el operador diga **«continúa con Claude»**, retomar sin pedir contexto:

1. Leer `PROTOCOL.md`, ambos inboxes, `LEASES.md`, ambos status, `CONTRACTS.md` y este handoff.
2. Consultar `LEASES.md` inmediatamente antes de cada patch; no revisar working trees mutables.
3. Preservar archivos runtime/unrelated dirty y nunca leer secretos.
4. Coordinar cada acción material por `INBOX-CLAUDE.md`; lease antes de editar y review bilateral.

## Estado sellado

- HEAD observado: `74a4f4f2` (auditoría Claude actualizada tras R3).
- C-010 R3 está sellado en `3078ce06` y **APROBADO bilateralmente** por CXD-345.
- Pin `fabric-v1` sellado en `98cefd2d` y **APROBADO bilateralmente** por CLD-350.
- Pin no significa apply: no se ejecutó conexión DB, DDL, apply ni migración.
- C-010 R3 no seleccionó una policy, no cambió `pipelines.yaml`, no promovió estados y no vuelve
  BL-45 DONE; quedan sus incrementos posteriores.
- No push. No reinicios/destrucción de Docker. No aplicar el plan sin autorización separada.

## Evidencia C-010 R3

- Baseline/final: `python -m pytest tests/unit/test_c010_policy_runs.py -q` → **9 passed**.
- Ataque A: todas las policies vigentes inelegibles → cero referencias resueltas.
- Ataque B: promoción temporal de `btc_hodl_b1` a `PARITY_GREEN` → una referencia rule_based;
  el candado que exige cero policies vigentes falló causalmente 1F.
- Ataque C: promoción temporal de `smart_simple_v11` (`engine.type=composite`) →
  `PolicyRunConfigError`, fail-closed.
- Ataque D: retirar temporalmente `resolve_feature_snapshot` → **2 failed**, caller + cutoff.
- YAML restaurados a sus SHA256 iniciales. Factory/test/YAML quedaron con blobs Git idénticos a
  `3078ce06`, diff cero y sin status propio. El SHA físico del factory cambió al rematerializar
  EOL en Windows; el blob canónico verificado es `17c633182b58a66831558acfab459ff18cbf3a44`.

## Coordinación Claude

- CLD-349 entregó R3; CXD-345 publica aprobación y libera leases.
- CLD-350 aprobó adversarialmente el pin `98cefd2d`.
- Claude actualizó su auditoría stale en `74a4f4f2`.
- CLD-352 detectó que `db_migrate.py --status` no es read-only: llama a
  `ensure_migrations_table()` y puede crear `_migrations`. No se ejecutó.

## Siguiente acción exacta mañana

1. Leer mensajes posteriores a CXD-345 y verificar leases/HEAD.
2. Diagnosticar **solo en lectura de código** el comportamiento de `db_migrate.py --status`.
3. Proponer/remediar un modo status genuinamente read-only bajo lease Codex y TDD fail-first,
   avisando a Claude antes; no conectarse a la DB durante esa remediación.
4. Pedir cross-review Claude del hash resultante.
5. Mantener separado cualquier futuro apply de `fabric-v1`: requiere autorización específica.

## Árbol ajeno que debe preservarse

Al cierre había cambios no propios en evidencia Codex antigua, coordinación runtime, DAGs
auxiliares, `data/health/metric_events.jsonl` y zscore. No revertirlos, stagearlos ni incluirlos
en commits. Confirmar con `git status --short` al retomar.
