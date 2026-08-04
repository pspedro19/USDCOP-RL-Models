# HANDOFF CODEX — 2026-08-05 02:42 -05:00

## Punto de reanudación

- Objetivo activo: llegar honestamente a 19/47 DONE coordinando con Claude.
- Corte cofirmado: 14/47. Corte propuesto: 15/47, pendiente de cross-review Claude.
- BL-40 fue promovido por Codex en `1fb83da7` (`[codex] BL-40: close governed quarantine flow`).
- Claude debe revisar ese hash según `CXD-508` en `INBOX-CLAUDE.md`; no contar 15 hasta recibir APROBADO.

## Evidencia BL-40 ya ejecutada

- Suite focal: 60 passed.
- Gates conocimiento: 1165 passed, 47 skipped.
- Inventory/doc-index check, links y grafo: verdes.
- PostgreSQL read-only: canonical USD/MXN imposible = 0; cuarentena durable = 0 porque los E2E revierten; feature status 2026-08-04T18Z = 7/7 UNAVAILABLE.
- E2E reversible previo: invalid→quarantine contextual; correction→canonical/event/CORRECTED; retry idempotente; rollback.
- C028: productor/consumer alineados al cutoff exacto; Claude R2 `3042155b`, Codex consumer `9c9b0bcd`.

## Al encender

1. Leer `INBOX-CODEX.md`, `CLAUDE-STATUS.md`, `LEASES.md` y `git log --oneline -12`.
2. Si Claude aprobó `1fb83da7`, cofirmar 15/32/0 y retirar de la ficha la línea pendiente de cross-review mediante commit acotado y gates afectados.
3. Si lo rechazó, volver BL-40 a PARTIAL o remediar sólo con evidencia concreta.
4. Elegir el siguiente PARTIAL realmente cerrable; no forzar BL-18 (allowlist 22) ni BL-45 (contrato de feature abierto).

## Working tree esperado

- Runtime propio sin commit: `INBOX-CLAUDE.md`, `LEASES.md`, este handoff.
- Cambio ajeno que debe preservarse: `data/health/metric_events.jsonl`.
- No hay leases activos de implementación; el release BL-40 está anotado en `LEASES.md`.

