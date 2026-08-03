# INTEGRATION-AUDIT — Claude ↔ Codex

> **ÍNDICE HISTÓRICO — NO ES SSOT.** Desde 2026-07-28T13:58:00-05:00, el
> único canal canónico es [`integration/README.md`](integration/README.md), con
> auditorías, contratos y matrices BDD/TDD bajo `integration/`. Se conserva
> íntegro este inicio de IA-R001 para no reescribir la historia; no se añaden
> aquí nuevos resultados.

Registro compartido y append-only para la integración final de los 47 BL.
No sustituye `reviews/BL-XX.md`, `CONTRACTS.md` ni `PROGRESS.md`.

## Protocolo de rondas

Cada ronda debe registrar:

- `round_id`: `IA-RNNN`.
- `reviewer` y `owner`.
- snapshot revisado: commit cuando exista; si es WIP, lista exacta de paths y hashes.
- hallazgos: ID, severidad `P0|P1|P2`, contrato esperado, evidencia reproducible y dueño.
- escenario BDD en Given/When/Then.
- ciclo TDD: rojo reproducido, cambio mínimo, verde focal, refactor y regresión.
- estado: `OPEN`, `ACCEPTED`, `FIXED_UNVERIFIED`, `VERIFIED` o `WONT_FIX_OPERATOR`.

Reglas:

1. El revisor no corrige archivos del otro; entrega el hallazgo.
2. SSOT, DRY, SOLID, fail-closed, idempotencia y separación
   ACTION/DIAGNOSTIC son criterios explícitos, no comentarios estéticos.
3. Un test que no demuestra rojo antes del arreglo no prueba el defecto.
4. Los resultados deben indicar el comando exacto, entorno y snapshot.
5. Docker permanece fuera de alcance hasta nueva autorización explícita del operador.
6. `IMPLEMENTATION_COMPLETE_UNVERIFIED` nunca equivale a `DONE`.

## Matriz de capas

| Capa | Evidencia mínima |
|---|---|
| Contrato | schemas/tipos y casos positivos/negativos compartidos |
| Dominio | unit tests deterministas, límites y propiedades |
| Persistencia | migración forward/rollback permitido, constraints, ACL e idempotencia |
| Integración | repositorio/servicio/adaptador con dobles controlados |
| BDD | comportamiento de negocio en lenguaje Given/When/Then |
| Seguridad | deny-by-default, secretos externos, inputs hostiles |
| Operación | runbook, health, reconciliación y recuperación |
| E2E | flujo por superficie; se agenda cuando la infraestructura sea autorizada |

## Ronda IA-R001 — iniciada 2026-07-28T11:54:00-05:00

- reviewer primario Codex sobre lote Codex WIP; contra-review solicitado a Claude.
- reviewer Codex sobre entregas Claude: hashes anunciados en `CLAUDE-STATUS.md` e inbox.
- alcance inicial Codex: BL-16..30, BL-35/37/38/40 y las bases ya escritas de BL-41/43/44.
- prioridad: defectos de contrato/seguridad/contabilidad antes de estilo.
- estado: `OPEN`.

### Cola de revisión Codex → Claude

1. C-004/BL-45 `4c40dbb`: fixture único, serialización cerrada, paridad Py/TS.
2. C-006/BL-20 `57c3e1c`: RBAC antes de filesystem, schema compartido, no-public data.
3. BL-31 `6f76934`: gates fail-closed y dependencias declaradas BL-28/BL-30.
4. Entregas FASE-B posteriores: solo cuando Claude publique hash y allowlist.

### Cola de revisión Claude → Codex

1. Migraciones 070–078 y orden de dependencias.
2. Identidad/canonical writer/fingerprints.
3. Legalidad 96→26 y bloqueos operacionales.
4. Catálogo/motor de métricas.
5. Event sourcing, facts, linaje, snapshot y allocator.
6. Dataset URI, factories y diff semántico.
7. `qlab`, cutoff de lectura y backfill anti-supervivencia.
8. Execution service independiente, reconciliación e idempotencia.
