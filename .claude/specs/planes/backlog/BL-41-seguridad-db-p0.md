---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-31
supersedes: []
code_anchors:
  - config/governance/bl41_secret_cutover.yaml
  - scripts/validation/check_bl41_secret_cutover.py
  - tests/regression/test_bl41_secret_cutover.py
  - database/migrations/055_rbac_monetization.sql
  - services/signalbridge_api/app/models.py
---

# BL-41 — Seguridad DB P0: referencias externas, roles y timestamps

**Fuente:** Plan Consolidado §8 / plan institucional §3.3 · **Contrato breaking:** C-007,
ACK condicionado.

## Estado verificable de este corte

**PARTIAL y bloqueado por operador.** Se adoptó un
[SSOT fail-closed](../../../../config/governance/bl41_secret_cutover.yaml) que mantiene
`cutover_allowed: false` y todas las precondiciones sin evidencia. El
[validador estático](../../../../scripts/validation/check_bl41_secret_cutover.py) y su
[gate de regresión](../../../../tests/regression/test_bl41_secret_cutover.py) impiden habilitar el
corte si falta una precondición, una evidencia o la autorización explícita del operador. El gate no
se conecta a PostgreSQL ni al secret store y no puede atestar readiness externa.

El código productivo sigue usando los flujos legacy:

- [la migración RBAC vigente](../../../../database/migrations/055_rbac_monetization.sql) crea
  `user_exchange_keys` con ciphertext en PostgreSQL;
- [los modelos SignalBridge](../../../../services/signalbridge_api/app/models.py) todavía declaran
  `sb_exchange_credentials` y timestamps SQLAlchemy sin zona explícita;
- [la ruta tenant](../../../../services/signalbridge_api/app/api/routes/tenant.py) inserta y lee
  `user_exchange_keys`.

No se repite como hecho actual la afirmación histórica de que las relaciones están vacías: C-007
exige demostrarlo bajo lock en la ventana real de migración. Tampoco se afirma que los roles runtime
sean no-superuser o que `public` tenga ya los grants correctos; ambos requieren evidencia de la base
real.

## Contrato estático reference-only

El destino previsto es `secret.external_account`. PostgreSQL sólo puede guardar identidad,
metadatos operativos y `secret_reference`; material como API keys, ciphertext, passphrases,
fingerprints o máscaras derivadas se rechaza por nombre. El backend del material debe ser un Vault
o KMS externo real.

Antes de cambiar `cutover_allowed`, el SSOT exige evidencia para:

- canario write/read/delete contra el secret store externo;
- relaciones legacy vacías bajo lock;
- identidades runtime no-superuser;
- runner de migración fail-closed;
- corte coordinado de SignalBridge y dashboard;
- autorización explícita del operador.

Marcar una condición `ready: true` con evidencia vacía también falla. El estado bloqueado es un
resultado válido del gate; no se disfraza como error de CI ni como readiness.

## Evidencia TDD

Rojo inicial:

```text
ModuleNotFoundError: No module named 'scripts.validation.check_bl41_secret_cutover'
1 error during collection
```

Verde después del validador y del SSOT:

```powershell
python -m pytest tests/regression/test_bl41_secret_cutover.py -q
# 8 passed
```

Las sondas cubren habilitación prematura, evidencia vacía, ausencia de autorización, shape con
material secreto, omisión de una relación legacy, DDL prematuro y naturaleza completamente
estática del gate.

## Fuera de alcance y bloqueos reales

- No existe ni se crea en este corte la migración `069_secret_external_account.sql`.
- No se ejecuta DDL, lock, canario, grant, revoke, backfill ni cutover de consumidores.
- No se leen valores de credenciales ni archivos de entorno.
- Cualquier migración futura cambia contratos Python/TypeScript y consumidores en un commit
  coordinado; C-007 mantiene `PreTradeGate` fail-safe y el rechazo de llaves con withdraw.
- BL-08 debe resolverse con el operador; BL-41 no puede certificar rotación o revocación remota.

## Done-when pendiente

El cierre requiere evidencia externa sellada, migración nueva revisada (nunca editar una aplicada),
paridad de contratos/consumidores, pruebas con roles no-superuser, verificación de catálogo/ACL,
rollback definido y cross-review. Hasta entonces BL-41 permanece `PARTIAL` y no autoriza capital ni
operación live.

La [readiness matrix](../04b-readiness-matrix.md) registra este control como `BLOCKED_EXTERNAL`.
