---
kind: rule
status: IMPLEMENTED
contract: CTR-APPROVAL-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - usdcop-trading-dashboard/lib/contracts/production-approval.contract.ts
---
# Rule: Approval Gates (2 votos)

> **SSOT de las invariantes de aprobación.** Detalle completo (secuencia, schemas, deploy,
> componentes): `../specs/platform/approval-lifecycle.md`.

## Invariantes

1. **Dos votos, siempre.** Vote 1 = gates automáticos del script de export
   (`--phase backtest`). Vote 2 = **humano** en `/dashboard`. Ninguno sustituye al otro.
2. **Vote 2 se emite sobre los números del bundle publicado** (`summary_*.json` +
   `approval_state.json.gates`), nunca sobre métricas recomputadas por el frontend.
   El replay puede mostrar un preview, pero se etiqueta como tal.
3. **`/production` es read-only.** Los botones de aprobación viven solo en `/dashboard`.
4. **El deploy re-valida server-side.** `forecast_h5_l4b_production_deploy::guard_approved`
   comprueba `status == APPROVED` otra vez; la UI no es la autoridad.
5. **Solo `admin` puede emitir Vote 2, promover o accionar el kill global**, y queda en
   `audit_log` (append-only). Ver `rbac.md`.
6. **Estados válidos**: `PENDING_APPROVAL → APPROVED → LIVE`, o `→ REJECTED → PENDING_APPROVAL`
   vía `--reset-approval`.
7. **5 gates por defecto**: retorno > -15%, Sharpe > 0, maxDD < 20%, trades >= 10, p < 0.05.
   `PROMOTE` solo con 5/5.

## DO NOT

- Do NOT saltarte el doble voto — los gates automáticos NO aprueban por sí solos.
- Do NOT editar `approval_state.json` a mano — usa `--reset-approval` o la API.
- Do NOT aprobar sobre métricas recomputadas en el frontend (integridad de Vote 2, audit I-4).
- Do NOT poner botones de aprobación en `/production`.
- Do NOT añadir un gate sin actualizar **ambos** contratos (TS y Python).
- Do NOT confiar en la UI como gate: el deploy revalida server-side.
