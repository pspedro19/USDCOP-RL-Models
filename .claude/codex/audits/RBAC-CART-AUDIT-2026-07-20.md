---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Auditoría RBAC y carrito

## Resultado

Las rutas `/api/cart`, `/api/cart/[assetId]` y `/api/cart/checkout` usan `requireSession`, por lo que las filas quedan aisladas por `user_id`; checkout nunca concede entitlements y delega la activación al webhook de pago. Se añadió una comprobación de defensa en profundidad para rechazar roles desconocidos o identificadores vacíos aun cuando un handler sea invocado sin pasar por middleware.

## Matriz verificada

| Superficie | Regla | Estado |
|---|---|---|
| GET/POST cart | sesión + `user_id` propio | OK |
| DELETE cart item | `WHERE user_id=$1 AND asset_id=$2` | OK |
| checkout | sólo `signals`/`auto`; filtra activos ya poseídos | OK |
| concesión de acceso | únicamente webhook/proveedor | OK |
| role spoofing | roles fuera de `ROLES` rechazados (401) | OK |
| admin APIs | middleware + guard `admin:all` | OK |

## Pendiente operativo

Ejecutar pruebas E2E con base de datos efímera y webhook firmado (idempotencia, replay y tenant isolation). No habilitar producción hasta observar esos eventos reales.
