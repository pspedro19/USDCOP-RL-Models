---
name: rbac-qa
description: Run the full access-control and UI quality gate for the dashboard — RBAC route coverage, contract invariants, role matrix, functional and visual QA. Use after touching routes, middleware, the RBAC contract, entitlements, or any page a role can reach.
---

# RBAC & QA gate

Los scripts ya existen y son buenos; el problema es que **se corren sueltos y a medias**. Esta
skill es el orden completo y qué significa cada fallo.

## El gate corto (lo que debe pasar siempre)

```powershell
cd usdcop-trading-dashboard
npm run qa:gate     # = rbac:check + rbac:test + qa:functional + qa:matrix
```

| Paso | Qué prueba | Si falla significa |
|---|---|---|
| `rbac:check` | **Cobertura**: toda ruta real está en la matriz de permisos | Hay una ruta sin entrada → deny-by-default no la cubre |
| `rbac:test` | **Invariantes** del contrato RBAC | La matriz es internamente incoherente |
| `qa:functional` | Flujos funcionales | Una página rompe para algún rol |
| `qa:matrix` | Matriz rol × vista | Un rol ve lo que no debe, o no ve lo que pagó |

## Complementos según lo que tocaste

```powershell
npm run qa:registration   # si tocaste registro/aprobación de usuarios
npm run qa:promotion      # si tocaste el flujo de promoción de estrategias
npm run qa:visual         # regresión visual (mobile/wide/zoom)
npm run test:e2e          # Playwright completo
```

## Qué NO cubre (y por qué importa)

- **`contracts-check.yml` no se dispara con cambios solo en TypeScript** — vigila
  `src/**/*.py`, `services/**/*.py`, `airflow/**/*.py`. Si editaste `rbac.contract.ts` y nada más,
  **corre los scripts a mano**: CI no lo hará por ti.
- `rbac-gate.yml` sí corre `rbac:check` + `rbac:test` en PR, pero **no** `qa:functional` ni
  `qa:matrix` (necesitan stack vivo). Esos son responsabilidad tuya en local.
- La cobertura verifica que la ruta *exista* en la matriz, **no que el permiso sea el correcto**.
  Una ruta de admin mapeada a `free` pasa el check de cobertura. Léelo tú.

## Reglas que el gate hace cumplir

1. **Deny-by-default**: toda página (≠ landing/pricing/login) y todo `/api/**` exige sesión +
   permiso server-side. Ocultar UI no es control de acceso.
2. **Rol ≠ plan**: el JWT es caché; `sb_users.entitlements` es la verdad. Vencido ⇒ se sirve como
   `free`.
3. **Nada monetizado anónimo**: `/data/**` y `/forecasting/**` exigen sesión en el edge.
4. **Vote 2, promover y kill global: solo `admin`**, siempre al `audit_log` (append-only).

## Constraints

- Do NOT añadir una ruta sin entrada en `rbac.contract.ts` — el CI la detecta, pero descubrirlo
  en el PR cuesta más que hacerlo bien.
- Do NOT poner artefactos monetizables nuevos en `public/` sin gate.
- Do NOT actualizar entitlements desde el cliente — solo webhook o `/admin`.
- Do NOT editar ni borrar filas de `audit_log` (un trigger lo impide; no lo quites).
- Do NOT dar por bueno un `qa:gate` parcial: si `qa:functional` no corrió por falta de stack,
  dilo explícitamente en vez de reportar "gate verde".
