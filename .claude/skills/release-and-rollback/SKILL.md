---
name: release-and-rollback
description: Assess release readiness and rollback options before shipping. Use when asked to deploy, promote a canary, cut a release, or roll back — and to check whether the deployment workflows actually do what their names suggest.
---

# Release & rollback

## Lo primero: los workflows de deploy NO despliegan

Medido, no supuesto — `.github/workflows/deploy.yml`:

| Métrica | Valor |
|---|---|
| Líneas totales | 358 |
| Líneas `echo` | 54 |
| Comandos `docker` / `kubectl` / `helm` / `aws` / `ssh` | **0** |

El workflow **imprime** lo que haría (blue-green, canary, smoke tests) pero **no ejecuta ninguna
operación de infraestructura**. Lo mismo aplica a `canary-promote.yml` hasta que se verifique lo
contrario.

**Esta skill no debe representarlos como funcionales.** Si alguien pide "desplegá a producción",
la respuesta honesta es que ese pipeline hoy es un placeholder, y preguntar qué mecanismo real de
despliegue quiere usar. Fingir un release es peor que no tener release.

## Lo que SÍ es un despliegue real en este repo

El único camino productivo verificado es el de estrategias, no el de infraestructura:

```
Vote 2 humano en /dashboard
  → POST /api/production/deploy
  → Airflow DAG forecast_h5_l4b_production_deploy
      guard_approved (re-valida APPROVED server-side)
      → run_production → validate_output → register_bundle
```

Para eso existe la skill `approval-cycle`. **Úsala; no reimplementes el flujo aquí.**

## Checklist de preparación (lo que sí puedes verificar hoy)

```powershell
python -m pytest tests/regression/ tests/contracts/ -q      # nada roto
python scripts/diagnostics/generate_inventory.py --check    # docs sincronizadas
cd usdcop-trading-dashboard; npm run qa:gate                # acceso y UI
```

Más: rama limpia, migraciones pendientes identificadas, `.env` con secretos reales **fuera** del
árbol, y el bundle a promover registrado en `registry.json`.

## Rollback

Lo que existe de verdad:

- **Estrategias**: los bundles son **inmutables y versionados**; volver atrás es re-apuntar el
  registry a la versión previa, no re-entrenar. El replay por versión permite comparar antes de
  decidir.
- **Base de datos**: `docs/operations/DATABASE_ROLLBACK_RUNBOOK.md` + los backups diarios de
  `core_l0_05_seed_backup` (Lun-Vie; **no hay backup de fin de semana**).
- **Kill switch**: `ResetKillSwitchCommand(confirmed=True)`, con rastro en el audit trail.

Lo que **no** existe: rollback automatizado de infraestructura. No lo prometas.

## Constraints

- Do NOT describir `deploy.yml` / `canary-promote.yml` como operativos: hoy no ejecutan nada.
- Do NOT poner órdenes en real sin haber pasado por `EXECUTION_MODE=testnet` y su semana de
  validación.
- Do NOT saltarse `PreTradeGate` en ningún camino de orden nuevo — es fail-safe: error ⇒ BLOCK.
- Do NOT desactivar el kill switch en producción.
- Do NOT hacer rollback de una estrategia re-entrenando: re-apunta el registry al bundle anterior.
- Do NOT asumir que hay backup del fin de semana.
- Do NOT anunciar un despliegue como hecho sin evidencia del DAG y del `summary.json` resultante.
