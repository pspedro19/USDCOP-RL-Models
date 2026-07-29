---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/PassportView.tsx
  - usdcop-trading-dashboard/lib/passport/compose.ts
  - usdcop-trading-dashboard/lib/contracts/passport.contract.ts
  - src/contracts/passport.py
  - scripts/pipeline/export_control_tower.py
  - .claude/specs/platform/passport-control-tower.md
---

# BL-32 — Passport (vista live + MV) + Control Tower

**Fuente**: FABRIC §24.4-§24.5 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Hub/Production muestran piezas; no existe el SELECT único (identidad+gobierno+linaje+desempeño 5 entornos+riesgo) ni la portada de 10 segundos.

## Qué falta exactamente
v_strategy_passport_live (NO materializada) + mv_strategy_performance_daily (nocturna) + composición; Control Tower: LIBRO/SLEEVES/DATOS (§24.5) incl. N_global vs N_MAX, test pareado v11 vs v12/v14 con p/e-value, semáforo de retiro.

## Impacto frontend
Página/sección nueva (o evolución de /hub) — la cara del sistema.

## Dependencias
BL-18, BL-22, BL-24; BL-05 es el primer ladrillo.

## Verificación
El paseo Airflow→MLflow→JSONs→SQL se reemplaza por un SELECT (demo).

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando:   python -m pytest tests/unit/test_passport_contract.py -q
verde:     83 passed   (era 65 antes del cierre)

comando-2: npx vitest run tests/unit/contracts/passport-contract.test.ts
           (desde usdcop-trading-dashboard/)
verde-2:   41 passed

muta:      src/contracts/passport.py — el bucle
           `for block, fields in PASSPORT_BLOCK_FIELDS.items()` deja de recorrer nada
espera:    8 failed, 75 passed

muta-2:    recortar de 8 a 3 la lista de bloques obligatorios del passport
espera-2:  5 failed, 60 passed   (medido en el cierre previo, 2a608feb)
```

**Historial honesto**: hasta el 2026-07-28 se comprobaba **PRESENCIA de 8 claves y nada más**,
en los DOS lenguajes. Recortar los bloques obligatorios de 8 a 3 (fuera `identity`,
`governance`, `lineage`, `risk`, `live`) dejaba la suite en **44 passed idéntico**; y en TS,
vaciar `governance` a `{}` —cero trials, cero DSR, cero `policy_hash`— pasaba **30/30**:
borrar la CLAVE caía, vaciarla no. **El fixture era literalmente el agujero**:
`minimalPassport()` usaba `identity:{}`, `governance:{}`, `lineage:{}`, `live:{}`, `risk:{}` —
un fixture más vacío que el payload real no puede detectar un payload vacío. Era un hueco de
**PRODUCCIÓN**, no de test: ni `validate_strategy_passport` (Py) ni `validateStrategyPassport`
(TS) tenían requisitos de contenido por bloque, así que se podía publicar una estrategia sin
gobernanza, sin linaje y sin riesgo.

**La regla aplicada, que es la parte que importa**: NO se exigen valores no-nulos.
`policy_hash` no tiene productor hasta BL-45, `dsr_family` está `unavailable` para casi todas
las estrategias y un activo sin trials es un estado legítimo. Lo que se exige es la forma que
el propio contrato ya define: `Sourced` con `value: null` y un `pending` NO VACÍO. **Sin
agujeros MUDOS, no sin agujeros.** Hay un test que protege esa frontera por los dos lados: un
hueco declarado (`pending: 'BL-45 …'`) debe ser ACEPTADO, y convertir la regla en "sin
agujeros" lo pone rojo y además revienta el payload real con *"quant-constitution §6 violated
on the real published surface"*.

**Payload real**: 18/18 estrategias del registry (4 activas + 14 archivadas) componen y validan
con 0 errores.

**Gaps declarados, no arreglados**: `validate_control_tower` / `validateControlTower` siguen
con el hueco idéntico (book, data y las filas de sleeves solo se validan por presencia de
clave); y las rutas `/api/passport/**` **NO llaman a los validadores**, solo componen — hoy el
único gate real es la suite.

## Notas constitución
MV de Postgres no puede sostener estado live (se reemplaza en refresh) — por eso la división.

## Estado de implementación (2026-07-28, CLAUDE lane2)

**PARTIAL — la superficie completa existe; las fuentes de hechos no.**

Entregado: contrato `CTR-PASSPORT-001` espejado Py↔TS, compositor BFF sobre
artefactos publicados, rutas `/api/passport/tower` y `/api/passport/[strategyId]`,
página `/passport` (`research:read`, RBAC verde), proyección de gobernanza
`scripts/pipeline/export_control_tower.py` (trials FT/AT + N×3 + familias +
protocolos de retiro), y spec `.claude/specs/platform/passport-control-tower.md`
con la **DDL de referencia** de `v_strategy_passport_live` /
`mv_strategy_performance_daily` / `v_strategy_passport` (no aplicada: sus tablas de
hechos no existen).

Pendiente de CODEX (declarado como interfaz legible por máquina en
`PENDING_INTERFACES`, renderizada en la propia página): BL-18 (motor de métricas,
p/e-value del test pareado, DSR por entorno), BL-21 (estado live, canary),
BL-22 (PnL del libro, timing_ratio, turnover, atribución), BL-23 (held-out),
BL-24 (grafo de linaje, vintage), BL-25 (reloj exec + semáforo por estrategia),
BL-26/27 (riesgo, ρ, multiplicadores), BL-28 (paridad replay).

Verificación: `rbac:check` 97 API/32 páginas OK · `rbac:test` PASS ·
`tsc --noEmit` delta 0 · exportador corrido (ledger 239 líneas, N_global=239/989,
4 activos con `n_trials_total` == total del ledger).
