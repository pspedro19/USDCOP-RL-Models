# Brief de arranque para la próxima raíz CODEX — 2026-07-28

> Escrito por `claude-root-9c3f1e42` (raíz Claude nueva, 11:06 COT).
> Pégale esto a la terminal Codex al abrirla, o déjala que lo lea.
> **Este brief NO sustituye al protocolo**: complementa
> `briefs/CODEX-HANDOFF-2026-07-28-0836.md`, que sigue siendo tu SSOT de continuidad.

---

## 0. Lo primero (antes de escribir una sola línea)

1. Lee, en este orden: `PROTOCOL.md` (v1.0 + enmiendas v1.1 y v1.2),
   `ASSIGNMENTS.md`, `INBOX-CODEX.md`, `INBOX-CLAUDE.md`, `CONTRACTS.md`,
   `KNOWLEDGE.md`, `LEASES.md`, `CLAUDE-STATUS.md` y
   `briefs/CODEX-HANDOFF-2026-07-28-0836.md`.
2. Anúnciate con un **`instance_id` NUEVO** en `CODEX-STATUS.md`. No hay sucesión
   silenciosa: `codex-root-5d968ac6` cerró en `CXD-047` y su auxiliar
   `codex-helper-7f9f3b832dd1` también.
3. Verifica índice Git vacío antes de cualquier trabajo:
   `git diff --cached --name-status` debe salir vacío.

Estado del árbol al escribir este brief:
`HEAD = b67b8e4e505f73d198411e558ca8ec778f1ff990`, índice vacío.

---

## 1. Cambio de fase ordenado por el operador (lo más importante)

El operador ordenó, textualmente: **implementar TODAS las tareas del backlog
aunque no estén probadas ni aprobadas, y cuando estén listas, coordinar TDD/BDD
para probar que todo esté bien.**

Consecuencias operativas (detalle en `INBOX-CODEX.md` → `CLD-140`):

- **`para_review` deja de bloquear el avance.** No te detengas a re-revisar cada
  entrega ciclo a ciclo. Los veredictos se recogen en la ola final.
- **Tus rechazos siguen siendo la especificación del remedio.** Claude los está
  implementando ahora mismo sin esperar re-review. No los retires ni los relajes.
- **Estado nuevo `IMPLEMENTED_UNVERIFIED`**: implementación completa +
  verificación propia, SIN cross-review del otro. **No es DONE.** El marcador
  estricto sigue en **1/47** (BL-07) y no se infla.
- **FASE B** = barrido de implementación por lanes disjuntos hasta 47/47
  `IMPLEMENTED_UNVERIFIED`.
  **FASE C** = ola conjunta TDD/BDD + cross-review cruzado por hash. Solo ahí se
  mueve a DONE.

### Lo que NO se difiere (sigue siendo no negociable)

Diferir la verificación **no** relaja la gobernanza:

1. **Constitución quant**: 0 trials. Ninguna decisión de modelado sin
   pre-registro del operador. El bar DSR 0.95 no se toca sin ADR.
2. **`git push` PROHIBIDO** hasta cerrar BL-08 (el `.env` filtrado sigue público).
3. **BL-41: sin DDL ni cutover** hasta Vault real, roles no-super, lock/preflight
   y TDD — tus cinco condiciones de `C-007` siguen en pie y Claude las ACKeó.
4. **Fronteras de `ASSIGNMENTS.md` y leases** antes de escribir. El incidente de
   índice (`CXD-045`) no se repite: allowlist explícita de rutas +
   `git diff --cached --name-status` inmediatamente antes de cada commit, y
   ningún permiso Git pendiente pasado el TTL.

---

## 2. Cerrado de mi lado (no necesitas re-litigarlo)

- **`CXD-045` (incidente de índice BL-10): ACK completo** — ver `CLD-139`.
  Acepto tu cronología, no te atribuyo violación de lease, y **la autoría de las
  seis rutas BL-10 es tuya**. No las he tocado ni las tocaré. Reemite
  `reviews/BL-10.md` designando el objeto `b86083e` cuando reanudes; el
  cross-review de esas seis rutas lo hago en la ola final.
- **`CXD-046` y los veredictos del handoff: ACK** — ver `CLD-141`. Retiré de
  `para_review` los cuatro rechazados y están en remedio ACTIVO ahora mismo:
  BL-02/03/04 (texto zoo falso Gold/BTC, BTC early-return, a11y de `✓`),
  BL-05 (`th scope=row`, tipografía relativa, Playwright real post-hash),
  BL-12-r3 (coherencia ledger↔header contra 239/111, guard estructural no
  evadible), BL-15-r2 (timezone mixto, fechas imposibles en TS con runner real,
  CSV que descarta el contrato, ingest wall).
- **`CXD-047` (handoff): ACK**.

---

## 3. Lo que tienes esperando veredicto (para la FASE C, no ahora)

En el orden que tú mismo dejaste en el handoff:

1. `C-004-r4` @ `4c40dbbce17dc89aa93089518486d19c78f8abe8` (`CLD-134`)
2. `C-006`/`BL-20-r2` @ `57c3e1cb0558e5dd1f27651d7c5ba119d5603e90` (`CLD-133`)
3. `BL-13-r4` + `BL-39-r2` @ `386156810ea707979223b21771501149f33cf7f2`
4. `BL-34-r2` @ `a18be019ad85e8ea0302830e7d723c877904cfe2` (`CLD-137`)
5. `BL-14` @ `5a2cf5d`+`ecbfca5`, `BL-25` @ `254ce8f`, kafka @ `3a42a48`,
   `BL-01-r2` @ `aa25516`

---

## 4. Tu lote (24 BLs) — sugerencia de orden para FASE B

Ya cerrado: **BL-07** (único DONE estricto del backlog).

Sin arrancar, en orden de dependencia según el grafo del README del backlog:

- **BL-16** (CI constitucional Etapa 0) — tu handoff dice GO solo para la mecánica
  TDD/matriz `96→26`; las decisiones de autoridad/serialización siguen NO-GO.
- **BL-17** (fingerprints + canonical writer + spine) — desbloquea 21/22/24/28.
- **BL-18** (catálogo de métricas + motor único), **BL-19** (esquema DB forecast.*).
- **BL-37** (identidades canónicas), **BL-38** (mercado canónico + caggs),
  **BL-35** (URIs de datasets), **BL-40** (cuarentena).
- **BL-21→BL-22→BL-23→BL-24** (event sourcing → facts → anti-supervivencia → linaje).
- **BL-26→BL-27→BL-28→BL-29→BL-30** (snapshot → allocator → factories → qlab →
  executor).
- **BL-33** (readiness matrix), **BL-44** (Timescale ops).
- **BL-43**: seguía bloqueado por `C-005`. Con la fase nueva puedes implementarlo
  contra el shape ACKeado y declarar la dependencia, o mantenerlo bloqueado si
  crees que acopla código a un registry no sellado — tu criterio, pero dilo.
- **BL-41**: contrato + TDD sí; DDL/cutover NO (ver §1.3).
- **BL-08**: necesita al operador (rotar credenciales, purgar historial,
  privatizar el repo). Es el que bloquea el push de todo lo demás.

---

## 5. Lanes de Claude ACTIVOS ahora (no los toques)

Leases publicados en `LEASES.md` hasta las 12:30 COT, dueño
`claude-root-9c3f1e42`:

| Lane | BL | Rutas principales |
|---|---|---|
| 1 | BL-31 | strangler COP |
| 2 | BL-32 | Passport / Control Tower (dashboard + BFF) |
| 3 | BL-36 | inventario DB (**solo lectura de DB; cero DDL**) |
| 4 | BL-46 | motor de políticas R4-R5 |
| 5 | BL-02/03/04 | `ForecastingView.tsx`, `WeeklyInferenceView.tsx`, `lib/ui/forecast-disclaimer.ts` |
| 6 | BL-05 | `PaperCandidatesPanel.tsx` + e2e |
| 7 | BL-09/11/12 | `registries/ledger.jsonl` (**append-only**), `registries/families/` |
| 8 | BL-15 | contrato `forecast_output` Py+TS |

**Aviso explícito sobre el lane 7** (ver `CLD-142`): toca `registries/` para
BL-09/11, pero **el contenido BL-10 tuyo no se modifica** — solo se appendean
asientos donde BL-09 lo exige y no se rompe la hash-chain. Si necesitas congelar
`registries/` para sellar tu pack de BL-10, dilo en `INBOX-CLAUDE.md` y paro esa
lane al instante.

---

## 6. Runtime al corte (contexto, no tarea)

- Scheduler Airflow estaba `unhealthy` con picos de ~1557% CPU y ~166 PIDs; el
  watchdog de forecasting corría largo. **La orden vigente es NO reiniciar ni
  matar procesos**: se espera a que el PID termine y el log muestre completion.
- Postgres, webserver y dashboard estaban healthy; sin recurrencia reciente del
  error `week` ni de fallos de autenticación.
- Warnings de predicciones constantes en XGBoost/LightGBM/CatBoost: diagnóstico,
  **no** decisión de modelado.

---

## 7. Contrato de comunicación en esta fase

- Mensajes `CXD-NNN` con HECHO / EVIDENCIA / IMPACTO / PROPUESTA / DONE-WHEN,
  timestamps de reloj de sistema.
- Heartbeat en `CODEX-STATUS.md` cada ≤5 min mientras trabajas.
- En FASE B basta con que anuncies **qué BL pasó a `IMPLEMENTED_UNVERIFIED` y con
  qué hash**. No hace falta pack completo de review por BL hasta la FASE C, pero
  sí el hash inmutable — sin hash no hay nada que revisar después.
- Cuando termines tu lote, dilo en `INBOX-CLAUDE.md` y arrancamos la FASE C
  juntos: batería completa (pytest + Vitest + Playwright/BDD + monitores
  `test_knowledge_frontmatter` / `test_strategy_manifests` / `test_scripts_layout`
  / `rbac:check`) y cross-review cruzado por hash.
