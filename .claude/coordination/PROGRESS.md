# PROGRESS — tablero CONJUNTO
# Refresco CLAUDE 2026-07-28T11:35 (claude-root-9c3f1e42) — PENDIENTE COFIRMA CODEX.
# Vocabulario v2.2 §2.2 + estado nuevo IMPLEMENTED_UNVERIFIED (CLD-140, pendiente ACK Codex).

## Métrica oficial (K-015): BLs en DONE / 47

- **DONE estricto: 5/47**  (co-firmado 2026-07-28T23:36:44-05:00)
  - **BL-07** (CODEX) — implementación `d0427d6`, packet `ed11c9a`, cross-review
    `CLD-118` APROBADO, cierre `d9fe3bf`.
  - **BL-34** (CLAUDE) — implementación `531c9eb4`, cross-review **CXD-087 APROBADO**.
    CODEX **ejecutó la mutación** `canPromote = true`: pytest 1 failed/7 passed y Vitest
    2 failed/3 passed, y restauró con SHA256 idéntico al inicial. Es el primer BL que
    cierra bajo el protocolo de CLD-209 (comando + mutación + fallo esperado, verificado
    por el otro).
  - **BL-10** (CODEX) — implementacion + backfill, cross-review **CLD-234 APROBADO** por
    CLAUDE, sellado bilateral en `6c9f6138`. La mutacion que lo aprobo no fue de conteo
    sino de CIRCULARIDAD: `n_trials_total` 111 -> 112 en el HYPOTHESIS-REGISTRY, que es un
    SSOT **independiente** del ledger, => 2 failed. Un conteo que solo se comprueba contra
    si mismo no prueba nada. Desbloqueo ademas el tripwire de BL-14: al cerrarse, los
    manifiestos COP tuvieron que dejar de aplazar y declarar su linaje FT real (48 trials
    derivados por consulta, no elegidos).
  - **BL-09** y **BL-11** (CLAUDE) — implementacion `cb1241b2`, cross-review **CXD-089
    APROBADO**. CODEX ejecuto las tres mutaciones: mataron **10, 3 y 2** tests, con SHA de
    restauracion exacto. Los 10 son el parametrizado por introspeccion sobre las funciones
    `check_*`: antes de este cierre, el agregador del gate podia quedarse con **1 de 11
    checks** sin que se moviera un test.
- **BL-06 NO está en DONE.** Este tablero lo listaba como DONE hasta las 2026-07-28T21:17 de hoy;
  era una afirmación rancia. Se retiró en `e144ede` porque se cerró un BL de CI con
  CERO CI y su candado no mordía, y la **verificación por mutación del 2026-07-28 lo
  confirma**: reescribiendo el widget como `fetch('/api/produc' + 'tion/approve')` y
  `<button>Comprar ahora</button>` la suite vuelve a **28 passed, 0 failed** — misma
  capacidad de acción, cero rojo. Fue el último BL abierto del lote CLAUDE; **cerrado el 2026-07-28** en `5ec84a19` con la evasión completa demostrada en rojo.
- APPROVED_PENDING_CLOSE: 0/47.
- **DONE no se infla por tests verdes aislados.** Sigue siendo cross-review del otro.

### Métrica de SUSTANCIA (nueva, 2026-07-28) — no sustituye a DONE, lo complementa

El marcador DONE mide **acuerdo entre los dos ingenieros**. No mide si una garantía
está protegida. Por eso se añade lo que sí se puede medir hoy, con la mutación como
juez (`integration/MUTATION-SCOREBOARD.md`):

| Lote | BLs | Con rojo demostrado | Muerden con hueco documentado | No muerden | Sin verificar |
|---|---|---|---|---|---|
| CLAUDE | 23 | **23** | 0 | 0 | 0 |

**Revisiones cerradas que NO son cierres de BL** (2026-07-28): **BL-20, BL-25 y BL-42** quedan en
`APROBADO_PARCIAL`, y **BL-32 se suma el 2026-07-29** (candado bilateral Py/TS aprobado: la mutacion
Python mata 8/75 y la analoga TS 8/33, mas 2/2 en la proyeccion del productor). CODEX aprobo sus
candados con mutantes y restauracion exacta, pero **los cuatro MD declaran brechas de alcance** y no se
convierten en DONE. Aprobar un candado no es aprobar un alcance;
contarlos seria la misma jugada que este tablero existe para impedir.

| CODEX | 24 | — | — | — | 11 (13 verificados por CLAUDE ⇒ 0 DONE-ABLE) |

**Lectura honesta**: el marcador DONE **no se ha movido** (sigue 1/47) porque ningún
cierre de hoy ha pasado todavía el cross-review del otro — es la Propuesta 6 de
CLD-209 aplicada a nosotros mismos. Lo que sí se movió es el suelo: se pasó de *"no
sabemos si algo está protegido"* a *"23 de 23 tienen un rojo demostrado y 16 defectos
que hacían el verde irrelevante están cerrados"*.

## Cambio de fase (orden del operador, 2026-07-28 ~11:05) — ver CLD-140

Implementar **47/47 primero** aunque no estén probados ni aprobados; la verificación
TDD/BDD y el cross-review cruzado se hacen en una **ola final conjunta**.

- **FASE B** (en curso): barrido de implementación por lanes disjuntos hasta que
  todos los BLs estén `IMPLEMENTED_UNVERIFIED`.
- **FASE C** (al terminar ambos lotes): batería completa — pytest + Vitest +
  Playwright/BDD + monitores (`test_knowledge_frontmatter`,
  `test_strategy_manifests`, `test_scripts_layout`, `rbac:check`) — y cross-review
  cruzado por hash. Solo ahí se mueve a DONE.
- `IMPLEMENTED_UNVERIFIED` := implementación completa + verificación propia, SIN
  cross-review del otro. **No es DONE.**
- `para_review` **deja de bloquear el avance**. Los veredictos pendientes se
  recogen en FASE C.

**No se difiere** (sigue no negociable): constitución quant (0 trials, ninguna
decisión de modelado sin pre-registro del operador, bar DSR 0.95 sin ADR),
`git push` prohibido hasta BL-08, BL-41 sin DDL/cutover hasta Vault real y roles
no-super, fronteras de ASSIGNMENTS y leases antes de escribir.

## CLAUDE (23 BL) — claude-root-9c3f1e42

**DONE (0)**. BL-06 retirado en `e144ede`; su candado no muerde (mutación 2026-07-28).

**FASE B en vuelo (8 lanes disjuntos, leases hasta 12:30)**:
| Lane | BL | Naturaleza |
|---|---|---|
| 1 | BL-31 | strangler COP — sin arrancar |
| 2 | BL-32 | Passport / Control Tower — sin arrancar |
| 3 | BL-36 | inventario DB — sin arrancar (matriz desde código; **cero DDL, sin Docker**) |
| 4 | BL-46 | motor de políticas R4-R5 — sin arrancar |
| 5 | BL-02/03/04 | remedio del rechazo `b86083e` (zoo falso Gold/BTC, BTC early-return, a11y `✓`) |
| 6 | BL-05 | remedio a11y (`th scope=row`, tipografía relativa, spec Playwright sin ejecutar) |
| 7 | BL-09/11/12 | ledger doble FT/AT + familias + provenance (**append-only sobre el BL-10 de CODEX**) |
| 8 | BL-15 | contrato `forecast_output` — remedio de los 5 puntos del rechazo |

**Entregado, esperando veredicto de CODEX (se recoge en FASE C, no bloquea)**:
`C-004-r4`@`4c40dbb` · `C-006`/`BL-20-r2`@`57c3e1c` · `BL-13-r4`+`BL-39-r2`@`3861568` ·
`BL-34-r2`@`a18be01` · `BL-01-r2`@`aa25516` · `BL-14`@`5a2cf5d`+`ecbfca5` ·
`BL-25`@`254ce8f` · kafka@`3a42a48` · `BL-42-r2` integrado en `e5c72b5`.

**Pendiente de arrancar**: BL-47 (espera a que cierre BL-46 — comparten rutas del
motor de políticas; no se paralelizan).

## CODEX (24 BL)

**DONE (1)**: BL-07.

**Cierre administrativo pendiente**: BL-10 — contenido íntegro y verde
(`239` globales / `55 FT` / `184 AT` / USD-COP `111`), capturado accidentalmente
dentro del commit Claude `b86083e` (incidente `CXD-045`). **Claude ACKeó el
incidente y la autoría de las seis rutas es de CODEX** (`CLD-139`): reemitir
`reviews/BL-10.md` designando ese objeto y Claude hace cross-review de solo esas
seis rutas en FASE C.

**Sin arrancar (~21)**: BL-08 (necesita al operador; bloquea el push de todo),
BL-16..19, BL-21..24, BL-26..30, BL-33, BL-35, BL-37, BL-38, BL-40, BL-41
(contrato+TDD sí, DDL no), BL-43, BL-44.

Brief de arranque para la próxima raíz Codex:
`briefs/CLAUDE-ARRANQUE-CODEX-2026-07-28.md`.

## Contratos (último estado en CONTRACTS.md)

C-001/002/003 ACK · C-004 rechazado ×4, remedio r4 `4c40dbb` esperando veredicto ·
C-005 ACK-shape, APPLIED objetado, remedio en `3861568` esperando veredicto ·
C-006 rechazado, remedio `57c3e1c` esperando veredicto · C-007 (BL-41, breaking)
PROPOSED con cinco condiciones ACKeadas por Claude, NO APPLIED.

## Runtime (fuera de backlog)

Scheduler Airflow `unhealthy` al último corte de Codex (picos ~1557% CPU, ~166 PIDs;
watchdog de forecasting corriendo largo). **Orden vigente: no reiniciar ni matar
procesos.** Postgres, webserver y dashboard healthy. Por orden del operador
(2026-07-28 ~11:2x) esta fase **no arranca Docker ni ejecuta pruebas de
infraestructura**.

## Estimado

Tocado/en pipeline: ~24/47 · sin iniciar por su dueño: ~21/47 (casi todos de CODEX).
El cuello real sigue siendo la FASE C: los rechazos previos fueron de sustancia
(RBAC, timezone, paridad Py↔TS, honestidad de copy), no de cosmética, así que la
ola de verificación reabrirá trabajo. Se asume conscientemente por orden del
operador.

FIRMAS:
- claude-root-9c3f1e42 · 2026-07-28T11:35:00-05:00 · refresco propio, **pendiente
  cofirma de la nueva raíz Codex** (corrige o cofirma; no reescribas la historia).
- codex-root-5d968ac6 · 2026-07-27T23:36:00-05:00 · firma de la raíz Codex
  ANTERIOR, ya cerrada (`CXD-047`). Se conserva como referencia histórica.
