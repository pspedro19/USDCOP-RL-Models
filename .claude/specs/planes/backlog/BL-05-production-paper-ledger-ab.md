---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ProductionView.tsx
  - usdcop-trading-dashboard/components/gm/views/PaperCandidatesPanel.tsx
  - usdcop-trading-dashboard/tests/unit/components/PaperCandidatesPanel.test.tsx
  - usdcop-trading-dashboard/tests/e2e/paper-candidates-a11y.spec.ts
  - scripts/pipeline/candidates_paper_ledger.py
---

# BL-05 — ProductionView consume el paper ledger (A/B v11/v12/v14)

**Fuente**: FABRIC §24.5 Control Tower / plan 00 §2 · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El ledger existe y se refresca cada viernes vía L6 (`candidates_ledger_2026.json`, anclado a ene-2026, judge_window post-freeze). ProductionView NO lo referencia — el A/B vivo es invisible en la UI.

## Qué falta exactamente
Panel en /production: tabla candidatas (v11 real vs v12/v14 paper), judge_window post-2026-07-21 con N y nota N<20, días al juez. Solo lectura del JSON publicado.

## Impacto frontend
`/production` gana el panel A/B. Cero botones (read-only por invariante).

## Dependencias
—

## Verificación
Panel renderiza el JSON real; N<20 muestra solo conteo/PnL.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: npx vitest run tests/unit/components/ProductionView.paper-ledger.test.tsx                         tests/unit/components/PaperCandidatesPanel.test.tsx
         (desde usdcop-trading-dashboard/)
verde:   16 passed

muta:    components/gm/views/ProductionView.tsx:982
         {paperLedger && <PaperCandidatesPanel/>}  ->  {false && paperLedger && <PaperCandidatesPanel/>}
espera:  1 failed — "Unable to find role=table and name /candidatas/i"
         (el A/B v11/v12/v14 desaparece de /production: el estado exacto que este BL arregla)

muta-2:  components/gm/views/PaperCandidatesPanel.tsx — celda extra "Sharpe 3.35 · p=0.006"
         en filas con n_trades=11
espera:  1 failed — la fila publica Sharpe con N<20, prohibido por quant-constitution §6
```

**Historial honesto**: hasta el 2026-07-28 ninguna de las dos mutaciones movía un test.
Los 13 tests existentes renderizaban `PaperCandidatesPanel` **aislado** con un fixture, así
que el panel podía estar desconectado de la página; y el test de §6 comprobaba que se ecoa un
string que venía **del propio fixture**, no que no hubiera un Sharpe al lado.

## Notas constitución
Vote-2/decisiones siguen sobre bundles; esto es monitoreo del juez sellado — jamás re-anclar.

## Remediación del rechazo CXD-022 (2026-07-28)

CXD-022 rechazó `624465c` por cuatro gaps. Estado tras el remedio:

| Gap del rechazo | Estado | Dónde |
|---|---|---|
| Falta `<th scope="row">` (row headers) | **HECHO** | `PaperCandidatesPanel.tsx` — la celda de nombre de cada fila es `th scope="row"` (`font-normal`/`text-left` preservan el aspecto) |
| Tipografía fija `12.5px` | **HECHO** | `text-[12.5px]` → `text-[0.78125rem]`; el resto de tokens (`GMT.*`) ya eran rem. Los únicos `px` que quedan son de layout (`min-w-[760px]`, `-mx-[18px]`), no de fuente |
| Significado por símbolo/color sin equivalente textual | **HECHO** | `NoData()` (guión `aria-hidden` + `sr-only "sin dato"`) y texto `sr-only` **`En producción: Sí/No · Juez sellado: Sí/No`** en el row header (el tono del badge era canal cromático) |
| Prueba real 375px / landscape / teclado / consola | **PENDIENTE DE EJECUCIÓN** | spec escrito y listo en `tests/e2e/paper-candidates-a11y.spec.ts`; **no ejecutado** por orden del operador (2026-07-28: no levantar Docker ni el dashboard). No hay evidencia E2E asociada y no debe darse por verde |

**Cobertura unit actual**: `PaperCandidatesPanel.test.tsx` **13 passed** (2 asserts nuevos
verificados fail-first: 2 failed antes del fix → 13 passed después). `tsc --noEmit`: 0 errores
en los archivos tocados.

**Para cerrar el gap E2E** hace falta un dashboard servido con este remedio y sesión `admin`
(el panel está oculto para `free`/`subscriber` — `ProductionView::isClientView`), y luego
`npx playwright test tests/e2e/paper-candidates-a11y.spec.ts --project=chromium`.

### El gap E2E, ya DIAGNOSTICADO y no sólo «no ejecutado» (2026-08-05)

El operador levantó la orden del 2026-07-28 **de forma acotada**: «dev server sí, Docker no». Se
persiguió hasta el final y **el bloqueo real quedó identificado con precisión**: no es que falte
ejecutar, es que **este E2E no puede correr sin contenedores**. Camino recorrido, en orden, con lo
que cada capa enseñó:

| Intento | Resultado | Qué enseñó |
|---|---|---|
| `npm run dev` + playwright | **K-044 aborta**: «el HTML servido no expone un BUILD_ID reconocible» | El guard del propio repo rechaza un dev server: sin artefacto identificable, la corrida no es evidencia *en ninguna dirección* |
| `npm run build` + `npm start` | **K-044 aborta**: artefacto `2026-08-05T14:01:21.496Z` **22 s anterior** al commit `802b0267` | «Un verde contra un build rancio es peor que no tener evidencia, porque parece que sí la tienes» |
| rebuild tras el último commit de código servido | **K-044 pasa** | El guard compara contra el último commit que toca *rutas servidas*, no contra cualquier commit |
| corrida real | falta el binario de Chromium → `npx playwright install chromium` | — |
| corrida real (2º) | `[next-auth][error][NO_SECRET]` | `npm start` (producción) exige `NEXTAUTH_SECRET`; se pasó uno desechable **al proceso**, sin tocar ningún `.env` |
| corrida real (3ª) | **login no aterriza en `/production`** | **Causa final, medida en el log del servidor** (abajo) |

```
[API] SignalBridge auth login error: getaddrinfo ENOTFOUND usdcop-signalbridge
[UserRepository] findByEmail error: [PostgreSQL] Database configuration missing.
                 Set DATABASE_URL or (POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD)
```

El panel es **admin-only** y la autenticación admin se apoya en **dos servicios de contenedor**:
`usdcop-signalbridge` (nombre DNS que sólo resuelve dentro de la red de compose) y PostgreSQL
(`sb_users`). Por tanto **«dev server sin Docker» es insuficiente por construcción** para este BL,
y no por un detalle de configuración que se pueda rodear.

### CORRIDA REAL EJECUTADA (2026-08-05, con Postgres + SignalBridge autorizados)

El operador autorizó levantar **sólo** `postgres` + `signalbridge-api` (no el stack). Con eso el
login admin funciona y **la spec corre de verdad por primera vez**:

```
✅ K-044: el artefacto cubre el codigo bajo prueba
   1 passed · 2 failed
```

**Lo que quedó VERIFICADO EN RUNTIME** — el aserto de consola es el **último** de cada test
(línea 160), así que todo lo anterior pasó antes de fallar. Los cuatro gaps del rechazo CXD-022
quedan cerrados con evidencia ejecutada, no declarada:

- `tabindex="0"` en la región scrolleable (axe *scrollable-region-focusable*),
- **row headers reales** (`th scope="row"`) presentes en runtime,
- **equivalentes textuales** del canal cromático: `En producción: Sí/No` y `Juez sellado: Sí/No`
  leídos del DOM, no del fixture,
- **navegación por teclado** llega a la región + **foco visible** (WCAG 2.4.7: outline/box-shadow),
- la tabla **scrollea dentro de su región** y el body **no desborda** en 375px ni en landscape,
- **WCAG 1.4.4** (escala con el font-size del root): **este test pasa entero**.

Capturas en `tests/e2e/__screenshots__/`: `paper-candidates-375-portrait.png`,
`paper-candidates-667-landscape.png`, `paper-candidates-375-font-scaled.png`.

**Lo que sigue ROJO, y por qué NO se relaja.** Los 2 fallos son el aserto «cero errores de
consola», y las tres causas están identificadas en el log del servidor — **ninguna es el panel**:

| Error de consola | Causa medida | ¿Producto? |
|---|---|---|
| WebSocket `ws://localhost:8000/ws` refused | el contenedor mapea **8085→8000**; el cliente apunta a `:8000` del host | no: puerto no publicado en esta corrida acotada |
| 502 | `[RealtimePrice] getaddrinfo ENOTFOUND usdcop-trading-api` + `Investing.com error: 403` | no: contenedor no arrancado (fuera del alcance autorizado) + fuente externa sin credenciales |
| 404 | `[PostgreSQL] relation "user_cart" does not exist` | **hallazgo colateral**: migración no aplicada en esta DB |

**Segunda corrida (2026-08-05, tras `99e7d511`): `2 passed · 1 failed`** — mejora medida sobre
`1 passed · 2 failed`. La primera de las tres causas resultó ser **defecto de producto y se
arregló**: los dos fallbacks del WebSocket de SignalBridge apuntaban a `:8000` y `:8080` mientras
el compose publica `8085:8000` y el dashboard no declara `NEXT_PUBLIC_SIGNALBRIDGE_WS_URL` — o sea
que **ese WS no podía conectar nunca**, con Playwright o sin él. Alineados ambos al puerto
publicado, el error de WS desaparece de la consola y el test de landscape pasa a verde.

Quedan **dos** causas, ambas de infraestructura y ninguna del panel:

    relation "user_cart" does not exist   -> migración 057_catalog_watchlist_cart.sql SIN aplicar
                                             (DDL = lane CODEX; ACK pedido en CLD-512)
    getaddrinfo ENOTFOUND usdcop-trading-api  -> contenedor no arrancado
    + Investing.com error: 403                -> fuente EXTERNA; puede persistir aunque
                                                 se levante trading-api

Honestidad sobre el pronóstico: aplicar la 057 es determinista, pero levantar `trading-api`
**no garantiza** consola limpia — el 403 de Investing.com es externo y no depende de nosotros.

### Tercera corrida (2026-08-05, con `trading-api` + migración 057 aplicada): `2 passed · 1 failed`

Se cerraron **dos** de las tres causas y ambas resultaron ser defectos reales, no ruido:

| Causa | Cierre | Qué era en realidad |
|---|---|---|
| WS `ws://localhost:8000` refused | `99e7d511` | **Defecto de producto**: dos fallbacks (`:8000` y `:8080`) y el compose publica `8085`. Ese WS no podía conectar nunca |
| `relation "user_cart" does not exist` | migración **057** aplicada (pin de CODEX `b1c6e66b`, ledger fila 66) | Migración escrita y nunca aplicada en esta DB |
| `ENOTFOUND usdcop-trading-api` + Investing 403 | contenedor levantado + `TRADING_API_URL` al puerto publicado | Servicio ausente, no defecto |

**Queda UNA sola causa, con nombre y apellido — y es un defecto propio, no de este BL:**

```
404  http://localhost:5000/api/models
```

Medido: `app/api/models/` contiene **sólo** `[modelId]/` (con `equity-curve` y `metrics`); **no
existe `route.ts`** para el listado. Y sin embargo el endpoint está **declarado**
(`lib/config/models.config.ts:212 → list: '/api/models'`), tiene **entrada RBAC propia**
(`rbac.contract.ts:168`, `research:read`) y lo llaman **dos** clientes
(`contexts/ModelContext.tsx:134`, `lib/services/model.service.ts:50`). `ModelContext` incluso
degrada con `defaultModels` ante un `>= 500`, pero ante un **404 lanza**: nadie contempló que la
ruta no existiera.

Es la misma familia que el WebSocket: **una superficie declarada que no existe**. Arreglarla es
decidir si la ruta debe crearse o si los llamantes son legado a retirar — decisión de producto que
**no pertenece a BL-05**, y construir un endpoint para poner verde un test sería exactamente la
motivación equivocada.

**BL-05 sigue `PARTIAL`, con el gap reducido a una única ruta nombrada.** Toda la accesibilidad
está verificada en runtime con capturas; el aserto de consola no se relaja.

El aserto se mantiene intacto **a propósito**. La ficha ya documenta que en su día se estabilizó
esta spec dejando de *fabricar* ruido, nunca bajando el listón; relajarlo ahora para cobrar un
verde sería exactamente el falso verde que este backlog persigue.

**BL-05 sigue `PARTIAL`** — pero ya no por «no ejecutado» ni por una orden genérica: **la parte de
accesibilidad está cerrada con evidencia runtime**, y lo único pendiente es una consola limpia, que
exige `usdcop-trading-api`, publicar el puerto del WS y aplicar la migración de `user_cart`. Es
decisión de infraestructura del operador, no trabajo de código de este BL.
