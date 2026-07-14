# Admin Console — Especificación Completa

> **Estado: PRIORIDADES §10.1–§10.4 IMPLEMENTADAS (2026-07-07)** · Complementa CTR-RBAC-001
> (permisos) y `ux-navigation.md` (§3.5).
>
> Contract: CTR-ADMIN-CONSOLE-001
> - TypeScript: `usdcop-trading-dashboard/lib/contracts/admin-console.contract.ts`
> - Backend: endpoints admin de SignalBridge (`services/signalbridge_api`) + BFF `/api/admin/**`
>
> ### AS-BUILT (2026-07-07)
>
> | Pieza | Archivo |
> |---|---|
> | Contrato | `lib/contracts/admin-console.contract.ts` (secciones, `normalizeRole` C2, `planForDisplay` C3, `isTestEmail` C4, `AdminWidget<T>` C5, `categorizeAuditAction` §7, umbrales de freshness espejo de `data-freshness.md`) |
> | Migración | `database/migrations/056_admin_console_is_test.sql` — `sb_users.is_test` + backfill heurístico + **trigger BEFORE INSERT** (altas nuevas se auto-clasifican) |
> | BFF | `app/api/admin/{kpis,queue,users/list,audit,system}/route.ts` + `users/[id]/flag-test` (auditado) · guard DRY en `lib/admin/guard.ts` · predicado de cola único en `lib/admin/queue-sql.ts` (C1) |
> | UI | `app/admin/page.tsx` (shell con pestañas) + `components/admin/{OverviewSection,QueueSection,UsersSection,SystemSection,AuditSection,primitives,useAdminWidget}` |
>
> **Decisiones de diseño (desviaciones justificadas):**
> - **C1**: la cola SE LEE de la DB (misma Postgres que escribe SignalBridge) — contador y tabla
>   comparten `PENDING_QUEUE_WHERE`; las ACCIONES approve/reject siguen retransmitiendo el bearer
>   del admin a SignalBridge (única autoridad de transición + temp password + email).
> - **C4/auditoría**: `is_test` NO se agrega a `audit_log` (su trigger append-only de la 055
>   impide el backfill); la vista de auditoría deriva el flag con `LEFT JOIN sb_users` — el
>   ledger no se muta jamás.
> - **§6.1**: la card Vote 2 es contador + deep-link a `/dashboard` (una sola superficie de
>   aprobación); nunca un segundo botón de Aprobar.
> - Los `data-testid` del flujo de aprobación (`approval-queue`, `pending-row-*`, `approve-*`,
>   `admin-flash`) se preservaron; `scripts/registration-qa.mjs` ahora clickea
>   `admin-tab-registros` antes de localizar la cola.
>
> **Pendiente (por fase):** Suscripciones (§5, Fase 6 billing) · Configuración (§8) ·
> Comunicaciones (§9) · ficha de usuario drawer con editor de entitlements + "ver como" (§4,
> siguiente incremento) · alertas push (§7 reglas de notificación) · `registration_mode` auto (§3).
>
> Principio: la consola admin responde dos preguntas en todo momento —
> **"¿cómo va el negocio?"** y **"¿está sano el sistema?"** — y permite ejecutar las
> acciones privilegiadas del producto (aprobar usuarios, Vote 2 de modelos, kill global,
> entitlements) **siempre con auditoría y confirmación**.

---

## 0. Correcciones inmediatas al estado actual (antes de agregar nada)

| # | Hallazgo observado | Corrección |
|---|---|---|
| C1 | Cola dice 0 con 4 usuarios `pending` visibles | Contador y tabla leen del mismo query/endpoint. Si la cola excluye tests, lo dice: "0 (4 test ocultos)" |
| C2 | Roles `user`/`free`/`subscriber` mezclados | Migración a enum del contrato `{admin, developer, subscriber, free}`; el enum se importa de `rbac.contract.ts`, nunca strings sueltos |
| C3 | Staff con plan (`admin` plan=free) | `plan` es NULL para staff; UI muestra "—"; las APIs ignoran entitlements si rol ∈ staff |
| C4 | Tráfico de test contamina auditoría y métricas | Flag `is_test` en users y audit_log (heurística: dominios @test.com/@example.com + flag manual). Toda métrica de negocio filtra `is_test=false`; la auditoría lo muestra con toggle |
| C5 | Error "Sesión de SignalBridge no encontrada" bloquea el panel | Auth unificada (misma JWT, §3 CTR-RBAC-001). Cada widget degrada independiente: su error no tumba el resto |
| C6 | Dos cuentas admin | Un humano = una cuenta admin nominal. Cuentas de servicio marcadas `service` con permisos mínimos |
| C7 | `Vence` sin poblar | Nace con la Fase 6 (billing) junto con el job diario de degradación |

---

## 1. Estructura del portal (8 secciones)

```
/admin
├── Overview          ← home: negocio + sistema en una pantalla
├── Registros         ← cola de aprobación de usuarios
├── Usuarios          ← tabla + ficha detalle
├── Suscripciones     ← billing, webhooks, dunning        (Fase 6)
├── Sistema           ← Vote 2 · bridge global · salud de datos · señal semanal
├── Auditoría         ← filtros, alertas, export
├── Configuración     ← flags, parámetros, legales versionados
└── Comunicaciones    ← emails, anuncios                   (fase 2)
```

---

## 2. Overview (la pantalla que abres cada mañana)

**Fila 1 — Negocio (filtrado `is_test=false`):**

| KPI | Definición exacta |
|---|---|
| Usuarios totales / nuevos 7d | count aprobados; delta semanal |
| Activos 7d / 30d | usuarios con ≥1 sesión en la ventana |
| MRR | Σ precio de suscripciones activas (proveedor = fuente de verdad) |
| Conversión free→pago 30d | primeros pagos / registros aprobados de la cohorte |
| Churn mensual | cancelaciones del mes / activos al inicio del mes |
| Pendientes | cola de registro + pagos fallidos sin resolver |

**Fila 2 — Sistema:**
semáforo de frescura de datos por fuente (el de `data-freshness.md`, elevado a UI) ·
estado del bridge (system + nº usuarios con ejecución activa) · señal de la semana
(estado + difusión: recibieron / ejecutaron) · última corrida de pipelines · errores 24h.

**Fila 3 — Alertas accionables** (cada una con botón que lleva a resolverla):
cola > 0 con SLA vencido · pagos fallidos · `*_denied` repetidos (§7) · datos en ámbar/rojo
· usuarios a punto de pasar paper→live esta semana.

Regla: cada número del Overview clica hacia su detalle. Un KPI que no lleva a ningún lado
es decoración.

---

## 3. Registros (cola de aprobación)

**Política — la aprobación manual es un modo, no la arquitectura:**
flag `registration_mode`:
- `manual` (beta actual): todo registro entra a la cola.
- `auto` (lanzamiento): free se auto-aprueba con **verificación de email**; a la cola solo
  van los señalados por heurísticas (dominio desechable, velocity por IP/dispositivo,
  patrón de email) y las solicitudes de rol `developer` (siempre manual).

**Dos estados independientes, nunca mezclados:** `email_verified` (lo controla el usuario)
y `approval_status` (lo controla el admin). Un usuario puede estar verificado y no
aprobado, o al revés; la tabla muestra ambos.

**Item de cola:** email · fecha · fuente de registro · señales de riesgo · acciones:
**Aprobar** / **Rechazar (motivo obligatorio)** / **Marcar como test**. Acciones en bulk.
SLA visible ("esperando hace 3h"). Notificación al admin en cada alta nueva (email o
Telegram) con approve de un clic. Todo → `audit_log`.

---

## 4. Usuarios

**Tabla:** búsqueda + filtros (rol, plan, estado, `is_test`, vence) + columnas que hoy
faltan: **último acceso** y **modo de ejecución** (—/paper/live). Orden por defecto: alta
descendente.

**Ficha de usuario (drawer):**
- Identidad y estado (verificado, aprobado, suspendido) + sesiones activas con botón
  "cerrar todas".
- **Rol** (cambio con confirmación tipada + audit) y **entitlements**: editor con presets
  por plan + validación contra el schema §1.2 del contrato; edición libre del JSON solo
  tras confirmación explícita.
- **SignalBridge del usuario:** modo paper/live, semanas de paper cumplidas, límites
  vigentes, su kill switch (visible; operable por admin solo con confirmación — es
  intervenir la cuenta de un cliente y queda auditado), últimas ejecuciones.
- Pagos del usuario (Fase 6) y su historial de auditoría filtrado.
- Acciones: extender vencimiento / cortesía · suspender · **eliminar (soft delete + purga
  de llaves de exchange inmediata)**.
- **"Ver como" (impersonación read-only):** banner permanente "Viendo como X — solo
  lectura", cero acciones habilitadas, entrada en auditoría con motivo. Es la herramienta
  de soporte más útil que existe y la más peligrosa si no es read-only.

---

## 5. Suscripciones y facturación (nace con Fase 6)

Suscripciones: plan · estado (`activa/past_due/cancelada`) · próximo cobro · proveedor.
**Log de webhooks del proveedor** con estado procesado/fallido y botón **re-procesar**
(con Wompi/PayU esto no es opcional: los webhooks fallan y el estado del cliente no puede
depender de un retry manual en la DB). Dunning: pagos fallidos, reintentos, aviso al
usuario. Cupones/cortesías con vencimiento automático. KPIs: MRR por plan, ARPU, upgrades
signals→auto, add-ons por activo.

---

## 6. Sistema (lo que hace admin a ESTE producto y no a un SaaS genérico)

**6.1 Aprobaciones de modelo (Vote 2).** La cola de promociones pendientes (hoy: Smart
Simple v1.1 `PENDING_APPROVAL`). Cada item muestra: diff de versión (parámetros que
cambian) · métricas del bundle candidato vs activo, con badges ●◆○ · resultado de gates ·
trades del período. Acciones **Aprobar / Rechazar** con confirmación que repite en texto
qué se está promoviendo. Registro en auditoría con snapshot del bundle. *(Si la acción ya
vive en /experimentos, aquí va el contador + deep-link; nunca dos botones que aprueban lo
mismo desde dos lugares.)*

**6.2 SignalBridge global.** Cuenta system (estado, exchanges, últimas ejecuciones) ·
**kill global** con doble confirmación tipada ("STOP") y motivo · vista agregada de
ejecución de clientes (por defecto anonimizada; el detalle nominal requiere clic que se
audita) · pipeline paper→live: quiénes cumplen las semanas esta semana (el paso a live lo
activa el usuario; el admin solo lo ve venir).

**6.3 Salud de datos.** El semáforo por fuente con última actualización, umbral y acción
("señales pausadas por seguridad" cuando aplica) — espejo de la franja que ve el cliente.

**6.4 Señal de la semana.** Estado (sin operación / esperando entrada / en posición /
cerrada) · difusión: usuarios elegibles, notificados, ejecutados (auto) · divergencia
promedio ejecución vs señal.

---

## 7. Auditoría (mejoras sobre lo existente)

- **Filtros:** acción, categoría, usuario, objeto, rango de fechas. Export CSV.
- **Resolución de identidad:** mostrar email (hover para UUID completo); `c172f00c…` no es
  legible para un humano a las 2 am.
- **Categorías con severidad:** `security` (denials, logins admin, cambios de rol) ·
  `execution` (kills, límites, live-enable) · `billing` · `governance` (Vote 2, promote).
- **Reglas de alerta mínimas:** ≥3 `*_denied` del mismo usuario en 60 min → notificación ·
  login admin desde IP nueva → notificación · cambio de rol a admin → notificación
  inmediata siempre.
- **Toggle `is_test`** por defecto oculto; retención definida (p.ej. 24 meses) y export
  antes de purga.
- Sigue siendo append-only: la UI no ofrece editar ni borrar, ni para admin.

---

## 8. Configuración

Flags de producto (`registration_mode`, tiers activos, activos en venta) · parámetros de
negocio con historial de cambios (`paper_weeks_required`, techos de riesgo — cambiarlos es
acción auditada) · textos legales **versionados** (qué versión de ToS aceptó cada usuario
ya vive en audit_log; aquí se publican versiones nuevas) · modo mantenimiento con mensaje.

## 9. Comunicaciones (fase 2)

Plantillas transaccionales (verificación, aprobación, pago fallido, señal semanal para el
plan signals) · anuncios in-app · estado de envíos. No bloquea nada de lo anterior.

---

## 10. Prioridad de construcción (según el estado actual: 16 usuarios, pre-billing)

1. **C1–C6** (correcciones): taxonomía de roles desde el contrato, `is_test`, staff sin
   plan, widgets independientes, reconciliación cola↔tabla.
2. **Ficha de usuario** con editor de entitlements + cambio de rol auditado — es la
   herramienta que usarás a diario en beta.
3. **Auditoría v2**: filtros + resolución de email + las 3 alertas de seguridad.
4. **Sistema §6.1 y §6.3**: contador/enlace de Vote 2 y semáforo de datos en Overview.
5. **Registros** con `registration_mode` (manual ahora, auto listo para lanzamiento).
6. **Suscripciones (§5)** cuando exista la Fase 6 de billing — no antes.
7. Overview completo de KPIs de negocio cuando haya usuarios reales que medir.

---

*Regla final: en esta consola no existe ninguna acción privilegiada sin (a) confirmación
proporcional al daño y (b) fila en audit_log. Si una acción nueva no puede cumplir ambas,
no se agrega.*

---

## As-built increment (2026-07-11) — Sistema observability + admin mutations

- **Sistema tab wired ON**: `GET /api/admin/system` now calls the (previously dormant) `readSlos`/`readPromTargets`/`readActiveAlerts`/`readPipeline`/`readResources` under the `settle()` fault-tolerant pattern → `SystemSection` renders SLO p50/p95/p99, Prometheus target health, active AlertManager alerts, L0→L6 DAG pipeline, CPU/mem. Unreachable source degrades to `partial_errors` (no blank console). Obs deep-links derive from `NEXT_PUBLIC_*_URL` (dead localhost links hidden).
- **New audited endpoints** (all INSERT `audit_log`, typed-confirm): `POST /api/admin/system/recover` (whitelisted Airflow DAG trigger), `POST /api/admin/risk/keys/[id]` (approve/reject pending exchange keys — approve fail-closed unless withdraw-disabled confirmed), `PATCH /api/admin/users/[id]` (role/plan/entitlements editor — validates `PlanId`/`Role`, writes canonical `PLAN_DEFAULTS[plan]` to `sb_users.entitlements`).
