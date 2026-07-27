---
kind: roadmap
status: PLANNED
version: 1.0.0
supersedes: []
last_verified: 2026-07-27
code_anchors:
  - .claude/rules/rbac.md
  - .claude/rules/approval-gates.md
  - services/signalbridge_api/app/services/pretrade.py
  - src/risk/
---

# PLAN — Institutional Readiness: qué tendrías al completar el plan (y qué NO)

> Cuarta pieza de `planes/`. Evaluación honesta del techo alcanzable: **con el plan
> completamente implementado, probado y operando como fue diseñado, tendrías una
> plataforma cuantitativa y de trading de grado institucional. Pero NO podrías afirmar,
> solo por eso, que tienes un hedge fund completo de grado institucional.**

Un hedge fund institucional no es únicamente el software:

```text
Plataforma cuantitativa
+ gestión de riesgo
+ ejecución
+ gobierno corporativo
+ cumplimiento regulatorio
+ custodia y administración
+ contabilidad y valoración
+ ciberseguridad
+ continuidad operativa
+ personal y segregación de funciones
+ relación con inversionistas
```

Los planes 00-02 cubren muy bien la primera mitad: investigación, datos, estrategias,
linaje, capital, ejecución y control técnico.

## 1. Qué SÍ tendrías

### Plataforma de investigación institucional: SÍ

Datos point-in-time · snapshots versionados · linaje proveedor→PnL · registro global de
hipótesis y trials · DSR por familia/clúster/global · separación investigación-producción
· estrategias congeladas y reproducibles · paper/canary/champion/retiro · bundles
inmutables · paridad replay-paper-live · forecasting separado de las decisiones ·
backfills sin sesgo de supervivencia · Passport derivado de hechos. **Claramente por
encima de un proyecto personal o un backtest convencional.**

### Plataforma de trading institucional: SÍ, condicionalmente

Si se implementa realmente: ejecución fuera de Airflow · órdenes event-sourced ·
`idempotency_key` · pre-trade risk bloqueante · reconciliación pre-operación y EOD ·
kill switch independiente · portfolio snapshots coherentes · allocator con optimización
restringida · retiro `EXIT_ONLY` · auditoría de overrides · separación paper/canary/live.
La muralla de los planes 00-01 es central: **solo `strategy_output` alcanza allocator y
executor; las predicciones diagnósticas no tocan capital.**

### Sistema operativo de un hedge fund: PARCIALMENTE

El núcleo tecnológico sí; falta el **operating model empresarial** alrededor.

## 2. La diferencia fundamental: Caso A vs Caso B

**Caso A — operar únicamente capital propio**: tras implementar y superar pruebas
operativas, se puede describir como **"sistema cuantitativo propietario de grado
institucional"**. No exige replicar la infraestructura administrativa de un fondo con
terceros, aunque siguen siendo necesarios seguridad, continuidad, control de riesgo,
contabilidad y disciplina operacional.

**Caso B — administrar capital de terceros**: la plataforma NO basta. Se necesita
además: vehículo legal · administrador y estructura regulatoria · custodio/prime broker
· contabilidad independiente · NAV y valoración · auditor externo · documentos de oferta
· KYC/AML · gestión de conflictos · compliance officer · reportes a inversionistas y
reguladores · políticas de asignación · gestión de liquidez y redenciones · segregación
de activos y funciones.

Referencias regulatorias (verificar vigencia al activar Caso B): EE.UU. — asesores
registrados deben adoptar políticas escritas con revisión anual y función de
cumplimiento (SEC, Compliance Programs Rule); ciertos asesores de fondos privados tienen
obligaciones de Form PF (fecha de cumplimiento de modificaciones actualmente
**2026-10-01**, sujeta a propuestas posteriores); futuros/swaps/opciones vía vehículo
colectivo pueden activar registro como commodity pool operator (CFTC). Colombia — la
estructura NO se asume automáticamente como Fondo de Capital Privado; depende de
activos, liquidez, tipo de inversionista y forma de captación (Superfinanciera enfatiza
capacidad técnica/administrativa/tecnológica, gestión de riesgos, conflictos e
información de portafolio a inversionistas). Ver también `rbac.md` regla 9: gate legal
SFC antes de `auto` con dinero real de terceros — hasta entonces paper-only.

## 3. Qué falta para que sea institucional de verdad

### 3.1 Gobierno independiente

Hoy el sistema concentra en una sola persona: investigador, desarrollador, operador,
responsable de riesgo, aprobador y administrador de infraestructura. Aunque los
controles estén automatizados, existe riesgo de concentración de funciones. Estructura
mínima (conceptual; en organización pequeña una persona cubre varias, pero **los
permisos y aprobaciones críticas no deben depender de una sola identidad**):

```text
Research · Portfolio management · Risk · Execution · Compliance
· Operations/reconciliation · Technology/security
```

### 3.2 Compliance y gobierno corporativo

Manual de cumplimiento · código de ética · política de conflictos · operaciones
personales · best execution · errores de trading · asignación entre cuentas ·
conservación de registros · registro de excepciones · revisión periódica independiente.
**La existencia del software no reemplaza estas políticas.**

### 3.3 Ciberseguridad y continuidad (programa completo, no piezas)

MFA · gestión de secretos · rotación de credenciales · separación de redes · backups
cifrados · recuperación en otra región · escaneo de vulnerabilidades · gestión de
dependencias · incident response · business continuity · disaster recovery · pruebas
periódicas de restauración · RTO/RPO definidos · gestión de proveedores. Marco de
referencia: NIST CSF 2.0 (governar/identificar/proteger/detectar/responder/recuperar).

### 3.4 Contabilidad y valoración independientes

`fact_pnl` no reemplaza: libro contable · NAV oficial · valoración independiente ·
conciliación de caja · devengos · fees de gestión/desempeño · high-water mark · clases
de participaciones · estados financieros · confirmaciones del custodio. **El PnL
operativo y el NAV legal se reconcilian, pero no son la misma cosa.**

### 3.5 Riesgo de contraparte y liquidez

Exposición por broker/banco · cash disponible · margin calls · haircuts · concentración
de garantías · liquidez de cierre · participación máxima en volumen · gaps · mercado
cerrado · riesgo cambiario del fondo · financiación · **capacidad real de la estrategia**
(rentable con poco capital ≠ rentable al crecer: spread, impacto, liquidez).

### 3.6 Operaciones con inversionistas (solo Caso B)

Suscripciones/redenciones · registro de inversionistas · KYC/AML · side letters ·
restricciones por inversionista · reportes periódicos · cálculo de fees · auditoría de
performance · comunicación de drawdowns · gates de redención. Nada de esto vive en el
Quant Control Plane — su objeto es la estrategia, no el pasivo del fondo.

## 4. Cómo DEMOSTRAR el grado institucional (evidencia, no checkboxes)

**Técnica**: 100% señales con `decision_fingerprint` · 100% órdenes con
`idempotency_key` · 100% fills conciliados · 0 órdenes duplicadas · 0 posiciones
huérfanas · 100% del PnL con linaje hasta raw_snapshot · restauración de backups
demostrada · failover probado · kill switch probado · retiro EXIT_ONLY probado.

**Cuantitativa**: trials correctamente cobrados · DSR family/cluster/global publicado ·
paridad semántica replay-paper-live · canary con mínimo de decisiones y fills · costos
reales dentro del rango modelado · sin reanclaje de jueces · baselines conservados ·
candidatas y retiradas incluidas · allocator shadow supera al baseline neto de costos.

**Operativa**: runbooks ejecutados POR OTRA PERSONA · reconciliaciones diarias firmadas
· incidentes registrados y cerrados · pruebas de pérdida de: proveedor de datos, broker,
Airflow, PostgreSQL, conectividad · recuperación desde backup.

**Organizacional**: segregación de permisos · aprobaciones de cuatro ojos · revisión
independiente de código, seguridad, legal, contable, compliance · auditoría de
resultados.

## 5. Evaluación por niveles

| Nivel | Después de implementar el plan |
|---|---|
| Backtesting profesional | Sí |
| Research quant institucional | Sí |
| MLOps/StrategyOps institucional | Sí |
| Trading system institucional | Sí, tras pruebas live y resiliencia |
| Gestor propietario sistemático | Sí |
| Hedge fund listo para administrar terceros | **Todavía no** |
| Hedge fund institucional completo | Solo al sumar estructura legal, compliance, administración, custodia, auditoría y equipo |

## 6. Veredicto final

> **El plan te puede dar el motor, el sistema nervioso y la caja negra de un hedge fund
> institucional. No crea por sí solo la entidad completa.**

Con todo implementado y probado: **"Institutional-grade systematic investment and
trading platform"**. Para "Institutional-grade hedge fund" falta la envolvente
empresarial, regulatoria y fiduciaria — y demostrar que los controles sobreviven
operaciones reales durante un período suficiente.

**El siguiente hito NO es agregar más modelos: es crear una Institutional Readiness
Matrix** con evidencias verificables para tecnología, riesgo, ejecución, seguridad,
compliance, operaciones e inversionistas.

## 7. Mapeo as-built (2026-07-27) — qué existe HOY contra esta vara

| Pieza | Estado real verificado |
|---|---|
| Kill switch + audit trail | ✅ (`ResetKillSwitchCommand(confirmed=True)`, audit_log append-only con trigger) |
| Pre-trade risk bloqueante paper-first | ✅ `PreTradeGate` fail-safe (error ⇒ BLOCK), default PAPER |
| RBAC deny-by-default server-side + rol≠plan | ✅ (CTR-RBAC-001, middleware, cobertura en CI) |
| Secretos | parcial: Vault AES-256-GCM para llaves de exchange ✅ · **incidente abierto: `.env` real en historial público de git (rotar+purgar+privatizar ANTES de cualquier push)** · sin MFA · sin rotación programada |
| Backup/restore probado | ✅ en frío (tarea #24) · ❌ failover cross-región, RTO/RPO no definidos |
| Doble voto + revalidación server-side | ✅ (Vote 1 gates + Vote 2 humano + L4b guard) |
| Segregación de funciones | ❌ una sola identidad opera todo — el mayor gap del Caso A |
| Contabilidad/NAV independiente | ❌ (solo fact_pnl operativo) |
| Gate legal terceros | ✅ declarado (rbac.md regla 9: SFC antes de `auto`, paper-only hasta entonces) |
| Runbooks ejecutables por otro | parcial (specs/operations existen; nunca ejecutados por un tercero) |

**Lectura honesta**: el repo está hoy en **Caso A en construcción** — motor y sistema
nervioso avanzados, con dos deudas de seguridad concretas y nombradas (leak de `.env`
en historial; identidad única) que son exactamente el tipo de ítem que la Readiness
Matrix debe rastrear con evidencia, no con intención.
