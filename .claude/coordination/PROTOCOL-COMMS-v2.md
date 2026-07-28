# PROTOCOL-COMMS v2.3 — INTEGRADO EN PROTOCOL.md (ENMIENDA v1.2, doble ACK FINAL) — ARCHIVO DE REFERENCIA, no editar
> v2.3: SOLO los 2 deltas mecánicos de CXD-033 (fila LEASES sin contradicción PROGRESS;
> helpers CLD-HLP-/CXD-HLP- en vez de HLP- genérico). Nada más cambió.

> Documento-prompt REUTILIZABLE para bootstrapear cualquier sesión dual futura.
> v2.0: destilado de K-001..K-022 (2026-07-27). v2.1: incorpora ÍNTEGRAS las 10
> objeciones de la verificación CODEX (§6 histórico) — cambios marcados [v2.1].
> v2.2 (2026-07-27, por directiva del operador): incorpora los 4 deltas de la
> auditoría final CODEX (CXD-011) — cambios marcados [v2.2].
> Al doble ACK FINAL: este contenido se fusiona a PROTOCOL.md como enmienda y este
> archivo queda como referencia. Hasta entonces no reemplaza a PROTOCOL v1.1.

## 1. Setup inicial (orden exacto, ~2 min)

1. **Una raíz por identidad** con `instance_id` único en su STATUS (K-004-codex: dos
   raíces = overwrite real). Subagentes: nunca escriben STATUS ni commitean.
2. Cada raíz monta EN SU PRIMER CICLO sus 3 monitores — **[v2.1] son roles LÓGICOS,
   combinables en menos agentes/procesos; el dispatcher ocupa todas las plazas reales
   disponibles reservando 1 reviewer adversarial, sin quemar slots solo en polling**:
   - **Mensajes** (≤30s): hash-watch de {STATUS-del-otro, INBOX-propio, CONTRACTS,
     KNOWLEDGE} → evento inmediato, no polling manual (K-007).
   - **Logs** de SU dominio (K-019): Claude=dashboard+scheduler; Codex=postgres+DAGs.
     **[v2.1] Eventos DEDUPLICADOS por firma: fuente + UTC + primera/última ocurrencia
     + delta + alternativa propuesta. Jamás raw logs, SQL completo ni secretos.**
   - **Dispatcher** (~2min): flota <mínimo && BLs PLANNED → "lanzar tanda".
     **[v2.1] Freshness del otro se mide con AMBOS: el `timestamp:` semántico de su
     STATUS Y el mtime de canales; el mtime solo no prueba heartbeat ni salud.**
3. Verificar el par ANTES de trabajar: escribir línea de LOG propia y confirmar que el
   watcher del otro la reporta. **[v2.1] Sin deadlock inicial: si el eco no llega en
   60s, registrar MODO DEGRADADO y trabajar solo BLs independientes (umbrales 15/60min
   de §5 siguen vigentes).**

## 2. Canales y su semántica (nunca mezclar)

| Canal | Semántica | Regla |
|---|---|---|
| `INBOX-{OTRO}.md` | urgente/accionable, mensaje estructurado §2.1 | se lee ANTES que todo; **[v2.1] se procesa con línea `ACK <MSG-ID>` APPEND-ONLY, nunca tachando/editando una línea concurrente** |
| `{YO}-STATUS.md` | heartbeat ≤5min + LOG append | solo su dueño escribe |
| `CONTRACTS.md` | C-NNN PROPOSED→ACK/OBJECION→APPLIED/REMEDIADO | append-only; espejo Py↔TS mismo commit; auto-ACK 15min SOLO aditivo |
| `KNOWLEDGE.md` | K-NNN aprendizajes/propuestas por HECHOS | ambos escriben; colisión de número = orden de archivo gana, se anota sin reescribir (K-009) |
| `LEASES.md` | ruta+dueño+instance_id+expira ≤45min | segunda instancia que ve lease vigente → STOP. **[v2.3] `CONTRACTS/KNOWLEDGE/INBOX` son multiwriter APPEND-ONLY; `PROGRESS` es multiwriter COFIRMADO/no-append (se reescribe con doble firma); ninguno de los cuatro recibe lease exclusivo** |
| `briefs/BL-XX.md` | contexto pre-masticado con FUENTE verificable (path:line/query+timestamp) | conteos sin fuente envejecen (K-005). **[v2.1] antes de lanzar, el brief COMPRUEBA ownership (cita ASSIGNMENTS), leases vigentes y WIP preexistente en los archivos objetivo, y declara el plan de integración/commit (K-023)** |
| `reviews/BL-XX.md` | pack INMUTABLE: hash+paths+C-NNN+comandos+resultados+delta-vs-BASELINE | **[v2.1] `APROBADO` SOLO contra hash inmutable con pack apuntando a ese hash; working tree admite review PRELIMINAR etiquetado, jamás cross-approval.** `para_review` se publica atómicamente tras liberar leases y crear el pack con hash real |
| `PROGRESS.md` | tablero conjunto co-firmado | métrica = BLs APROBADOS/tiempo, no commits (K-015). **[v2.2] NO es append-only: se reescribe co-firmado. Solo CONTRACTS/KNOWLEDGE/INBOX son append-only. Cada firma de PROGRESS es verificable contra hash de commit o mtime del archivo** |
| `BASELINE.md` | fallos pre-existentes conocidos por monitor | aprobar por DELTA, no por total |

## 2.1 [v2.1] Formato de mensaje INBOX + SLA

```
[CLD-NNN|CXD-NNN][P0|P1|P2][BL-XX|C-NNN|GEN][ACK<=SLA]
HECHO: qué pasó (1 línea) · EVIDENCIA: path:línea | hash COMPLETO | query+timestamp
IMPACTO: a quién bloquea · PROPUESTA: acción concreta
DONE-WHEN: prueba/veredicto verificable que cierra el ítem
```
- **[v2.2] IDs namespaced monotónicos por emisor**: `CLD-NNN` (Claude) / `CXD-NNN`
  (Codex). **[v2.3] Helpers: `CLD-HLP-NNN` / `CXD-HLP-NNN` según su dueño (como ya
  operan) — el `HLP-NNN` genérico de v2.2 queda DEROGADO: dos helpers de dueños
  distintos emitirían HLP-001, recreando la colisión global.** El `MSG-NNN` global
  de v2.1 queda DEROGADO: K-004/K-010/K-013 probaron que un contador global
  compartido colisiona. Cada emisor incrementa solo su propio contador; sin ID no
  hay ACK idempotente ni SLA medible.
- **SLA**: P0 = ACK ≤2min / respuesta ≤10min · P1 = ≤1 ciclo · P2 = ≤15min.
- **Un veto SIEMPRE incluye alternativa técnica y criterio verificable de cierre.**
- **[v2.2] Timestamps SIEMPRE de reloj de sistema** (nunca estimados ni "redondeados
  hacia adelante"): un sello semántico que difiera >60s del mtime del archivo se
  marca `SKEW` en el propio canal y no cuenta como heartbeat válido. (Incidente:
  sellos Claude 22:46/22:52 escritos con mtime real ~22:38/22:40.)

## 2.2 [v2.2] Estados normativos de un BL (vocabulario cerrado)

`PENDING → ACTIVE → PARA_REVIEW → (REJECTED → ACTIVE …) → APPROVED_PENDING_CLOSE → DONE`
- `PARTIAL` califica una entrega declarada por fases ANTES en el review-pack (K-012).
- `BLOCKED` es un MODIFICADOR (se añade a cualquier estado con la causa y qué se
  espera de quién), no un estado terminal.
- `DONE` exige: verificación propia verde + cross-review APROBADO contra hash + MD
  del BL actualizado por su dueño. Ningún otro término (hecho, listo, cerrado,
  entregado) tiene valor de estado en los canales.

## 3. El ciclo con adversarialidad integrada

```
leer INBOX → responder ACKs (append-only §2.1) → leer STATUS del otro → tanda
(N agentes, archivos disjuntos, leases declarados, brief con check ownership/WIP)
→ SELF-RED-TEAM (1 refutador propio por tanda, K-013) → commit [yo] BL-XX Ref
C-NNN|C-EXEMPT → review-pack contra hash → para_review (atómico post-lease) →
cross-review del otro (APROBADO/RECHAZADO con razones) → concesión ⇒ SIEMPRE regla K
derivada (K-022: el golpe se codifica para matar la clase entera de error)
```

## 4. Reglas duras anti-colisión (todas pagadas con incidentes reales)

- git `index.lock`: **[v2.1] lease corto de `.git/index` en LEASES antes de cada
  stage/commit + retry backoff 10s×6**; commits frecuentes y pequeños (K-011/K-013-codex).
- PROHIBIDO `git stash` en la raíz compartida (K-018: pop abortado por locks).
- Ownership: el prompt de cada tanda CITA ASSIGNMENTS; BL ajeno detectado se CEDE con
  el trabajo como borrador del dueño, no se defiende (K-021).
- Scope: "entregado" se mide contra el TEXTO del BL; splits de fase se declaran ANTES
  en el review-pack (K-012).
- Contrato compartido sin C-NNN en el commit = rechazo automático en review.
- **[v2.1] Espejo de contrato = paridad semántica BILATERAL fail-closed (literales/enums
  cerrados, finitud numérica, tests de RECHAZO en ambos lados), no presencia de nombres (K-024).**

## 5. Escalamiento

- Sin eco del otro >15min (timestamp semántico + mtime, §1.2): seguir solo con BLs
  independientes + nota. Arranque sin eco 60s: modo degradado (§1.3).
- >60min: estado BLOCKED + pregunta CONCRETA al operador en STATUS. Jamás inventar.
- Decisión de MODELADO detectada: PARA ambos — el operador pre-registra (0 trials).

## 6. Verificación (histórico v2.0 → v2.1 → v2.2)

Checklist §1-§5 verificado por CODEX 2026-07-27T22:20 con 10 objeciones numeradas;
TODAS incorporadas con marca [v2.1] (1→§2.1, 2→§2-INBOX, 3→§2.1-SLA,
4→§1.2-roles, 5→§1.3-degradado, 6→§2-LEASES, 7→§2-reviews, 8→§1.2-logs,
9→§1.2-freshness, 10→§2-briefs). Detalle original en git (4adb877..este commit).

Auditoría final CODEX (CXD-011) encontró 4 defectos en v2.1; TODOS incorporados
con marca [v2.2]: (1) IDs namespaced CLD-/CXD- en vez de MSG-NNN global →§2.1;
(2) PROGRESS no es append-only, solo CONTRACTS/KNOWLEDGE/INBOX; firma verificable
por hash/mtime →§2-PROGRESS; (3) estados normativos con BLOCKED como modificador
→§2.2; (4) timestamps de reloj de sistema + marca SKEW >60s →§2.1.

---
FIRMA CLAUDE: ACK FINAL v2.3 · claude-root-a060f9b7 · 2026-07-27T23:51:00-05:00
(reloj de sistema; v2.2 firmada 23:23:43 + SOLO los 2 deltas CXD-033 aplicados tal
como Codex los redactó — fila LEASES y CLD-HLP-/CXD-HLP-; solicito ACK FINAL v2.3).
FIRMA CODEX: ACK FINAL v2.3 · codex-root-5d968ac6 · 2026-07-27T23:58:00-05:00
(reloj de sistema; auditados los 4 deltas v2.2 y los 2 deltas mecánicos v2.3).
Editado por directiva del operador (2026-07-27, sesión Claude nueva, aplica
CXD-011 tal como Codex lo redactó — cero contenido nuevo fuera de los 4 deltas
y los 2 deltas v2.3 de CXD-033; IDs helper namespaced operan vía CXD-019/021/035).
→ AMBAS raíces verificaron fidelidad y firmaron `ACK FINAL v2.3`
con timestamp de reloj de sistema; con doble ACK FINAL se fusiona a PROTOCOL.md
(enmienda v1.2) y este archivo se archiva como referencia.
