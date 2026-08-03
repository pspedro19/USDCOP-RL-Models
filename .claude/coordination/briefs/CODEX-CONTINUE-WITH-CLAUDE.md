# Handoff — «continúa con Claude»

Actualizado por CODEX: 2026-08-03T16:35:00-05:00.

Cuando el operador diga **«continúa con Claude»**, retomar este protocolo sin pedir que repita
el contexto:

1. Leer `PROTOCOL.md`, ambos inboxes, `LEASES.md`, ambos status, `CONTRACTS.md` y este handoff.
2. Comprobar mensajes/commits nuevos y respetar todos los leases antes de escribir.
3. Coordinar por `INBOX-CLAUDE.md`; registrar heartbeat en `CODEX-STATUS.md`.
4. Implementar el siguiente incremento local verificable, pedir cross-review y no inflar DONE.

## Estado cofirmado

- Backlog: **10 IMPLEMENTED / 35 PARTIAL / 2 PLANNED = 47 (21.3%)**.
- Tablero cofirmado en `f5e46267`.
- No push: BL-08 sigue PARTIAL por acciones externas.
- No DDL ni actualización del pin FABRIC sin autorización del operador.
- Digest FABRIC actual `023ebf...` difiere del pin `b83bf...`; el rojo es esperado.

## Último trabajo

- BL-06 cerrado IMPLEMENTED: implementación `96d4c361`, cierre owner `c30bd666`.
- R3 causal + remedio Metric: `bf1e02f8`.
  - `resolve_feature_snapshot` exige `available_at <= decision_cutoff` y conserva el contrato
    `{nombre: valor}`.
  - Selección final 11P; amplia 27P/1F sólo por digest FABRIC conocido.
  - Claude tiene leases de mutación para cross-review de `feature_snapshot.py` y
    `src/metrics/engine.py`; esperar restauración/veredicto antes de tocarlos.
- BL-15 `8552d7ea` y BL-45 `0d79e59e` siguen PARTIAL honestamente.
- BL-40 `2d3ded21` no es cerrable: Banxico CF373 refuta 2.5 como mínimo histórico universal.
  Próximo diseño debe modelar ventana/proveedor moderno de forma enforced; no inventar fuente.

## Infraestructura

- Docker Desktop 29.6.2 está instalado en perfil de usuario.
- Motor Linux bloqueado hasta instalar WSL/VirtualMachinePlatform y reiniciar.
- El operador está ejecutando `wsl --install`; tras reinicio verificar `wsl --status`,
  `docker version` y `docker info`.
- Aunque Docker quede listo, BL-18 integration sigue bloqueada por digest FABRIC/DDL.
- Monitor runtime reactivado con PID 18900; si no existe tras reinicio, reactivarlo o hacer polling.

## Siguiente acción exacta

1. Leer el último `CLD-*`; obtener resultado adversarial de `bf1e02f8` y cierre de leases.
2. Si aprobado, registrar ACK; R3 sigue PARTIAL hasta wiring Airflow verificable.
3. Tras reinicio validar Docker, sin aplicar FABRIC ni cambiar pin.
4. Acordar con Claude el siguiente BL/path antes de reclamar lease.
