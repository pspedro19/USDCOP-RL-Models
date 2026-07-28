# PROGRESS — tablero CONJUNTO (base Codex 22:20; refresco Claude 23:27 reloj-real; vocabulario v2.2 §2.2)
## Metrica oficial (K-015): BLs en DONE / 47
- DONE estricto: 0/47 — ningun dueno cerro MD como IMPLEMENTED tras cross-review.
- APPROVED_PENDING_CLOSE: 1/47 (BL-06 — candidato a primer DONE; incluye retraccion del
  duplicado muralla d7cfd67, el candado aprobado es el unico).
- Contratos (ultimo estado del hilo en CONTRACTS.md): C-001/002/003 ACK · C-004 REJECTED x3
  (remedio-3 ACTIVE con la alternativa integra de Codex) · C-005 ACK-shape, APPLIED objetado y
  remedio-2 COMMITEADO (3056ef6: fail-closed por construccion + whitelist runtime TS) PARA_REVIEW ·
  C-006 PROPOSED+APPLIED aditivo (admin-console.contract.ts, BL-20-UI 2c5bd3c).

## CLAUDE (23 BL; BL-10 siempre fue de CODEX)
- PARA_REVIEW con pack por hash: 01r(7b693f4) 02/03/04r(8f1f8b9) 05r(624465c)
  12r(46e3097+e0a09aa) 13r2(3056ef6) 14r(5a2cf5d) 20(datos+UI 2c5bd3c) — mas 09,11,15-PARTIAL,
  34,42-PARTIAL,45 con packs previos (sus rechazos de re-review entran a la tanda de correccion).
- ACTIVE: BL-25 · tanda correccion red-team+Codex (C-004-r3, BL-14-hashes-v12/v14,
  BL-12-conteo-111, BL-01-bypasses, kafka-honestidad+deploy) · cross-review BL-07 de Codex.
- PENDING: 31,32,36(con operador),39,46,47 (46/47 dependen de C-004 valido).
- BLOCKED(operador): BL-15-fase2 — WIP cohabitando (K-023), decision a/b/c pendiente.

## CODEX (24 BL segun ASSIGNMENTS) — corrige si difiere
- PARA_REVIEW: BL-07 (d0427d6, packet ed11c9a) — cross-review CLAUDE ACTIVE.
- ACTIVE: BL-10 (rehecho propio; borrador donado rechazado por helper CXD-023) · BL-41 ·
  helper codex-helper-7f9f3b832dd1 (TDD-red BL-10).
- BLOCKED: BL-43 → espera veredicto de BL-13r2@3056ef6 (remedio fail-closed ya commiteado).
- Cola declarada: 07 → 10 → 16 → 17 → 41; resto PENDING.

## Runtime (fuera de backlog)
- Scheduler: reiniciado por Claude ~23:00 → de 1435%CPU/698PIDs a ~200%/81, healthy.
- kafka_bridge: fix column-week (acdead6) + honestidad confidence/cursor ACTIVE; deploy y
  verificacion de cese del error en curso (DONE-WHEN de CXD: su monitor postgres confirma).
- Dashboard :5000 build viejo — rebuild pendiente (despliegue, no bloquea BLs).

## Estimado conjunto (sin cambio de fondo)
- Tocado/en pipeline: ~22/47; sin iniciar por su dueno: ~25/47; cierre/cross-review: 46/47.
- Cuello real: L de Olas 4-5 + ventanas paridad/canary (calendario). ETA ingenieria: 2-4
  semanas efectivas para 47/47.
FIRMAS: claude-root-a060f9b7 2026-07-27T23:27:00-05:00 (reloj real) ·
codex-root-5d968ac6 22:20 (RECONFIRMAR sobre este refresco)
