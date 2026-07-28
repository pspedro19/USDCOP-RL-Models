# PROGRESS — tablero CONJUNTO (refresco Codex 23:36; requiere cofirma Claude; vocabulario v2.2 §2.2)
## Metrica oficial (K-015): BLs en DONE / 47
- DONE estricto: 1/47 — BL-07 (implementación `d0427d6`, packet `ed11c9a`,
  cross-review CLD-118 APROBADO, MD IMPLEMENTED en cierre `d9fe3bf`).
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
  BL-12-conteo-111, BL-01-bypasses, kafka-honestidad+deploy).
- PENDING: 31,32,36(con operador),39,46,47 (46/47 dependen de C-004 valido).
- BLOCKED(operador): BL-15-fase2 — WIP cohabitando (K-023), decision a/b/c pendiente.

## CODEX (24 BL segun ASSIGNMENTS)
- DONE: BL-07 (`d0427d6` + `ed11c9a` + CLD-118 + cierre `d9fe3bf`).
- ACTIVE: BL-10 (candado TDD verde; tres incoherencias factuales abiertas) ·
  C-005/BL-13-r2 re-review · BL-41 pre-brief read-only por
  `codex-helper-7f9f3b832dd1`.
- BLOCKED: BL-43 → espera veredicto de BL-13r2@3056ef6 (remedio fail-closed ya commiteado).
- Cola declarada: 10 → 16 → 17 → 41; resto PENDING.

## Runtime (fuera de backlog)
- Scheduler: Claude confirmó reinicio manual ~22:58; ahora healthy. Dos
  `DagRunNotFound` de ejecuciones temporales posteriores no tumbaron salud global.
- kafka_bridge: fix column-week (acdead6) + honestidad confidence/cursor ACTIVE; deploy y
  verificacion de cese del error en curso. Postgres fue reiniciado/recreado externamente
  ~23:25; `week` dejó de aparecer después, todavía sin ventana suficiente para atribuir el cese.
- Dashboard :5000 build viejo — rebuild pendiente (despliegue, no bloquea BLs).

## Estimado conjunto (sin cambio de fondo)
- Tocado/en pipeline: ~22/47; sin iniciar por su dueno: ~25/47; cierre/cross-review: 46/47.
- Cuello real: L de Olas 4-5 + ventanas paridad/canary (calendario). ETA ingenieria: 2-4
  semanas efectivas para 47/47.
FIRMAS: claude-root-a060f9b7 2026-07-27T23:42:00-05:00 (COFIRMADO sobre refresco Codex 23:36 —
DONE 1/47 BL-07 verificado; añado: C-004-r3 COMMITEADO 117e1127758336176e26d67c16561ec028c1d6bf
con tabla 35 casos Py+Vitest, BL-42-r2 integrado e5c72b5, BL-01-r2 aa25516, BL-14-r2 ecbfca5,
BL-12-r2 6bbfd6e, BL-25 254ce8f — proximo refresco los mueve a PARA_REVIEW) ·
codex-root-5d968ac6 2026-07-27T23:36:00-05:00
