# CLD-HLP-002 — Protocolo de sucesión de la raíz Claude (pre-firmado, espejo CXD-HLP-002)

1. Sucesora designada: `claude-helper-417962fe` (terminal ayudante autenticada por el
   operador, CLD-HLP-001). Ninguna otra instancia puede asumir.
2. La sucesión ocurre SOLO por una de estas vías:
   a. `HANDOFF READY` explícito de la raíz `claude-root-a060f9b7` en su LOG (acta: agentes
      en vuelo, commits pendientes de integrar, decisiones abiertas del operador, leases
      vigentes), o
   b. takeover por ausencia: heartbeat >15 min + leases Claude expirados + proceso raíz
      ausente (mismo estándar CXD-019).
3. Al asumir: la sucesora toma `instance_id` NUEVO, lo anuncia en CLAUDE-STATUS y por
   INBOX-CODEX ANTES de cualquier otra escritura, y arranca LEYENDO (INBOX-CLAUDE →
   CLAUDE-STATUS LOG completo → CONTRACTS → KNOWLEDGE → LEASES) antes de hacer.
4. Nunca dos escribiendo: la raíz vieja deja de escribir ANTES de que la nueva empiece.
5. Mientras convivan: la ayudante es manos, no voz — sin STATUS/commits/canales por
   iniciativa propia (excepción: orden directa del operador, anotada).

Firmado: claude-root-a060f9b7 · 2026-07-27T23:31:00-05:00 (reloj real)
