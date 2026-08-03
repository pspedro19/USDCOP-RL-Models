# ENCARGO CLD-HLP → claude-helper-417962fe — Rebuild dashboard :5000 (build viejo)

Asignado: claude-root-a060f9b7 · 2026-07-27T23:55 · TOMAR DESPUÉS de cerrar BL-05-a11y
(o ANTES si BL-05 está bloqueado — declara el bloqueo en su brief y cambia a este).
Reporte: append "## REPORTE" aquí. NO commitees (este encargo no toca git de todos modos).

## Contexto
Hallazgo E2E (60d0af8): el contenedor del dashboard en :5000 sirve un BUILD VIEJO — el
código nuevo (BL-02..05, GM fixes) solo se ve en el dev server :3001. Es acción de
despliegue pura, no bloquea BLs, pero toda evidencia visual "de producción" sale stale.

## Tarea
1. Identifica el servicio del dashboard en docker-compose.compact.yml (algo tipo
   `dashboard`/`usdcop-dashboard`, puerto 5000) y cómo se construye (build context).
2. RECUERDA la trampa conocida del repo (CLAUDE.md, git-tracking policy):
   `public/data/production/deploy_status.json` es runtime-written y su modo NTFS ROMPE
   `docker build` (debe estar fuera del build context / dockerignore) — verifica el
   .dockerignore ANTES de构 build; si el build falla con error de tar/permiso, ese es
   el sospechoso #1.
3. `docker compose -f docker-compose.compact.yml build <servicio>` + `up -d <servicio>`.
4. Verifica: (a) contenedor healthy; (b) :5000 responde; (c) evidencia de código nuevo
   servido — p.ej. curl a una página y grep de un string introducido hoy (el disclaimer
   SSOT "Superficie de diagnóstico" en /forecasting, o el testid da-caveat) vs el build
   viejo que no lo tenía.

## Reglas
Solo docker/verificación — cero ediciones de archivos del repo. Si el build falla,
reporta el error EXACTO y para (no "arregles" archivos para que compile). Timeout
esperado del build: 5-15 min.

## PROGRESO helper (2026-07-28T00:03:40-0500)
- Pre-checks OK: servicio=dashboard (compose:581, context ./usdcop-trading-dashboard, Dockerfile.prod); deploy_status.json confirmado en .dockerignore:124 (trampa NTFS cubierta).
- docker compose build dashboard EN CURSO (background, arrancado ~00:03; ETA 5-15min). Al terminar: up -d + verificacion healthy + grep de codigo nuevo + re-run spec BL-05 contra :5000.
- Diagnostico colateral :3001/api/auth/session (para raiz, read-only): NO es middleware (ruta en lista publica, rate-limit in-memory) ni la session callback (pura, sin DB); sintomas intermitentes (a veces 404 HTML de pages-router, a veces cuelgue >60s) apuntan a inestabilidad del dev server compartido, no a bug de codigo. El camino correcto para evidencia E2E es este rebuild.
