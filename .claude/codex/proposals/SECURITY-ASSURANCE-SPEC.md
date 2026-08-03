---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .github/workflows/security.yml
  - .github/workflows/security-scan.yml
  - scripts/validation/pentest_checklist.py
  - usdcop-trading-dashboard/middleware.ts
  - services/signalbridge_api/app
---

# Security Assurance

## Baseline normativa

- OWASP Top 10:2025: A01 acceso, A02 configuración, A03 supply chain, A04 criptografía, A05 inyección,
  A06 diseño, A07 autenticación, A08 integridad, A09 logging/alertas y A10 excepciones.
- OWASP ASVS 5.0.0 como catálogo verificable; objetivo inicial L2 para aplicación financiera.
- OWASP API Security Top 10 para BOLA/BFLA, consumo de recursos, SSRF e inventario de endpoints.
- OWASP GenAI/LLM Top 10 para chat, análisis y agentes.

## Gates requeridos

| Superficie | Control automático | Política |
|---|---|---|
| Código Python/TS | Semgrep/CodeQL + lint seguro | Critical/High bloquea |
| Dependencias | `pip-audit`, npm audit/OSV, lockfile diff | Critical/High bloquea; SLA Medium |
| Secretos | Gitleaks + baseline revisada | cualquier secreto nuevo bloquea |
| IaC/containers | Trivy/Checkov + imágenes pinneadas por digest | Critical/High bloquea |
| Web/API | ZAP baseline autenticado por rol en entorno aislado | High bloquea |
| Auth/RBAC | matriz negativa + BOLA/BFLA | cualquier escalación bloquea |
| Pagos | firma, orden server-side, monto, moneda, replay | cualquier grant incorrecto bloquea |
| IA | prompt injection, exfiltración, tool abuse, output encoding | acceso a secreto/acción no autorizada bloquea |
| Trading | llaves sin retiro, paper default, kill y límites | bypass bloquea live |

## Amenazas específicas

- Impersonación o preview que escala en lugar de reducir permisos.
- Cookies/JWT robados, fijación de sesión, revocación tardía y CSRF en mutaciones.
- IDOR en rutas con `userId`, `id`, `assetId`, modelId o strategyId.
- Inyección SQL/OS/path en BFF de archivos, rutas catch-all y comandos de deploy.
- SSRF mediante URLs de proveedor, proxies, scrapers, webhooks y callbacks.
- Zip-slip/symlinks y ejecución arbitraria mediante skills, modelos o artefactos importados.
- Poisoning de datasets/modelos, deserialización insegura y reemplazo de bundles.
- Prompt injection desde noticias/documentos; tool calls con autoridad excesiva.
- Filtración de PII, tokens, claves o posiciones en logs, traces, videos y screenshots.
- Fail-open ante caída de Redis, DB, Vault, proveedor de identidad o risk engine.

## Resultados ejecutados — 2026-07-20

`npm audit --omit=dev` reportó **50 vulnerabilidades**: 3 critical, 17 high, 27 moderate y 3 low.
Critical directas: `next` y `jspdf`; critical transitiva: `fast-xml-parser`. High directas incluyen `fabric`,
`jspdf-autotable`, `js-yaml`, `prisma` y `xlsx`; para `xlsx` npm no ofrece fix automático. Este resultado
bloquea release hasta analizar reachability, actualizar/remover dependencias y repetir build/E2E.

El checklist local de pentest produjo `CRITICAL_ISSUES`: 16 pass, 2 fail y 1 warning. Detectó construcción
dinámica de SQL y ejecución de comandos. Evidencia localizada:

- SQL dinámico en `scripts/ops/backup/feature_data_backup.py`; nombres de tabla/columnas requieren allowlist
  estricta + `psycopg.sql.Identifier`.
- `os.system` en `scripts/ops/backup_restore_system.py`.
- `shell=True` en scripts de presentación/análisis.
- `eval` con builtins vacíos en builders de fórmulas; requiere parser/AST allowlist si la expresión no es
  una constante interna congelada.
- warning por contenedores sin `read_only: true` donde sea viable.

El checklist es heurístico: cada match requiere triage, pero no debe descartarse sin evidencia.

## Seguridad de IA

El LLM nunca concede permisos, modifica entitlements, aprueba producción ni ejecuta órdenes por texto libre.
Toda acción usa herramientas allowlisted, parámetros tipados, autorización server-side y confirmación humana
para acciones financieras. El contenido recuperado se trata como datos no confiables; sus instrucciones no
pueden cambiar políticas del sistema. Las salidas se validan y codifican antes de renderizarse o persistirse.

## Excepciones

Una excepción requiere owner, riesgo residual, compensating control, fecha de expiración y aprobación. “Solo
es demo” no permite credenciales débiles, datos falsos etiquetados live ni bypasses compilados en producción.
