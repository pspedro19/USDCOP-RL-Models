---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Matriz maestra de validación pendiente

## Validado localmente

- Estadística descriptiva, frecuencia, gaps y missingness de datasets locales.
- Inventario y lectura de seeds/backups.
- Reconciliación no destructiva con hashes.
- Contratos cuantitativos y harnesses unitarios.
- RBAC: 93 rutas API y 30 páginas.
- Checkout/webhook/idempotencia mediante mocks.
- Compilación y registry básico de DAGs.
- Evidencia reproducible con logs y SHA-256.

## Parcial o bloqueado

### Modelos y rentabilidad

- PIT/vintages: bloqueado.
- OOS walk-forward por activo: bloqueado.
- DSR/PBO/trials: sin artefactos finales.
- Calmar, drawdown, costes, slippage y benchmark neto: no aprobados.
- Estabilidad por régimen y bootstrap/block-CI: pendiente.
- Reentrenamiento 2026: no listo.

### Adquisición y datos

- TwelveData, MT5, BCRP, Suameca y scraping: adaptadores presentes, ejecución
  real con credenciales no demostrada.
- Manifiestos de proveedor: faltantes.
- `usdcop_m5` y macro: discrepancias seed/backup sin explicación de negocio.
- Restore drill, backup offsite y prueba de recuperación: pendientes.
- Calendarios oficiales, feriados y timezone por activo: pendientes de evidencia.

### Noticias y scraping

- Fetch exitoso, HTTP status, retries, latencia y tasa de error por fuente.
- Dedupe por URL/hash, canonicalización y detección de contenido repetido.
- Validación de `published_at` frente a `available_at` para evitar leakage.
- Calidad de extracción, idioma, encoding, artículos vacíos y noticias futuras.
- Prueba de rate limit, robots/ToS, circuit breaker y DLQ.
- Evidencia de alerta, digest diario y recuperación ante caída de fuente.

### Autenticación, sesiones y auditoría

- Login/logout, expiración, refresh/revocación de sesión.
- Cookies `Secure`, `HttpOnly`, `SameSite`, CSRF y rotación de sesión.
- MFA si aplica, bloqueo por intentos y recuperación de cuenta.
- Logs de inicio/cierre de sesión, IP/device hash, actor, resultado y correlation ID.
- Redacción de tokens, contraseñas, PII y secretos en logs.
- Retención, acceso RBAC a logs y trazabilidad de cambios administrativos.

### Commerce/RBAC

- Sandbox real del PSP: checkout, webhook, replay, refund, chargeback y renovación.
- Suscripciones canceladas/expiradas y revocación de entitlements.
- Pruebas negativas por cada rol sobre modelos, backtests y señales.
- Impuestos, moneda, redondeo, importe firmado y conciliación contable.

### DAGs y operación

- `airflow dags list-import-errors` en scheduler real.
- Ejecución dry-run de adquisición → feature → train → backtest → promotion.
- Retry/idempotencia, sensores, DLQ, alertas y timeouts.
- Drift, freshness, latencia, errores, uptime y SLO.
- Canary/testnet de una semana y rollback real.
- No confundir workflows CI placeholder con despliegue de infraestructura.

### Evidencia visual

- Screenshots y videos autenticados del dashboard, gates, carrito, RBAC,
  ejecución DAG, aprobación, canary y rollback.
- Cada captura debe incluir timestamp, entorno, usuario/rol y referencia al log.

## Criterio de cierre

El estado solo puede cambiar a `GO` cuando cada fila bloqueada tiene artefacto
reproducible, hash, responsable, fecha de ejecución y resultado PASS. Sin ello,
la decisión correcta continúa `NO-GO`.
