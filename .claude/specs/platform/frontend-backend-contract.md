# Contrato Frontend ↔ Backend — GlobalMarkets Terminal

> **Responsabilidad**: fuente autoritativa de cómo el dashboard (Next.js) se conecta con el
> backend (microservicios + TimescaleDB + Airflow + stack de observabilidad). Define la
> arquitectura de conexión, las convenciones/mejores prácticas y **cada endpoint** que consume
> el rediseño (Var A / Var B), incluyendo los nuevos de **catálogo · watchlist · carrito**.
>
> - Contrato: `CTR-FE-BE-001`
> - Versión: `1.0.0`
> - Fecha: 2026-07-10
> - Alcance: dashboard `usdcop-trading-dashboard` ↔ servicios de la plataforma
> - Relacionados: `CTR-RBAC-001`, `CTR-STRAT-REGISTRY-001`, `CTR-OBS-001`, `CTR-DQ-OPS-001`
> - SSOT de tipos: `usdcop-trading-dashboard/lib/contracts/*`, `usdcop-trading-dashboard/types/*`

---

## Índice

1. [Arquitectura de conexión (BFF)](#1-arquitectura-de-conexión-bff)
2. [Convenciones y mejores prácticas](#2-convenciones-y-mejores-prácticas)
3. [Autenticación, sesión y RBAC](#3-autenticación-sesión-y-rbac)
4. [Catálogo de endpoints por módulo](#4-catálogo-de-endpoints-por-módulo)
5. [Contratos que NO se deben romper](#5-contratos-que-no-se-deben-romper)
6. [Matriz RBAC (endpoint × rol/plan)](#6-matriz-rbac-endpoint--rolplan)
7. [Mapa vista → endpoints](#7-mapa-vista--endpoints)
8. [Checklist de implementación backend](#8-checklist-de-implementación-backend)
9. [DO / DON'T](#9-do--dont)

---

## 1. Arquitectura de conexión (BFF)

El frontend **no habla directo** con los microservicios. Usa el patrón **Backend-For-Frontend (BFF)**:
las rutas `app/api/**/route.ts` de Next.js reciben la petición del navegador (con **cookie de sesión
httpOnly**), aplican RBAC, adjuntan el **bearer** del usuario y hacen `fetch` al servicio upstream.

```
┌──────────────┐   cookie de sesión (httpOnly)   ┌───────────────────────────┐
│  Navegador   │ ──────────────────────────────▶ │  Next.js BFF              │
│  (React)     │   /api/**                        │  app/api/**/route.ts      │
└──────────────┘ ◀────────────────────────────── │  · valida sesión NextAuth │
                    JSON / SSE / WS               │  · re-check RBAC          │
                                                  │  · relay Bearer upstream  │
                                                  │  · shaping + fallback     │
                                                  └───┬───────────────────────┘
                     ┌────────────────────────────────┼──────────────────────────────────┐
                     ▼                ▼                ▼                ▼                   ▼
             trading-api       analytics-api    signalbridge-api   backtest-api     mlops-inference-api
              :8000             :8001            :8000 (BACKEND_URL) :8000/api/v1     :8090
                     │                │                │                                    │
                     └── TimescaleDB (forecast_h5_*, OHLCV) ── Redis ── Airflow ── MLflow ──┘
                                     │
                       Observabilidad: Prometheus :9090 · Grafana :3002 · Loki :3100
                                      AlertManager :9093 · Jaeger :16686
```

**Por qué BFF (y no llamar directo):**

| Motivo | Detalle |
|---|---|
| Secretos | API keys/secrets de exchange, tokens de proveedor de pago y credenciales de DB **nunca** llegan al bundle del navegador. |
| RBAC defensa en profundidad | El middleware protege la ruta y el handler **re-verifica** el permiso (`admin:all`, etc.) contra la matriz. |
| Relay de identidad | El BFF adjunta el `Authorization: Bearer <token>` del usuario hacia SignalBridge (fuente de verdad de usuarios/ejecución). |
| Shaping + fallback | Une varias respuestas (p.ej. balances por credencial), normaliza el envelope y sirve *fallback* sintético cuando el upstream cae. |
| CORS | El navegador solo habla con su propio origen; el fan-out a microservicios ocurre server-side. |

### Variables de entorno (contrato de despliegue)

| Variable | Uso |
|---|---|
| `BACKEND_URL` | Base de **SignalBridge API** (auth, usuarios, exchanges, ejecución, kill-switch). |
| `BACKTEST_API_URL` | Base de **backtest-api** (`/api/v1/backtest/real[/stream]`, `/api/v1/health`). |
| `TRADING_API_URL` / `ANALYTICS_API_URL` / `MLOPS_INFERENCE_URL` | Bases de trading-api, analytics-api, inferencia. |
| `NEXT_PUBLIC_*_URL` | Solo lo que el navegador puede conocer (p. ej. base de WebSocket público). Nunca secretos. |
| `POSTGRES_*` | Conexión TimescaleDB (solo server-side, para `/api/production/live`, `/api/analysis/*`). |
| `DEPLOY_DAG_ID` | DAG de Airflow a disparar en promoción a producción (default `forecast_h5_l4b_production_deploy`). |
| `NEXTAUTH_SECRET` / `NEXTAUTH_URL` | Firma de sesión y callback de NextAuth. |
| `ANTHROPIC_API_KEY` | Solo server-side, para `/api/analysis/chat`. |
| `SLACK_WEBHOOK_URL` | AlertManager → Slack (hoy vacío ⇒ notificaciones inactivas, ver `CTR-OBS-001`). |
| `GRAFANA_USER` / `GRAFANA_PASSWORD` | Acceso a Grafana (deep-links desde Sistema). |

---

## 2. Convenciones y mejores prácticas

Estas reglas aplican a **todos** los endpoints salvo que se indique lo contrario.

### 2.1 Versionado y base
- Microservicios exponen `/{api/}v1/...`. El BFF puede montar sin versión (`/api/backtest`) y mapear
  internamente a `${BACKTEST_API_URL}/api/v1/backtest/...`. **Nunca** romper `v1`; cambios incompatibles ⇒ `v2`.

### 2.2 Envelope de respuesta (uniforme)
Éxito y error comparten forma (ver `types/api.ts` → `ApiResponse<T>`, `ApiError`):

```jsonc
// Éxito
{ "ok": true, "data": <T>, "meta": { "requestId": "…", "ts": "2026-07-10T22:00:00Z" } }

// Error (problem-style)
{ "ok": false,
  "error": { "message": "…", "code": "FORBIDDEN", "status": 403,
             "details": {}, "timestamp": "…", "path": "/api/admin/users" } }
```

### 2.3 Códigos HTTP
`200` OK · `201` creado · `202` aceptado (jobs async: backtest, deploy) · `204` sin contenido ·
`400` validación · `401` no autenticado · `403` sin permiso/plan · `404` no existe ·
`409` conflicto/idempotencia · `422` payload inválido · `429` rate limit · `5xx` upstream.
Distinguir siempre **401 (loguéate)** de **403 (no tienes plan/rol)** — el frontend redirige distinto.

### 2.4 Paginación, filtro y orden
`?page=1&page_size=25&sort=-created_at&status=pending`. Respuesta con `PaginatedResponse<T>`:
`{ items, page, page_size, total, has_next }`. Listas de admin (usuarios, ejecuciones, auditoría) **siempre** paginadas.

### 2.5 Tiempo real: elige el transporte correcto
| Transporte | Cuándo | Ejemplos |
|---|---|---|
| **WebSocket** | Streams de alta frecuencia bidireccionales | precios/ticker, señal en vivo (`hooks/useNRTWebSocket.ts` → `/ws/quotes`, `/ws/signals`) |
| **SSE** (`text/event-stream`) | Progreso unidireccional de un job | `/api/backtest/stream`, `/api/backtest/real/stream` |
| **Polling** (`refreshMs`) | Widgets semi-estáticos | admin queue/system (`REFRESH.queue`, `REFRESH.system`) |
Reglas WS/SSE: reconexión con **backoff exponencial + jitter**, *heartbeat* cada ≤30 s, y **fallback** a polling/JSON si el socket no abre (el dashboard ya sirve *fallback* sintético en backtest).

### 2.6 Caché
- Datos **vivos** (producción, balances, kill-switch, quotes): `Cache-Control: no-store` + `cache: 'no-store'` en el fetch del BFF.
- Datos **semi-estáticos** (registry, catálogo, semanas de análisis): `ETag` + `stale-while-revalidate`; el cliente usa SWR.
- Imágenes/JSON de forecasting: cacheables por semana, pero **respetando el delay por plan** (§5).

### 2.7 Idempotencia
Mutaciones sensibles (checkout, promoción de modelo, connect de exchange, aprobar/rechazar usuario)
aceptan header `Idempotency-Key: <uuid>`; el backend deduplica durante ≥24 h y devuelve el mismo resultado.

### 2.8 Rate limiting
Respuestas incluyen `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`. Al exceder ⇒ `429` + `Retry-After`.
Endpoints públicos (login, register, reset-password, chat) llevan **captcha** (`GET /api/captcha`, reto aritmético firmado).

### 2.9 Trazabilidad end-to-end (observabilidad)
- El navegador/BFF propaga `traceparent` (W3C) hacia los servicios → correlación en **Jaeger**.
- Cada servicio expone `GET /metrics` (Prometheus) y logs estructurados JSON con `request_id`, `model_id`,
  `action`, `duration_ms` → **Loki** (ver `CTR-OBS-001`).
- Todo handler del BFF loguea `{ requestId, path, status, ms }`.

### 2.10 Seguridad
- **Secretos jamás al cliente.** API key/secret de exchange se envían del navegador → BFF → SignalBridge una sola vez; el backend los cifra (AES-256) y **nunca** los devuelve (solo máscara).
- **Retiros deshabilitados** por diseño: al conectar un exchange se exige que la API key NO tenga permiso de retiro; el backend valida y rechaza si lo tiene.
- Cookies de sesión `httpOnly`, `Secure`, `SameSite=Lax`. Bearer solo server-side.
- Validación de esquema (zod/pydantic) en **ambos** extremos; el BFF nunca confía en el cliente.

### 2.11 Contratos como fuente de verdad
Los tipos viven en `lib/contracts/*.ts` (frontend) y en las specs `.claude/specs/*` (backend). Un cambio de forma
requiere: (1) actualizar el contrato, (2) subir versión, (3) test de contrato en CI (estilo Pact/OpenAPI) que
valide request/response contra el esquema. **PR que cambie un endpoint sin tocar su contrato = bloqueado.**

---

## 3. Autenticación, sesión y RBAC

### 3.1 Flujo de sesión
```
Registro:  POST /api/execution/auth/register  → SignalBridge POST /api/auth/register
           ⇒ estado "pending" → entra a la cola de aprobación (Admin › Registros)
Login:     POST /api/auth/signin (NextAuth Credentials)
           → BFF valida en SignalBridge POST /api/auth/login (obtiene bearer)
           → rol/plan vía SignalBridge GET /api/users/me + GET /api/billing/me
           ⇒ crea sesión (cookie httpOnly)
Sesión:    GET /api/auth/session   ·  Logout: POST /api/auth/signout
Reset:     POST /api/execution/auth/reset-password → SB /api/auth/reset-password
```
El **redirect de invitado**: si un invitado abre una sección protegida, el BFF responde `401`; el frontend
redirige a login con `?next=<ruta>` y, tras autenticar, vuelve a `next`.

### 3.2 Roles y planes (CTR-RBAC-001)
- **Roles**: `admin`, `developer`, `subscriber`, `free` (guest = sin sesión).
- **Planes**: `free`, `signals` (Señales Pro), `auto` (Auto Premium) + **add-ons por activo**.
- **Permisos** (ejemplos): `research:read` (backtest/experimentos — admin/dev), `signals:read`,
  `forecast:read`, `analysis:read`, `execution:self`, `execution:global`, `admin:all`.
- **Entitlements efectivos** los resuelve el servidor en `GET /api/billing/me` (plan + add-ons + delays). El
  cliente **nunca** decide acceso; solo pinta/oculta según lo que devuelve el backend.

---

## 4. Catálogo de endpoints por módulo

> Columna **Upstream**: servicio real detrás del BFF. `SB` = SignalBridge (`BACKEND_URL`).
> Los marcados **🆕** no existen aún — se definen aquí para el catálogo/watchlist/carrito.

### 4.1 Auth & sesión
| Método · Ruta (BFF) | Upstream | Auth | Descripción |
|---|---|---|---|
| `POST /api/auth/signin` | NextAuth → SB `/api/auth/login` | público + captcha | Inicia sesión, crea cookie. |
| `POST /api/auth/signout` | NextAuth | sesión | Cierra sesión. |
| `GET /api/auth/session` | NextAuth | sesión | Sesión + rol + plan efectivo. |
| `POST /api/execution/auth/register` | SB `/api/auth/register` | público + captcha | Registro ⇒ estado `pending`. |
| `POST /api/execution/auth/reset-password` | SB `/api/auth/reset-password` | público + captcha | Recuperar contraseña. |
| `GET /api/captcha` | BFF | público | Reto aritmético firmado para formularios públicos. |

### 4.2 Billing & entitlements
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/billing/me` | proveedor de pago + DB | sesión | Entitlements efectivos: plan, add-ons, delays. **Única fuente de acceso.** |
| `POST /api/billing/checkout` | proveedor de pago | sesión | Crea sesión de checkout (plan + add-ons del carrito). Idempotente. |
| `POST /api/billing/webhook` | proveedor de pago → BFF | firma HMAC | Eventos de pago (CTR-RBAC-001 regla 9). Verificar firma. |

### 4.3 Catálogo · Watchlist · Carrito 🆕
Diseñados sobre `GET /api/registry` (activos/estrategias) + `GET /api/billing/me` (entitlements).
Categorías UI ← `asset_class` del registry: `fx→Forex`, `crypto→Cripto`, `equity_index→Acciones`, `commodity→Materias primas`.

| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/catalog` 🆕 | registry + billing | sesión/guest | Catálogo completo por categoría con estado y precio de add-on. |
| `GET /api/watchlist` 🆕 | DB (por usuario) | sesión | Activos que el usuario sigue (solo lectura, gratis). |
| `POST /api/watchlist` 🆕 | DB | sesión | Agrega `{ asset_id }` a la watchlist. |
| `DELETE /api/watchlist/{asset_id}` 🆕 | DB | sesión | Quita de la watchlist. |
| `GET /api/cart` 🆕 | DB (por usuario) | sesión | Carrito actual: plan elegido + add-ons. |
| `POST /api/cart` 🆕 | DB | sesión | Agrega add-on `{ asset_id }` al carrito. |
| `DELETE /api/cart/{asset_id}` 🆕 | DB | sesión | Quita add-on. |
| `POST /api/cart/checkout` 🆕 | → `/api/billing/checkout` | sesión | Convierte carrito en checkout. Idempotente. |

```ts
// GET /api/catalog  →  200
interface CatalogResponse {
  categories: { id: 'fx'|'crypto'|'equity_index'|'commodity'; label: string; count: number }[];
  assets: {
    asset_id: string;            // 'usdcop' | 'btcusdt' | 'spx500' | 'xauusd' | …
    symbol: string;              // 'USD/COP'
    name: string;                // 'Peso colombiano'
    asset_class: 'fx'|'crypto'|'equity_index'|'commodity';
    status: 'available' | 'coming_soon';
    price: number | null;        // último precio (si available)
    change_pct: number | null;
    addon_price_month: number | null;  // COP; null si incluido/próximamente
    entitled: boolean;           // si el usuario ya lo tiene desbloqueado
    in_watchlist: boolean;
  }[];
}
```

### 4.4 Registry (estrategias/modelos/activos) — `CTR-STRAT-REGISTRY-001`
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/registry` | fs `public/data/registry.json` (RegistryBuilder) | sesión | SSOT dinámico de estrategias/activos. Alimenta dropdowns de estrategia, catálogo y modelos. |
| `POST /api/registry/promote` | registry + Airflow | `admin:all` | Promueve una versión de modelo a `active` para una estrategia. Idempotente. |
| `GET /api/data/{...path}` | fs `public/data` | según recurso | Sirve JSON del data-lake (path-traversal-guarded). |
| `GET /api/strategies/{id}/manifest` | fs | sesión | Manifiesto de estrategia (`strategy-manifest.contract.ts`). |

### 4.5 Backtest — `backtest.contract.ts`, `backtest-ssot.contract.ts`
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `POST /api/backtest` | `${BACKEND_URL}/v1/backtest` | `research:read` | Backtest no-streaming. |
| `GET /api/backtest/stream?model_id=` | SSE (o JSON precomputado para estrategias forecast) | `research:read` | Progreso incremental. |
| `POST /api/backtest/real` | `${BACKTEST_API_URL}/api/v1/backtest/real` | `research:read` | Backtest OOS real por `proposal_id` + rango de fechas. |
| `GET /api/backtest/real/stream?proposal_id=&start_date=&end_date=` | `${BACKTEST_API_URL}/api/v1/backtest/real/stream` | `research:read` | Replay OOS (SSE). |
| `GET /api/backtest/status/{modelId}` | backtest-api | `research:read` | Estado del job. |
| `GET /api/models/{modelId}/metrics` | mlops/registry | sesión | KPIs del modelo (Sharpe, DA, PSR, Calmar, MaxDD…). |
| `GET /api/models/{modelId}/equity-curve` | mlops/registry | sesión | Curva de equity para el gráfico. |

### 4.6 Producción / Señales (live)
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/production/live` | TimescaleDB `forecast_h5_signals/executions/subtrades/paper_trading` | `signals:read` | Señal de la semana, posición, equity y P&L en vivo. |
| `GET /api/trading/performance/multi-strategy` | analytics/DB | `signals:read` | KPIs por estrategia (YTD, Sharpe, Calmar, MaxDD, win rate). |
| `GET /api/trading/trades/history` | DB | `signals:read` | Historial de operaciones (compartido con backtest). |
| `POST /api/production/deploy` | Airflow DAG `${DEPLOY_DAG_ID}` | `admin:all` | Promoción a producción (2 votos + gates, ver `production-approval.contract.ts`). `202`. |
| `WS /ws/quotes`, `WS /ws/signals` | trading-api | `signals:read` | Ticker y señal en tiempo real (`useNRTWebSocket`). |

### 4.7 Forecasting (imágenes/JSON con delay) — `forecasting.contract.ts`
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/forecasting/{...path}` | fs `public/forecasting` | sesión/guest | Sirve PNG/JSON/CSV del forecast. **Aplica delay por plan** (§5). Path-traversal-guarded. |

Layout on-disk (contrato): `public/forecasting/<asset>/<model>_h<n>.png`, `weekly_inference_<year>.json`.
El delay: `free ⇒ forecast_delay_hours = 168` (T-1 semana; se eliminan las entradas de la semana actual);
plan pago ⇒ live. Legacy estático `/forecasting/<asset>/...` sigue existiendo.

### 4.8 Análisis semanal + chat — `weekly-analysis.contract.ts`, `analysis-assets.ts`
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/analysis/assets` | fs | sesión/guest | Activos con análisis disponible. |
| `GET /api/analysis/weeks?asset=` | fs `analysis/<asset>/analysis_index.json` | sesión/guest | Semanas disponibles (para el selector). |
| `GET /api/analysis/week/{year}/{week}?asset=` | fs `weekly_YYYY_WXX.json` | `analysis:read` (free con delay) | Recap OHLC, técnico, régimen macro (Z-Score), timeline diario, noticias. |
| `GET /api/analysis/calendar?asset=` | fs `upcoming_events.json` | sesión/guest | Calendario económico. |
| `POST /api/analysis/chat` | Anthropic (server-side) | `analysis:read` | Chatbot contextual (streaming). Rate-limited. |

Toda ruta pasa por `resolveAnalysisAsset()` (sin path traversal; id desconocido ⇒ activo por defecto USD/COP).

### 4.9 SignalBridge / Ejecución — `lib/contracts/execution/*`
| Método · Ruta | Upstream (SB) | Auth | Descripción |
|---|---|---|---|
| `GET /api/execution/exchanges` | `/api/exchanges/credentials` | `execution:self` | Exchanges conectados del usuario. |
| `POST /api/execution/exchanges/{exchange}/connect` | `/api/exchanges/credentials` | `execution:self` | Registra API key/secret (valida sin permiso de retiro). Secreto no se devuelve. |
| `GET /api/execution/exchanges/{exchange}/balance` | `/api/exchanges/credentials/{id}/balances` | `execution:self` | Balance por exchange. |
| `GET /api/execution/executions` | `/api/executions` | `execution:self` (admin ⇒ global) | Historial de ejecuciones. |
| `GET /api/execution/signal-bridge/history` | `/api/signal-bridge/history` | `execution:self` | Historial de señales ejecutadas. |
| `GET/POST /api/execution/signal-bridge/kill-switch` | `/api/signal-bridge/kill-switch[/status]` | `execution:self` (admin ⇒ global) | Estado / activar kill-switch. |

### 4.10 Admin — `admin-console.contract.ts`
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/admin/kpis` | DB/billing | `admin:all` | KPIs de negocio (usuarios activos, MRR, suscriptores, churn). |
| `GET /api/admin/overview` | DB + audit | `admin:all` | Resumen: usuarios, auditoría reciente, mezcla de planes. |
| `GET /api/admin/queue` | relay SB `/api/admin/users?status=pending` | `admin:all` | Cola de aprobación (KYC, origen, plan solicitado). |
| `POST /api/admin/users/{id}/approve` | relay SB | `admin:all` | Aprueba registro. Idempotente. |
| `POST /api/admin/users/{id}/reject` | relay SB | `admin:all` | Rechaza registro (con motivo). |
| `GET /api/admin/users/list` | relay SB | `admin:all` | Tabla de usuarios (rol, plan, add-ons, LTV, KYC, último acceso, estado). |
| `POST /api/admin/users/{id}/suspend` 🆕 | relay SB | `admin:all` | Suspender/reactivar. |
| `PATCH /api/admin/users/{id}/plan` 🆕 | relay SB + billing | `admin:all` | Cambiar plan/add-ons. |
| `GET /api/admin/revenue` 🆕 | proveedor de pago | `admin:all` | MRR/ARR/ARPU/LTV, ingresos por plan y por activo, movimiento de MRR, cobros, dunning. |
| `GET /api/admin/models` 🆕 | registry + mlops | `admin:all` | Registro de modelos (estado, gates, drift). |
| `POST /api/admin/models/{id}/vote` 🆕 | registry + `production-approval.contract` | `admin:all` | Voto de promoción (2/2 + gates). Idempotente. |
| `GET /api/admin/risk` 🆕 | SB + production-monitor | `admin:all` | Kill-switch global, límites, guardrails disparados, conexiones API por aprobar, usuarios/IPs bloqueados. |
| `POST /api/admin/exchanges/{id}/approve` 🆕 | SB | `admin:all` | Aprueba/rechaza conexión API. |
| `GET /api/admin/system` | Prometheus/Grafana/Airflow (ver §4.11) | `admin:all` | Salud del sistema para Resumen/Sistema. Spec §6. |
| `GET /api/admin/audit` | DB (ledger) | `admin:all` | Auditoría (append-only). Spec §7. |
| `GET/PATCH /api/admin/catalog` 🆕 | registry | `admin:all` | Habilitar/despublicar activos, precio de add-on (mapea 1:1 al registry). |

### 4.11 Observabilidad / Sistema — `CTR-OBS-001`
El BFF **no reemplaza** Grafana; enlaza a él y resume señales clave.
| Método · Ruta | Upstream | Auth | Descripción |
|---|---|---|---|
| `GET /api/admin/system` | agrega los de abajo | `admin:all` | Estado del stack + SLOs + targets + alertas + pipeline + recursos. |
| (proxy) Prometheus | `http://prometheus:9090/api/v1/query`, `/targets`, `/alerts` | server-side | SLO de inferencia (p50/p95/p99), targets UP/DOWN, reglas. |
| (proxy) AlertManager | `http://alertmanager:9093/api/v2/alerts` | server-side | Alertas activas (53 reglas / 16 grupos). |
| (deep-link) Grafana | `http://…:3002/d/<uid>` | usuario Grafana | 4 dashboards: `trading-performance`, `mlops-monitoring`, `system-health`, `macro-ingestion`. |
| (deep-link) Loki | `http://…:3002/explore` (LogQL) | usuario Grafana | Logs por servicio/nivel (retención 31 d). |
| (deep-link) Jaeger | `http://…:16686` | — | Trazas por `traceparent`. |
| (proxy) Airflow | `airflow-webserver:8080` | server-side | Estado de DAGs L0→L6 + heartbeat del scheduler. |

---

## 5. Contratos que NO se deben romper

1. **Forecasting con delay por plan** — `GET /api/forecasting/{...path}`: mantener la estructura de archivos
   (`<asset>/<model>_h<n>.png`, `weekly_inference_<year>`) y el `forecast_delay_hours` (free = 168h). Cualquier
   cambio de layout rompe los lectores de imágenes por semana/año.
2. **Registry SSOT** — `GET /api/registry` lee `public/data/registry.json` (producido por RegistryBuilder). El
   catálogo, los dropdowns de estrategia y la pestaña Modelos derivan de aquí; no duplicar la lista en el front.
3. **Análisis por activo** — resolución vía `resolveAnalysisAsset()`; ids desconocidos colapsan al activo por
   defecto (no romper el fallback legacy de USD/COP).
4. **Producción sobre TimescaleDB** — `/api/production/live` consulta `forecast_h5_*`; no cambiar nombres de tabla
   sin migración + versión de contrato.
5. **Backtest vs Producción** comparten el framework de velas (Lightweight Charts) y el shape de `trades`/`equity`;
   propósitos distintos (revisión OOS + aprobación vs monitoreo en vivo) pero **mismo contrato de datos**.

---

## 6. Matriz RBAC (endpoint × rol/plan)

| Dominio | guest | free | subscriber (signals) | subscriber (auto) | admin/dev |
|---|:--:|:--:|:--:|:--:|:--:|
| Catálogo / Watchlist | ver | ✔ | ✔ | ✔ | ✔ |
| Carrito / Checkout | → login | ✔ | ✔ | ✔ | ✔ |
| Forecasting | T-1 | T-1 (1 activo) | live (entitled) | live | live |
| Análisis semanal | resumen T-7 | resumen T-7 | completo | completo | completo |
| Señales / Producción | ✖ | ✖ | ✔ | ✔ | ✔ (global) |
| Backtest / Experimentos | ✖ | ✖ | ✖ | ✖ | ✔ (`research:read`) |
| SignalBridge / Ejecución | ✖ | ✖ | ✖ (paper) | ✔ (self) | ✔ (global) |
| Admin (todo) | ✖ | ✖ | ✖ | ✖ | ✔ (`admin:all`) |

Enforcement: **middleware** (protege ruta) **+ re-check en handler** (contra la matriz `rbac.contract.ts`). El
front solo pinta según `GET /api/billing/me`; el backend es la autoridad.

---

## 7. Mapa vista → endpoints

| Vista (rediseño) | Endpoints que consume |
|---|---|
| Landing / Login / Registro | `/api/captcha`, `/api/auth/*`, `/api/execution/auth/*` |
| Hub | `/api/billing/me`, `/api/watchlist`, `/ws/quotes` |
| Catálogo + Carrito | `/api/catalog`, `/api/watchlist`, `/api/cart`, `/api/cart/checkout` |
| Backtest | `/api/registry`, `/api/backtest/real[/stream]`, `/api/models/{id}/metrics`, `/api/models/{id}/equity-curve`, `/api/trading/trades/history` |
| Producción / Señales | `/api/production/live`, `/api/trading/performance/multi-strategy`, `/api/trading/trades/history`, `/ws/signals` |
| Forecasting | `/api/registry`, `/api/forecasting/{...path}` |
| Análisis semanal | `/api/analysis/assets`, `/api/analysis/weeks`, `/api/analysis/week/{y}/{w}`, `/api/analysis/calendar`, `/api/analysis/chat` |
| SignalBridge | `/api/execution/exchanges*`, `/api/execution/executions`, `/api/execution/signal-bridge/*` |
| Admin · Resumen | `/api/admin/kpis`, `/api/admin/overview`, `/api/admin/system` |
| Admin · Ingresos | `/api/admin/revenue`, `/api/billing/*` |
| Admin · Registros | `/api/admin/queue`, `/api/admin/users/{id}/approve|reject` |
| Admin · Usuarios | `/api/admin/users/list`, `.../suspend`, `.../plan` |
| Admin · Modelos | `/api/admin/models`, `/api/admin/models/{id}/vote`, `/api/registry/promote` |
| Admin · Riesgo | `/api/admin/risk`, `/api/admin/exchanges/{id}/approve`, kill-switch |
| Admin · Sistema | `/api/admin/system` (+ deep-links Grafana/Prometheus/Loki/Jaeger) |
| Admin · Auditoría | `/api/admin/audit` |

---

## 8. Checklist de implementación backend

- [ ] Todos los servicios exponen `GET /health` (readiness) y `GET /metrics` (Prometheus).
- [ ] Envelope `ApiResponse<T>` / `ApiError` uniforme en todos los endpoints.
- [ ] RBAC en middleware **y** re-check en handler; 401 vs 403 correctos.
- [ ] `GET /api/billing/me` resuelve entitlements efectivos (plan + add-ons + delays) — única autoridad de acceso.
- [ ] Endpoints 🆕 (catálogo/watchlist/carrito, revenue, models/vote, risk, catalog admin) implementados según §4.
- [ ] Secretos de exchange cifrados, nunca devueltos; conexión rechaza keys con permiso de retiro.
- [ ] SSE con `text/event-stream` + heartbeat; WS con reconexión/backoff; polling con `refreshMs`.
- [ ] Idempotency-Key en checkout, promote, connect, approve/reject.
- [ ] Rate limit + `Retry-After`; captcha en formularios públicos.
- [ ] `traceparent` propagado; logs JSON con `request_id`; alertas Prometheus (53 reglas) activas.
- [ ] Tests de contrato en CI para cada endpoint (request/response vs `lib/contracts/*`).
- [ ] Forecasting/registry/analysis/producción respetan los contratos de §5 (sin romper layout ni delays).

---

## 9. DO / DON'T

**DO**
- Mantener el BFF como único punto de contacto del navegador.
- Versionar contratos y correr tests de contrato antes de mergear.
- Devolver siempre el envelope uniforme y códigos HTTP correctos.
- Resolver acceso en el servidor (`billing/me` + RBAC), pintar en el cliente.

**DON'T**
- No exponer secretos, tokens o credenciales de DB al bundle del navegador.
- No permitir API keys con permiso de retiro.
- No romper el layout/delay de forecasting ni los nombres de tabla `forecast_h5_*`.
- No duplicar la lista de activos/estrategias fuera del registry (SSOT).
- No decidir permisos en el frontend.
