/**
 * API module — CTR-FE-BE-001 surface (spec `.claude/specs/platform/frontend-backend-contract.md`).
 *
 * Server (route handlers): `ok`/`fail` envelope + `requirePermission`/`requireSession`
 * + `relayUpstream`. Browser: `apiFetch` (envelope-aware, legacy-tolerant).
 *
 * (The pre-contract zod `apiClient` was removed 2026-07-10 — it had zero consumers.)
 */

export {
  ok, fail, UpstreamError, upstreamCode,
  type ApiErrorShape, type Envelope, type Meta,
} from './envelope';
export {
  bearerFrom, relayUpstream, requirePermission, requireSession,
  type HandlerIdentity, type RelayOpts,
} from './relay';
export { apiFetch, ClientApiError, goToLogin, type ApiResult } from './gm-client';
