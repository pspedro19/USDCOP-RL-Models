/**
 * Admin Console contract (CTR-ADMIN-CONSOLE-001) — spec `.claude/specs/platform/admin-console.md`.
 *
 * The console answers two questions at all times — "¿cómo va el negocio?" and
 * "¿está sano el sistema?" — and every privileged action requires (a) confirmation
 * proportional to the damage and (b) an audit_log row. Roles/plans are NEVER loose
 * strings here: they derive from `rbac.contract.ts` (SSOT). Every widget degrades
 * independently (spec C5): one failing source never blanks the console.
 */

import type { PlanId, Role } from './rbac.contract';
import { ROLES } from './rbac.contract';

// ─────────────────────────────────────────────────────────── sections (portal nav)

export interface AdminSectionDef {
  id: 'overview' | 'ingresos' | 'registros' | 'usuarios' | 'roles' | 'modelos' | 'interpretabilidad' | 'riesgo' | 'catalogo-admin' | 'sistema' | 'auditoria';
  label: string;
  /** Sections gated on a future phase render as disabled with the phase note. */
  phaseGate?: string;
}

/**
 * Pestañas en orden del prototipo Var B (VISUAL-SPEC-CHECKLIST §Admin).
 * Ingresos absorbe la vieja "Suscripciones" (fase-gated); configuracion/comunicaciones
 * salen del tab-bar (alias abajo). Badges dinámicos: registros=queue.count,
 * modelos=candidatos PENDING, riesgo=llaves API pendientes.
 * "Interpretabilidad" (BL-20, 2026-07-27) va tras Modelos: diagnóstico SHAP/atribución
 * de los artefactos publicados por scripts/analysis/generate_interpretability.py.
 */
export const ADMIN_SECTIONS: readonly AdminSectionDef[] = [
  { id: 'overview', label: 'Resumen' },
  { id: 'ingresos', label: 'Ingresos' },
  { id: 'registros', label: 'Registros' },
  { id: 'usuarios', label: 'Usuarios' },
  { id: 'roles', label: 'Roles y vistas' },
  { id: 'modelos', label: 'Modelos' },
  { id: 'interpretabilidad', label: 'Interpretabilidad' },
  { id: 'riesgo', label: 'Riesgo y bloqueos' },
  { id: 'catalogo-admin', label: 'Catálogo' },
  { id: 'sistema', label: 'Sistema' },
  { id: 'auditoria', label: 'Auditoría' },
] as const;

export type AdminSectionId = AdminSectionDef['id'];

/** Ids viejos de ?tab= → sección nueva (deep-links guardados siguen funcionando). */
export const LEGACY_TAB_ALIASES: Record<string, AdminSectionId> = {
  suscripciones: 'ingresos',
  configuracion: 'sistema',
  comunicaciones: 'overview',
};

// ─────────────────────────────────────────────────────────── roles & plans (C2/C3)

/** Staff roles have no plan: they are not customers (spec C3). */
export const STAFF_ROLES: readonly Role[] = ['admin', 'developer'] as const;

/**
 * C2 — SignalBridge legacy rows still carry role='user' (its UserRole enum predates
 * the RBAC contract). Normalize at the read boundary so the UI only ever sees the
 * contract enum; the DB migration to retire 'user' happens on the backend's clock.
 */
export function normalizeRole(raw: string | null | undefined): Role {
  if (raw && (ROLES as readonly string[]).includes(raw)) return raw as Role;
  return 'free';
}

export function isStaff(role: Role): boolean {
  return STAFF_ROLES.includes(role);
}

/** C3 — plan shown for customers only; staff renders '—' and APIs ignore their plan. */
export function planForDisplay(role: Role, plan: string | null | undefined): PlanId | null {
  if (isStaff(role)) return null;
  if (plan === 'signals' || plan === 'auto') return plan;
  return 'free';
}

// ─────────────────────────────────────────────────────────── test traffic (C4)

/** Heuristic per spec C4 — manual flag (`sb_users.is_test`) always wins over this. */
const TEST_EMAIL_PATTERNS = [
  /@test\.com$/i, /@example\.(com|org|net)$/i, /@.*\.local$/i,
  /@mailinator\.com$/i, /^qa[._+-]/i, /\+test@/i,
];

export function isTestEmail(email: string | null | undefined): boolean {
  if (!email) return false;
  return TEST_EMAIL_PATTERNS.some((re) => re.test(email));
}

// ─────────────────────────────────────────────────────────── widget envelope (C5)

/**
 * Every /api/admin widget endpoint resolves to this envelope; a widget owns its
 * error and retry — a SignalBridge 502 must never blank the users table (spec C5).
 */
export interface AdminWidget<T> {
  data: T | null;
  error: string | null;
}

// ─────────────────────────────────────────────────────────── overview · negocio

/** Business KPIs (always computed over is_test=false). Nulls = phase not live yet. */
export interface BusinessKpis {
  total_users: number;
  new_7d: number;
  active_7d: number;
  active_30d: number;
  pending_queue: number;
  /** Hidden test accounts behind the queue counter, per spec C1 ("0 (4 test ocultos)"). */
  pending_test_hidden: number;
  plan_mix: Record<string, number>;
  /** Fase 6 (billing) — null until the provider is the source of truth. */
  mrr_cop: number | null;
  conversion_30d_pct: number | null;
  churn_monthly_pct: number | null;
}

// ─────────────────────────────────────────────────────────── overview · sistema

export type FreshnessStatus = 'ok' | 'warn' | 'stale' | 'unknown';

export interface FreshnessSource {
  id: string;
  label: string;
  latest: string | null;       // ISO of the newest row/file
  age_hours: number | null;
  threshold_hours: number;     // from .claude/rules/data-freshness.md (SSOT)
  status: FreshnessStatus;
  detail?: string;
}

/** Thresholds mirror `data-freshness.md` — do not invent new numbers here. */
export const FRESHNESS_THRESHOLDS_HOURS = {
  ohlcv_m5: 72,        // OHLCV 3 days
  macro_daily: 168,    // macro 7 days
  news: 24,            // news articles 24h
  production_bundle: 240, // models/bundle 10 days (soft)
} as const;

export interface ServiceCheck {
  name: string;
  ok: boolean;
  latency_ms: number | null;
  error?: string;
}

/** Vote 2 queue summary (§6.1) — deep-links to /dashboard, never a second approve button. */
export interface Vote2Summary {
  status: string;                 // PENDING_APPROVAL | APPROVED | REJECTED | LIVE
  strategy: string;
  strategy_name: string;
  backtest_year: number | null;
  recommendation: string | null;  // PROMOTE | REVIEW | REJECT
  gates_passed: number;
  gates_total: number;
  approved_by: string | null;
  approved_at: string | null;
  pending: boolean;
}

// ── Sistema ampliado (§4.11 CTR-OBS-001): proxies server-side Prometheus/AM/Airflow.

export interface SloMetric {
  id: string;
  label: string;
  /** Valor medido (unidad en `unit`); null = Prometheus sin datos para la query. */
  value: number | null;
  unit: 'ms' | '%' | 'rps';
  target: string;             // e.g. "< 50 ms"
  ok: boolean | null;
}

export interface PromTargetRow {
  job: string;
  instance: string;
  health: 'up' | 'down' | 'unknown';
  last_scrape_ms: number | null;
}

export interface ActiveAlertRow {
  name: string;
  severity: string;           // critical | warning | info | none
  state: string;              // active | suppressed
  since: string | null;
  summary: string | null;
}

export interface PipelineStageChip {
  stage: string;              // L0 … L7
  dag_id: string;
  name: string;
  /** null = sin runs registrados aún. */
  ok: boolean | null;
  state: string | null;       // success | running | failed | …
  last_run: string | null;    // ISO end/start date of latest run
  paused: boolean | null;
}

export interface ResourceGauge {
  id: string;
  label: string;
  pct: number | null;
  detail?: string;
}

export interface SystemStatus {
  freshness: FreshnessSource[];
  services: ServiceCheck[];
  vote2: Vote2Summary | null;
  deploy: { phase: string | null; runner: string | null; updated_at: string | null } | null;
  /** Sub-checks that failed — surfaced, never silently dropped. */
  partial_errors: string[];
  /** Ampliado 2026-07 (opcionales: cada proxy degrada a null + partial_errors). */
  slos?: SloMetric[] | null;
  prom_targets?: PromTargetRow[] | null;
  alerts_active?: ActiveAlertRow[] | null;
  pipeline?: PipelineStageChip[] | null;
  resources?: ResourceGauge[] | null;
}

// ─────────────────────────────────────────────────────────── overview · alertas

export interface AdminAlert {
  id: string;
  severity: 'info' | 'warn' | 'critical';
  message: string;
  /** Every alert carries the section that resolves it (spec: "botón que lleva a resolverla"). */
  section: AdminSectionId;
}

// ─────────────────────────────────────────────────────────── registros (queue)

export interface QueueItem {
  id: string;
  email: string;
  name: string | null;
  status: string;
  role: Role;
  created_at: string;
  waiting_hours: number;
  is_test: boolean;
}

/** SLA for the approval queue — beyond this the Overview raises a warn alert. */
export const QUEUE_SLA_HOURS = 24;

export interface QueueResponse {
  items: QueueItem[];
  /** Same query as `items` (spec C1): count = items minus hidden test accounts. */
  count: number;
  test_hidden: number;
}

// ─────────────────────────────────────────────────────────── usuarios

export type ExecutionMode = 'paper' | 'live' | null;

export interface AdminUserRow {
  id: string;
  email: string;
  name: string | null;
  role: Role;
  /** Raw DB value before normalization — visible on hover for debugging C2 leftovers. */
  raw_role: string;
  status: string;
  plan: PlanId | null;          // null ⇒ staff, render '—' (C3)
  expires_at: string | null;
  created_at: string;
  last_login: string | null;
  is_test: boolean;
  execution_mode: ExecutionMode; // from user_risk_limits_v2.mode; null = no execution
  kill_switch: boolean | null;
}

export interface UsersListResponse {
  users: AdminUserRow[];
  total: number;
}

// ─────────────────────────────────────────────────────────── auditoría

export type AuditCategory = 'security' | 'execution' | 'billing' | 'governance' | 'admin' | 'other';

export interface AuditEntry {
  id: number;
  user_id: string | null;
  /** Resolved via LEFT JOIN sb_users — a human reads emails at 2am, not UUIDs (§7). */
  user_email: string | null;
  action: string;
  category: AuditCategory;
  severity: 'info' | 'warn' | 'critical';
  object_type: string | null;
  object_id: string | null;
  detail: unknown;
  ip: string | null;
  created_at: string;
  is_test: boolean;
}

/**
 * Categorization of the migration-055 action vocabulary + as-built tenant actions.
 * Unknown actions fall to 'other'/info — never crash on a new action (open/closed).
 */
const AUDIT_RULES: Array<{ re: RegExp; category: AuditCategory; severity: AuditEntry['severity'] }> = [
  { re: /_denied$/, category: 'security', severity: 'warn' },
  { re: /^login/, category: 'security', severity: 'info' },
  { re: /^(role_change|key_add|key_remove|user_flag_test)$/, category: 'security', severity: 'warn' },
  { re: /^kill_global/, category: 'execution', severity: 'critical' },
  { re: /^(kill_user|go_live|live_enable|fanout|execution|limits_update|risk_accepted)$/, category: 'execution', severity: 'warn' },
  { re: /^(plan_change|billing|payment)/, category: 'billing', severity: 'info' },
  { re: /^(vote2_|promote)/, category: 'governance', severity: 'warn' },
  { re: /^user_(approve|reject)/, category: 'admin', severity: 'info' },
];

export function categorizeAuditAction(action: string): { category: AuditCategory; severity: AuditEntry['severity'] } {
  const hit = AUDIT_RULES.find((r) => r.re.test(action));
  return hit ? { category: hit.category, severity: hit.severity } : { category: 'other', severity: 'info' };
}

export interface AuditQuery {
  action?: string;
  category?: AuditCategory;
  user?: string;      // email or uuid fragment
  from?: string;      // ISO date
  to?: string;        // ISO date
  include_test?: boolean; // default false (§7 toggle)
  limit?: number;     // server caps at 500
}

export interface AuditResponse {
  entries: AuditEntry[];
  total_scanned: number;
}

// ─────────────────────────────────────────────────────────── modelos (GET /api/admin/models)

export type AdminModelEstado = 'produccion' | 'candidato' | 'experimental' | 'deprecado';

export interface AdminModelGate {
  gate: string;
  label: string;
  passed: boolean;
  value: number | null;
  threshold: number | null;
}

export interface AdminModelRow {
  strategy_id: string;
  display_name: string;
  asset_id: string;
  asset_symbol: string;
  timeframe: string | null;
  active_version: string | null;
  sharpe: number | null;
  return_pct: number | null;
  p_value: number | null;
  /** DA no está en el registry hoy — null hasta que el bundle lo publique. */
  da_pct: number | null;
  gates_passed: number | null;
  gates_total: number | null;
  estado: AdminModelEstado;
}

/** Candidato = approval_state con PENDING_APPROVAL. El voto vive en /dashboard (una sola superficie). */
export interface AdminModelCandidate {
  strategy_id: string;
  strategy_name: string;
  asset_id: string | null;
  backtest_year: number | null;
  recommendation: string | null;   // PROMOTE | REVIEW | REJECT
  status: string;
  gates: AdminModelGate[];
}

export interface AdminModelsResponse {
  models: AdminModelRow[];
  candidates: AdminModelCandidate[];
  /** Badge de la pestaña Modelos. */
  pending_count: number;
}

// ─────────────────────────────────── interpretabilidad (BL-20 · GET /api/admin/interpretability)

/**
 * Espejo TS de los artefactos que publica `scripts/analysis/generate_interpretability.py`
 * en `<repo>/data/interpretability/<surface>/<asset>/<model_id>/<version>/summary.json`
 * (FUERA de public/ — C-006/CXD-040: solo se sirven vía API con gate admin:all; el
 * schema runtime compartido con Python vive en
 * `app/api/admin/interpretability/_schema/interp-summary.schema.json`).
 * La UI SOLO renderiza estos JSON (jamás recomputa SHAP ni condiciones — regla
 * strategy-engines I-7). FABRIC A.7: "SHAP explica el modelo, no el mercado" —
 * diagnóstico 0 trials para RECHAZAR modelos absurdos, nunca "importancia para operar".
 */
export interface InterpIndexEntry {
  /** 'zoo' (SHAP lineal) | 'rule_based' (atribución). Abierto: fase 2 añade árboles. */
  surface: string;
  asset: string;
  model_id: string;
  /** Descendente — [0] es la más reciente. */
  versions: string[];
}

export interface InterpIndexResponse {
  entries: InterpIndexEntry[];
}

export interface InterpFeatureRow {
  rank: number;
  feature: string;
  coef: number;
  mean_abs_shap: number;
  mean_shap: number;
}

export interface InterpYearFeatureRow {
  feature: string;
  mean_abs_shap: number;
  mean_shap: number;
}

/**
 * Huellas que COMPROMETEN las entradas del artefacto (espejo de `provenance` en
 * interp-summary.schema.json y de `_write` en scripts/analysis/generate_interpretability.py).
 *
 * Por qué existe: `version` es la fecha del último dato, y esa clave NO distingue dos
 * corridas con distinto código, distinto dataset (mismo último día) o distintos
 * hiperparámetros — el generador sobrescribía y `generated_at` cambiaba en cada corrida,
 * así que la MISMA versión podía mutar la evidencia en silencio.
 */
export interface InterpProvenance {
  data_fingerprint: string;
  code_fingerprint: string;
  config_fingerprint: string;
  model_fingerprint: string;
  /** Qué cubre EXACTAMENTE `model_fingerprint` (coeficientes ajustados, receta congelada…). */
  model_fingerprint_basis: string;
  nota?: string;
}

interface InterpSummaryBase {
  /** Header constitucional obligatorio del generador — la UI lo muestra tal cual. */
  nota: string;
  surface: string;
  asset: string;
  model_id: string;
  version: string;
  generated_at: string;
  scope: string;
  /** sha256 de TODO el contenido no volátil: la identidad real de la evidencia. */
  artifact_id: string;
  provenance: InterpProvenance;
  /** artifact_id sustituido por un `--supersede` explícito del operador (si hubo). */
  supersedes?: string | null;
}

/**
 * SHAP lineal cerrado (phi_j = coef_j·(z_j−mu_j)) — ridge / bayesian_ridge / ard.
 *
 * Atribución SOLO sobre filas OOS del MISMO walk-forward expanding anual que la ruta
 * de árbol (corrección 2026-07-28: hasta esa fecha esta ruta hacía UN fit global y
 * atribuía sobre su propio train — 1649 de 1654 filas — mientras el artefacto llevaba
 * el header "solo test-folds"). De ahí que su forma sea la de `InterpTreeSummary` más
 * `coef` en `top_features`: con un fit por fold, ese `coef` es la MEDIA por fold y el
 * detalle exacto vive en `provenance.model_fingerprint`.
 */
export interface InterpLinearSummary extends InterpSummaryBase {
  model_type: 'linear';
  method: 'linear_shap_closed_form';
  attribution_not_shap: false;
  /** max |sum(φ) + base − pred_cruda|: la forma cerrada es exacta ⇒ ~1e-16. */
  additivity_max_abs_err: number;
  /**
   * N de entrenamiento NO ambiguo. En expanding los trains son ANIDADOS: `sum(n_train)`
   * por folds contaría las mismas filas varias veces, así que la suma no se publica.
   */
  fit: {
    scheme: string;
    origin: string;
    /** Un fit POR FOLD: `n_fits` > 1 es lo que hace CIERTO el header "solo test-folds". */
    n_fits: number;
    n_train_last_fit: number;
    n_train_distinct_rows: number;
    n_train_by_fold: number[];
    n_train_note: string;
    horizon: number;
    purge_days: number;
    scaler: string;
    params: Record<string, number | string | boolean | null>;
  };
  folds: InterpFold[];
  base_value: number;
  n_rows: number;
  n_features: number;
  n_folds: number;
  top_features: InterpFeatureRow[];
  by_year: Record<string, InterpYearFeatureRow[]>;
  /** Corte por el gate Hurst CONGELADO, igual que la ruta de árbol. */
  by_regime: Record<string, InterpYearFeatureRow[]>;
  regime_gate: string;
  /** Kill-flag A.7 calculado por el GENERADOR: signo del aporte medio que cambia entre años. */
  kill_flags_sign_change_by_year: string[];
  /** Mismo kill-flag A.7 sobre el corte por régimen. */
  kill_flags_sign_change_by_regime: string[];
}

/**
 * TreeSHAP EXACTO nativo del booster (xgboost/lightgbm/catboost), BL-20 fase 2.
 * Sin `coef` (un árbol no lo tiene) y con `by_regime` (gate Hurst congelado).
 * Atribución SOLO sobre filas OOS de folds anuales expanding (una fila jamás se
 * atribuye con el modelo que la vio en su train).
 */
export interface InterpTreeFeatureRow {
  rank: number;
  feature: string;
  mean_abs_shap: number;
  mean_shap: number;
}

/**
 * Bitácora de un fold del walk-forward expanding anual. La comparten las DOS rutas
 * del zoo (lineal y árbol) — desde 2026-07-28 la lineal también refitea por fold.
 */
export interface InterpFold {
  year: number;
  n_train: number;
  n_test: number;
  train_start: string;
  train_end: string;
  base_value: number;
  /** Huella del train EXACTO de este fold (mismas filas ⇒ misma huella). */
  fold_fingerprint: string;
}

export interface InterpTreeSummary extends InterpSummaryBase {
  model_type: 'tree';
  method: 'tree_shap';
  attribution_not_shap: false;
  /** Implementación TreeSHAP concreta usada (no el paquete `shap`, sino el booster). */
  shap_backend: string;
  /** ¿Importaba el paquete `shap` en la corrida? Informativo — el backend es el nativo. */
  shap_package_available: boolean;
  /** max |sum(φ) + base − pred_cruda| sobre todas las filas: TreeSHAP es exacto ⇒ ~0. */
  additivity_max_abs_err: number;
  /**
   * N de entrenamiento NO ambiguo. En expanding los trains son ANIDADOS: `sum(n_train)`
   * por folds contaba las mismas filas varias veces (4959 "filas" sobre ~1.6k reales),
   * así que la suma no se publica en ningún campo.
   */
  fit: {
    scheme: string;
    origin: string;
    n_train_last_fit: number;
    n_train_distinct_rows: number;
    n_train_by_fold: number[];
    n_train_note: string;
    horizon: number;
    purge_days: number;
    scaler: string;
    params: Record<string, number | string | boolean | null>;
  };
  folds: InterpFold[];
  base_value: number;
  n_rows: number;
  n_features: number;
  n_folds: number;
  top_features: InterpTreeFeatureRow[];
  by_year: Record<string, InterpYearFeatureRow[]>;
  by_regime: Record<string, InterpYearFeatureRow[]>;
  regime_gate: string;
  kill_flags_sign_change_by_year: string[];
  kill_flags_sign_change_by_regime: string[];
}

/** Degradación HONESTA: sin backend TreeSHAP no se emite ni un valor de atribución. */
export interface InterpTreeUnavailableSummary extends InterpSummaryBase {
  model_type: 'tree';
  method: 'tree_shap_unavailable';
  attribution_not_shap: false;
  status: 'tree_shap_unavailable';
  reason: 'backend_import_failed' | 'shap_computation_failed' | 'model_artifact_missing';
  detail: string;
}

export interface InterpRuleYear {
  n_days: number;
  pnl_gross: number;
  pnl_beta: number;
  pnl_timing_cov_pos_ret: number;
  costs: number;
  pnl_net: number;
  pct_days_trend_on?: number;
  pct_days_position_active?: number;
  avg_exposure?: number;
}

/** Atribución de reglas (attribution_not_shap=true) — pnl_gross = beta + timing. */
export interface InterpRuleSummary extends InterpSummaryBase {
  model_type: 'rule_based';
  method: 'rule_attribution';
  attribution_not_shap: true;
  rules: {
    gate: string;
    pct_days_trend_on: number;
    pct_days_position_active: number;
    avg_exposure: number;
    n_trades: number;
  };
  pnl_decomposition: InterpRuleYear;
  by_year: Record<string, InterpRuleYear>;
}

export type InterpSummary =
  | InterpLinearSummary
  | InterpTreeSummary
  | InterpTreeUnavailableSummary
  | InterpRuleSummary;

// ─────────────────────────────────────────────────────────── riesgo (GET /api/admin/risk)

export interface AdminRiskUserKill {
  user_id: string;
  email: string | null;
  mode: string;
  kill_switch: boolean;
}

export interface AdminRiskApiPending {
  id: string;
  user_id: string;
  email: string | null;
  exchange: string;
  created_at: string;
}

export interface AdminRiskSuspended {
  id: string;
  email: string;
  status: string;
  created_at: string | null;
}

export interface AdminRiskResponse {
  /** null = SignalBridge inalcanzable (partial). */
  global_kill: { active: boolean | null; detail: string | null };
  user_kills: AdminRiskUserKill[];
  modes: { paper: number; live: number };
  api_pending: AdminRiskApiPending[];
  suspended: AdminRiskSuspended[];
  /** Techos de sistema (espejo de tenant.py CEILING_*). */
  limits: { max_notional_usd: number; max_open_positions: number; max_daily_loss_pct: number };
  partial_errors: string[];
}

// ─────────────────────────────────────────────────────────── ingresos (GET /api/admin/revenue)

/**
 * Estructura fiel del prototipo con TODOS los valores null hasta Fase 6 (billing):
 * la UI pinta "— · Fase 6". Nunca mocks.
 */
export interface AdminRevenueResponse {
  kpis: { mrr: number | null; arr: number | null; arpu: number | null; ltv: number | null };
  por_plan: Array<{ plan: string; amount: number | null; pct: number | null }>;
  movimiento: {
    nuevo: number | null; expansion: number | null; contraccion: number | null;
    churn: number | null; neto: number | null;
  };
  cobros: {
    exitosos: number | null; fallidos: number | null;
    reembolsos: number | null; contracargos: number | null;
  };
  por_activo: Array<{ symbol: string; amount: number | null }>;
  dunning: Array<{ user: string; plan: string; amount: number | null; attempts: string | null; reason: string | null }>;
  phase_note: string;
}

// ─────────────────────────────────────────────────────────── catálogo (GET /api/admin/catalog)

export interface AdminCatalogRow {
  asset_id: string;
  symbol: string;
  name: string;
  asset_class: string;
  status: 'available' | 'coming_soon';
  addon_price_month: number | null;
}

export interface AdminCatalogResponse {
  assets: AdminCatalogRow[];
  /** El registry lo genera el pipeline (RegistryBuilder) — solo lectura desde la consola. */
  mutable: false;
  note: string;
}

// ─────────────────────────────────────────────────────────── "ver como" (impersonate read-only)

/** Cookie httpOnly firmada (HMAC NEXTAUTH_SECRET): `role|exp|sig`. */
export const VIEW_AS_COOKIE = 'gm-view-as';
/** Espejo legible por el cliente (solo el rol, no sensible) para banner/hooks. */
export const VIEW_AS_ROLE_COOKIE = 'gm-view-as-role';
export const VIEW_AS_TTL_SECONDS = 30 * 60;

export interface ImpersonateRequest {
  /** Uno de los dos: usuario concreto (se resuelve su rol) o rol directo. */
  user_id?: string;
  role?: Role;
  /** Obligatorio — va al audit_log ('view_as_start'). */
  motivo: string;
}

export interface ImpersonateResponse {
  role: Role;
  expires_at: string;
}
