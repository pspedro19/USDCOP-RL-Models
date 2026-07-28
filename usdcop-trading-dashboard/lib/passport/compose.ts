/**
 * Passport / Control Tower composer (BL-32, CTR-PASSPORT-001) — SERVER ONLY.
 * ==========================================================================
 *
 * FABRIC §24.4 says the Passport should be "un SELECT": one read that reconstructs
 * identity + governance + lineage + performance × 5 envs + execution/risk, replacing
 * the Airflow → MLflow → JSONs → SQL walk. The SQL views that would back that SELECT
 * (`v_strategy_passport_live`, `mv_strategy_performance_daily`) require fact tables
 * that DO NOT EXIST yet (BL-18 metric engine, BL-21/BL-22 exec + fact_pnl, BL-24
 * lineage). See `.claude/specs/platform/passport-control-tower.md` for the DDL that
 * lands the day those tables do.
 *
 * Until then this module IS the composition, over the same shape, reading only
 * PUBLISHED artifacts under `public/data/**` plus the governance projection that
 * `scripts/pipeline/export_control_tower.py` writes to `data/control-tower/` —
 * deliberately OUTSIDE the web root, because it carries INTERNALS (see
 * `controlTowerRoot` below). Three rules govern it:
 *
 *  1. It NEVER computes a performance statistic. It copies numbers other systems
 *     published and does date arithmetic over published dates. The one exception
 *     is COUNTING (how many strategies are in each state) — that is not a metric.
 *  2. Any figure with no published artifact becomes `unavailable(pending)` naming
 *     the backlog item that will supply it. Never 0, never "—", never an estimate.
 *  3. N<20 ⇒ the §6 guard strips ratios and p-values before the payload leaves.
 *
 * The frontend renders this; it does not re-evaluate any of it (§24.1).
 */

import { promises as fs } from 'fs';
import path from 'path';

import {
  BOOK_STATES,
  HEALTH_CLOCKS,
  MIN_TRADES_FOR_RATIOS,
  N_MAX_TRIALS,
  PASSPORT_CONTRACT_ID,
  PASSPORT_CONTRACT_VERSION,
  PASSPORT_ENVS,
  DSR_BAR,
  smallSampleReason,
  sourced,
  suppressSmallSample,
  unavailable,
  type BookState,
  type ControlTowerSnapshot,
  type EnvPerformance,
  type HealthClock,
  type PassportEnv,
  type PassportGate,
  type PendingInterface,
  type Sourced,
  type StrategyPassport,
  type TowerBook,
  type TowerData,
  type TowerFamilyTrials,
  type TowerPairedTest,
  type TowerSleeve,
} from '@/lib/contracts/passport.contract';

// ────────────────────────────────────────────────────────────── fs plumbing

const PUBLIC_DATA = path.join(process.cwd(), 'public', 'data');
/** Prefix used in `source.path` so every figure is traceable from the repo root. */
const REPO_PREFIX = 'usdcop-trading-dashboard/public/data';

/**
 * Governance projection root — deliberately OUTSIDE `public/` (CODEX P0, precedent
 * C-006/`data/interpretability/`). The projection carries trials, DSR inputs and gate
 * state: INTERNALS. Under `public/` it is reachable through the `/data/**` static path,
 * which the edge middleware gates with *a session only*, so a `free`/`subscriber` could
 * read it and bypass the `research:read` the Passport requires (`rbac.md` §8). Living
 * here, the ONLY reader is this server-side composer behind `/api/passport/**`.
 *
 * Default `<repo>/data/control-tower` (the dashboard's cwd is `usdcop-trading-dashboard/`);
 * override with `CONTROL_TOWER_DATA_DIR` for containers/tests.
 */
function controlTowerRoot(): string {
  const env = process.env.CONTROL_TOWER_DATA_DIR;
  if (env && env.trim()) return path.resolve(env.trim());
  return path.resolve(process.cwd(), '..', 'data', 'control-tower');
}

/** Read a JSON artifact under `public/data/`. Missing/corrupt ⇒ null (never throws:
 *  a missing artifact is a DEGRADED state the contract models, not an error). */
async function readData<T>(relPath: string): Promise<T | null> {
  return readJsonAt(path.join(PUBLIC_DATA, relPath));
}

/** Same contract as `readData`, for artifacts that must NOT be web-servable. */
async function readGovernance<T>(fileName: string): Promise<T | null> {
  return readJsonAt(path.join(controlTowerRoot(), fileName));
}

async function readJsonAt<T>(absPath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(absPath, 'utf-8')) as T;
  } catch {
    return null;
  }
}

const src = (relPath: string) => `${REPO_PREFIX}/${relPath}`;

// ─────────────────────────────────────────────────────── published artifact shapes

interface RegistryStrategy {
  strategy_id: string;
  asset_id: string;
  status?: string;
  display_name?: string;
  pipeline_type?: string;
  timeframe?: string;
  manifest?: string;
  surface?: string;
  active_version?: string;
  has_production?: boolean;
  backtest_years?: number[];
  return_pct?: number | null;
  sharpe?: number | null;
  p_value?: number | null;
}

interface RegistryFile {
  generated_at?: string;
  assets?: Array<{ asset_id: string; symbol: string; display_name?: string; asset_class?: string }>;
  strategies?: RegistryStrategy[];
  default?: { asset_id: string; strategy_id: string };
}

interface ManifestBacktest {
  model_version: string;
  year: number;
  immutable_id?: string;
  summary?: string;
  gates?: { passed?: number; of?: number; recommendation?: string };
  headline?: Record<string, number | null | undefined>;
}

interface StrategyManifest {
  strategy_id: string;
  asset_id?: string;
  display_name?: string;
  pipeline_type?: string;
  timeframe?: string;
  status?: string;
  surface?: string;
  backtests?: ManifestBacktest[];
  production?: { model_version?: string; year?: number; summary?: string; trades?: string; updated_at?: string };
  approval?: { file?: string; status?: string };
  model_versions?: Array<{
    version: string; active: boolean; trained_at: string | null; train_window: string | null;
    feature_hash: string | null; norm_stats_hash: string | null; artifact_uri: string | null;
  }>;
}

interface SummaryStrategyBlock {
  total_return_pct?: number | null;
  sharpe?: number | null;
  calmar?: number | null;
  max_dd_pct?: number | null;
  win_rate_pct?: number | null;
  profit_factor?: number | null;
  n_long?: number | null;
  n_short?: number | null;
  insufficient_trades?: boolean;
}

interface SummaryFile {
  generated_at?: string;
  strategy_id?: string;
  year?: number;
  initial_capital?: number | null;
  n_trades?: number | null;
  insufficient_trades?: boolean;
  strategies?: Record<string, SummaryStrategyBlock>;
  statistical_tests?: { p_value?: number | null };
}

interface ApprovalStateFile {
  status?: string;
  strategy?: string;
  gates?: PassportGate[];
  backtest_metrics?: Record<string, number | null>;
  last_updated?: string;
}

interface PaperJudgeWindow {
  starts_after: string;
  n_trades: number | null;
  pnl_pct_compound: number | null;
  note?: string | null;
}

interface PaperLedgerFile {
  anchor?: string;
  generated_at?: string;
  labels?: Record<string, string>;
  judge_note?: string;
  strategies?: Record<string, {
    ret_2026_ytd_pct: number | null;
    n_trades: number | null;
    note_n?: string | null;
    judge_window: PaperJudgeWindow | null;
  }>;
  book?: {
    weights_frozen?: Record<string, number>;
    policy?: string;
    weeks?: Array<{ iso_week: string; book_ret_pct: number | null; status?: string }>;
  };
}

interface SystemHealthFile {
  generated_at?: string;
  promotions_frozen?: boolean;
  withdrawal_triggered?: boolean;
  clocks?: Record<string, {
    clock?: string; signal?: string; actions?: string[]; evaluated_at?: string;
    metrics?: Record<string, unknown>;
  }>;
}

interface GovernanceFile {
  generated_at?: string;
  ledger_available?: boolean;
  n_global?: number | null;
  n_max_trials?: number;
  assets?: Record<string, {
    n_trials_total?: number | null;
    ledger_forecast_trials?: number | null;
    ledger_action_trials?: number | null;
    ledger_total?: number | null;
    n_family?: number | null;
    n_cluster?: number | null;
    n_global?: number | null;
    hypothesis_registry?: string | null;
    withdrawal_protocol?: string | null;
    withdrawal_protocol_signed?: boolean;
  }>;
  families?: Array<{
    family_id: string; kind?: string | null; cluster_id?: string | null;
    n_trials?: number; closed?: boolean;
  }>;
}

// ────────────────────────────────────────────────── artifact paths (single place)

const P = {
  registry: 'registry.json',
  systemHealth: 'production/system_health.json',
  paperLedger: 'production/paper/candidates_ledger_2026.json',
  deployStatus: 'production/deploy_status.json',
  manifest: (id: string) => `strategies/${id}/manifest.json`,
} as const;

/** Governance projection: file name + repo-root path, NOT under `public/` (see
 *  `controlTowerRoot`). Kept apart from `P` so nobody re-adds it to the web root. */
const GOVERNANCE_FILE = 'governance.json';
const GOVERNANCE_PATH = `data/control-tower/${GOVERNANCE_FILE}`;

// ──────────────────────────────────────────── pending interfaces (the acople map)

/**
 * The single, machine-readable list of everything the Passport/Tower declares but
 * cannot source yet. Each entry names the backlog item and the artifact/table the
 * composer will read the day it exists. THIS is the hand-off contract with the
 * other lane: when a producer lands, only this file changes — never the contract,
 * never the view.
 */
export const PENDING_INTERFACES: PendingInterface[] = [
  {
    backlog_id: 'BL-18',
    field: 'passport.metric_engine · performance.*',
    produced_by: 'catálogo de métricas + motor único + metric_event',
    note: 'FABRIC §24.4 exige la MISMA métrica del MISMO motor en las cinco columnas. '
      + 'Hoy cada columna trae la métrica que su propio bundle publicó: comparables en '
      + 'orden de magnitud, NO idénticas en definición. Se declara, no se disimula.',
  },
  {
    backlog_id: 'BL-21',
    field: 'passport.live.* · performance.canary',
    produced_by: 'exec.* event sourcing (órdenes/fills/idempotencia)',
    note: 'Sin event sourcing no hay órdenes abiertas, último fill ni entorno canary reales.',
  },
  {
    backlog_id: 'BL-22',
    field: 'tower.book.pnl_* · book.capital · sleeves.timing_ratio · sleeves.turnover',
    produced_by: 'fact_position / fact_pnl + timing_ratio persistido',
    note: 'El PnL del LIBRO y la atribución timing/beta/carry salen de hechos contables. '
      + 'BL-07 calculó timing_ratio una sola vez y NO lo persistió (por diseño): no hay de dónde leerlo.',
  },
  {
    backlog_id: 'BL-24',
    field: 'passport.lineage.lineage_graph · tower.data.last_vintage_revision',
    produced_by: 'nodes/edges de linaje + camino dorado + revisiones tipificadas',
    note: 'Los fingerprints existen por bundle; el GRAFO de linaje y la última revisión de '
      + 'vintage detectada no tienen productor.',
  },
  {
    backlog_id: 'BL-25',
    field: 'tower.data.clocks.pnl · sleeves.retirement_signal',
    produced_by: 'control__system_health (tres relojes: datos/modelo/PnL) por estrategia',
    // Qué relojes trae hoy el artefacto NO se afirma aquí: lo dice `clocks[*].source.pending`,
    // derivado del payload (F-07). Esta nota solo declara lo que sigue sin productor.
    note: 'system_health.json publica relojes y banderas GLOBALES. No existe evaluación de '
      + 'retiro POR ESTRATEGIA: por eso el semáforo de cada sleeve es "unknown" y no se le '
      + 'imputa la bandera global (sería atribuir un hecho del sistema a una sleeve).',
  },
  {
    backlog_id: 'BL-26 / BL-27',
    field: 'tower.book.vol_* · gross/net · cvar · correlation_matrix · risk.*',
    produced_by: 'portfolio_snapshot + allocator v1 (inverse-vol + caps)',
    note: 'Sin barrera temporal del libro ni allocator no hay exposición, vol objetivo, '
      + 'CVaR, ρ entre sleeves ni multiplicadores m_forward/m_dd.',
  },
  {
    backlog_id: 'BL-23',
    field: 'performance.held_out',
    produced_by: 'backfill anti-supervivencia (campeonas+candidatas+retiradas+baselines)',
    note: 'No hay entorno held-out reconstruido para ninguna estrategia.',
  },
  {
    backlog_id: 'BL-32 (SQL)',
    field: 'v_strategy_passport_live · mv_strategy_performance_daily',
    produced_by: 'PostgreSQL (DDL en .claude/specs/platform/passport-control-tower.md)',
    note: 'La MV nocturna no puede sostener estado live (se reemplaza en el refresh): por eso '
      + 'la división vista-live / MV-histórica. Ambas quedan como DDL de referencia hasta que '
      + 'existan sus tablas de hechos.',
  },
];

// ─────────────────────────────────────────────────────────────── env performance

/** Why a trade count is missing. Its presence in `pending` is what turns "we do
 *  not know N" into a visible fact instead of a silent licence to publish. */
const NO_TRADE_COUNT_PENDING =
  'BL-18 — ningún artefacto publicado de esta estrategia declara el conteo de trades '
  + '(sin N no se publica Sharpe/p-value/DSR: quant-constitution §6, fail-closed)';

/** An env with no published artifact at all — all fields unavailable, never zeros. */
function emptyEnv(env: PassportEnv, pending: string): EnvPerformance {
  const u = <T = number>() => unavailable<T>(pending);
  return {
    env,
    period_label: u<string>(),
    return_pct: u(), n_trades: u(), max_dd_pct: u(), win_rate_pct: u(),
    profit_factor: u(), sharpe: u(), calmar: u(), p_value: u(),
    dsr_family: u(), timing_ratio: u(),
    insufficient_trades: false,
  };
}

/** Build an env block from a published `summary_*.json`-shaped artifact. */
function envFromSummary(
  env: PassportEnv,
  summary: SummaryFile,
  strategyId: string,
  artifactPath: string,
  periodLabel: string,
  note?: string,
): EnvPerformance {
  const block = summary.strategies?.[strategyId] ?? {};
  const nTrades = nTradesFromSummary(summary, strategyId);
  const perf: EnvPerformance = {
    env,
    period_label: sourced<string>(periodLabel, artifactPath, note),
    return_pct: sourced(block.total_return_pct ?? null, artifactPath, note),
    n_trades: nTrades != null
      ? sourced(nTrades, artifactPath, note)
      : unavailable(NO_TRADE_COUNT_PENDING),
    max_dd_pct: sourced(block.max_dd_pct ?? null, artifactPath, note),
    win_rate_pct: sourced(block.win_rate_pct ?? null, artifactPath, note),
    profit_factor: sourced(block.profit_factor ?? null, artifactPath, note),
    sharpe: sourced(block.sharpe ?? null, artifactPath, note),
    calmar: sourced(block.calmar ?? null, artifactPath, note),
    p_value: sourced(summary.statistical_tests?.p_value ?? null, artifactPath, note),
    // The DSR lives in the approval gates, not in the summary — merged by the caller.
    dsr_family: unavailable('BL-18 (DSR por entorno); hoy solo el gate de aprobación lo publica'),
    timing_ratio: unavailable('BL-22 (timing_ratio persistido); BL-07 fue one-off sin persistencia'),
    insufficient_trades: !!summary.insufficient_trades || !!block.insufficient_trades,
  };
  // §6 guard runs LAST so nothing inferential can slip through on a small sample.
  return suppressSmallSample(perf);
}

/**
 * The trade count of a published bundle summary. `n_trades` when the exporter
 * wrote it, else `n_long + n_short` from the strategy block — the fallback that
 * already existed for the live env and was NOT used for the backtest one (S-04).
 */
function nTradesFromSummary(summary: SummaryFile | null, strategyId: string): number | null {
  if (!summary) return null;
  if (typeof summary.n_trades === 'number') return summary.n_trades;
  const block = summary.strategies?.[strategyId];
  if (!block) return null;
  const long = block.n_long;
  const short = block.n_short;
  if (typeof long !== 'number' && typeof short !== 'number') return null;
  return (typeof long === 'number' ? long : 0) + (typeof short === 'number' ? short : 0);
}

/** Build the backtest env from the manifest headline (published, immutable bundle).
 *
 *  S-04: `headline.trades` exists in exactly THREE manifests (`smart_simple_*`).
 *  Every Gold/BTC bundle omits it and `spx500_*` spells it `n_trades`, so the
 *  count silently became `null` — and the §6 guard, which keyed off that null,
 *  did nothing precisely where N was smallest (`btc_hodl_b1` = 1 trade). The
 *  count is now taken from the headline in either spelling, else from the
 *  bundle's own `summary_*.json` (an artifact the manifest itself points at),
 *  and when neither yields a number the env is `unavailable` — never published
 *  with a null that reads as "no small-sample problem here". */
async function envFromManifestBacktest(
  entry: ManifestBacktest,
  manifestPath: string,
): Promise<EnvPerformance> {
  const h = entry.headline ?? {};
  const headlineN = (h.trades ?? h.n_trades ?? null) as number | null;
  const summary = entry.summary ? await readData<SummaryFile>(entry.summary) : null;
  const summaryN = nTradesFromSummary(summary, summary?.strategy_id ?? '');
  const nTrades = typeof headlineN === 'number' ? headlineN : summaryN;
  const nPath = typeof headlineN === 'number' || !entry.summary
    ? manifestPath
    : src(entry.summary);
  const perf: EnvPerformance = {
    env: 'backtest',
    period_label: sourced<string>(`${entry.model_version} · ${entry.year}`, manifestPath),
    return_pct: sourced(h.return_pct ?? null, manifestPath),
    n_trades: nTrades != null
      ? sourced(nTrades, nPath, nPath === manifestPath ? undefined : 'conteo del summary del bundle')
      : unavailable(NO_TRADE_COUNT_PENDING),
    max_dd_pct: sourced(h.max_dd_pct ?? null, manifestPath),
    win_rate_pct: sourced(h.win_rate_pct ?? null, manifestPath),
    profit_factor: unavailable('BL-18 (profit_factor no está en el headline del manifiesto)'),
    sharpe: sourced(h.sharpe ?? null, manifestPath),
    calmar: unavailable('BL-18 (Calmar no está en el headline del manifiesto)'),
    p_value: sourced(h.p_value ?? null, manifestPath),
    dsr_family: unavailable('BL-18 (DSR por entorno); hoy solo el gate de aprobación lo publica'),
    timing_ratio: unavailable('BL-22 (timing_ratio persistido); BL-07 fue one-off sin persistencia'),
    insufficient_trades: false,
  };
  return suppressSmallSample(perf);
}

/** Paper env from the sealed judge window of the A/B ledger (BL-05's artifact). */
function envFromPaperLedger(
  entry: NonNullable<PaperLedgerFile['strategies']>[string],
  ledgerPath: string,
): EnvPerformance {
  const jw = entry.judge_window;
  if (!jw) {
    return emptyEnv('paper', 'candidata sin judge_window sellada en el paper ledger');
  }
  const note = jw.note ?? entry.note_n ?? null;
  const perf: EnvPerformance = {
    env: 'paper',
    period_label: sourced<string>(`juez sellado desde ${jw.starts_after}`, ledgerPath, note ?? undefined),
    return_pct: sourced(jw.pnl_pct_compound, ledgerPath, note ?? undefined),
    n_trades: sourced(jw.n_trades, ledgerPath, note ?? undefined),
    max_dd_pct: unavailable('BL-22 (el ledger publica conteo y PnL compuesto, no DD)'),
    win_rate_pct: unavailable('BL-22 (el ledger publica conteo y PnL compuesto)'),
    profit_factor: unavailable('BL-22 (el ledger publica conteo y PnL compuesto)'),
    sharpe: unavailable('juez sellado con N<20 — quant-constitution §6'),
    calmar: unavailable('juez sellado con N<20 — quant-constitution §6'),
    p_value: unavailable('juez sellado con N<20 — quant-constitution §6'),
    dsr_family: unavailable('BL-18 (DSR del juez forward)'),
    timing_ratio: unavailable('BL-22 (timing_ratio persistido)'),
    insufficient_trades: (jw.n_trades ?? 0) < 20,
  };
  return suppressSmallSample(perf);
}

// ────────────────────────────────────────────────────────────── passport (per id)

/** Locate the approval artifact a manifest points at, with the legacy fallback. */
async function readApproval(manifest: StrategyManifest | null, strategyId: string) {
  const candidates = [
    manifest?.approval?.file,
    `production/approval_state_${strategyId}.json`,
    strategyId === 'smart_simple_v11' ? 'production/approval_state.json' : null,
  ].filter((x): x is string => !!x);
  for (const rel of candidates) {
    const data = await readData<ApprovalStateFile>(rel);
    if (data) return { data, path: src(rel) };
  }
  return { data: null, path: null };
}

/** The DSR the published approval gate carries — the ONLY DSR with a source today. */
function dsrFromGates(gates: PassportGate[] | undefined): number | null {
  const gate = gates?.find((g) => g.gate === 'deflated_sharpe');
  return gate?.value ?? null;
}

export async function composeStrategyPassport(strategyId: string): Promise<StrategyPassport | null> {
  const registry = await readData<RegistryFile>(P.registry);
  const entry = registry?.strategies?.find((s) => s.strategy_id === strategyId);
  const manifest = await readData<StrategyManifest>(P.manifest(strategyId));
  if (!entry && !manifest) return null;

  const assetId = entry?.asset_id ?? manifest?.asset_id ?? 'unknown';
  const manifestPath = src(P.manifest(strategyId));
  const registryPath = src(P.registry);
  const governance = await readGovernance<GovernanceFile>(GOVERNANCE_FILE);
  const governancePath = GOVERNANCE_PATH;
  const gov = governance?.assets?.[assetId];
  const approval = await readApproval(manifest, strategyId);

  // ── performance × 5 envs. Every env starts unavailable and is only filled from a
  //    real artifact — that is what keeps "sin datos" from being rendered as zero.
  const performance = Object.fromEntries(
    PASSPORT_ENVS.map((env) => [env, emptyEnv(env, `sin artefacto publicado para el entorno ${env}`)]),
  ) as Record<PassportEnv, EnvPerformance>;

  performance.held_out = emptyEnv('held_out',
    'BL-23 (backfill anti-supervivencia) — ningún activo tiene held-out reconstruido');
  performance.canary = emptyEnv('canary',
    'BL-21 (event sourcing exec.*) — no existe entorno canary');

  // backtest ← the manifest entry for the active version, most recent year.
  const activeVersion = entry?.active_version ?? manifest?.production?.model_version ?? null;
  const backtests = manifest?.backtests ?? [];
  const chosen = [...backtests]
    .filter((b) => !activeVersion || b.model_version === activeVersion)
    .sort((a, b) => (b.year ?? 0) - (a.year ?? 0))[0]
    ?? [...backtests].sort((a, b) => (b.year ?? 0) - (a.year ?? 0))[0];
  if (chosen) performance.backtest = await envFromManifestBacktest(chosen, manifestPath);

  // live ← the published production/forward bundle. Labelled honestly: it is a
  // forward from the pipeline, NOT reconciled against exchange fills (BL-21/22).
  const prodSummaryRel = manifest?.production?.summary
    ?? (strategyId === 'smart_simple_v11' ? 'production/summary.json' : null);
  if (prodSummaryRel) {
    const prodSummary = await readData<SummaryFile>(prodSummaryRel);
    if (prodSummary) {
      performance.live = envFromSummary(
        'live', prodSummary, strategyId, src(prodSummaryRel),
        `forward ${prodSummary.year ?? manifest?.production?.year ?? ''}`.trim(),
        'forward publicado por el pipeline — NO reconciliado contra fills (BL-21/BL-22)',
      );
    }
  }

  // paper ← the sealed judge window of the A/B ledger (BL-05).
  const ledger = await readData<PaperLedgerFile>(P.paperLedger);
  const paperEntry = ledger?.strategies?.[strategyId];
  if (paperEntry) performance.paper = envFromPaperLedger(paperEntry, src(P.paperLedger));

  const gates = approval.data?.gates;
  const dsr = dsrFromGates(gates);

  const passport: StrategyPassport = {
    contract: PASSPORT_CONTRACT_ID,
    contract_version: PASSPORT_CONTRACT_VERSION,
    strategy_id: strategyId,
    generated_at: new Date().toISOString(),
    identity: {
      strategy_id: strategyId,
      asset_id: assetId,
      display_name: entry?.display_name ?? manifest?.display_name ?? strategyId,
      surface: entry?.surface ?? manifest?.surface ?? null,
      engine_type: entry?.pipeline_type ?? manifest?.pipeline_type ?? null,
      status: entry?.status ?? manifest?.status ?? null,
      active_version: activeVersion
        ? sourced<string>(activeVersion, entry ? registryPath : manifestPath)
        : unavailable<string>('el manifiesto no declara versión activa'),
      timeframe: entry?.timeframe ?? manifest?.timeframe ?? null,
    },
    governance: {
      n_trials_total: gov?.n_trials_total != null
        ? sourced(gov.n_trials_total, governancePath, gov.hypothesis_registry ?? undefined)
        : unavailable('BL-09/BL-10 — el activo no tiene conteo de trials proyectado'),
      n_trials_forecast: gov?.ledger_forecast_trials != null
        ? sourced(gov.ledger_forecast_trials, governancePath, 'linaje FT (ADR-0022)')
        : unavailable('BL-09 — ledger FT/AT no proyectado para este activo'),
      n_trials_action: gov?.ledger_action_trials != null
        ? sourced(gov.ledger_action_trials, governancePath, 'linaje AT (ADR-0022)')
        : unavailable('BL-09 — ledger FT/AT no proyectado para este activo'),
      n_family: gov?.n_family != null
        ? sourced(gov.n_family, governancePath)
        : unavailable('BL-09/BL-11 — N_family no proyectado'),
      n_cluster: gov?.n_cluster != null
        ? sourced(gov.n_cluster, governancePath)
        : unavailable('BL-09/BL-11 — N_cluster no proyectado'),
      n_global: gov?.n_global != null
        ? sourced(gov.n_global, governancePath)
        : unavailable('BL-09 — N_global no proyectado'),
      dsr_family: dsr != null && approval.path
        ? sourced(dsr, approval.path, `gate deflated_sharpe (bar ${DSR_BAR})`)
        : unavailable('BL-18 — sin gate DSR publicado para esta estrategia'),
      dsr_bar: DSR_BAR,
      approval_status: approval.data?.status && approval.path
        ? sourced<string>(approval.data.status, approval.path)
        : unavailable<string>('sin approval_state publicado'),
      gates: gates && approval.path
        ? sourced<PassportGate[]>(gates, approval.path)
        : unavailable<PassportGate[]>('sin approval_state publicado'),
      withdrawal_protocol: gov?.withdrawal_protocol
        ? sourced<string>(gov.withdrawal_protocol, governancePath, 'protocolo firmado ex-ante')
        : unavailable<string>('quant-constitution §5 — protocolo de retiro NO firmado para este activo'),
      // Deliberately NOT derived from the GLOBAL health flags: imputing a
      // system-level trigger to one sleeve would be attributing a fact that is
      // not about it. Per-strategy retirement evaluation is BL-25.
      retirement_signal: 'unknown',
      retirement_reason: gov?.withdrawal_protocol_signed
        ? 'protocolo firmado, pero no existe evaluación de retiro POR ESTRATEGIA publicada (BL-25)'
        : 'sin protocolo de retiro firmado para el activo (quant-constitution §5)',
    },
    lineage: {
      model_versions: manifest?.model_versions
        ? sourced(manifest.model_versions, manifestPath)
        : unavailable<StrategyManifest['model_versions']>('el manifiesto no publica model_versions') as never,
      spec_fingerprint: unavailable<string>('BL-17 — fingerprints canónicos + canonical writer'),
      feature_set_hash: unavailable<string>('BL-39 — feature contracts por estrategia-versión'),
      policy_hash: unavailable<string>('BL-45 — motor de políticas (policy_hash/params_hash)'),
      lineage_graph: unavailable<unknown>('BL-24 — nodes/edges de linaje + camino dorado'),
    },
    performance,
    live: {
      open_orders: unavailable('BL-21 — event sourcing exec.*'),
      last_fill_at: unavailable<string>('BL-21 — event sourcing exec.*'),
      quarantined: unavailable<boolean>('BL-21 — cuarentena por reconciliación'),
      reconciled: unavailable<boolean>('BL-21/BL-22 — reconciliación contra fills'),
      kill_switch_engaged: unavailable<boolean>('BL-30 — kill switch independiente de Airflow'),
      deploy_status: await deployStatusFor(strategyId),
      last_signal_at: manifest?.production?.updated_at
        ? sourced<string>(manifest.production.updated_at, manifestPath, 'updated_at del bundle de producción')
        : unavailable<string>('sin bundle de producción publicado'),
    },
    risk: {
      current_exposure: unavailable('BL-26 — portfolio_snapshot'),
      vol_target_pct: unavailable('BL-27 — allocator v1'),
      vol_forecast_pct: unavailable('BL-27 — allocator v1'),
      m_forward: unavailable('BL-27 — multiplicadores m_forward/m_dd'),
      m_dd: unavailable('BL-27 — multiplicadores m_forward/m_dd'),
      rho_max: unavailable('BL-26 — matriz de correlación entre sleeves'),
      turnover: unavailable('BL-22 — fact_position'),
    },
    metric_engine: unavailable<string>(
      'BL-18 — catálogo de métricas + motor único',
      'Las cinco columnas NO vienen todavía del mismo motor: cada una trae la métrica que su '
      + 'propio bundle publicó. Compararlas exige leer la fuente de cada celda.',
    ),
    pending_interfaces: PENDING_INTERFACES,
  };

  return passport;
}

/** Deploy status is a runtime file: present ⇒ published, absent ⇒ unavailable. */
async function deployStatusFor(strategyId: string): Promise<Sourced<string>> {
  const deploy = await readData<{ status?: string; strategy_id?: string; error?: string }>(P.deployStatus);
  if (!deploy || (deploy.strategy_id && deploy.strategy_id !== strategyId)) {
    return unavailable<string>('sin deploy_status.json publicado para esta estrategia');
  }
  return sourced<string>(deploy.status ?? null, src(P.deployStatus), deploy.error ?? undefined);
}

// ─────────────────────────────────────────────────────────────── control tower

/**
 * Map registry status → FABRIC book state. This is a LABEL mapping over a published
 * vocabulary, not an inference: `production` is the deployed champion, `paper`/
 * `experimental` are paper sleeves. `CANARY`/`REDUCED`/`QUARANTINED` have NO producer
 * (BL-27 tiering / BL-30 ops), so they stay `null` — never 0, which would read as
 * "we checked and there are none".
 */
const REGISTRY_TO_BOOK_STATE: Record<string, BookState> = {
  production: 'CHAMPION',
  paper: 'PAPER',
  experimental: 'PAPER',
};

/** Days between a published ISO date and today, at calendar-day granularity.
 *  Date arithmetic over published dates — not a metric (§24.1 stays intact). */
export function judgeElapsedDays(dateIso: string | null | undefined, now: Date): number | null {
  if (!dateIso) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(dateIso).trim());
  if (!m) return null;
  const start = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  return Math.round((today - start) / 86_400_000);
}

async function composeSleeve(
  entry: RegistryStrategy,
  governance: GovernanceFile | null,
  ledger: PaperLedgerFile | null,
  now: Date,
): Promise<TowerSleeve> {
  const registryPath = src(P.registry);
  const governancePath = GOVERNANCE_PATH;
  const gov = governance?.assets?.[entry.asset_id];
  const manifest = await readData<StrategyManifest>(P.manifest(entry.strategy_id));
  const approval = await readApproval(manifest, entry.strategy_id);
  const dsr = dsrFromGates(approval.data?.gates);

  // Which envs actually carry published data — the honest version of "env".
  const envsWithData: PassportEnv[] = [];
  if ((manifest?.backtests ?? []).length) envsWithData.push('backtest');
  const paperEntry = ledger?.strategies?.[entry.strategy_id];
  if (paperEntry?.judge_window) envsWithData.push('paper');
  if (entry.has_production || manifest?.production?.summary) envsWithData.push('live');

  // Headline figures: registry.json publishes them per strategy (it is itself a
  // published projection built by the Python RegistryBuilder).
  //
  // S-04: the count is resolved the same way as the backtest env — headline in
  // either spelling, else the bundle summary the manifest points at — and an
  // UNRESOLVED count is treated as insufficient (fail-closed), because the rows
  // that hid their N were the 1-trade ones.
  const chosenBacktest = manifest?.backtests?.find(
    (b) => b.model_version === entry.active_version)
    ?? [...(manifest?.backtests ?? [])].sort((a, b) => (b.year ?? 0) - (a.year ?? 0))[0];
  const headlineN = (chosenBacktest?.headline?.trades
    ?? chosenBacktest?.headline?.n_trades ?? null) as number | null;
  const backtestSummary = chosenBacktest?.summary
    ? await readData<SummaryFile>(chosenBacktest.summary)
    : null;
  const nTrades = typeof headlineN === 'number'
    ? headlineN
    : nTradesFromSummary(backtestSummary, backtestSummary?.strategy_id ?? '');
  const nTradesPath = typeof headlineN === 'number' || !chosenBacktest?.summary
    ? src(P.manifest(entry.strategy_id))
    : src(chosenBacktest.summary);
  const insufficient = nTrades == null || nTrades < MIN_TRADES_FOR_RATIOS;

  const sleeve: TowerSleeve = {
    strategy_id: entry.strategy_id,
    asset_id: entry.asset_id,
    display_name: entry.display_name ?? entry.strategy_id,
    research_state: entry.status ?? null,
    tier: null,       // BL-27
    ops_state: null,  // BL-30
    envs_with_data: envsWithData,
    rho_max: unavailable('BL-26 — matriz de correlación entre sleeves'),
    sharpe: entry.sharpe != null
      ? sourced(entry.sharpe, registryPath, 'headline del registry (bundle publicado)')
      : unavailable('sin Sharpe publicado en el registry'),
    n_trades: nTrades != null
      ? sourced(nTrades as number, nTradesPath)
      : unavailable(NO_TRADE_COUNT_PENDING),
    return_pct: entry.return_pct != null
      ? sourced(entry.return_pct, registryPath, 'headline del registry (bundle publicado)')
      : unavailable('sin retorno publicado en el registry'),
    dsr_family: dsr != null && approval.path
      ? sourced(dsr, approval.path, `gate deflated_sharpe (bar ${DSR_BAR})`)
      : unavailable('BL-18 — sin gate DSR publicado para esta estrategia'),
    n_family: gov?.n_family != null ? sourced(gov.n_family, governancePath)
      : unavailable('BL-09/BL-11 — N_family no proyectado'),
    n_cluster: gov?.n_cluster != null ? sourced(gov.n_cluster, governancePath)
      : unavailable('BL-09/BL-11 — N_cluster no proyectado'),
    n_global: gov?.n_global != null ? sourced(gov.n_global, governancePath)
      : unavailable('BL-09 — N_global no proyectado'),
    timing_ratio: unavailable('BL-22 — timing_ratio persistido (BL-07 fue one-off, no persiste)'),
    m_forward: unavailable('BL-27 — multiplicadores vigentes'),
    m_dd: unavailable('BL-27 — multiplicadores vigentes'),
    turnover: unavailable('BL-22 — fact_position'),
    judge_days: unavailable('sin ventana de juez publicada para esta estrategia'),
    judge_starts_after: unavailable<string>('sin ventana de juez publicada para esta estrategia'),
    retirement_signal: 'unknown',
    retirement_reason: gov?.withdrawal_protocol_signed
      ? 'protocolo firmado; falta evaluación de retiro POR ESTRATEGIA (BL-25)'
      : 'sin protocolo de retiro firmado (quant-constitution §5)',
    insufficient_trades: insufficient,
  };

  // Judge clock: paper candidates run from their sealed `starts_after`; the live
  // champion's judge IS the forward anchored by the ledger. Both dates are published.
  const ledgerPath = src(P.paperLedger);
  if (paperEntry?.judge_window?.starts_after) {
    sleeve.judge_starts_after = sourced<string>(paperEntry.judge_window.starts_after, ledgerPath, 'juez sellado post-freeze');
    const days = judgeElapsedDays(paperEntry.judge_window.starts_after, now);
    sleeve.judge_days = days != null
      ? sourced(days, ledgerPath, 'días transcurridos desde la fecha publicada (aritmética de fechas)')
      : unavailable('fecha de juez no parseable');
  } else if (paperEntry && ledger?.anchor) {
    sleeve.judge_starts_after = sourced<string>(ledger.anchor, ledgerPath, 'ancla del forward (producción)');
    const days = judgeElapsedDays(ledger.anchor, now);
    sleeve.judge_days = days != null
      ? sourced(days, ledgerPath, 'días transcurridos desde el ancla publicada')
      : unavailable('ancla no parseable');
  }

  // §6: without a published N >= 20 the sleeve row must not carry Sharpe or DSR.
  if (insufficient) {
    const reason = smallSampleReason(nTrades ?? null);
    sleeve.sharpe = unavailable(reason);
    sleeve.dsr_family = unavailable(reason);
  }
  return sleeve;
}

function composeBook(
  ledger: PaperLedgerFile | null,
  strategies: RegistryStrategy[],
): TowerBook {
  const ledgerPath = src(P.paperLedger);
  const registryPath = src(P.registry);

  // COUNTING published statuses is not a metric. States with no producer stay null.
  const counts = Object.fromEntries(BOOK_STATES.map((s) => [s, null])) as Record<BookState, number | null>;
  for (const state of ['CHAMPION', 'PAPER'] as BookState[]) counts[state] = 0;
  for (const s of strategies) {
    if (s.status === 'archived') continue;      // archived sleeves are not in the book
    const mapped = REGISTRY_TO_BOOK_STATE[s.status ?? ''];
    if (mapped) counts[mapped] = (counts[mapped] ?? 0) + 1;
  }

  const weights = ledger?.book?.weights_frozen ?? null;

  return {
    capital: unavailable('BL-26 — portfolio_snapshot (el capital del LIBRO, no el del bundle)'),
    pnl_d_pct: unavailable('BL-22 — fact_pnl'),
    pnl_m_pct: unavailable('BL-22 — fact_pnl'),
    pnl_y_pct: unavailable('BL-22 — fact_pnl'),
    vol_forecast_pct: unavailable('BL-27 — allocator v1'),
    vol_target_pct: unavailable('BL-27 — allocator v1'),
    max_dd_pct: unavailable('BL-22 — fact_pnl del LIBRO (el DD por sleeve sí está en su bundle)'),
    gross_exposure: unavailable('BL-26 — portfolio_snapshot'),
    net_exposure: unavailable('BL-26 — portfolio_snapshot'),
    cvar_pct: unavailable('BL-26/BL-27 — riesgo del libro'),
    state_counts: counts,
    state_counts_source: {
      path: registryPath, status: 'published', pending: null,
      note: 'conteo de estados publicados en el registry; CANARY/REDUCED/QUARANTINED no '
        + 'tienen productor (BL-27/BL-30) y quedan en null, nunca en 0',
    },
    correlation_matrix: unavailable('BL-26 — matriz de correlación entre sleeves'),
    diversification_ratio: unavailable('BL-26 — ratio de diversificación'),
    pnl_attribution: unavailable(
      'BL-22 — atribución timing/beta/carry persistida',
      'BL-07 la calculó one-off y por diseño NO la persistió; v11 quedó UNAVAILABLE por falta '
      + 'del retorno subyacente en el adapter. No hay número que leer.',
    ),
    weights: weights
      ? sourced(weights, ledgerPath, ledger?.book?.policy ?? 'pesos congelados del book')
      : unavailable<Record<string, number>>('sin pesos de libro publicados en el paper ledger'),
  };
}

function composeData(
  health: SystemHealthFile | null,
  governance: GovernanceFile | null,
): TowerData {
  const healthPath = src(P.systemHealth);
  const governancePath = GOVERNANCE_PATH;

  // F-07: la nota se DERIVA del payload. La versión anterior afirmaba a mano
  // "system_health publica data y model": una aseveración sobre un artefacto que
  // el artefacto puede desmentir (y desmiente en cuanto BL-25 publique el reloj de
  // PnL). Lo que el usuario lee ahora es lo que el fichero trae hoy.
  const publishedClocks = Object.entries(health?.clocks ?? {})
    .filter(([, v]) => !!v)
    .map(([k]) => k)
    .sort();

  const clocks = Object.fromEntries(HEALTH_CLOCKS.map((c) => {
    const raw = health?.clocks?.[c];
    if (!raw) {
      return [c, unavailable(
        `BL-25 — system_health.json no publica el reloj ${c} (publica: ${publishedClocks.join(', ')})`,
      )];
    }
    return [c, sourced({
      clock: c as HealthClock,
      signal: raw.signal ?? 'unknown',
      actions: raw.actions ?? [],
      evaluated_at: raw.evaluated_at ?? null,
    }, healthPath)];
  })) as TowerData['clocks'];

  const probesNotOk = health?.clocks?.data?.metrics?.probes_not_ok;

  const families: TowerFamilyTrials[] = (governance?.families ?? []).map((f) => ({
    family_id: f.family_id,
    kind: (f.kind === 'action' ? 'action' : 'forecast'),
    cluster_id: f.cluster_id ?? null,
    n_trials: f.n_trials ?? 0,
    closed: !!f.closed,
  }));

  return {
    clocks,
    promotions_frozen: health?.promotions_frozen != null
      ? sourced<boolean>(health.promotions_frozen, healthPath)
      : unavailable<boolean>('system_health.json no publica promotions_frozen'),
    withdrawal_triggered: health?.withdrawal_triggered != null
      ? sourced<boolean>(health.withdrawal_triggered, healthPath, 'bandera GLOBAL del sistema, no por estrategia')
      : unavailable<boolean>('system_health.json no publica withdrawal_triggered'),
    stale_probes: typeof probesNotOk === 'number'
      ? sourced(probesNotOk, healthPath, 'sondas del reloj de datos que no están OK')
      : unavailable('system_health.json no publica el conteo de sondas'),
    replay_parity: unavailable<string>('BL-28 — diff semántico de bundles (paridad replay)'),
    trials_by_family: governance
      ? sourced(families, governancePath, 'proyección de registries/ledger.jsonl + families/')
      : unavailable<TowerFamilyTrials[]>('BL-09/BL-11 — proyección de gobernanza no generada '
        + '(corre scripts/pipeline/export_control_tower.py)'),
    n_global: governance?.n_global != null
      ? sourced(governance.n_global, governancePath, 'contador corriente del ledger append-only')
      : unavailable('BL-09 — ledger de trials no proyectado'),
    // Always the constant: it is a declared spend cap, never a measurement.
    n_max_trials: sourced(N_MAX_TRIALS, 'src/contracts/passport.py',
      'FABRIC §9.7 — cota de GASTO; JAMÁS entra en el DSR'),
    last_vintage_revision: unavailable<string>('BL-24 — revisiones de vintage tipificadas'),
  };
}

/**
 * The paired v11-live vs v12/v14-paper comparison (§24.5). The ledger publishes the
 * two return series' headline numbers; it does NOT publish a paired p-value or
 * e-value, and this module will not manufacture one — running a test here would be
 * a new statistic computed in the BFF, i.e. exactly what §24.1 forbids AND a trial.
 */
function composePairedTests(ledger: PaperLedgerFile | null): TowerPairedTest[] {
  const ledgerPath = src(P.paperLedger);
  const strategies = ledger?.strategies ?? {};
  const baselineId = 'smart_simple_v11';
  const baseline = strategies[baselineId];
  if (!baseline) return [];
  return Object.entries(strategies)
    .filter(([id]) => id !== baselineId)
    .map(([candidateId, candidate]) => ({
      baseline_id: baselineId,
      candidate_id: candidateId,
      baseline_return_pct: sourced(baseline.ret_2026_ytd_pct, ledgerPath, baseline.note_n ?? undefined),
      candidate_return_pct: sourced(candidate.ret_2026_ytd_pct, ledgerPath, candidate.note_n ?? undefined),
      n_paired: sourced(candidate.judge_window?.n_trades ?? null, ledgerPath, 'trades dentro del juez sellado'),
      p_value: unavailable('BL-18 — test pareado con p-value publicado por el motor de métricas'),
      e_value: unavailable('BL-18 — e-value del test pareado'),
      note: ledger?.judge_note
        ?? 'el juez sellado consume SOLO judge_window; la serie completa es monitoreo, no evidencia',
    }));
}

export async function composeControlTower(now: Date = new Date()): Promise<ControlTowerSnapshot> {
  const [registry, governance, health, ledger] = await Promise.all([
    readData<RegistryFile>(P.registry),
    readGovernance<GovernanceFile>(GOVERNANCE_FILE),
    readData<SystemHealthFile>(P.systemHealth),
    readData<PaperLedgerFile>(P.paperLedger),
  ]);

  const strategies = (registry?.strategies ?? []).filter((s) => s.status !== 'archived');
  const sleeves = await Promise.all(
    strategies.map((entry) => composeSleeve(entry, governance, ledger, now)),
  );

  return {
    contract: PASSPORT_CONTRACT_ID,
    contract_version: PASSPORT_CONTRACT_VERSION,
    generated_at: now.toISOString(),
    book: composeBook(ledger, strategies),
    sleeves,
    data: composeData(health, governance),
    paired_tests: composePairedTests(ledger),
    pending_interfaces: PENDING_INTERFACES,
  };
}

/** Lightweight index for the strategy selector — no per-strategy fs fan-out. */
export async function listPassportStrategies(): Promise<Array<{
  strategy_id: string; asset_id: string; display_name: string; status: string | null;
}>> {
  const registry = await readData<RegistryFile>(P.registry);
  return (registry?.strategies ?? [])
    .filter((s) => s.status !== 'archived')
    .map((s) => ({
      strategy_id: s.strategy_id,
      asset_id: s.asset_id,
      display_name: s.display_name ?? s.strategy_id,
      status: s.status ?? null,
    }));
}
