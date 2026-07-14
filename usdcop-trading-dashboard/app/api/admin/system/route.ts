/**
 * GET /api/admin/system — system health for Overview/Sistema (admin:all). Spec §6.
 *
 * Composes INDEPENDENT sub-checks (spec C5): data freshness from the DB (thresholds
 * mirror `.claude/rules/data-freshness.md`), Vote 2 pending count from the published
 * approval_state.json (deep-link only — the approve action stays on /dashboard, never
 * a second button), service reachability, and deploy status. Each sub-check that
 * fails lands in partial_errors; the rest still render.
 */
import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

import { requireAdminRole } from '@/lib/admin/guard';
import {
  FRESHNESS_THRESHOLDS_HOURS,
  type ActiveAlertRow, type FreshnessSource, type FreshnessStatus, type PipelineStageChip,
  type PromTargetRow, type ResourceGauge, type ServiceCheck, type SloMetric,
  type SystemStatus, type Vote2Summary,
} from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

const PROD_DIR = path.join(process.cwd(), 'public', 'data', 'production');
const SB_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';
const TRADING_API_URL = process.env.TRADING_API_URL || 'http://usdcop-trading-api:8000/';
const PROM_URL = process.env.PROMETHEUS_URL || 'http://prometheus:9090';
const ALERTMANAGER_URL = process.env.ALERTMANAGER_URL || 'http://alertmanager:9093';
const AIRFLOW_API_URL = process.env.AIRFLOW_API_URL || '';
const AIRFLOW_API_USER = process.env.AIRFLOW_API_USER || '';
const AIRFLOW_API_PASSWORD = process.env.AIRFLOW_API_PASSWORD || '';

async function fetchJson(url: string, init?: RequestInit, timeoutMs = 3500): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal, cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

/** Un valor escalar de una query instantánea de Prometheus, o null si no hay datos. */
async function promScalar(promql: string): Promise<number | null> {
  const j = (await fetchJson(`${PROM_URL}/api/v1/query?query=${encodeURIComponent(promql)}`)) as
    { data?: { result?: Array<{ value?: [number, string] }> } };
  const v = j?.data?.result?.[0]?.value?.[1];
  return v != null && v !== 'NaN' ? Number(v) : null;
}

/** SLOs de inferencia (§4.11): histogram_quantile sobre el request-duration del inference-api. */
async function readSlos(): Promise<SloMetric[]> {
  const HIST = 'inference_request_duration_seconds_bucket';
  const [p50, p95, p99, errRate, thru] = await Promise.all([
    promScalar(`histogram_quantile(0.50, sum(rate(${HIST}[5m])) by (le))`).catch(() => null),
    promScalar(`histogram_quantile(0.95, sum(rate(${HIST}[5m])) by (le))`).catch(() => null),
    promScalar(`histogram_quantile(0.99, sum(rate(${HIST}[5m])) by (le))`).catch(() => null),
    promScalar(`sum(rate(inference_requests_total{status=~"5.."}[5m])) / clamp_min(sum(rate(inference_requests_total[5m])),1)`).catch(() => null),
    promScalar(`sum(rate(inference_requests_total[5m]))`).catch(() => null),
  ]);
  const ms = (s: number | null) => (s == null ? null : Math.round(s * 1000));
  const pct = (r: number | null) => (r == null ? null : Math.round(r * 1000) / 10);
  return [
    { id: 'p50', label: 'Latencia p50', value: ms(p50), unit: 'ms', target: '< 20 ms', ok: p50 == null ? null : p50 * 1000 < 20 },
    { id: 'p95', label: 'Latencia p95', value: ms(p95), unit: 'ms', target: '< 50 ms', ok: p95 == null ? null : p95 * 1000 < 50 },
    { id: 'p99', label: 'Latencia p99', value: ms(p99), unit: 'ms', target: '< 100 ms', ok: p99 == null ? null : p99 * 1000 < 100 },
    { id: 'err', label: 'Tasa de error', value: pct(errRate), unit: '%', target: '< 1 %', ok: errRate == null ? null : errRate < 0.01 },
    { id: 'thru', label: 'Throughput', value: thru == null ? null : Math.round(thru * 100) / 100, unit: 'rps', target: '> 0.1/s', ok: thru == null ? null : thru > 0.1 },
  ];
}

async function readPromTargets(): Promise<PromTargetRow[]> {
  const j = (await fetchJson(`${PROM_URL}/api/v1/targets?state=active`)) as
    { data?: { activeTargets?: Array<{ labels?: Record<string, string>; scrapeUrl?: string; health?: string; lastScrapeDuration?: number }> } };
  return (j?.data?.activeTargets ?? []).map((t) => ({
    job: t.labels?.job ?? '—',
    instance: t.labels?.instance ?? t.scrapeUrl ?? '—',
    health: t.health === 'up' ? 'up' : t.health === 'down' ? 'down' : 'unknown',
    last_scrape_ms: typeof t.lastScrapeDuration === 'number' ? Math.round(t.lastScrapeDuration * 1000) : null,
  }));
}

async function readActiveAlerts(): Promise<ActiveAlertRow[]> {
  const j = (await fetchJson(`${ALERTMANAGER_URL}/api/v2/alerts?active=true`)) as
    Array<{ labels?: Record<string, string>; annotations?: Record<string, string>; status?: { state?: string }; startsAt?: string }>;
  return (j ?? []).map((a) => ({
    name: a.labels?.alertname ?? '—',
    severity: a.labels?.severity ?? 'none',
    state: a.status?.state ?? 'active',
    since: a.startsAt ?? null,
    summary: a.annotations?.summary ?? a.annotations?.description ?? null,
  }));
}

/** Recursos básicos vía node_exporter (si Prometheus los expone). */
async function readResources(): Promise<ResourceGauge[]> {
  const [cpu, mem] = await Promise.all([
    promScalar(`100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`).catch(() => null),
    promScalar(`100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))`).catch(() => null),
  ]);
  const round = (n: number | null) => (n == null ? null : Math.round(n));
  return [
    { id: 'cpu', label: 'CPU', pct: round(cpu) },
    { id: 'mem', label: 'Memoria', pct: round(mem) },
  ];
}

/** Estado de los DAGs L0→L6 vía Airflow REST (basic-auth); vacío si no hay env. */
const PIPELINE_DAGS: Array<{ stage: string; dag_id: string; name: string }> = [
  { stage: 'L0', dag_id: 'core_l0_02_ohlcv_realtime', name: 'Ingesta OHLCV' },
  { stage: 'L0', dag_id: 'core_l0_04_macro_update', name: 'Macro update' },
  { stage: 'L3', dag_id: 'forecast_h5_l3_weekly_training', name: 'Entrenamiento H5' },
  { stage: 'L4', dag_id: 'forecast_h5_l4b_production_deploy', name: 'Deploy producción' },
  { stage: 'L5', dag_id: 'forecast_h5_l5_weekly_signal', name: 'Señal H5' },
  { stage: 'L6', dag_id: 'forecast_h5_l6_weekly_monitor', name: 'Monitoreo H5' },
];

async function readPipeline(): Promise<PipelineStageChip[]> {
  if (!AIRFLOW_API_URL || !AIRFLOW_API_USER) throw new Error('Airflow REST no configurado');
  const auth = 'Basic ' + Buffer.from(`${AIRFLOW_API_USER}:${AIRFLOW_API_PASSWORD}`).toString('base64');
  const out = await Promise.all(PIPELINE_DAGS.map(async (d) => {
    try {
      const j = (await fetchJson(
        `${AIRFLOW_API_URL.replace(/\/$/, '')}/api/v1/dags/${d.dag_id}/dagRuns?order_by=-execution_date&limit=1`,
        { headers: { Authorization: auth } },
      )) as { dag_runs?: Array<{ state?: string; end_date?: string; start_date?: string }> };
      const run = j?.dag_runs?.[0];
      const state = run?.state ?? null;
      return {
        stage: d.stage, dag_id: d.dag_id, name: d.name,
        ok: state == null ? null : state === 'success' || state === 'running',
        state, last_run: run?.end_date ?? run?.start_date ?? null, paused: null,
      } satisfies PipelineStageChip;
    } catch {
      return { stage: d.stage, dag_id: d.dag_id, name: d.name, ok: null, state: null, last_run: null, paused: null };
    }
  }));
  return out;
}

function freshnessOf(latest: Date | null, thresholdHours: number): { status: FreshnessStatus; age: number | null } {
  if (!latest || Number.isNaN(latest.getTime())) return { status: 'unknown', age: null };
  const age = (Date.now() - latest.getTime()) / 3_600_000;
  const status: FreshnessStatus = age <= thresholdHours ? 'ok' : age <= thresholdHours * 1.5 ? 'warn' : 'stale';
  return { status, age: Math.round(age * 10) / 10 };
}

async function dbFreshness(id: string, label: string, sql: string, thresholdHours: number): Promise<FreshnessSource> {
  const res = await query(sql);
  const raw = res.rows[0]?.latest ?? null;
  const latest = raw ? new Date(raw) : null;
  const { status, age } = freshnessOf(latest, thresholdHours);
  return { id, label, latest: latest?.toISOString() ?? null, age_hours: age, threshold_hours: thresholdHours, status };
}

async function fileFreshness(id: string, label: string, file: string, thresholdHours: number): Promise<FreshnessSource> {
  const stat = await fs.stat(path.join(PROD_DIR, file));
  const { status, age } = freshnessOf(stat.mtime, thresholdHours);
  return { id, label, latest: stat.mtime.toISOString(), age_hours: age, threshold_hours: thresholdHours, status, detail: file };
}

async function checkService(name: string, url: string, timeoutMs = 3000): Promise<ServiceCheck> {
  const start = Date.now();
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    clearTimeout(timer);
    return { name, ok: res.ok, latency_ms: Date.now() - start };
  } catch (e) {
    return { name, ok: false, latency_ms: Date.now() - start, error: String((e as Error)?.message ?? e) };
  }
}

async function readVote2(): Promise<Vote2Summary | null> {
  const raw = JSON.parse(await fs.readFile(path.join(PROD_DIR, 'approval_state.json'), 'utf-8'));
  const gates: Array<{ passed: boolean }> = raw.gates ?? [];
  return {
    status: raw.status ?? 'UNKNOWN',
    strategy: raw.strategy ?? '',
    strategy_name: raw.strategy_name ?? raw.strategy ?? '',
    backtest_year: raw.backtest_year ?? null,
    recommendation: raw.backtest_recommendation ?? null,
    gates_passed: gates.filter((g) => g.passed).length,
    gates_total: gates.length,
    approved_by: raw.approved_by ?? null,
    approved_at: raw.approved_at ?? null,
    pending: raw.status === 'PENDING_APPROVAL',
  };
}

async function readDeploy(): Promise<SystemStatus['deploy']> {
  const raw = JSON.parse(await fs.readFile(path.join(PROD_DIR, 'deploy_status.json'), 'utf-8'));
  return {
    phase: raw.phase ?? raw.status ?? null,
    runner: raw.runner ?? null,
    updated_at: raw.updated_at ?? raw.last_updated ?? null,
  };
}

export async function GET(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;

  const partial: string[] = [];
  const settle = async <T>(label: string, p: Promise<T>): Promise<T | null> => {
    try { return await p; } catch (e) { partial.push(`${label}: ${String((e as Error)?.message ?? e)}`); return null; }
  };

  const T = FRESHNESS_THRESHOLDS_HOURS;
  const [
    ohlcv, macro, news, bundle, vote2, deploy, sb, tradingApi,
    slos, promTargets, alertsActive, pipeline, resources,
  ] = await Promise.all([
    settle('ohlcv', dbFreshness('ohlcv_m5', 'OHLCV 5-min (USD/COP)',
      `SELECT MAX(time) AS latest FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'`, T.ohlcv_m5)),
    settle('macro', dbFreshness('macro_daily', 'Macro diario',
      `SELECT MAX(fecha) AS latest FROM macro_indicators_daily`, T.macro_daily)),
    settle('news', dbFreshness('news', 'Noticias',
      `SELECT MAX(published_at) AS latest FROM news_articles`, T.news)),
    settle('bundle', fileFreshness('production_bundle', 'Bundle producción (summary.json)',
      'summary.json', T.production_bundle)),
    settle('vote2', readVote2()),
    settle('deploy', readDeploy()),
    settle('signalbridge', checkService('signalbridge', `${SB_URL}/health`)),
    settle('trading-api', checkService('trading-api', TRADING_API_URL)),
    // Ampliado 2026-07 (§4.11 CTR-OBS-001): cada proxy degrada a null + partial_errors (C5).
    settle('slos (prometheus)', readSlos()),
    settle('prom targets', readPromTargets()),
    settle('alertmanager', readActiveAlerts()),
    settle('pipeline (airflow)', readPipeline()),
    settle('resources (node_exporter)', readResources()),
  ]);

  // Frontend is trivially up (it served this request) — listed so the services
  // table enumerates the full surface (spec admin-ui-polish §2.2).
  const frontend: ServiceCheck = { name: 'frontend', ok: true, latency_ms: 0 };

  const body: SystemStatus = {
    freshness: [ohlcv, macro, news, bundle].filter((f): f is FreshnessSource => !!f),
    services: [sb, tradingApi, frontend].filter((s): s is ServiceCheck => !!s),
    vote2,
    deploy,
    partial_errors: partial,
    slos,
    prom_targets: promTargets,
    alerts_active: alertsActive,
    pipeline,
    resources,
  };
  return NextResponse.json(body);
}
