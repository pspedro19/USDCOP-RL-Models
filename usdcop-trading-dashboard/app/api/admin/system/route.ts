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
  type FreshnessSource, type FreshnessStatus, type ServiceCheck,
  type SystemStatus, type Vote2Summary,
} from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

const PROD_DIR = path.join(process.cwd(), 'public', 'data', 'production');
const SB_URL = process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000';
const TRADING_API_URL = process.env.TRADING_API_URL || 'http://usdcop-trading-api:8000/';

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
  const [ohlcv, macro, news, bundle, vote2, deploy, sb, tradingApi] = await Promise.all([
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
  };
  return NextResponse.json(body);
}
