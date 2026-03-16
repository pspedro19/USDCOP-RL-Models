/**
 * Enhanced health check endpoint for Docker container healthcheck.
 * Checks dashboard dependencies: trading-api, signalbridge, data files.
 * Returns 200 with degraded status if non-critical deps are down.
 */
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

interface HealthCheck {
  name: string;
  status: 'ok' | 'error';
  latency_ms?: number;
  error?: string;
}

async function checkUrl(url: string, timeoutMs: number = 3000): Promise<{ ok: boolean; latency_ms: number; error?: string }> {
  const start = Date.now();
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    return { ok: res.ok, latency_ms: Date.now() - start };
  } catch (e: any) {
    return { ok: false, latency_ms: Date.now() - start, error: e?.message || 'unreachable' };
  }
}

export async function GET() {
  const checks: HealthCheck[] = [];

  // Check trading-api
  const tradingApi = await checkUrl(
    process.env.TRADING_API_URL || 'http://usdcop-trading-api:8000/'
  );
  checks.push({
    name: 'trading_api',
    status: tradingApi.ok ? 'ok' : 'error',
    latency_ms: tradingApi.latency_ms,
    ...(tradingApi.error && { error: tradingApi.error }),
  });

  // Check signalbridge
  const signalbridge = await checkUrl(
    (process.env.SIGNALBRIDGE_BACKEND_URL || 'http://usdcop-signalbridge:8000') + '/health'
  );
  checks.push({
    name: 'signalbridge',
    status: signalbridge.ok ? 'ok' : 'error',
    latency_ms: signalbridge.latency_ms,
    ...(signalbridge.error && { error: signalbridge.error }),
  });

  // Check critical data files exist
  const dataDir = path.join(process.cwd(), 'public', 'data', 'production');
  const summaryExists = fs.existsSync(path.join(dataDir, 'summary.json'));
  const approvalExists = fs.existsSync(path.join(dataDir, 'approval_state.json'));
  checks.push({
    name: 'data_files',
    status: summaryExists || approvalExists ? 'ok' : 'error',
    ...(!(summaryExists || approvalExists) && { error: 'No production data files found' }),
  });

  const allOk = checks.every((c) => c.status === 'ok');
  const anyOk = checks.some((c) => c.status === 'ok');

  return NextResponse.json({
    status: allOk ? 'healthy' : anyOk ? 'degraded' : 'unhealthy',
    timestamp: new Date().toISOString(),
    service: 'usdcop-dashboard',
    checks,
  });
}
