/**
 * Server-side entitlements resolution (CTR-RBAC-001 rule 3: JWT is a cache, the DB is
 * the truth). Route handlers call `getEntitlements(userId)`; expired plans degrade to
 * `free` automatically via `effectiveEntitlements()`.
 *
 * A short in-process TTL cache keeps hot paths (chart polling) off the DB while still
 * honoring revocation within CACHE_TTL_MS.
 */
import { query } from '@/lib/db/postgres-client';
import {
  effectiveEntitlements,
  PLAN_DEFAULTS,
  type Entitlements,
} from '@/lib/contracts/rbac.contract';

const CACHE_TTL_MS = 60_000;
const cache = new Map<string, { at: number; value: Entitlements }>();

export async function getEntitlements(userId: string | null | undefined): Promise<Entitlements> {
  if (!userId) return PLAN_DEFAULTS.free;
  const hit = cache.get(userId);
  if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.value;

  try {
    const res = await query<{ role: string; entitlements: Entitlements }>(
      'SELECT role, entitlements FROM sb_users WHERE id = $1',
      [userId],
    );
    const row = res.rows[0];
    // INTERNAL ROLES (admin/developer) always operate at full data capability regardless
    // of plan bookkeeping (§B.2: Forecasting/Análisis completos para internos). This is
    // ROLE-based so it survives DB resets — after a cold boot migration 055 re-seeds
    // admin with plan=auto but assets=['usdcop'], which 403'd Gold/BTC until this fix.
    // Asset scope + delays are lifted; EXECUTION stays plan/permission-based (developer
    // never gets execution:self — role matrix blocks it upstream).
    const isInternal = row?.role === 'admin' || row?.role === 'developer';
    const value: Entitlements = isInternal
      ? {
          ...effectiveEntitlements(row?.entitlements),
          plan: row.role === 'admin' ? 'auto' : effectiveEntitlements(row?.entitlements).plan,
          assets: ['usdcop', 'xauusd', 'btcusdt'],
          forecast_delay_hours: 0,
          analysis_delay_days: 0,
          signals_realtime: true,
        }
      : effectiveEntitlements(row?.entitlements);
    cache.set(userId, { at: Date.now(), value });
    return value;
  } catch {
    // DB unavailable: fail CLOSED to free (delayed content), never open.
    return PLAN_DEFAULTS.free;
  }
}

/**
 * Content-freshness gate for dated artifacts. Filenames in the gated trees carry either
 * an ISO week (`..._2026-W15.json`) or a date (`..._2026-07-04.png`). Returns true when
 * the artifact is NEWER than the user's allowed horizon (i.e. must be blocked).
 */
export function isFresherThanAllowed(fileName: string, delayDays: number): boolean {
  if (delayDays <= 0) return false;
  const cutoff = Date.now() - delayDays * 86_400_000;

  const week = fileName.match(/(\d{4})-W(\d{2})/);
  if (week) {
    const monday = mondayOfIsoWeek(Number(week[1]), Number(week[2]));
    return monday.getTime() > cutoff;
  }
  const day = fileName.match(/(\d{4})-(\d{2})-(\d{2})/);
  if (day) return new Date(`${day[0]}T00:00:00Z`).getTime() > cutoff;
  return false; // undated artifacts are not freshness-gated
}

function mondayOfIsoWeek(year: number, week: number): Date {
  const jan4 = new Date(Date.UTC(year, 0, 4));
  const monday = new Date(jan4);
  monday.setUTCDate(jan4.getUTCDate() - ((jan4.getUTCDay() + 6) % 7) + (week - 1) * 7);
  return monday;
}
