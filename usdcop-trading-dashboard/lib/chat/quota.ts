/**
 * Chat quota (CTR-CHAT-001 × CTR-RBAC-001) — monetization lever.
 *
 * The chatbot is part of the analysis surface (analysis:read). The PLAN drives
 * how much you can use it: free is capped, paid tiers are generous. Enforced
 * server-side per user per day. In-memory counter (resets on restart) — good
 * enough for abuse control; a durable counter can back the same interface later.
 */

import type { Entitlements, PlanId } from '@/lib/contracts/rbac.contract';

export interface ChatQuota {
  plan: PlanId;
  dailyLimit: number;
}

const DAILY_LIMIT_BY_PLAN: Record<PlanId, number> = {
  free: 15,
  signals: 100,
  auto: 250,
};

export function chatQuotaFor(entitlements: Entitlements): ChatQuota {
  const plan = entitlements.plan;
  return { plan, dailyLimit: DAILY_LIMIT_BY_PLAN[plan] ?? DAILY_LIMIT_BY_PLAN.free };
}

// userId → { day: 'YYYY-MM-DD', count }
const usage = new Map<string, { day: string; count: number }>();

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export interface QuotaState {
  allowed: boolean;
  remaining: number;
  limit: number;
}

/** Check + increment atomically. Returns whether this message is allowed. */
export function consumeQuota(userId: string, quota: ChatQuota): QuotaState {
  const day = today();
  const cur = usage.get(userId);
  const count = cur && cur.day === day ? cur.count : 0;
  if (count >= quota.dailyLimit) {
    return { allowed: false, remaining: 0, limit: quota.dailyLimit };
  }
  usage.set(userId, { day, count: count + 1 });
  return { allowed: true, remaining: quota.dailyLimit - (count + 1), limit: quota.dailyLimit };
}
