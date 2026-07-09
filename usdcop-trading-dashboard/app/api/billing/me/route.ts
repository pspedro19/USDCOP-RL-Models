/**
 * GET /api/billing/me — the logged-in user's EFFECTIVE entitlements (server-resolved:
 * DB truth + expiry degradation via `effectiveEntitlements`). Read-only; the client
 * never writes entitlements (CTR-RBAC-001 rule 9).
 */
import { NextResponse } from 'next/server';

import { getEntitlements } from '@/lib/auth/entitlements';

export async function GET(req: Request) {
  const userId = req.headers.get('x-user-id');
  if (!userId) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });
  const ent = await getEntitlements(userId);
  return NextResponse.json(ent, { headers: { 'Cache-Control': 'private, max-age=30' } });
}
