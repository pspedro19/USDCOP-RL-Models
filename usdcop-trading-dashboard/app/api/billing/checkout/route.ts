/**
 * POST /api/billing/checkout — create a hosted-checkout session for the logged-in user.
 * Body: { plan: 'signals' | 'auto', addOnAssets?: string[] }
 * Auth: middleware guarantees a session ('authenticated') and stamps x-user-id.
 */
import { NextResponse } from 'next/server';

import { query } from '@/lib/db/postgres-client';
import { getBillingProvider } from '@/lib/billing';

export async function POST(req: Request) {
  const userId = req.headers.get('x-user-id');
  if (!userId) return NextResponse.json({ error: 'unauthenticated' }, { status: 401 });

  let body: { plan?: string; addOnAssets?: string[] };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: 'invalid JSON' }, { status: 400 });
  }
  if (body.plan !== 'signals' && body.plan !== 'auto') {
    return NextResponse.json({ error: "plan must be 'signals' or 'auto'" }, { status: 400 });
  }

  const res = await query<{ email: string }>('SELECT email FROM sb_users WHERE id = $1', [userId]);
  const email = res.rows[0]?.email;
  if (!email) return NextResponse.json({ error: 'user not found' }, { status: 404 });

  try {
    const session = await getBillingProvider().createCheckout({
      userId, email, plan: body.plan, addOnAssets: body.addOnAssets ?? [],
    });
    return NextResponse.json(session);
  } catch (e) {
    // Provider not configured yet (no keys): explicit, actionable error — never a 500 stack.
    return NextResponse.json(
      { error: 'billing provider not configured', detail: String(e) },
      { status: 503 },
    );
  }
}
