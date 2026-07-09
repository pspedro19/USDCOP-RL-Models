/**
 * POST /api/admin/users/:id/reject — reject a pending user (admin:all).
 * Optional { reason } is relayed to SignalBridge, which emails the outcome + audits.
 */
import { relayAdmin } from '@/lib/signalbridge/admin-proxy';

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  return relayAdmin(req, `/api/admin/users/${encodeURIComponent(id)}/reject`, {
    method: 'POST',
    body: body?.reason ? { reason: String(body.reason) } : {},
  });
}
