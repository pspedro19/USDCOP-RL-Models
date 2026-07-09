/**
 * POST /api/admin/users/:id/approve — approve a pending user (admin:all).
 * SignalBridge generates a temporary password, emails it, and writes the audit row.
 */
import { relayAdmin } from '@/lib/signalbridge/admin-proxy';

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  return relayAdmin(req, `/api/admin/users/${encodeURIComponent(id)}/approve`, { method: 'POST' });
}
