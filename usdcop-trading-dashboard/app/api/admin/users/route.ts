/**
 * GET /api/admin/users?status=pending — approval queue (admin:all, middleware-gated).
 * Relays the admin's SignalBridge bearer to SB /api/admin/users (source of truth).
 */
import { relayAdmin } from '@/lib/signalbridge/admin-proxy';

export async function GET(req: Request) {
  const status = new URL(req.url).searchParams.get('status') ?? 'pending';
  return relayAdmin(req, `/api/admin/users?status=${encodeURIComponent(status)}`);
}
