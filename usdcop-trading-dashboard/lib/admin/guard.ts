/**
 * Shared guard + actor identity for /api/admin/** handlers (CTR-ADMIN-CONSOLE-001).
 *
 * Defense in depth: middleware already enforces `admin:all` via the RBAC matrix and
 * stamps anti-spoof `x-user-id`/`x-user-role`; every handler re-checks the role here
 * instead of copy-pasting the check (DRY — one guard, N routes).
 */
import { NextResponse } from 'next/server';

/** Returns a 403 response if the caller is not an admin, else null (proceed). */
export function requireAdminRole(req: Request): NextResponse | null {
  if (req.headers.get('x-user-role') !== 'admin') {
    return NextResponse.json({ error: 'admin only' }, { status: 403 });
  }
  return null;
}

/** The acting admin's identity as stamped by the middleware (for audit rows). */
export function adminActor(req: Request): { id: string | null; ip: string | null } {
  return {
    id: req.headers.get('x-user-id'),
    ip: req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null,
  };
}
