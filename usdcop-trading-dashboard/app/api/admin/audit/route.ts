/**
 * GET /api/admin/audit — audit ledger view (admin:all). Spec §7 (Auditoría v2).
 *
 * READ-ONLY over the append-only audit_log: filters (action/category/user/date),
 * email resolution via LEFT JOIN sb_users (humans read emails, not UUIDs), test
 * traffic hidden by default (derived from the ACTOR's sb_users.is_test — the
 * ledger itself is never mutated), CSV export via ?format=csv.
 */
import { NextResponse } from 'next/server';

import { requireAdminRole } from '@/lib/admin/guard';
import {
  categorizeAuditAction,
  type AuditEntry, type AuditResponse,
} from '@/lib/contracts/admin-console.contract';
import { query } from '@/lib/db/postgres-client';

const MAX_LIMIT = 500;

function toCsv(entries: AuditEntry[]): string {
  const esc = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const header = 'created_at,action,category,severity,user_email,user_id,object_type,object_id,ip,detail';
  const rows = entries.map((e) => [
    e.created_at, e.action, e.category, e.severity, e.user_email, e.user_id,
    e.object_type, e.object_id, e.ip, JSON.stringify(e.detail),
  ].map(esc).join(','));
  return [header, ...rows].join('\n');
}

export async function GET(req: Request) {
  const denied = requireAdminRole(req);
  if (denied) return denied;

  const url = new URL(req.url);
  const p = url.searchParams;
  const limit = Math.min(Number(p.get('limit')) || 200, MAX_LIMIT);
  const includeTest = p.get('include_test') === 'true';

  const where: string[] = [];
  const params: unknown[] = [];
  const bind = (v: unknown) => { params.push(v); return `$${params.length}`; };

  if (p.get('action')) where.push(`a.action ILIKE ${bind(`%${p.get('action')}%`)}`);
  if (p.get('user')) where.push(`(u.email ILIKE ${bind(`%${p.get('user')}%`)} OR a.user_id::text ILIKE ${bind(`%${p.get('user')}%`)})`);
  if (p.get('from')) where.push(`a.created_at >= ${bind(p.get('from'))}`);
  if (p.get('to')) where.push(`a.created_at <= ${bind(p.get('to'))}`);
  if (!includeTest) where.push(`NOT COALESCE(u.is_test, FALSE)`);

  try {
    const res = await query(
      `SELECT a.id, a.user_id, u.email AS user_email, a.action, a.object_type,
              a.object_id, a.detail, a.ip, a.created_at,
              COALESCE(u.is_test, FALSE) AS is_test
       FROM audit_log a
       LEFT JOIN sb_users u ON u.id = a.user_id
       ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
       ORDER BY a.created_at DESC
       LIMIT ${limit}`,
      params,
    );

    let entries: AuditEntry[] = res.rows.map((r) => {
      const { category, severity } = categorizeAuditAction(r.action ?? '');
      return {
        id: Number(r.id),
        user_id: r.user_id ?? null,
        user_email: r.user_email ?? null,
        action: r.action,
        category,
        severity,
        object_type: r.object_type ?? null,
        object_id: r.object_id ?? null,
        detail: r.detail,
        ip: r.ip ?? null,
        created_at: r.created_at,
        is_test: !!r.is_test,
      };
    });

    // Category is derived (not a DB column), so this filter applies post-query.
    const category = p.get('category');
    if (category) entries = entries.filter((e) => e.category === category);

    if (p.get('format') === 'csv') {
      return new NextResponse(toCsv(entries), {
        headers: {
          'Content-Type': 'text/csv; charset=utf-8',
          'Content-Disposition': `attachment; filename="audit_${new Date().toISOString().slice(0, 10)}.csv"`,
        },
      });
    }
    const body: AuditResponse = { entries, total_scanned: res.rowCount ?? entries.length };
    return NextResponse.json(body);
  } catch (e) {
    return NextResponse.json({ error: 'db unavailable', detail: String(e) }, { status: 503 });
  }
}
