'use client';

/**
 * Usuarios (CTR-ADMIN-UI-001 §2.4): segmented Reales|Test|Todos, 250ms-debounced
 * search with match highlight, execution badges (○ PAPER / ● LIVE), row → read-only
 * user drawer reflected in the URL (?user=). Roles from the RBAC enum (C2), staff
 * without plan (C3). Filters live in the querystring (§3.3).
 */
import { useEffect, useMemo, useState } from 'react';
import { ChevronRight, Users as UsersIcon, SearchX } from 'lucide-react';

import { ROLES, type Role } from '@/lib/contracts/rbac.contract';
import type { AdminUserRow, UsersListResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, HIGHLIGHT_MARK, ROLE_BADGE, SURFACE, TYPE } from '@/lib/ui/tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { useUrlState } from './useUrlState';
import {
  Badge, Card, DrawerHost, EmptyState, SkeletonRows, TestBadge,
  fmtDate, fmtDateTime, fmtRelative, useNow,
} from './ui';
import { useToast } from './ui/toast';

const STATUS_OPTIONS = ['pending', 'approved', 'rejected'] as const;
const SEGMENTS = [
  { id: 'real', label: 'Reales' },
  { id: 'test', label: 'Test' },
  { id: 'all', label: 'Todos' },
] as const;

function RolePill({ role, rawRole }: { role: Role; rawRole?: string }) {
  const legacy = rawRole && rawRole !== role;
  return (
    <span
      title={legacy ? `valor en DB: '${rawRole}' (normalizado a '${role}')` : undefined}
      className={`inline-block rounded-full border px-2 py-0.5 text-[10px] font-semibold ${ROLE_BADGE[role]}`}
    >
      {role}{legacy ? '*' : ''}
    </span>
  );
}

function Highlight({ text, needle }: { text: string; needle: string }) {
  if (!needle) return <>{text}</>;
  const i = text.toLowerCase().indexOf(needle.toLowerCase());
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark className={HIGHLIGHT_MARK}>{text.slice(i, i + needle.length)}</mark>
      {text.slice(i + needle.length)}
    </>
  );
}

function ExecutionBadge({ mode, kill }: { mode: AdminUserRow['execution_mode']; kill: boolean | null }) {
  if (!mode) return <span className={COLOR.textSecondary}>—</span>;
  return (
    <span className="inline-flex items-center gap-1.5">
      <Badge tone={mode === 'live' ? 'ok' : 'warn'}>{mode === 'live' ? '● LIVE' : '○ PAPER'}</Badge>
      {kill && <Badge tone="error">KILL</Badge>}
    </span>
  );
}

export function UsersSection() {
  const list = useAdminWidget<UsersListResponse>('/api/admin/users/list', { refreshMs: REFRESH.users });
  const { toast } = useToast();
  const now = useNow();
  const url = useUrlState();

  const segment = (url.get('seg') || 'real') as (typeof SEGMENTS)[number]['id'];
  const role = url.get('rol') as '' | Role;
  const status = url.get('estado');
  const openUserId = url.get('user');

  // Debounced search (250 ms) mirrored to URL (§2.4 + §3.3).
  const [qInput, setQInput] = useState(url.get('q'));
  useEffect(() => {
    const t = setTimeout(() => { if (qInput !== url.get('q')) url.setMany({ q: qInput || null }); }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qInput]);
  const q = url.get('q');

  const filtered = useMemo(() => {
    const rows = list.data?.users ?? [];
    const needle = q.trim().toLowerCase();
    return rows.filter((u) =>
      (segment === 'all' || (segment === 'test') === u.is_test)
      && (!role || u.role === role)
      && (!status || u.status === status)
      && (!needle || u.email.toLowerCase().includes(needle) || (u.name ?? '').toLowerCase().includes(needle)),
    );
  }, [list.data, q, role, status, segment]);

  const openUser = (list.data?.users ?? []).find((u) => u.id === openUserId) ?? null;

  const flagTest = async (u: AdminUserRow) => {
    const r = await fetch(`/api/admin/users/${u.id}/flag-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_test: !u.is_test }),
    });
    if (r.ok) { toast(`${u.email} marcado como ${u.is_test ? 'cuenta real' : 'cuenta de test'}`, 'ok'); list.reload(); }
    else toast(`Error (${r.status}) al marcar ${u.email}`, 'error');
  };

  return (
    <>
      <Card
        title="Usuarios"
        icon={<UsersIcon className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        badge={<Badge tone="neutral">{filtered.length} de {list.data?.total ?? '…'}</Badge>}
        meta={list.updatedAt ? <span title={new Date(list.updatedAt).toISOString()}>{fmtRelative(new Date(list.updatedAt).toISOString(), now)}</span> : null}
        stale={list.stale}
      >
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <input
            value={qInput} onChange={(e) => setQInput(e.target.value)}
            placeholder="Buscar email o nombre…" aria-label="buscar usuarios"
            className={`${SURFACE.input} ${CTA.focusRing} w-56`}
          />
          <select value={role} onChange={(e) => url.setMany({ rol: e.target.value || null })} aria-label="filtrar por rol" className={`${SURFACE.input} ${CTA.focusRing}`}>
            <option value="">Rol: todos</option>
            {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
          <select value={status} onChange={(e) => url.setMany({ estado: e.target.value || null })} aria-label="filtrar por estado" className={`${SURFACE.input} ${CTA.focusRing}`}>
            <option value="">Estado: todos</option>
            {STATUS_OPTIONS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <div role="group" aria-label="segmento de cuentas" className="inline-flex rounded-lg border border-slate-700 overflow-hidden ml-auto">
            {SEGMENTS.map((s) => (
              <button
                key={s.id}
                aria-pressed={segment === s.id}
                onClick={() => url.setMany({ seg: s.id === 'real' ? null : s.id })}
                className={`px-3 py-1.5 text-xs font-semibold ${CTA.focusRing}
                  ${segment === s.id ? CTA.segmentActive : `${COLOR.textSecondary} hover:bg-slate-800/60`}`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {list.error && !list.data && (
          <EmptyState icon={<SearchX className="w-8 h-8" aria-hidden />} cause={<>No se pudo cargar: {list.error}</>}
            action={<button onClick={list.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>} />
        )}
        {list.loading && !list.data && <SkeletonRows rows={4} cols={6} />}

        {list.data && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                  <th className="py-2 pr-3">Email</th><th className="pr-3">Rol</th>
                  <th className="pr-3">Plan</th><th className="pr-3">Estado</th>
                  <th className="pr-3">Ejecución</th><th className="pr-3">Último acceso</th>
                  <th className="pr-3">Vence</th><th className="pr-3">Alta</th><th className="w-6" aria-hidden />
                </tr>
              </thead>
              <tbody>
                {filtered.map((u) => (
                  <tr
                    key={u.id}
                    onClick={() => url.setMany({ user: u.id })}
                    className={`h-10 border-b border-slate-800/50 cursor-pointer group ${SURFACE.tableRowHover}`}
                  >
                    <td className={`pr-3 ${COLOR.textPrimary}`} title={u.id}>
                      <Highlight text={u.email} needle={q} /> {u.is_test && <TestBadge />}
                    </td>
                    <td className="pr-3"><RolePill role={u.role} rawRole={u.raw_role} /></td>
                    <td className="pr-3">
                      {u.plan
                        ? <Badge tone={u.plan === 'auto' ? 'ok' : u.plan === 'signals' ? 'info' : 'neutral'} className="uppercase">{u.plan}</Badge>
                        : <span className={COLOR.textSecondary}>—</span>}
                    </td>
                    <td className="pr-3">
                      <span className={u.status === 'approved' ? COLOR.ok.text : u.status === 'rejected' ? COLOR.error.text : COLOR.warn.text}>
                        {u.status}
                      </span>
                    </td>
                    <td className="pr-3"><ExecutionBadge mode={u.execution_mode} kill={u.kill_switch} /></td>
                    <td className="pr-3" title={u.last_login ?? undefined}>{u.last_login ? fmtRelative(u.last_login, now) : '—'}</td>
                    <td className={`pr-3 ${TYPE.mono}`}>{fmtDate(u.expires_at)}</td>
                    <td className={`pr-3 ${TYPE.mono}`}>{fmtDate(u.created_at)}</td>
                    <td className="text-right">
                      <ChevronRight className={`w-3.5 h-3.5 opacity-0 group-hover:opacity-60 ${COLOR.textSecondary}`} aria-hidden />
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={9}>
                      <EmptyState
                        icon={<SearchX className="w-8 h-8" aria-hidden />}
                        cause={
                          segment !== 'all' && (list.data.users.length - filtered.length) > 0
                            ? <>0 visibles — {list.data.users.length} ocultos por el segmento «{SEGMENTS.find((s) => s.id === segment)?.label}».</>
                            : 'Sin resultados con estos filtros.'
                        }
                        action={segment !== 'all'
                          ? <button onClick={() => url.setMany({ seg: 'all' })} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Ver todos</button>
                          : undefined}
                      />
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* ficha read-only (§2.4 — el editor de entitlements es el Bloque 2 de admin-console.md) */}
      <DrawerHost
        open={!!openUser}
        title={openUser?.email ?? ''}
        onClose={() => url.setMany({ user: null })}
        fields={openUser ? [
          ['ID', <span key="id" className={TYPE.mono}>{openUser.id}</span>],
          ['Nombre', openUser.name ?? '—'],
          ['Rol', <RolePill key="r" role={openUser.role} rawRole={openUser.raw_role} />],
          ['Plan', openUser.plan ?? '— (staff sin plan)'],
          ['Estado', openUser.status],
          ['Ejecución', <ExecutionBadge key="e" mode={openUser.execution_mode} kill={openUser.kill_switch} />],
          ['Último acceso', fmtDateTime(openUser.last_login)],
          ['Vence', fmtDate(openUser.expires_at)],
          ['Alta', fmtDateTime(openUser.created_at)],
          ['Cuenta de test', openUser.is_test ? 'sí' : 'no'],
        ] : []}
        footer={openUser ? (
          <div className="space-y-3">
            <button onClick={() => flagTest(openUser)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>
              {openUser.is_test ? 'Marcar como cuenta real' : 'Marcar como cuenta de test'}
            </button>
            <p className={TYPE.meta}>
              Editor de rol/entitlements y «ver como»: próximo incremento (admin-console.md §10.2).
            </p>
          </div>
        ) : null}
      />
    </>
  );
}
