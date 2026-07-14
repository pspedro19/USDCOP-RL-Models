'use client';

/**
 * Usuarios (CTR-ADMIN-UI-001 §2.4): segmented Reales|Test|Todos, 250ms-debounced
 * search with match highlight, execution badges (○ PAPER / ● LIVE), row → read-only
 * user drawer reflected in the URL (?user=). Roles from the RBAC enum (C2), staff
 * without plan (C3). Filters live in the querystring (§3.3).
 */
import { useEffect, useMemo, useState } from 'react';
import { ChevronRight, Eye, Users as UsersIcon, SearchX } from 'lucide-react';

import { PERMISSIONS, PLAN_DEFAULTS, ROLES, type PlanId, type Role } from '@/lib/contracts/rbac.contract';
import type { AdminUserRow, UsersListResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, HIGHLIGHT_MARK, ROLE_BADGE, SURFACE, TYPE } from '@/lib/ui/tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { useUrlState } from './useUrlState';
import {
  Badge, Card, DrawerHost, EmptyState, SkeletonRows, TestBadge,
  fmtDate, fmtDateTime, fmtRelative, useNow,
} from './ui';
import { useToast } from './ui/toast';

/**
 * Overrides de permisos por usuario (CTR-RBAC-001 / migración 056): conceder o
 * denegar un permiso puntual por encima de su rol. Efectivo = rol ∪ grants − denies;
 * aplica en el próximo login del usuario. Auditado (`user_override_change`).
 */
function OverridesEditor({ userId }: { userId: string }) {
  const { toast } = useToast();
  const [effects, setEffects] = useState<Record<string, 'grant' | 'deny'>>({});
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(`/api/admin/users/${userId}/overrides`, { cache: 'no-store' })
      .then((r) => r.json()).then((b) => {
        if (!alive) return;
        const map: Record<string, 'grant' | 'deny'> = {};
        for (const o of (b?.data?.overrides ?? [])) map[o.permission] = o.effect;
        setEffects(map);
      }).catch(() => { /* editor degrada a vacío */ });
    return () => { alive = false; };
  }, [userId]);

  async function set(permission: string, effect: 'grant' | 'deny' | 'clear') {
    setBusy(permission);
    try {
      const res = await fetch(`/api/admin/users/${userId}/overrides`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ permission, effect }),
      });
      const b = await res.json().catch(() => ({}));
      if (!res.ok || b?.ok === false) throw new Error(b?.error?.message ?? `HTTP ${res.status}`);
      const map: Record<string, 'grant' | 'deny'> = {};
      for (const o of (b?.data?.overrides ?? [])) map[o.permission] = o.effect;
      setEffects(map);
      toast(`Override ${effect === 'clear' ? 'quitado' : effect} · ${permission}`, effect === 'deny' ? 'warn' : 'ok');
    } catch (e) {
      toast(String((e as Error)?.message ?? e), 'error');
    } finally { setBusy(null); }
  }

  const opt = (permission: string, value: 'inherit' | 'grant' | 'deny', label: string) => {
    const current = effects[permission] ?? 'inherit';
    const active = current === value;
    const tone = value === 'grant' ? CTA.primary : value === 'deny' ? CTA.ghost : CTA.ghost;
    return (
      <button
        key={value}
        disabled={busy === permission}
        onClick={() => set(permission, value === 'inherit' ? 'clear' : value)}
        aria-pressed={active}
        className={`${active ? tone : CTA.ghost} ${CTA.focusRing} px-2 py-0.5 text-[10px] ${active ? '' : 'opacity-55'} disabled:opacity-30`}
      >
        {label}
      </button>
    );
  };

  return (
    <div className="space-y-2 pt-1 border-t border-[rgba(148,163,184,.08)]">
      <div className={TYPE.sectionTitle}>Permisos por usuario (override)</div>
      <p className={TYPE.meta}>Sobre el rol. «Heredado» = sin override. Aplica en el próximo inicio de sesión del usuario.</p>
      <div className="space-y-1">
        {PERMISSIONS.map((perm) => (
          <div key={perm} className="flex items-center justify-between gap-2">
            <span className={`${TYPE.mono} text-[11px] ${COLOR.textPrimary}`}>{perm}</span>
            <div className="inline-flex gap-1">
              {opt(perm, 'inherit', 'Heredado')}
              {opt(perm, 'grant', 'Conceder')}
              {opt(perm, 'deny', 'Denegar')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const STATUS_OPTIONS = ['pending', 'approved', 'rejected'] as const;
const PLAN_OPTIONS = Object.keys(PLAN_DEFAULTS) as PlanId[];
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

  // Editor de rol/plan/entitlements (§10.2 admin-console.md): el servidor valida plan∈PlanId
  // y role∈Role y escribe entitlements canónicos; aquí solo la conveniencia del formulario.
  const [editPlan, setEditPlan] = useState<PlanId | ''>('');
  const [editExpires, setEditExpires] = useState('');
  const [editRole, setEditRole] = useState<Role | ''>('');
  const [editBusy, setEditBusy] = useState(false);

  useEffect(() => {
    setEditPlan(openUser?.plan ?? '');
    setEditExpires(openUser?.expires_at ? openUser.expires_at.slice(0, 10) : '');
    setEditRole(openUser?.role ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openUserId]);

  const curExpires = openUser?.expires_at ? openUser.expires_at.slice(0, 10) : '';
  const planChanged = !!openUser && editPlan !== '' && editPlan !== (openUser.plan ?? '');
  const expiresChanged = !!openUser && editPlan !== '' && editExpires !== curExpires;
  const roleChanged = !!openUser && editRole !== '' && editRole !== openUser.role;
  const editDirty = planChanged || expiresChanged || roleChanged;

  const saveUserEdits = async () => {
    if (!openUser || !editDirty) return;
    const patch: Record<string, unknown> = {};
    // Cambiar el vencimiento reescribe los entitlements ⇒ requiere reenviar el plan.
    if (editPlan !== '' && (planChanged || expiresChanged)) {
      patch.plan = editPlan;
      patch.expires_at = editExpires ? new Date(editExpires).toISOString() : null;
    }
    if (roleChanged) patch.role = editRole;
    setEditBusy(true);
    try {
      const r = await fetch(`/api/admin/users/${openUser.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      });
      const body = await r.json().catch(() => ({}));
      if (r.ok) { toast(`${openUser.email} actualizado`, 'ok'); list.reload(); }
      else toast(`Error (${r.status}): ${body?.error?.message ?? body?.error ?? 'no se pudo actualizar'}`, 'error');
    } catch (e) {
      toast(`Error de red: ${String(e)}`, 'error');
    } finally {
      setEditBusy(false);
    }
  };

  // "Ver como" (impersonación read-only): motivo obligatorio (min 3), POST fija la
  // cookie firmada httpOnly + el espejo legible; recargamos en /hub para que el banner
  // (TerminalShell) y la nav simulada tomen efecto. NO concede permisos: el servidor
  // sigue exigiendo el rol REAL para cualquier mutación (§10.2 admin-console.md).
  const [viewAsUser, setViewAsUser] = useState<AdminUserRow | null>(null);
  const [motivo, setMotivo] = useState('');
  const [viewBusy, setViewBusy] = useState(false);

  const startViewAs = async () => {
    if (!viewAsUser || motivo.trim().length < 3) return;
    setViewBusy(true);
    try {
      const r = await fetch('/api/admin/impersonate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: viewAsUser.id, motivo: motivo.trim() }),
      });
      if (r.ok) {
        window.location.href = '/hub'; // recarga → cookie efectiva + banner visible
        return;
      }
      const body = await r.json().catch(() => ({}));
      toast(`Error (${r.status}): ${body?.error?.message ?? body?.error ?? 'no se pudo iniciar «ver como»'}`, 'error');
    } catch (e) {
      toast(`Error de red: ${String(e)}`, 'error');
    } finally {
      setViewBusy(false);
      setViewAsUser(null);
      setMotivo('');
    }
  };

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
                    <td className="text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => { setViewAsUser(u); setMotivo(''); }}
                        aria-label={`ver como ${u.email}`}
                        className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-2 py-1 text-[10px]`}
                      >
                        <Eye className="w-3 h-3" aria-hidden /> Ver como
                      </button>
                      <ChevronRight className={`inline w-3.5 h-3.5 ml-1 opacity-0 group-hover:opacity-60 ${COLOR.textSecondary}`} aria-hidden />
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
          <div className="space-y-4">
            {/* Editor rol / plan / entitlements (§10.2) — validado server-side. */}
            <div className="space-y-2.5">
              <div className={TYPE.sectionTitle}>Editar acceso</div>
              <label className="flex items-center justify-between gap-3">
                <span className={TYPE.meta}>Rol</span>
                <select
                  value={editRole}
                  onChange={(e) => setEditRole(e.target.value as Role)}
                  aria-label="editar rol"
                  className={`${SURFACE.input} ${CTA.focusRing} w-44`}
                >
                  {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </label>
              <label className="flex items-center justify-between gap-3">
                <span className={TYPE.meta}>Plan</span>
                <select
                  value={editPlan}
                  onChange={(e) => setEditPlan(e.target.value as PlanId)}
                  aria-label="editar plan"
                  className={`${SURFACE.input} ${CTA.focusRing} w-44`}
                >
                  <option value="">— (sin plan / staff)</option>
                  {PLAN_OPTIONS.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
              <label className="flex items-center justify-between gap-3">
                <span className={TYPE.meta}>Vence</span>
                <input
                  type="date"
                  value={editExpires}
                  onChange={(e) => setEditExpires(e.target.value)}
                  disabled={editPlan === ''}
                  aria-label="fecha de vencimiento del plan"
                  className={`${SURFACE.input} ${CTA.focusRing} w-44 disabled:opacity-40`}
                />
              </label>
              <p className={TYPE.meta}>
                Cambiar el plan reescribe los entitlements canónicos (techos SSOT). Vacío en «Vence» =
                sin vencimiento. Toda edición queda en auditoría (plan_change / role_change).
              </p>
              <button
                onClick={saveUserEdits}
                disabled={!editDirty || editBusy}
                className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs disabled:opacity-40`}
              >
                {editBusy ? 'Guardando…' : 'Guardar cambios'}
              </button>
            </div>

            {/* Overrides de permisos por usuario (migración 056). */}
            <OverridesEditor userId={openUser.id} />

            <div className="flex flex-wrap gap-2 pt-1 border-t border-[rgba(148,163,184,.08)]">
              <button onClick={() => flagTest(openUser)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>
                {openUser.is_test ? 'Marcar como cuenta real' : 'Marcar como cuenta de test'}
              </button>
              <button
                onClick={() => { setViewAsUser(openUser); setMotivo(''); }}
                className={`${CTA.primary} ${CTA.focusRing} inline-flex items-center gap-1.5 px-3 py-1.5 text-xs`}
              >
                <Eye className="w-3.5 h-3.5" aria-hidden /> Ver como
              </button>
            </div>
            <p className={TYPE.meta}>
              «Ver como» es un preview real de solo lectura (30 min): el servidor autoriza las LECTURAS
              con el rol previsualizado (intersección con tu rol, nunca escala); toda MUTACIÓN sigue
              exigiendo tu rol admin real. Queda en auditoría.
            </p>
          </div>
        ) : null}
      />

      {/* modal «ver como» — motivo obligatorio (queda en auditoría) */}
      {viewAsUser && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="ver como usuario">
          <button aria-label="cancelar" onClick={() => setViewAsUser(null)} className={`absolute inset-0 w-full ${SURFACE.overlay}`} tabIndex={-1} />
          <div className={`relative ${SURFACE.card} p-5 w-[460px] max-w-[92vw] space-y-3`}>
            <h3 className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>
              Ver como <span className={COLOR.accent.text}>{viewAsUser.email}</span> ({viewAsUser.role})
            </h3>
            <p className={TYPE.meta}>
              Simulación de navegación read-only (30 min). El motivo es obligatorio y queda en auditoría.
              No concede permisos: toda mutación sigue exigiendo tu rol real.
            </p>
            <textarea
              value={motivo} onChange={(e) => setMotivo(e.target.value)} maxLength={200} rows={2} autoFocus
              className={`${SURFACE.input} ${CTA.focusRing} w-full`} placeholder="Motivo (mín. 3 caracteres)…"
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setViewAsUser(null)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Cancelar</button>
              <button
                disabled={motivo.trim().length < 3 || viewBusy}
                onClick={startViewAs}
                className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs disabled:opacity-40`}
              >
                {viewBusy ? 'Iniciando…' : 'Ver como este usuario'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
