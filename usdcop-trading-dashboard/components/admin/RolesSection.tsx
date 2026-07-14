'use client';

/**
 * Roles y vistas (CTR-RBAC-001 / migración 056). Editor de la matriz permiso×rol
 * (DB-backed, `rbac_role_permissions`) + previsualización "Ver como" por rol. Los
 * cambios se auditan (`role_perm_change`) y aplican en el PRÓXIMO inicio de sesión
 * del usuario (el JWT baked se re-emite); la aplicación server-side (relay) refleja
 * el cambio dentro del cache de 60s del resolver. Deny-by-default intacto: quitar
 * admin:all del rol admin está bloqueado en el servidor.
 */
import { useEffect, useMemo, useState } from 'react';
import { ShieldCheck, Eye } from 'lucide-react';

import { COLOR, CTA, TYPE } from '@/lib/ui/tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { Badge, Card, EmptyState, SkeletonRows, fmtRelative, useNow } from './ui';
import { useToast } from './ui/toast';

interface RolesMatrixResponse {
  roles: string[];
  permissions: string[];
  matrix: Record<string, string[]>;
}

const ROLE_LABEL: Record<string, string> = {
  admin: 'Admin', developer: 'Developer', subscriber: 'Suscriptor', free: 'Free',
};

export function RolesSection() {
  const roles = useAdminWidget<RolesMatrixResponse>('/api/admin/roles', { refreshMs: REFRESH.users });
  const now = useNow(30_000);
  const { toast } = useToast();
  const d = roles.data;

  // Copia local editable de la matriz (se re-siembra cuando llega data fresca).
  const [matrix, setMatrix] = useState<Record<string, Set<string>>>({});
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    if (!d) return;
    const next: Record<string, Set<string>> = {};
    for (const r of d.roles) next[r] = new Set(d.matrix[r] ?? []);
    setMatrix(next);
  }, [d]);

  const has = (role: string, perm: string) => matrix[role]?.has(perm) ?? false;

  async function toggle(role: string, perm: string) {
    const key = `${role}:${perm}`;
    const enabled = !has(role, perm);
    setBusy(key);
    // Optimista.
    setMatrix((m) => {
      const s = new Set(m[role] ?? []);
      if (enabled) s.add(perm); else s.delete(perm);
      return { ...m, [role]: s };
    });
    try {
      const res = await fetch('/api/admin/roles', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, permission: perm, enabled }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok || body?.ok === false) {
        throw new Error(body?.error?.message ?? `HTTP ${res.status}`);
      }
      toast(`${ROLE_LABEL[role] ?? role}: ${enabled ? 'concedido' : 'quitado'} ${perm}`, 'ok');
    } catch (e) {
      // Revertir en error.
      setMatrix((m) => {
        const s = new Set(m[role] ?? []);
        if (enabled) s.delete(perm); else s.add(perm);
        return { ...m, [role]: s };
      });
      toast(String((e as Error)?.message ?? e), 'error');
    } finally {
      setBusy(null);
    }
  }

  async function previewAs(role: string) {
    try {
      const res = await fetch('/api/admin/impersonate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, motivo: `Previsualización de la vista del rol ${role} desde la consola` }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        throw new Error(b?.error?.message ?? `HTTP ${res.status}`);
      }
      window.location.href = '/hub';
    } catch (e) {
      toast(`No se pudo previsualizar: ${String((e as Error)?.message ?? e)}`, 'error');
    }
  }

  const meta = roles.updatedAt
    ? <span title={new Date(roles.updatedAt).toISOString()}>{fmtRelative(new Date(roles.updatedAt).toISOString(), now)}</span>
    : null;

  const permCount = useMemo(() => (d ? d.permissions.length : 0), [d]);

  return (
    <div className="space-y-4" data-testid="admin-section-roles">
      <Card
        title="Roles y vistas"
        icon={<ShieldCheck className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Matriz permiso×rol (deny-by-default). Los cambios se auditan y aplican en el próximo inicio de sesión del usuario. La aplicación en APIs refleja el cambio en ≤60s."
        badge={d ? <Badge tone="neutral">{d.roles.length} roles · {permCount} permisos</Badge> : null}
        meta={meta} stale={roles.stale}
      >
        {roles.error && !d && (
          <EmptyState
            icon={<ShieldCheck className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar la matriz: {roles.error}</>}
            action={<button onClick={roles.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        )}
        {roles.loading && !d && <SkeletonRows rows={6} cols={5} />}
        {d && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className={`text-left ${COLOR.textSecondary} border-b border-[var(--gm-border)]`}>
                  <th className="py-2 pr-3">Permiso</th>
                  {d.roles.map((r) => (
                    <th key={r} className="px-3 text-center">
                      <div className="flex flex-col items-center gap-1">
                        <span className="font-semibold">{ROLE_LABEL[r] ?? r}</span>
                        <button
                          onClick={() => previewAs(r)}
                          className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-1.5 py-0.5`}
                          title={`Ver la vista del rol ${r} (solo lectura)`}
                        >
                          <Eye className="w-3 h-3" aria-hidden /> Ver como
                        </button>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {d.permissions.map((perm) => (
                  <tr key={perm} className="border-b border-[var(--gm-border)]">
                    <td className={`py-2 pr-3 font-mono ${COLOR.textPrimary}`}>{perm}</td>
                    {d.roles.map((role) => {
                      const locked = role === 'admin' && perm === 'admin:all';
                      const key = `${role}:${perm}`;
                      return (
                        <td key={role} className="px-3 text-center">
                          <input
                            type="checkbox"
                            checked={has(role, perm)}
                            disabled={locked || busy === key}
                            onChange={() => toggle(role, perm)}
                            aria-label={`${perm} para ${role}`}
                            title={locked ? 'admin:all no se puede quitar del rol admin' : undefined}
                            className="w-4 h-4 accent-[var(--gm-accent)] cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className={`${TYPE.meta} mt-3`}>
              Nota: los cambios de permisos aplican cuando el usuario vuelve a iniciar sesión (el token se re-emite).
              Deny-by-default: si la tabla queda vacía, el sistema usa la matriz estática del contrato.
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}
