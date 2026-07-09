'use client';

/**
 * Registros (CTR-ADMIN-UI-001 §2.3). READS from /api/admin/queue (DB, spec C1);
 * ACTIONS relay the admin's SignalBridge bearer (single authority).
 *
 * Approve = DEFERRED COMMIT (§3 decisión documentada): the row leaves instantly and a
 * 5s undo toast holds the POST; "Deshacer" cancels before anything reaches the server
 * (user stays pending, no audit row, no email). Reject = modal with mandatory motivo.
 * data-testids (approval-queue, pending-row-*, approve-*, reject-*, admin-flash) are
 * load-bearing for scripts/registration-qa.mjs.
 */
import { useMemo, useState } from 'react';
import { Check, Inbox, MoreHorizontal, UserCheck, X } from 'lucide-react';

import type { QueueItem, QueueResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, SURFACE, TYPE } from '@/lib/ui/tokens';

import type { WidgetState } from './useAdminWidget';
import {
  Badge, Card, DrawerHost, EmptyState, SkeletonRows, TestBadge,
  fmtDateTime, fmtHours, fmtRelative, useNow,
} from './ui';
import { useToast } from './ui/toast';

/** The admin's own SignalBridge token authorizes approvals — never a shared secret. */
const sbAuthHeaders = (): Record<string, string> => {
  const t = typeof window !== 'undefined' ? window.localStorage.getItem('auth-token') : null;
  return t ? { Authorization: `Bearer ${t}` } : {};
};

const SLA_WARN_H = 4;
const SLA_ERROR_H = 24;

async function postAction(id: string, action: 'approve' | 'reject', reason?: string) {
  const r = await fetch(`/api/admin/users/${id}/${action}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...sbAuthHeaders() },
    body: JSON.stringify(reason ? { reason } : {}),
  });
  const body = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, body };
}

export function QueueSection({ queue }: { queue: WidgetState<QueueResponse> }) {
  const { toast, toastUndo } = useToast();
  const now = useNow(15_000);
  const [hidden, setHidden] = useState<Set<string>>(new Set()); // optimistically removed
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [rejecting, setRejecting] = useState<QueueItem[] | null>(null);
  const [reason, setReason] = useState('');
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [detail, setDetail] = useState<QueueItem | null>(null);

  const items = useMemo(
    () => (queue.data?.items ?? []).filter((i) => !hidden.has(i.id)),
    [queue.data, hidden],
  );

  const hide = (ids: string[]) => setHidden((h) => new Set([...h, ...ids]));
  const unhide = (ids: string[]) => setHidden((h) => { const n = new Set(h); ids.forEach((id) => n.delete(id)); return n; });
  const clearSelection = () => setSelected(new Set());

  /** Deferred-commit approve for 1..N users (§2.3 + §3.1/§3.2). */
  const approve = (targets: QueueItem[]) => {
    const ids = targets.map((t) => t.id);
    hide(ids); clearSelection(); setDetail(null);
    const label = targets.length === 1 ? targets[0].email : `${targets.length} usuarios`;
    toastUndo(`Aprobando a ${label}…`, {
      testId: 'admin-toast-undo',
      onUndo: () => unhide(ids), // nothing reached the server: still pending, no audit row
      commit: async () => {
        const results = await Promise.all(targets.map((t) => postAction(t.id, 'approve').then((r) => ({ t, r }))));
        const okOnes = results.filter(({ r }) => r.ok);
        const failed = results.filter(({ r }) => !r.ok);
        if (okOnes.length === 1 && targets.length === 1) {
          const { t, r } = okOnes[0];
          toast(`Aprobado: ${t.email}${r.body?.email_sent ? ' · correo enviado' : ' · correo NO enviado'}`, 'ok', 'admin-flash');
        } else if (okOnes.length > 0) {
          toast(`Aprobados ${okOnes.length}/${targets.length}`, 'ok', 'admin-flash');
        }
        for (const { t, r } of failed) {
          unhide([t.id]); // revert with error (§3.2)
          toast(
            r.status === 401
              ? 'Sesión de SignalBridge no encontrada — vuelve a iniciar sesión.'
              : `Error (${r.status}) al aprobar ${t.email}: ${r.body?.message || r.body?.error || ''}`,
            'error', 'admin-flash',
          );
        }
        queue.reload();
      },
    });
  };

  /** Reject: modal with mandatory motivo; red lives only here (§1.3). */
  const confirmReject = async () => {
    if (!rejecting) return;
    const motivo = reason.trim();
    const targets = rejecting;
    setRejecting(null); setReason(''); setDetail(null);
    hide(targets.map((t) => t.id)); clearSelection();
    const results = await Promise.all(targets.map((t) => postAction(t.id, 'reject', motivo).then((r) => ({ t, r }))));
    for (const { t, r } of results) {
      if (r.ok) toast(`Rechazado: ${t.email}`, 'ok', 'admin-flash');
      else {
        unhide([t.id]);
        toast(`Error (${r.status}) al rechazar ${t.email}: ${r.body?.message || r.body?.error || ''}`, 'error', 'admin-flash');
      }
    }
    queue.reload();
  };

  const flagTest = async (u: QueueItem) => {
    setMenuFor(null);
    const r = await fetch(`/api/admin/users/${u.id}/flag-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_test: !u.is_test }),
    });
    if (r.ok) { toast(`${u.email} marcado como ${u.is_test ? 'cuenta real' : 'cuenta de test'}`, 'ok'); queue.reload(); }
    else toast(`Error (${r.status}) al marcar ${u.email}`, 'error');
  };

  const toggleSel = (id: string) => setSelected((s) => {
    const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n;
  });
  const selItems = items.filter((i) => selected.has(i.id));
  const waitTone = (h: number) => (h > SLA_ERROR_H ? 'error' : h > SLA_WARN_H ? 'warn' : null);

  const q = queue.data;
  return (
    <div className="space-y-4">
      <Card
        title="Cola de aprobación"
        icon={<UserCheck className={`w-4 h-4 ${COLOR.warn.text}`} aria-hidden />}
        testId="approval-queue"
        badge={
          <Badge tone="warn">{q ? `${Math.max(q.count - hidden.size, 0)}${q.test_hidden > 0 ? ` (${q.test_hidden} test)` : ''}` : '…'}</Badge>
        }
        meta={queue.updatedAt ? <span title={new Date(queue.updatedAt).toISOString()}>{fmtRelative(new Date(queue.updatedAt).toISOString(), now)}</span> : null}
        stale={queue.stale}
      >
        {queue.error && !queue.data && (
          <EmptyState
            icon={<Inbox className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar la cola: {queue.error}</>}
            action={<button onClick={queue.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        )}
        {queue.loading && !queue.data && <SkeletonRows rows={3} cols={5} />}
        {q && items.length === 0 && !queue.error && (
          <EmptyState
            icon={<Inbox className="w-8 h-8" aria-hidden />}
            cause="No hay solicitudes pendientes — las altas nuevas aparecen aquí al instante."
          />
        )}

        {q && items.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800 sticky top-0`}>
                  <th className="py-2 pr-3 w-8">
                    <input
                      type="checkbox" aria-label="seleccionar todos"
                      checked={selected.size > 0 && selected.size === items.length}
                      onChange={(e) => setSelected(e.target.checked ? new Set(items.map((i) => i.id)) : new Set())}
                    />
                  </th>
                  <th className="pr-3">Correo</th><th className="pr-3">Nombre</th>
                  <th className="pr-3">Solicitado</th><th className="pr-3">Esperando</th>
                  <th className="text-right">Acción</th>
                </tr>
              </thead>
              <tbody>
                {items.map((u) => {
                  const tone = waitTone(u.waiting_hours);
                  return (
                    <tr
                      key={u.id}
                      data-testid={`pending-row-${u.email}`}
                      onClick={() => setDetail(u)}
                      className={`h-10 border-b border-slate-800/50 cursor-pointer group ${SURFACE.tableRowHover}`}
                    >
                      <td className="pr-3" onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" aria-label={`seleccionar ${u.email}`} checked={selected.has(u.id)} onChange={() => toggleSel(u.id)} />
                      </td>
                      <td className={`pr-3 font-medium ${COLOR.textPrimary}`}>{u.email} {u.is_test && <TestBadge />}</td>
                      <td className="pr-3">{u.name ?? '—'}</td>
                      <td className="pr-3" title={u.created_at}>{fmtDateTime(u.created_at)}</td>
                      <td className={`pr-3 ${TYPE.mono} ${tone ? COLOR[tone].text : ''}`} title={`${u.waiting_hours} h`}>
                        {fmtHours(u.waiting_hours)}{tone === 'error' ? ' · SLA vencido' : tone === 'warn' ? ' · SLA' : ''}
                      </td>
                      <td className="text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                        <button
                          data-testid={`approve-${u.email}`}
                          onClick={() => approve([u])}
                          className={`${CTA.primary} ${CTA.focusRing} inline-flex items-center gap-1 px-2.5 py-1 mr-1.5 text-xs`}
                        >
                          <Check className="w-3 h-3" aria-hidden /> Aprobar
                        </button>
                        <button
                          data-testid={`reject-${u.email}`}
                          onClick={() => { setRejecting([u]); setReason(''); }}
                          className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-2.5 py-1 mr-1.5 text-xs`}
                        >
                          <X className="w-3 h-3" aria-hidden /> Rechazar
                        </button>
                        <span className="relative inline-block">
                          <button
                            aria-label={`más acciones para ${u.email}`} aria-haspopup="menu" aria-expanded={menuFor === u.id}
                            onClick={() => setMenuFor(menuFor === u.id ? null : u.id)}
                            className={`${CTA.ghost} ${CTA.focusRing} px-1.5 py-1`}
                          >
                            <MoreHorizontal className="w-3.5 h-3.5" aria-hidden />
                          </button>
                          {menuFor === u.id && (
                            <span role="menu" className={`absolute right-0 top-full mt-1 z-20 ${SURFACE.card} py-1 w-52 shadow-xl block text-left`}>
                              <button role="menuitem" onClick={() => flagTest(u)}
                                className={`block w-full text-left px-3 py-1.5 text-xs ${COLOR.textPrimary} hover:bg-slate-800/60 ${CTA.focusRing}`}>
                                {u.is_test ? 'Marcar como cuenta real' : 'Marcar como cuenta de test'}
                              </button>
                            </span>
                          )}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* sticky bulk bar (§3.5) */}
      {selItems.length > 0 && (
        <div className={`sticky bottom-4 z-30 ${SURFACE.card} px-4 py-2.5 flex items-center gap-3 shadow-xl`}>
          <span className={`${TYPE.body} ${COLOR.textPrimary} font-semibold`}>{selItems.length} seleccionado(s)</span>
          <button onClick={() => approve(selItems)} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Aprobar</button>
          <button onClick={() => { setRejecting(selItems); setReason(''); }} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Rechazar</button>
          <button onClick={clearSelection} className={`${TYPE.meta} underline ml-auto ${CTA.focusRing}`}>Limpiar</button>
        </div>
      )}

      {/* reject modal — mandatory motivo; destructive red lives here (§1.3/§3.1) */}
      {rejecting && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="confirmar rechazo">
          <button aria-label="cancelar" onClick={() => setRejecting(null)} className={`absolute inset-0 w-full ${SURFACE.overlay}`} tabIndex={-1} />
          <div className={`relative ${SURFACE.card} p-5 w-[440px] max-w-[92vw] space-y-3`}>
            <h3 className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>
              Rechazar {rejecting.length === 1 ? <>a <span className={COLOR.error.text}>{rejecting[0].email}</span></> : `${rejecting.length} solicitudes`}
            </h3>
            <p className={TYPE.meta}>El motivo es obligatorio: se envía al solicitante y queda en auditoría.</p>
            <textarea
              value={reason} onChange={(e) => setReason(e.target.value)} maxLength={500} rows={3} autoFocus
              className={`${SURFACE.input} w-full`} placeholder="Motivo del rechazo…"
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setRejecting(null)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Cancelar</button>
              <button
                disabled={reason.trim().length < 5}
                onClick={confirmReject}
                className={`${CTA.destructive} ${CTA.focusRing} px-3 py-1.5 text-xs disabled:opacity-40`}
              >
                Confirmar rechazo
              </button>
            </div>
          </div>
        </div>
      )}

      {/* row detail drawer (§2.3) */}
      <DrawerHost
        open={!!detail}
        title={detail?.email ?? ''}
        onClose={() => setDetail(null)}
        fields={detail ? [
          ['ID', detail.id],
          ['Nombre', detail.name ?? '—'],
          ['Estado', detail.status],
          ['Solicitado', fmtDateTime(detail.created_at)],
          ['Esperando', `${fmtHours(detail.waiting_hours)}${detail.waiting_hours > SLA_ERROR_H ? ' · SLA vencido' : ''}`],
          ['Señal de riesgo', detail.is_test ? 'dominio de test (heurística C4)' : 'ninguna detectada'],
        ] : []}
        footer={detail ? (
          <div className="flex gap-2">
            <button onClick={() => approve([detail])} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Aprobar</button>
            <button onClick={() => { setRejecting([detail]); setReason(''); }} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Rechazar</button>
          </div>
        ) : null}
      />
    </div>
  );
}
