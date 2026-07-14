'use client';

/**
 * Riesgo y bloqueos (CTR-ADMIN-CONSOLE-001 · pestaña Risk). Lee /api/admin/risk (DB
 * directa + relay al SB para el kill GLOBAL, tolerante a fallo → partial_errors) y
 * expone: kill-switch global, conteo paper/live, llaves API por aprobar, usuarios
 * suspendidos y los techos del sistema.
 *
 * El kill global es un POST con confirmación TIPADA "STOP" (modal destructivo, mismo
 * idioma que el modal de rechazo de QueueSection). El bearer del admin real se relaya
 * vía sbAuthHeaders (localStorage 'auth-token') — SB audita con su identidad.
 */
import { useState } from 'react';
import { AlertTriangle, KeyRound, ShieldAlert, ShieldCheck, SlidersHorizontal, UserX } from 'lucide-react';

import type { AdminRiskResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, SURFACE, TYPE } from '@/lib/ui/tokens';
import { GM_KILL } from '@/lib/ui/gm-tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { Badge, Card, EmptyState, SkeletonRows, StatusDot, fmtDateTime, fmtRelative, useNow } from './ui';
import { useToast } from './ui/toast';

const DASH = '—';

/** El propio token de SignalBridge del admin autoriza el kill global — nunca un secreto compartido. */
const sbAuthHeaders = (): Record<string, string> => {
  const t = typeof window !== 'undefined' ? window.localStorage.getItem('auth-token') : null;
  return t ? { Authorization: `Bearer ${t}` } : {};
};

export function RiskSection() {
  const risk = useAdminWidget<AdminRiskResponse>('/api/admin/risk', { refreshMs: REFRESH.risk });
  const { toast } = useToast();
  const now = useNow(20_000);
  const [confirm, setConfirm] = useState<{ enable: boolean } | null>(null);
  const [typed, setTyped] = useState('');
  const [busy, setBusy] = useState(false);
  // Aprobar/rechazar llaves de exchange (confirmación tipada + check anti-retiro).
  const [keyAction, setKeyAction] = useState<{ id: string; email: string; exchange: string; action: 'approve' | 'reject' } | null>(null);
  const [keyTyped, setKeyTyped] = useState('');
  const [withdrawOk, setWithdrawOk] = useState(false);
  const [keyBusy, setKeyBusy] = useState(false);
  const d = risk.data;

  const meta = risk.updatedAt
    ? <span title={new Date(risk.updatedAt).toISOString()}>{fmtRelative(new Date(risk.updatedAt).toISOString(), now)}</span>
    : null;

  const doKill = async () => {
    if (!confirm || typed !== 'STOP') return;
    const enable = confirm.enable;
    setBusy(true);
    try {
      const r = await fetch('/api/admin/risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...sbAuthHeaders() },
        body: JSON.stringify({ enable, confirm: 'STOP' }),
      });
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        toast(enable ? 'Kill global ACTIVADO — ejecución detenida.' : 'Kill global desactivado — ejecución reanudada.', 'ok', 'admin-flash');
        risk.reload();
      } else {
        toast(
          r.status === 401
            ? 'Sesión de SignalBridge no encontrada — vuelve a iniciar sesión.'
            : `Error (${r.status}): ${body?.error?.message ?? body?.error ?? 'no se pudo aplicar el kill global.'}`,
          'error', 'admin-flash',
        );
      }
    } catch (e) {
      toast(`Error de red: ${String(e)}`, 'error', 'admin-flash');
    } finally {
      setBusy(false);
      setConfirm(null);
      setTyped('');
    }
  };

  const keyToken = keyAction?.action === 'approve' ? 'APPROVE' : 'REJECT';
  const keyConfirmReady = !!keyAction
    && keyTyped === keyToken
    && (keyAction.action === 'reject' || withdrawOk)
    && !keyBusy;

  const doKeyAction = async () => {
    if (!keyAction || !keyConfirmReady) return;
    const { id, action, exchange } = keyAction;
    setKeyBusy(true);
    try {
      const r = await fetch(`/api/admin/risk/keys/${id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          confirm: keyToken,
          withdraw_disabled_confirmed: action === 'approve' ? withdrawOk : undefined,
        }),
      });
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        toast(action === 'approve' ? `Llave ${exchange} aprobada (verificada).` : `Llave ${exchange} rechazada (revocada).`, 'ok', 'admin-flash');
        risk.reload();
      } else {
        toast(`Error (${r.status}): ${body?.error?.message ?? body?.error ?? 'no se pudo aplicar la acción sobre la llave.'}`, 'error', 'admin-flash');
      }
    } catch (e) {
      toast(`Error de red: ${String(e)}`, 'error', 'admin-flash');
    } finally {
      setKeyBusy(false);
      setKeyAction(null);
      setKeyTyped('');
      setWithdrawOk(false);
    }
  };

  const active = d?.global_kill.active ?? null;
  const killCard = active === true ? GM_KILL.cardActive : active === false ? GM_KILL.cardCalm : GM_KILL.cardUnknown;
  const killTone = active === true ? 'error' : active === false ? 'ok' : 'neutral';

  return (
    <div className="space-y-4" data-testid="admin-section-riesgo">
      {risk.error && !d && (
        <Card title="Riesgo y bloqueos" icon={<ShieldAlert className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}>
          <EmptyState
            icon={<ShieldAlert className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar riesgo: {risk.error}</>}
            action={<button onClick={risk.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        </Card>
      )}
      {risk.loading && !d && (
        <Card title="Riesgo y bloqueos" icon={<ShieldAlert className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}>
          <SkeletonRows rows={3} cols={3} />
        </Card>
      )}

      {d && (
        <>
          {d.partial_errors.length > 0 && (
            <p className={`${TYPE.meta} ${COLOR.warn.text} flex items-center gap-1.5`}>
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" aria-hidden />
              Checks no disponibles: {d.partial_errors.join(' · ')}
            </p>
          )}

          {/* Kill switch global */}
          <Card
            title="Kill switch global"
            icon={active === true
              ? <ShieldAlert className={`w-4 h-4 ${COLOR.error.text}`} aria-hidden />
              : <ShieldCheck className={`w-4 h-4 ${COLOR.ok.text}`} aria-hidden />}
            info="Detiene toda ejecución automática en todos los exchanges y cuentas al instante. Requiere escribir STOP para confirmar y queda en auditoría."
            meta={meta} stale={risk.stale}
          >
            <div className={`${killCard} p-4 flex flex-wrap items-center justify-between gap-4`}>
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <StatusDot tone={killTone} label={active === true ? 'ejecución detenida' : active === false ? 'ejecución activa' : 'desconocido'} />
                  <span className={`${TYPE.body} font-semibold ${active === true ? COLOR.error.text : active === false ? COLOR.ok.text : COLOR.textSecondary}`}>
                    {active === true ? 'EJECUCIÓN DETENIDA' : active === false ? 'Ejecución activa' : 'Estado desconocido'}
                  </span>
                </div>
                {d.global_kill.detail && <p className={TYPE.meta}>{d.global_kill.detail}</p>}
              </div>
              {active === true ? (
                <button
                  onClick={() => { setConfirm({ enable: false }); setTyped(''); }}
                  className={`${GM_KILL.resumeBtn} ${CTA.focusRing} px-4 py-2.5 text-xs`}
                >
                  Reanudar ejecución
                </button>
              ) : (
                <button
                  onClick={() => { setConfirm({ enable: true }); setTyped(''); }}
                  disabled={active === null}
                  className={`${GM_KILL.haltBtn} ${CTA.focusRing} px-4 py-2.5 text-xs disabled:opacity-40`}
                >
                  Activar kill global
                </button>
              )}
            </div>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            {/* Modos paper/live + techos */}
            <Card title="Modos de ejecución" icon={<SlidersHorizontal className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />} meta={meta} stale={risk.stale}>
              <div className="flex items-center gap-2 mb-3">
                <Badge tone="warn">○ PAPER {d.modes.paper}</Badge>
                <Badge tone="ok">● LIVE {d.modes.live}</Badge>
              </div>
              <div className={`${TYPE.sectionTitle} mb-1.5`}>Techos del sistema</div>
              <table className="w-full text-xs">
                <tbody>
                  {([
                    ['Máx. notional', `$${d.limits.max_notional_usd.toLocaleString()}`],
                    ['Máx. posiciones abiertas', String(d.limits.max_open_positions)],
                    ['Máx. pérdida diaria', `${d.limits.max_daily_loss_pct}%`],
                  ] as Array<[string, string]>).map(([label, v]) => (
                    <tr key={label} className="h-8 border-b border-slate-800/50">
                      <td className={`pr-3 ${COLOR.textSecondary}`}>{label}</td>
                      <td className={`text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            {/* Kill por usuario */}
            <Card
              title="Kill por usuario"
              icon={<ShieldAlert className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
              badge={<Badge tone="neutral">{d.user_kills.filter((k) => k.kill_switch).length} activos</Badge>}
              meta={meta} stale={risk.stale}
            >
              {d.user_kills.length === 0 ? (
                <EmptyState icon={<ShieldCheck className="w-8 h-8" aria-hidden />} cause="Sin límites de riesgo por usuario registrados." />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                        <th className="py-2 pr-3">Usuario</th><th className="pr-3">Modo</th><th className="pr-3">Kill</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.user_kills.map((k) => (
                        <tr key={k.user_id} className="h-9 border-b border-slate-800/50">
                          <td className={`pr-3 ${COLOR.textPrimary}`} title={k.user_id}>{k.email ?? k.user_id}</td>
                          <td className="pr-3"><Badge tone={k.mode === 'live' ? 'ok' : 'warn'}>{k.mode === 'live' ? '● LIVE' : '○ PAPER'}</Badge></td>
                          <td className="pr-3">{k.kill_switch ? <Badge tone="error">KILL</Badge> : <span className={COLOR.textSecondary}>—</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </div>

          {/* Llaves API por aprobar */}
          <Card
            title="Conexiones API por aprobar"
            icon={<KeyRound className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
            info="Nuevas llaves de exchange. Verifica que los permisos de RETIRO estén deshabilitados antes de aprobar: aprobar marca la llave como verificada (operable), rechazar la revoca. Ambas acciones piden confirmación tipada y quedan en auditoría."
            badge={<Badge tone={d.api_pending.length > 0 ? 'warn' : 'neutral'}>{d.api_pending.length}</Badge>}
            meta={meta} stale={risk.stale}
          >
            {d.api_pending.length === 0 ? (
              <EmptyState icon={<KeyRound className="w-8 h-8" aria-hidden />} cause="Sin llaves de exchange pendientes de aprobación." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                      <th className="py-2 pr-3">Usuario</th><th className="pr-3">Exchange</th>
                      <th className="pr-3">Solicitado</th><th className="text-right">Acciones</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.api_pending.map((k) => (
                      <tr key={k.id} className="h-9 border-b border-slate-800/50">
                        <td className={`pr-3 ${COLOR.textPrimary}`} title={k.user_id}>{k.email ?? k.user_id}</td>
                        <td className={`pr-3 ${TYPE.mono}`}>{k.exchange}</td>
                        <td className="pr-3" title={k.created_at}>{fmtRelative(k.created_at, now)}</td>
                        <td className="text-right whitespace-nowrap">
                          <span className="inline-flex items-center gap-1">
                            <button
                              onClick={() => { setKeyAction({ id: k.id, email: k.email ?? k.user_id, exchange: k.exchange, action: 'approve' }); setKeyTyped(''); setWithdrawOk(false); }}
                              className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-2 py-1 text-[10px]`}
                              aria-label={`aprobar llave ${k.exchange} de ${k.email ?? k.user_id}`}
                            >
                              Aprobar
                            </button>
                            <button
                              onClick={() => { setKeyAction({ id: k.id, email: k.email ?? k.user_id, exchange: k.exchange, action: 'reject' }); setKeyTyped(''); setWithdrawOk(false); }}
                              className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-2 py-1 text-[10px] ${COLOR.error.text}`}
                              aria-label={`rechazar llave ${k.exchange} de ${k.email ?? k.user_id}`}
                            >
                              Rechazar
                            </button>
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Usuarios suspendidos */}
          <Card
            title="Usuarios suspendidos / rechazados"
            icon={<UserX className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
            badge={<Badge tone="neutral">{d.suspended.length}</Badge>}
            meta={meta} stale={risk.stale}
          >
            {d.suspended.length === 0 ? (
              <EmptyState icon={<UserX className="w-8 h-8" aria-hidden />} cause="Sin usuarios suspendidos ni rechazados." />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                      <th className="py-2 pr-3">Usuario</th><th className="pr-3">Estado</th><th className="pr-3">Alta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.suspended.map((u) => (
                      <tr key={u.id} className="h-9 border-b border-slate-800/50">
                        <td className={`pr-3 ${COLOR.textPrimary}`} title={u.id}>{u.email}</td>
                        <td className="pr-3"><span className={COLOR.error.text}>{u.status}</span></td>
                        <td className={`pr-3 ${TYPE.mono}`}>{u.created_at ? fmtDateTime(u.created_at) : DASH}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}

      {/* modal aprobar/rechazar llave — confirmación tipada + gate anti-retiro en approve */}
      {keyAction && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="confirmar acción sobre llave">
          <button aria-label="cancelar" onClick={() => setKeyAction(null)} className={`absolute inset-0 w-full ${SURFACE.overlay}`} tabIndex={-1} />
          <div className={`relative ${SURFACE.card} p-5 w-[480px] max-w-[92vw] space-y-3`}>
            <h3 className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>
              {keyAction.action === 'approve'
                ? <>Aprobar la llave <span className={COLOR.accent.text}>{keyAction.exchange}</span> de {keyAction.email}</>
                : <>Rechazar la llave <span className={COLOR.error.text}>{keyAction.exchange}</span> de {keyAction.email}</>}
            </h3>
            <p className={TYPE.meta}>
              {keyAction.action === 'approve'
                ? 'Aprobar marca la llave como verificada (operable). Jamás custodiamos retiros: confirma que la llave NO tiene permiso de retiro. Escribe APPROVE; queda en auditoría.'
                : 'Rechazar revoca la llave (no operable). Escribe REJECT para confirmar; queda en auditoría.'}
            </p>
            {keyAction.action === 'approve' && (
              <label className={`flex items-start gap-2 ${TYPE.meta} ${COLOR.textPrimary} cursor-pointer`}>
                <input
                  type="checkbox"
                  checked={withdrawOk}
                  onChange={(e) => setWithdrawOk(e.target.checked)}
                  className="mt-0.5"
                  aria-label="confirmo que la llave no tiene permiso de retiro"
                />
                <span>Confirmo que verifiqué que esta llave <strong>no tiene permiso de retiro</strong>.</span>
              </label>
            )}
            <input
              value={keyTyped}
              onChange={(e) => setKeyTyped(e.target.value)}
              autoFocus
              aria-label={`escribe ${keyToken} para confirmar`}
              placeholder={`Escribe ${keyToken}`}
              className={`${SURFACE.input} ${CTA.focusRing} w-full`}
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setKeyAction(null)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Cancelar</button>
              <button
                disabled={!keyConfirmReady}
                onClick={doKeyAction}
                className={`${keyAction.action === 'approve' ? CTA.primary : CTA.destructive} ${CTA.focusRing} px-3 py-1.5 text-xs disabled:opacity-40`}
              >
                {keyBusy ? 'Aplicando…' : keyAction.action === 'approve' ? 'Confirmar aprobación' : 'Confirmar rechazo'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* modal de confirmación tipada "STOP" — rojo pleno vive aquí */}
      {confirm && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="confirmar kill global">
          <button aria-label="cancelar" onClick={() => setConfirm(null)} className={`absolute inset-0 w-full ${SURFACE.overlay}`} tabIndex={-1} />
          <div className={`relative ${SURFACE.card} p-5 w-[460px] max-w-[92vw] space-y-3`}>
            <h3 className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>
              {confirm.enable
                ? <>Detener <span className={COLOR.error.text}>TODA la ejecución</span> automática</>
                : 'Reanudar la ejecución automática global'}
            </h3>
            <p className={TYPE.meta}>
              {confirm.enable
                ? 'Esto activa el kill switch global en todos los exchanges y cuentas al instante. Escribe STOP para confirmar; queda en auditoría.'
                : 'Esto desactiva el kill switch global y permite que la ejecución automática se reanude. Escribe STOP para confirmar; queda en auditoría.'}
            </p>
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoFocus
              aria-label="escribe STOP para confirmar"
              placeholder="Escribe STOP"
              className={`${SURFACE.input} ${CTA.focusRing} w-full`}
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setConfirm(null)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Cancelar</button>
              <button
                disabled={typed !== 'STOP' || busy}
                onClick={doKill}
                className={`${CTA.destructive} ${CTA.focusRing} px-3 py-1.5 text-xs disabled:opacity-40`}
              >
                {busy ? 'Aplicando…' : confirm.enable ? 'Confirmar kill global' : 'Confirmar reanudación'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
