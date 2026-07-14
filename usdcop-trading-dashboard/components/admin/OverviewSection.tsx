'use client';

/**
 * Overview (CTR-ADMIN-UI-001 §2.2): 4 business KpiTiles (all clickable — a KPI that
 * leads nowhere is decoration), 3 equal system cards (freshness with progress-to-
 * threshold, state-dependent Vote 2, full services), alerts never mute. Business row
 * excludes test traffic server-side (C4).
 */
import { useState } from 'react';
import { ArrowRight, BellRing, Database, ExternalLink, Server, ShieldCheck } from 'lucide-react';

import type { AdminSectionId, BusinessKpis, SystemStatus, AdminAlert } from '@/lib/contracts/admin-console.contract';
import { QUEUE_SLA_HOURS } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, RING_ACCENT, TYPE, type SemanticTone } from '@/lib/ui/tokens';

import { PAID_PLANS } from '@/lib/billing/prices';

import { REFRESH, useAdminWidget, type WidgetState } from './useAdminWidget';
import {
  Badge, Card, EmptyState, KpiTile, ProgressBar, SkeletonRows, StatusDot,
  fmtCop, fmtHours, fmtPct, fmtRelative, useNow,
} from './ui';

const FRESH_TONE: Record<string, SemanticTone> = { ok: 'ok', warn: 'warn', stale: 'error', unknown: 'neutral' };

/** Alert rules armed today (§2.2 "reglas armadas: N" — keep in sync with the list below). */
const ARMED_RULES = ['SLA de cola', 'Vote 2 pendiente', 'datos degradados/servicio caído'];

export function OverviewSection({ onNavigate, queueWaitingMax, pendingQueue, system }: {
  onNavigate: (s: AdminSectionId) => void;
  queueWaitingMax: number | null;
  pendingQueue: { count: number; test_hidden: number } | null;
  /** Lifted to the shell so Overview/Sistema/app-bar share ONE poller (§3.4). */
  system: WidgetState<SystemStatus>;
}) {
  const kpis = useAdminWidget<BusinessKpis>('/api/admin/kpis', { refreshMs: REFRESH.kpis });
  const [activeWindow, setActiveWindow] = useState<'7d' | '30d'>('7d');
  const now = useNow(10_000);

  const k = kpis.data;
  const s = system.data;

  const alerts: AdminAlert[] = [];
  if (k && k.pending_queue > 0 && (queueWaitingMax ?? 0) > QUEUE_SLA_HOURS) {
    alerts.push({ id: 'queue-sla', severity: 'warn', section: 'registros',
      message: `Cola con SLA vencido: ${k.pending_queue} pendiente(s), espera máx ${fmtHours(queueWaitingMax)}` });
  } else if (k && k.pending_queue > 0) {
    alerts.push({ id: 'queue', severity: 'info', section: 'registros',
      message: `${k.pending_queue} registro(s) esperando aprobación` });
  }
  if (s?.vote2?.pending) {
    alerts.push({ id: 'vote2', severity: 'warn', section: 'sistema',
      message: `Vote 2 pendiente: ${s.vote2.strategy_name} (${s.vote2.gates_passed}/${s.vote2.gates_total} gates)` });
  }
  for (const f of s?.freshness ?? []) {
    if (f.status === 'stale' || f.status === 'warn') {
      alerts.push({ id: `fresh-${f.id}`, severity: f.status === 'stale' ? 'critical' : 'warn', section: 'sistema',
        message: `${f.label}: ${fmtHours(f.age_hours)} de antigüedad (umbral ${fmtHours(f.threshold_hours)})` });
    }
  }
  for (const svc of s?.services ?? []) {
    if (!svc.ok) alerts.push({ id: `svc-${svc.name}`, severity: 'critical', section: 'sistema', message: `Servicio ${svc.name} no responde` });
  }

  const metaOf = (w: { updatedAt: number | null }) =>
    w.updatedAt ? <span title={new Date(w.updatedAt).toISOString()}>{fmtRelative(new Date(w.updatedAt).toISOString(), now)}</span> : null;

  return (
    <div className="space-y-4">
      {/* Fila 1 — negocio (4 tiles §2.2) */}
      {kpis.error && !k && (
        <Card title="Negocio" icon={<Database className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}>
          <EmptyState icon={<Database className="w-8 h-8" aria-hidden />} cause={<>KPIs no disponibles: {kpis.error}</>}
            action={<button onClick={kpis.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>} />
        </Card>
      )}
      {kpis.loading && !k && <SkeletonRows rows={1} cols={4} />}
      {k && (
        <div className={`grid gap-3 sm:grid-cols-2 lg:grid-cols-3 ${kpis.stale ? 'opacity-60' : ''}`}>
          <KpiTile
            label="Usuarios" value={String(k.total_users)}
            delta={k.new_7d > 0 ? `+${k.new_7d} en 7d` : undefined} deltaTone={k.new_7d > 0 ? 'ok' : 'neutral'}
            note={k.new_7d === 0 ? 'sin altas en 7d' : undefined}
            onClick={() => onNavigate('usuarios')}
          />
          <KpiTile
            label={`Activos ${activeWindow}`}
            value={String(activeWindow === '7d' ? k.active_7d : k.active_30d)}
            note={
              <button
                onClick={(e) => { e.stopPropagation(); setActiveWindow(activeWindow === '7d' ? '30d' : '7d'); }}
                className={`underline ${CTA.focusRing}`}
              >
                ver {activeWindow === '7d' ? '30d' : '7d'}
              </button>
            }
            onClick={() => onNavigate('usuarios')}
          />
          <KpiTile
            label="Pendientes" value={String(pendingQueue?.count ?? k.pending_queue)}
            note={(pendingQueue?.test_hidden ?? k.pending_test_hidden) > 0 ? `(${pendingQueue?.test_hidden ?? k.pending_test_hidden} test)` : undefined}
            deltaTone={(pendingQueue?.count ?? k.pending_queue) > 0 ? 'warn' : 'neutral'}
            delta={(queueWaitingMax ?? 0) > QUEUE_SLA_HOURS ? 'SLA vencido' : undefined}
            onClick={() => onNavigate('registros')}
            testId="kpi-pendientes"
          />
          <KpiTile
            label="MRR" value={fmtCop(k.mrr_cop)}
            note={k.mrr_cop === 0 ? 'sin suscriptores de pago' : 'mensual'}
            dimmed={k.mrr_cop == null}
            onClick={() => onNavigate('ingresos')}
          />
          {(() => {
            const paidSubs = PAID_PLANS.reduce((s, p) => s + (k.plan_mix[p] ?? 0), 0);
            return (
              <KpiTile
                label="Suscriptores" value={String(paidSubs)}
                note={k.conversion_30d_pct != null ? `conv. ${fmtPct(k.conversion_30d_pct)}` : undefined}
                deltaTone={paidSubs > 0 ? 'ok' : 'neutral'}
                onClick={() => onNavigate('ingresos')}
              />
            );
          })()}
          <KpiTile
            label="Churn" value={k.churn_monthly_pct == null ? '—' : fmtPct(k.churn_monthly_pct)}
            note={k.churn_monthly_pct == null ? 'sin datos aún' : 'mensual'}
            dimmed={k.churn_monthly_pct == null}
            deltaTone={(k.churn_monthly_pct ?? 0) > 0 ? 'warn' : 'neutral'}
            onClick={() => onNavigate('ingresos')}
          />
        </div>
      )}

      {/* Fila 2 — sistema: 3 cards iguales (§2.2) */}
      <div className="grid gap-3 lg:grid-cols-3">
        <Card
          title="Frescura de datos" icon={<Database className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
          meta={metaOf(system)} stale={system.stale}
        >
          {system.loading && !s && <SkeletonRows rows={4} cols={2} />}
          {system.error && !s && <p className={`${TYPE.meta} ${COLOR.error.text}`}>{system.error}</p>}
          <ul className="space-y-2.5">
            {s?.freshness.map((f) => {
              const tone = FRESH_TONE[f.status];
              const ratio = f.age_hours != null ? f.age_hours / f.threshold_hours : 0;
              return (
                <li key={f.id}>
                  <div className="flex items-center gap-2 text-xs mb-1">
                    <StatusDot tone={tone} label={f.status} />
                    <span className={`flex-1 ${COLOR.textPrimary}`}>{f.label}</span>
                    <span className={`${TYPE.mono} ${COLOR.textSecondary}`} title={f.latest ?? undefined}>
                      {fmtHours(f.age_hours)} / {fmtHours(f.threshold_hours)}
                    </span>
                  </div>
                  <ProgressBar ratio={ratio} tone={tone} />
                </li>
              );
            })}
          </ul>
        </Card>

        {/* Vote 2 — estado-dependiente (§2.2): PENDIENTE ruidoso, APPROVED silencioso */}
        <div className={s?.vote2?.pending ? RING_ACCENT : ''}>
          <Card
            title="Vote 2 (modelos)" icon={<ShieldCheck className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
            info="El Vote 2 humano se emite sobre los números del bundle publicado, en /dashboard — aquí solo contador y enlace (una sola superficie de aprobación)."
            meta={metaOf(system)} stale={system.stale}
          >
            {system.loading && !s && <SkeletonRows rows={2} cols={2} />}
            {s && !s.vote2 && <p className={TYPE.meta}>Sin bundle publicado.</p>}
            {s?.vote2 && (
              <div className="space-y-2 text-xs">
                <div className={`font-semibold ${COLOR.textPrimary}`}>{s.vote2.strategy_name}</div>
                <div className="flex items-center gap-2">
                  <Badge tone={s.vote2.pending ? 'warn' : s.vote2.status === 'APPROVED' || s.vote2.status === 'LIVE' ? 'ok' : 'error'}>
                    {s.vote2.status}
                  </Badge>
                  <span className={COLOR.textSecondary}>gates {s.vote2.gates_passed}/{s.vote2.gates_total}</span>
                  {s.vote2.recommendation && <span className={COLOR.textSecondary}>· {s.vote2.recommendation}</span>}
                </div>
                {s.vote2.pending ? (
                  <a href="/dashboard" className={`${CTA.primary} ${CTA.focusRing} inline-flex items-center gap-1.5 px-3 py-1.5 text-xs`}>
                    Revisar y votar <ExternalLink className="w-3 h-3" aria-hidden />
                  </a>
                ) : (
                  <p className={TYPE.meta}>
                    {s.vote2.approved_by ? `aprobado por ${s.vote2.approved_by}` : 'sin aprobación registrada'}
                    {s.vote2.approved_at ? ` · ${s.vote2.approved_at.slice(0, 16).replace('T', ' ')}` : ''}
                  </p>
                )}
              </div>
            )}
          </Card>
        </div>

        <Card
          title="Servicios" icon={<Server className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
          meta={metaOf(system)} stale={system.stale}
        >
          {system.loading && !s && <SkeletonRows rows={3} cols={2} />}
          <ul className="space-y-2">
            {s?.services.map((svc) => (
              <li key={svc.name} className="flex items-center gap-2 text-xs">
                <StatusDot tone={svc.ok ? 'ok' : 'error'} label={svc.ok ? 'ok' : 'caído'} />
                <span className={`flex-1 ${COLOR.textPrimary}`}>{svc.name}</span>
                <span className={`${TYPE.mono} ${COLOR.textSecondary}`}>
                  {svc.ok ? `${svc.latency_ms} ms` : svc.error ?? 'sin respuesta'}
                </span>
              </li>
            ))}
          </ul>
          {s?.deploy?.phase && (
            <p className={`${TYPE.meta} mt-2`}>deploy: {s.deploy.phase}{s.deploy.runner ? ` · ${s.deploy.runner}` : ''}</p>
          )}
          {s && s.partial_errors.length > 0 && (
            <p className={`${TYPE.meta} ${COLOR.warn.text} mt-2`}>Checks no disponibles: {s.partial_errors.join(' · ')}</p>
          )}
        </Card>
      </div>

      {/* Fila 3 — alertas: nunca vacío mudo (§2.2) */}
      <Card title="Alertas" icon={<BellRing className={`w-4 h-4 ${COLOR.warn.text}`} aria-hidden />}>
        {alerts.length === 0 ? (
          <p className={TYPE.meta}>
            Sin alertas activas · reglas armadas: {ARMED_RULES.length} ({ARMED_RULES.join(', ')}) · última disparada: ninguna en esta sesión.
          </p>
        ) : (
          <ul className="space-y-2">
            {alerts.map((a) => (
              <li key={a.id}>
                <button
                  onClick={() => onNavigate(a.section)}
                  className={`w-full flex items-center justify-between gap-3 rounded-lg border px-3 py-2 text-xs text-left ${CTA.focusRing}
                    ${a.severity === 'critical' ? COLOR.error.badge : a.severity === 'warn' ? COLOR.warn.badge : COLOR.neutral.badge}`}
                >
                  <span>{a.message}</span>
                  <span className="inline-flex items-center gap-1 shrink-0 underline">resolver <ArrowRight className="w-3 h-3" aria-hidden /></span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
