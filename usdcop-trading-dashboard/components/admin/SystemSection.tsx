'use client';

/**
 * Sistema (CTR-ADMIN-UI-001 §2.5): state-dependent Vote 2 (prose → ⓘ tooltip),
 * freshness table with progress-to-threshold + copy-DAG action (the privileged
 * "Reintentar backfill" is the next increment), full services table.
 */
import { Activity, ClipboardCopy, Database, ExternalLink, ShieldCheck } from 'lucide-react';

import type { SystemStatus } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, RING_ACCENT, SURFACE, TYPE, type SemanticTone } from '@/lib/ui/tokens';

import type { WidgetState } from './useAdminWidget';
import {
  Badge, Card, EmptyState, ProgressBar, SkeletonRows, StatusDot,
  fmtDateTime, fmtHours, fmtRelative, useNow,
} from './ui';
import { useToast } from './ui/toast';

const FRESH_TONE: Record<string, SemanticTone> = { ok: 'ok', warn: 'warn', stale: 'error', unknown: 'neutral' };

/** Recovery DAG per freshness source (data-freshness.md — recovery procedures). */
const RECOVERY_DAG: Record<string, string> = {
  ohlcv_m5: 'core_l0_01_ohlcv_backfill',
  macro_daily: 'core_l0_03_macro_backfill',
  news: 'news_daily_pipeline',
  production_bundle: 'forecast_h5_l3_weekly_training',
};

export function SystemSection({ system }: { system: WidgetState<SystemStatus> }) {
  const { toast } = useToast();
  const now = useNow(10_000);
  const s = system.data;

  const meta = system.updatedAt
    ? <span title={new Date(system.updatedAt).toISOString()}>{fmtRelative(new Date(system.updatedAt).toISOString(), now)}</span>
    : null;

  const copyDag = (dag: string) => {
    navigator.clipboard?.writeText(`airflow dags trigger ${dag}`).then(
      () => toast(`Copiado: airflow dags trigger ${dag}`, 'ok'),
      () => toast('No se pudo copiar', 'error'),
    );
  };

  return (
    <div className="space-y-4">
      {/* §6.1 Vote 2 — estado-dependiente */}
      <div className={s?.vote2?.pending ? RING_ACCENT : ''}>
        <Card
          title="Aprobaciones de modelo (Vote 2)"
          icon={<ShieldCheck className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
          info="El Vote 2 humano se emite sobre los números del bundle publicado (summary_*.json), en la vista de aprobación de /dashboard. Aquí solo el contador y el enlace: nunca dos botones que aprueban lo mismo desde dos lugares (§6.1)."
          meta={meta} stale={system.stale}
        >
          {system.error && !s && (
            <EmptyState icon={<ShieldCheck className="w-8 h-8" aria-hidden />} cause={<>No disponible: {system.error}</>}
              action={<button onClick={system.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>} />
          )}
          {system.loading && !s && <SkeletonRows rows={2} cols={3} />}
          {s && !s.vote2 && <p className={TYPE.meta}>Sin bundle publicado en producción.</p>}
          {s?.vote2 && (
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="space-y-1.5 text-xs">
                <div className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>{s.vote2.strategy_name}</div>
                <div className="flex items-center gap-2">
                  <Badge tone={s.vote2.pending ? 'warn' : s.vote2.status === 'APPROVED' || s.vote2.status === 'LIVE' ? 'ok' : 'error'}>
                    {s.vote2.status}
                  </Badge>
                  <span className={COLOR.textSecondary}>gates {s.vote2.gates_passed}/{s.vote2.gates_total}</span>
                  {s.vote2.recommendation && <span className={COLOR.textSecondary}>· recomendación {s.vote2.recommendation}</span>}
                  {s.vote2.backtest_year && <span className={COLOR.textSecondary}>· backtest {s.vote2.backtest_year}</span>}
                </div>
                {!s.vote2.pending && s.vote2.approved_by && (
                  <p className={TYPE.meta}>aprobado por {s.vote2.approved_by} · {fmtDateTime(s.vote2.approved_at)}</p>
                )}
              </div>
              <a href="/dashboard" className={`${s.vote2.pending ? CTA.primary : CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1.5 px-4 py-2 text-xs shrink-0`}>
                {s.vote2.pending ? 'Revisar y votar' : 'Ver en /dashboard'} <ExternalLink className="w-3.5 h-3.5" aria-hidden />
              </a>
            </div>
          )}
        </Card>
      </div>

      {/* §6.3 salud de datos con barra hacia umbral + copiar DAG */}
      <Card
        title="Salud de datos" icon={<Database className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Umbrales según data-freshness.md (SSOT). El botón copia el comando de recuperación; el disparo directo del backfill (acción privilegiada auditada) llega en el siguiente incremento."
        meta={meta} stale={system.stale}
      >
        {system.loading && !s && <SkeletonRows rows={4} cols={5} />}
        {s && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                  <th className="py-2 pr-3">Fuente</th><th className="pr-3">Última actualización</th>
                  <th className="pr-3 w-44">Antigüedad / umbral</th><th className="pr-3">Estado</th>
                  <th className="text-right">Recuperación</th>
                </tr>
              </thead>
              <tbody>
                {s.freshness.map((f) => {
                  const tone = FRESH_TONE[f.status];
                  return (
                    <tr key={f.id} className={`h-10 border-b border-slate-800/50 ${SURFACE.tableRowHover}`}>
                      <td className={`pr-3 font-medium ${COLOR.textPrimary}`}>{f.label}</td>
                      <td className="pr-3" title={f.latest ?? undefined}>{f.latest ? fmtRelative(f.latest, now) : '—'}</td>
                      <td className="pr-3">
                        <div className={`${TYPE.mono} ${COLOR.textSecondary} mb-1`}>{fmtHours(f.age_hours)} / {fmtHours(f.threshold_hours)}</div>
                        <ProgressBar ratio={f.age_hours != null ? f.age_hours / f.threshold_hours : 0} tone={tone} />
                      </td>
                      <td className="pr-3">
                        <span className="inline-flex items-center gap-1.5">
                          <StatusDot tone={tone} label={f.status} /><span className={COLOR[tone].text}>{f.status}</span>
                        </span>
                      </td>
                      <td className="text-right">
                        {RECOVERY_DAG[f.id] && (
                          <button
                            onClick={() => copyDag(RECOVERY_DAG[f.id])}
                            className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-2 py-1 text-[10px]`}
                            aria-label={`copiar comando de recuperación de ${f.label}`}
                          >
                            <ClipboardCopy className="w-3 h-3" aria-hidden /> {RECOVERY_DAG[f.id]}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* servicios completos (§2.5) */}
      <Card title="Servicios y deploy" icon={<Activity className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />} meta={meta} stale={system.stale}>
        {system.loading && !s && <SkeletonRows rows={3} cols={3} />}
        {s && (
          <div className="space-y-3">
            <table className="w-full text-xs">
              <thead>
                <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                  <th className="py-2 pr-3">Servicio</th><th className="pr-3">Estado</th><th className="text-right">Latencia</th>
                </tr>
              </thead>
              <tbody>
                {s.services.map((svc) => (
                  <tr key={svc.name} className={`h-10 border-b border-slate-800/50 ${SURFACE.tableRowHover}`}>
                    <td className={`pr-3 font-medium ${COLOR.textPrimary}`}>{svc.name}</td>
                    <td className="pr-3">
                      <span className="inline-flex items-center gap-1.5">
                        <StatusDot tone={svc.ok ? 'ok' : 'error'} label={svc.ok ? 'ok' : 'caído'} />
                        <span className={svc.ok ? COLOR.ok.text : COLOR.error.text}>{svc.ok ? 'ok' : svc.error ?? 'sin respuesta'}</span>
                      </span>
                    </td>
                    <td className={`text-right ${TYPE.mono} ${COLOR.textSecondary}`}>{svc.latency_ms != null ? `${svc.latency_ms} ms` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {s.deploy && (
              <p className={TYPE.meta}>
                Último deploy: {s.deploy.phase ?? '—'}{s.deploy.runner ? ` · runner ${s.deploy.runner}` : ''}
                {s.deploy.updated_at ? ` · ${fmtDateTime(s.deploy.updated_at)}` : ''}
              </p>
            )}
            {s.partial_errors.length > 0 && (
              <p className={`${TYPE.meta} ${COLOR.warn.text}`}>Checks no disponibles: {s.partial_errors.join(' · ')}</p>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
