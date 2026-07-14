'use client';

/**
 * Sistema (CTR-ADMIN-UI-001 §2.5): state-dependent Vote 2 (prose → ⓘ tooltip),
 * freshness table with progress-to-threshold + copy-DAG action (the privileged
 * "Reintentar backfill" is the next increment), full services table.
 */
import { useState } from 'react';
import {
  Activity, AlertTriangle, ClipboardCopy, Cpu, Database, ExternalLink, Gauge,
  PlayCircle, Server, ShieldCheck, Timer, Workflow,
} from 'lucide-react';

import type { SystemStatus } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, RING_ACCENT, SURFACE, TYPE, type SemanticTone } from '@/lib/ui/tokens';

import type { WidgetState } from './useAdminWidget';
import {
  Badge, Card, EmptyState, ProgressBar, SkeletonRows, StatusDot,
  fmtDateTime, fmtHours, fmtRelative, useNow,
} from './ui';
import { useToast } from './ui/toast';

const FRESH_TONE: Record<string, SemanticTone> = { ok: 'ok', warn: 'warn', stale: 'error', unknown: 'neutral' };

/**
 * Observability deep-links (CTR-FE-BE-001 §4.11 / CTR-OBS-001): the console does NOT
 * replace Grafana — it links to it. Hosts come from NEXT_PUBLIC_*_URL; a link whose env
 * is unset is HIDDEN (Task 5) — never render a dead localhost link in prod. In local dev,
 * export the NEXT_PUBLIC_*_URL vars to surface the stack.
 */
const OBS_LINKS: Array<{ label: string; href: string | undefined; hint: string }> = [
  { label: 'Grafana', href: process.env.NEXT_PUBLIC_GRAFANA_URL, hint: '4 dashboards: trading · mlops · system · macro' },
  { label: 'Prometheus', href: process.env.NEXT_PUBLIC_PROMETHEUS_URL, hint: 'targets, reglas, SLOs' },
  { label: 'AlertManager', href: process.env.NEXT_PUBLIC_ALERTMANAGER_URL, hint: '53 reglas / 16 grupos' },
  { label: 'Jaeger', href: process.env.NEXT_PUBLIC_JAEGER_URL, hint: 'trazas por traceparent' },
  { label: 'Airflow', href: process.env.NEXT_PUBLIC_AIRFLOW_URL, hint: 'DAGs L0→L7 + watchdog' },
  { label: 'MLflow', href: process.env.NEXT_PUBLIC_MLFLOW_URL, hint: 'runs de entrenamiento H5' },
];

/** Only links with a configured host (Task 5 — hide dead localhost fallbacks). */
const VISIBLE_OBS_LINKS = OBS_LINKS.filter(
  (l): l is { label: string; href: string; hint: string } => !!l.href,
);

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

  // Recovery modal (Task 2): trigger the recovery DAG via Airflow REST with a typed
  // confirm; the clipboard command stays as an offline/degraded fallback.
  const [recover, setRecover] = useState<{ dag: string; label: string } | null>(null);
  const [typed, setTyped] = useState('');
  const [busy, setBusy] = useState(false);

  const meta = system.updatedAt
    ? <span title={new Date(system.updatedAt).toISOString()}>{fmtRelative(new Date(system.updatedAt).toISOString(), now)}</span>
    : null;

  const copyDag = (dag: string) => {
    navigator.clipboard?.writeText(`airflow dags trigger ${dag}`).then(
      () => toast(`Copiado: airflow dags trigger ${dag}`, 'ok'),
      () => toast('No se pudo copiar', 'error'),
    );
  };

  const doRecover = async () => {
    if (!recover || typed !== recover.dag) return;
    setBusy(true);
    try {
      const r = await fetch('/api/admin/system/recover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dag_id: recover.dag, confirm: recover.dag }),
      });
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        toast(`Recuperación disparada: ${recover.dag}`, 'ok', 'admin-flash');
        setTimeout(() => system.reload(), 1500);
      } else {
        const msg = body?.error?.message ?? body?.error ?? 'no se pudo disparar el DAG.';
        // Airflow no configurado (503) → ofrece copiar el comando manual.
        toast(`Error (${r.status}): ${msg} — usa «copiar» para el comando manual.`, 'error', 'admin-flash');
        copyDag(recover.dag);
      }
    } catch (e) {
      toast(`Error de red: ${String(e)}`, 'error', 'admin-flash');
    } finally {
      setBusy(false);
      setRecover(null);
      setTyped('');
    }
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
        info="Umbrales según data-freshness.md (SSOT). «Reintentar» dispara el DAG de recuperación vía Airflow REST (acción privilegiada, confirmación tipada + auditoría); el icono de copiar deja el comando manual como respaldo."
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
                      <td className="text-right whitespace-nowrap">
                        {RECOVERY_DAG[f.id] && (
                          <span className="inline-flex items-center gap-1">
                            <button
                              onClick={() => { setRecover({ dag: RECOVERY_DAG[f.id], label: f.label }); setTyped(''); }}
                              className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-2 py-1 text-[10px]`}
                              aria-label={`reintentar recuperación de ${f.label}`}
                              title={`Disparar ${RECOVERY_DAG[f.id]} en Airflow`}
                            >
                              <PlayCircle className="w-3 h-3" aria-hidden /> Reintentar
                            </button>
                            <button
                              onClick={() => copyDag(RECOVERY_DAG[f.id])}
                              className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-1.5 py-1 text-[10px]`}
                              aria-label={`copiar comando de recuperación de ${f.label}`}
                              title="Copiar comando manual"
                            >
                              <ClipboardCopy className="w-3 h-3" aria-hidden />
                            </button>
                          </span>
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

      {/* §4.11 SLOs de inferencia + recursos (Prometheus / node_exporter) */}
      {s && (s.slos?.length || s.resources?.length) ? (
        <Card
          title="SLOs de inferencia y recursos"
          icon={<Timer className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
          info="Latencia p50/p95/p99, tasa de error y throughput del inference-api (histogram_quantile sobre Prometheus), más CPU/memoria vía node_exporter. Sin datos = Prometheus no expone la métrica."
          meta={meta} stale={system.stale}
        >
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {(s.slos ?? []).map((m) => {
              const tone: SemanticTone = m.ok == null ? 'neutral' : m.ok ? 'ok' : 'error';
              return (
                <div key={m.id} className="rounded-lg border border-slate-800 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className={`text-xs font-semibold ${COLOR.textPrimary}`}>{m.label}</span>
                    <StatusDot tone={tone} label={m.ok == null ? 'sin datos' : m.ok ? 'ok' : 'fuera de objetivo'} />
                  </div>
                  <div className={`${TYPE.mono} ${COLOR.textPrimary} mt-1`}>
                    {m.value == null ? '—' : `${m.value} ${m.unit}`}
                  </div>
                  <div className={TYPE.meta}>objetivo {m.target}</div>
                </div>
              );
            })}
            {(s.resources ?? []).map((r) => {
              const tone: SemanticTone = r.pct == null ? 'neutral' : r.pct >= 90 ? 'error' : r.pct >= 75 ? 'warn' : 'ok';
              return (
                <div key={r.id} className="rounded-lg border border-slate-800 px-3 py-2">
                  <div className="flex items-center gap-1.5">
                    <Cpu className={`w-3.5 h-3.5 ${COLOR.textSecondary}`} aria-hidden />
                    <span className={`text-xs font-semibold ${COLOR.textPrimary}`}>{r.label}</span>
                  </div>
                  <div className={`${TYPE.mono} ${COLOR.textPrimary} mt-1 mb-1`}>{r.pct == null ? '—' : `${r.pct}%`}</div>
                  <ProgressBar ratio={r.pct != null ? r.pct / 100 : 0} tone={tone} />
                </div>
              );
            })}
          </div>
        </Card>
      ) : null}

      {/* §4.11 pipeline L0→L6 (Airflow REST) */}
      {s?.pipeline?.length ? (
        <Card
          title="Pipeline L0 → L6"
          icon={<Workflow className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
          info="Estado del último dag_run de cada etapa vía Airflow REST. Sin runs = etapa aún no ejecutada."
          meta={meta} stale={system.stale}
        >
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {s.pipeline.map((p) => {
              const tone: SemanticTone = p.ok == null ? 'neutral'
                : p.state === 'running' ? 'info' : p.ok ? 'ok' : 'error';
              return (
                <div key={p.dag_id} className="rounded-lg border border-slate-800 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="inline-flex items-center gap-1.5">
                      <Badge tone="neutral">{p.stage}</Badge>
                      <span className={`text-xs font-semibold ${COLOR.textPrimary}`}>{p.name}</span>
                    </span>
                    <StatusDot tone={tone} label={p.state ?? 'sin runs'} />
                  </div>
                  <div className={`${TYPE.meta} mt-1 flex items-center justify-between gap-2`}>
                    <span className={COLOR[tone].text}>{p.state ?? 'sin runs'}</span>
                    <span title={p.last_run ?? undefined}>{p.last_run ? fmtRelative(p.last_run, now) : '—'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      ) : null}

      {/* §4.11 targets de Prometheus + alertas activas (AlertManager) */}
      {s && (s.prom_targets?.length || s.alerts_active?.length) ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card
            title="Targets de Prometheus"
            icon={<Server className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
            badge={s.prom_targets?.length
              ? <Badge tone={s.prom_targets.some((t) => t.health === 'down') ? 'error' : 'ok'}>
                  {s.prom_targets.filter((t) => t.health === 'up').length}/{s.prom_targets.length} up
                </Badge>
              : undefined}
            meta={meta} stale={system.stale}
          >
            {!s.prom_targets?.length ? (
              <EmptyState icon={<Server className="w-8 h-8" aria-hidden />} cause="Prometheus no disponible o sin targets activos." />
            ) : (
              <div className="overflow-x-auto max-h-72">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                      <th className="py-2 pr-3">Job</th><th className="pr-3">Instancia</th>
                      <th className="pr-3">Salud</th><th className="text-right">Scrape</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.prom_targets.map((t, idx) => {
                      const tone: SemanticTone = t.health === 'up' ? 'ok' : t.health === 'down' ? 'error' : 'neutral';
                      return (
                        <tr key={`${t.job}-${t.instance}-${idx}`} className={`h-8 border-b border-slate-800/50 ${SURFACE.tableRowHover}`}>
                          <td className={`pr-3 font-medium ${COLOR.textPrimary}`}>{t.job}</td>
                          <td className={`pr-3 ${TYPE.mono} ${COLOR.textSecondary}`} title={t.instance}>{t.instance}</td>
                          <td className="pr-3">
                            <span className="inline-flex items-center gap-1.5"><StatusDot tone={tone} label={t.health} /><span className={COLOR[tone].text}>{t.health}</span></span>
                          </td>
                          <td className={`text-right ${TYPE.mono} ${COLOR.textSecondary}`}>{t.last_scrape_ms != null ? `${t.last_scrape_ms} ms` : '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card
            title="Alertas activas"
            icon={<AlertTriangle className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
            badge={<Badge tone={s.alerts_active?.length ? 'error' : 'ok'}>{s.alerts_active?.length ?? 0}</Badge>}
            meta={meta} stale={system.stale}
          >
            {!s.alerts_active?.length ? (
              <EmptyState icon={<ShieldCheck className="w-8 h-8" aria-hidden />} cause="Sin alertas activas en AlertManager." />
            ) : (
              <div className="overflow-x-auto max-h-72">
                <table className="w-full text-xs">
                  <thead>
                    <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                      <th className="py-2 pr-3">Alerta</th><th className="pr-3">Severidad</th><th className="text-right">Desde</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.alerts_active.map((a, idx) => {
                      const tone: SemanticTone = a.severity === 'critical' ? 'error' : a.severity === 'warning' ? 'warn' : 'info';
                      return (
                        <tr key={`${a.name}-${idx}`} className={`h-9 border-b border-slate-800/50 ${SURFACE.tableRowHover}`}>
                          <td className={`pr-3 font-medium ${COLOR.textPrimary}`} title={a.summary ?? undefined}>{a.name}</td>
                          <td className="pr-3"><Badge tone={tone}>{a.severity}</Badge></td>
                          <td className={`text-right ${TYPE.mono} ${COLOR.textSecondary}`} title={a.since ?? undefined}>{a.since ? fmtRelative(a.since, now) : '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      ) : null}

      {/* deep-links de observabilidad (§4.11 CTR-FE-BE-001) — solo hosts configurados (Task 5) */}
      {VISIBLE_OBS_LINKS.length > 0 && (
        <Card
          title="Observabilidad (deep-links)"
          icon={<Gauge className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
          info="La consola no reemplaza a Grafana: enlaza al stack. Hosts configurables vía NEXT_PUBLIC_*_URL; los no definidos se ocultan."
        >
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {VISIBLE_OBS_LINKS.map((l) => (
              <a
                key={l.label}
                href={l.href}
                target="_blank"
                rel="noreferrer"
                className={`${CTA.ghost} ${CTA.focusRing} flex items-center justify-between gap-2 px-3 py-2.5`}
              >
                <span>
                  <span className={`block text-xs font-semibold ${COLOR.textPrimary}`}>{l.label}</span>
                  <span className={`block ${TYPE.meta}`}>{l.hint}</span>
                </span>
                <ExternalLink className={`w-3.5 h-3.5 shrink-0 ${COLOR.textSecondary}`} aria-hidden />
              </a>
            ))}
          </div>
        </Card>
      )}

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

      {/* modal de recuperación — confirmación tipada (escribe el dag_id) + auditoría */}
      {recover && (
        <div className="fixed inset-0 z-[75] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="confirmar recuperación">
          <button aria-label="cancelar" onClick={() => setRecover(null)} className={`absolute inset-0 w-full ${SURFACE.overlay}`} tabIndex={-1} />
          <div className={`relative ${SURFACE.card} p-5 w-[480px] max-w-[92vw] space-y-3`}>
            <h3 className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>
              Reintentar recuperación de <span className={COLOR.accent.text}>{recover.label}</span>
            </h3>
            <p className={TYPE.meta}>
              Dispara el DAG <span className={TYPE.mono}>{recover.dag}</span> en Airflow. Es una acción
              privilegiada y queda en auditoría. Escribe <span className={TYPE.mono}>{recover.dag}</span> para confirmar.
            </p>
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              autoFocus
              aria-label={`escribe ${recover.dag} para confirmar`}
              placeholder={recover.dag}
              className={`${SURFACE.input} ${CTA.focusRing} w-full ${TYPE.mono}`}
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setRecover(null)} className={`${CTA.ghost} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Cancelar</button>
              <button
                disabled={typed !== recover.dag || busy}
                onClick={doRecover}
                className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs disabled:opacity-40`}
              >
                {busy ? 'Disparando…' : 'Disparar recuperación'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
