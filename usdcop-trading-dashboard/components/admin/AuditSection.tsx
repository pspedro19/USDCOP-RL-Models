'use client';

/**
 * Auditoría v2 (CTR-ADMIN-UI-001 §2.6): filters live in the URL (deep-linkable from
 * alerts), removable chips, SMART empty state (confesses how many rows the test
 * filter hides and offers the fix), expandable rows with formatted JSON, CSV export
 * that states the filtered count. Read-only over the append-only ledger.
 */
import { Fragment, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Download, ScrollText, SearchX } from 'lucide-react';

import type { AuditCategory, AuditEntry, AuditResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, SEVERITY_ROW_BG, SURFACE, TYPE, type SemanticTone } from '@/lib/ui/tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { useUrlState } from './useUrlState';
import { Badge, Card, EmptyState, SkeletonRows, TestBadge, fmtDateTime, fmtRelative, useNow } from './ui';

const CATEGORIES: AuditCategory[] = ['security', 'execution', 'billing', 'governance', 'admin', 'other'];
const CATEGORY_TONE: Record<AuditCategory, SemanticTone> = {
  security: 'error', execution: 'warn', billing: 'info', governance: 'accent', admin: 'neutral', other: 'neutral',
};

/** URL param → filter definition (labels for chips). */
const FILTER_DEFS = [
  { param: 'accion', label: 'acción' },
  { param: 'cat', label: 'categoría' },
  { param: 'usuario', label: 'usuario' },
  { param: 'desde', label: 'desde' },
  { param: 'hasta', label: 'hasta' },
] as const;

export function AuditSection() {
  const url = useUrlState();
  const now = useNow(10_000);
  const [expanded, setExpanded] = useState<number | null>(null);

  const action = url.get('accion');
  const category = url.get('cat') as '' | AuditCategory;
  const user = url.get('usuario');
  const from = url.get('desde');
  const to = url.get('hasta');
  const includeTest = url.get('test') === '1';

  const qs = useMemo(() => {
    const p = new URLSearchParams();
    if (action) p.set('action', action);
    if (category) p.set('category', category);
    if (user) p.set('user', user);
    if (from) p.set('from', from);
    if (to) p.set('to', to);
    if (includeTest) p.set('include_test', 'true');
    p.set('limit', '200');
    return p.toString();
  }, [action, category, user, from, to, includeTest]);

  const audit = useAdminWidget<AuditResponse>(`/api/admin/audit?${qs}`, { refreshMs: REFRESH.audit });
  // Probe with test traffic included — powers the smart empty state's hidden count (§2.6).
  const probeQs = qs.replace(/&?include_test=true/, '') + '&include_test=true';
  const probe = useAdminWidget<AuditResponse>(`/api/admin/audit?${probeQs}`, { refreshMs: REFRESH.audit });

  const entries = audit.data?.entries ?? [];
  const hiddenByTest = !includeTest && probe.data ? Math.max(probe.data.entries.length - entries.length, 0) : 0;

  const activeChips = FILTER_DEFS.filter((f) => url.get(f.param));

  return (
    <Card
      title="Auditoría (append-only)"
      icon={<ScrollText className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
      badge={<Badge tone="neutral">{audit.data ? `${entries.length} filas` : '…'}</Badge>}
      meta={audit.updatedAt ? <span title={new Date(audit.updatedAt).toISOString()}>{fmtRelative(new Date(audit.updatedAt).toISOString(), now)}</span> : null}
      stale={audit.stale}
    >
      {/* filtros → URL (§3.3) */}
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <input value={action} onChange={(e) => url.setMany({ accion: e.target.value || null })}
          placeholder="Acción (p.ej. kill)" aria-label="filtrar por acción" className={`${SURFACE.input} ${CTA.focusRing} w-36`} />
        <select value={category} onChange={(e) => url.setMany({ cat: e.target.value || null })} aria-label="filtrar por categoría" className={`${SURFACE.input} ${CTA.focusRing}`}>
          <option value="">Categoría: todas</option>
          {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <input value={user} onChange={(e) => url.setMany({ usuario: e.target.value || null })}
          placeholder="Usuario (email o id)" aria-label="filtrar por usuario" className={`${SURFACE.input} ${CTA.focusRing} w-44`} />
        <input type="date" value={from} onChange={(e) => url.setMany({ desde: e.target.value || null })} aria-label="desde" className={`${SURFACE.input} ${CTA.focusRing}`} />
        <input type="date" value={to} onChange={(e) => url.setMany({ hasta: e.target.value || null })} aria-label="hasta" className={`${SURFACE.input} ${CTA.focusRing}`} />
        <label className={`inline-flex items-center gap-1.5 text-xs ${COLOR.textSecondary}`}>
          <input type="checkbox" checked={includeTest} onChange={(e) => url.setMany({ test: e.target.checked ? '1' : null })} />
          incluir test
        </label>
        <a
          href={`/api/admin/audit?${qs}&format=csv`}
          className={`${CTA.ghost} ${CTA.focusRing} inline-flex items-center gap-1 px-3 py-1.5 text-xs ml-auto`}
        >
          <Download className="w-3.5 h-3.5" aria-hidden /> Exportar {entries.length} filas filtradas
        </a>
      </div>

      {/* chips removibles (§2.6) */}
      {activeChips.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {activeChips.map((f) => (
            <button
              key={f.param}
              onClick={() => url.setMany({ [f.param]: null })}
              className={`${COLOR.accent.badge} ${CTA.focusRing} inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold`}
              aria-label={`quitar filtro ${f.label}`}
            >
              {f.label}: {url.get(f.param)} ✕
            </button>
          ))}
          <button onClick={() => url.setMany(Object.fromEntries(FILTER_DEFS.map((f) => [f.param, null])))}
            className={`${TYPE.meta} underline ${CTA.focusRing}`}>
            limpiar todo
          </button>
        </div>
      )}

      {audit.error && !audit.data && (
        <EmptyState icon={<ScrollText className="w-8 h-8" aria-hidden />} cause={<>No se pudo cargar: {audit.error}</>}
          action={<button onClick={audit.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>} />
      )}
      {audit.loading && !audit.data && <SkeletonRows rows={5} cols={5} />}

      {audit.data && entries.length === 0 && (
        <EmptyState
          icon={<SearchX className="w-8 h-8" aria-hidden />}
          cause={
            hiddenByTest > 0
              ? <>0 visibles — <span className={COLOR.textPrimary}>{hiddenByTest}{probe.data && probe.data.entries.length >= 200 ? '+' : ''} ocultas</span> por el filtro de tráfico test.</>
              : 'Sin eventos con estos filtros.'
          }
          action={
            hiddenByTest > 0
              ? <button onClick={() => url.setMany({ test: '1' })} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Incluir test</button>
              : activeChips.length > 0
                ? <button onClick={() => url.setMany(Object.fromEntries(FILTER_DEFS.map((f) => [f.param, null])))} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Limpiar filtros</button>
                : undefined
          }
        />
      )}

      {audit.data && entries.length > 0 && (
        <div className="overflow-x-auto max-h-[32rem] overflow-y-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800 sticky top-0 bg-slate-950 z-10`}>
                <th className="py-2 pr-2 w-6" aria-label="expandir" />
                <th className="pr-3">Cuándo</th><th className="pr-3">Acción</th>
                <th className="pr-3">Categoría</th><th className="pr-3">Usuario</th>
                <th className="pr-3">Objeto</th><th className="text-right">IP</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((a: AuditEntry) => (
                <Fragment key={a.id}>
                  <tr
                    onClick={() => setExpanded(expanded === a.id ? null : a.id)}
                    className={`h-10 border-b border-slate-800/50 cursor-pointer ${SURFACE.tableRowHover} ${a.severity === 'critical' ? SEVERITY_ROW_BG : ''}`}
                    aria-expanded={expanded === a.id}
                  >
                    <td className="pr-2">
                      {expanded === a.id
                        ? <ChevronDown className={`w-3.5 h-3.5 ${COLOR.textSecondary}`} aria-hidden />
                        : <ChevronRight className={`w-3.5 h-3.5 ${COLOR.textSecondary}`} aria-hidden />}
                    </td>
                    <td className={`pr-3 whitespace-nowrap ${TYPE.mono}`} title={a.created_at}>{fmtRelative(a.created_at, now)}</td>
                    <td className={`pr-3 font-medium ${a.severity === 'critical' ? COLOR.error.text : a.severity === 'warn' ? COLOR.warn.text : COLOR.textPrimary}`}>
                      {a.action}
                    </td>
                    <td className="pr-3"><Badge tone={CATEGORY_TONE[a.category]}>{a.category}</Badge></td>
                    <td className="pr-3" title={a.user_id ?? undefined}>
                      {a.user_email ?? (a.user_id ? `${a.user_id.slice(0, 8)}…` : '—')} {a.is_test && <TestBadge />}
                    </td>
                    <td className={`pr-3 ${COLOR.textSecondary}`}>{a.object_type ?? '—'}{a.object_id ? ` · ${a.object_id.slice(0, 12)}` : ''}</td>
                    <td className={`text-right ${TYPE.mono} ${COLOR.textSecondary}`}>{a.ip ?? '—'}</td>
                  </tr>
                  {expanded === a.id && (
                    <tr className="border-b border-slate-800/50">
                      <td colSpan={7} className="py-2 px-6">
                        <pre className={`${TYPE.mono} text-[11px] ${COLOR.textSecondary} bg-slate-900/70 rounded-lg p-3 overflow-x-auto`}>
                          {JSON.stringify({ id: a.id, created_at: a.created_at, user_id: a.user_id, object_id: a.object_id, detail: a.detail }, null, 2)}
                        </pre>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
