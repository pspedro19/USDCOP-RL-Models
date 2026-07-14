'use client';

/**
 * Modelos (CTR-ADMIN-CONSOLE-001 · pestaña Models). Compone el registry SSOT con los
 * approval_state publicados: tabla activo/modelo/sharpe/gates/estado + una sub-tarjeta de
 * "Candidatos a promoción" (approval_state PENDING) con sus gates.
 *
 * SIN botón de aprobar/rechazar aquí: el Vote 2 humano vive en /dashboard sobre los
 * números del bundle publicado (regla una-sola-superficie, approval-gates.md I-4). Esta
 * vista SOLO deep-linkea a /dashboard. Valores desconocidos → "—" (nunca inventados).
 */
import { Boxes, CheckCircle2, ExternalLink, XCircle } from 'lucide-react';

import type {
  AdminModelCandidate, AdminModelEstado, AdminModelRow, AdminModelsResponse,
} from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, RING_ACCENT, TYPE } from '@/lib/ui/tokens';
import { GM_VIOLET } from '@/lib/ui/gm-tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { Badge, Card, EmptyState, SkeletonRows, fmtRelative, useNow } from './ui';

const DASH = '—';

const ESTADO_LABEL: Record<AdminModelEstado, string> = {
  produccion: 'Producción', candidato: 'Candidato', experimental: 'Experimental', deprecado: 'Deprecado',
};

/** Rol=matiz: producción verde · candidato cian · experimental violeta · deprecado gris. */
function EstadoBadge({ estado }: { estado: AdminModelEstado }) {
  if (estado === 'experimental') {
    return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${GM_VIOLET.badge}`}>{ESTADO_LABEL.experimental}</span>;
  }
  const tone = estado === 'produccion' ? 'ok' : estado === 'candidato' ? 'accent' : 'neutral';
  return <Badge tone={tone}>{ESTADO_LABEL[estado]}</Badge>;
}

function fmtNum(n: number | null, digits = 2): string {
  return n == null ? DASH : n.toFixed(digits);
}

function CandidateCard({ c }: { c: AdminModelCandidate }) {
  const passed = c.gates.filter((g) => g.passed).length;
  const total = c.gates.length;
  return (
    <div className={`${RING_ACCENT} p-4 space-y-3`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <div className={`${TYPE.body} font-semibold ${COLOR.textPrimary}`}>{c.strategy_name}</div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Badge tone="warn">{c.status}</Badge>
            {total > 0 && <span className={COLOR.textSecondary}>gates {passed}/{total}</span>}
            {c.recommendation && <span className={COLOR.textSecondary}>· recomendación {c.recommendation}</span>}
            {c.backtest_year && <span className={COLOR.textSecondary}>· backtest {c.backtest_year}</span>}
          </div>
        </div>
        <a
          href="/dashboard"
          className={`${CTA.primary} ${CTA.focusRing} inline-flex items-center gap-1.5 px-4 py-2 text-xs shrink-0`}
        >
          Revisar en /dashboard <ExternalLink className="w-3.5 h-3.5" aria-hidden />
        </a>
      </div>
      {total > 0 && (
        <ul className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
          {c.gates.map((g) => (
            <li key={g.gate} className="flex items-center gap-1.5 text-xs">
              {g.passed
                ? <CheckCircle2 className={`w-3.5 h-3.5 shrink-0 ${COLOR.ok.text}`} aria-hidden />
                : <XCircle className={`w-3.5 h-3.5 shrink-0 ${COLOR.error.text}`} aria-hidden />}
              <span className={COLOR.textSecondary}>{g.label}</span>
              {g.value != null && <span className={`${TYPE.mono} ${COLOR.textPrimary}`}>{g.value}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ModelsSection() {
  const models = useAdminWidget<AdminModelsResponse>('/api/admin/models', { refreshMs: REFRESH.models });
  const now = useNow(30_000);
  const d = models.data;

  const meta = models.updatedAt
    ? <span title={new Date(models.updatedAt).toISOString()}>{fmtRelative(new Date(models.updatedAt).toISOString(), now)}</span>
    : null;

  return (
    <div className="space-y-4" data-testid="admin-section-modelos">
      <Card
        title="Registro de modelos"
        icon={<Boxes className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="Registry SSOT (RegistryBuilder) + approval_state publicados. El voto de promoción NO vive aquí: se emite en /dashboard sobre los números del bundle (una sola superficie)."
        badge={d ? <Badge tone="neutral">{d.models.length}</Badge> : null}
        meta={meta} stale={models.stale}
      >
        {models.error && !d && (
          <EmptyState
            icon={<Boxes className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar el registro: {models.error}</>}
            action={<button onClick={models.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        )}
        {models.loading && !d && <SkeletonRows rows={4} cols={5} />}
        {d && d.models.length === 0 && !models.error && (
          <EmptyState icon={<Boxes className="w-8 h-8" aria-hidden />} cause="Sin modelos en el registry — publica un bundle para verlos aquí." />
        )}
        {d && d.models.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className={`text-left ${COLOR.textSecondary} border-b border-slate-800`}>
                  <th className="py-2 pr-3">Modelo</th><th className="pr-3">Activo</th>
                  <th className="pr-3">Marco</th><th className="pr-3 text-right">Sharpe</th>
                  <th className="pr-3">Gates</th><th className="pr-3">Estado</th>
                </tr>
              </thead>
              <tbody>
                {d.models.map((m: AdminModelRow) => (
                  <tr key={m.strategy_id} className="h-10 border-b border-slate-800/50">
                    <td className={`pr-3 font-medium ${COLOR.textPrimary}`} title={m.active_version ?? undefined}>{m.display_name}</td>
                    <td className={`pr-3 ${TYPE.mono}`}>{m.asset_symbol}</td>
                    <td className={`pr-3 ${COLOR.textSecondary}`}>{m.timeframe ?? DASH}</td>
                    <td className={`pr-3 text-right ${TYPE.mono} ${COLOR.textPrimary}`}>{fmtNum(m.sharpe)}</td>
                    <td className={`pr-3 ${TYPE.mono} ${COLOR.textSecondary}`}>
                      {m.gates_passed != null && m.gates_total != null ? `${m.gates_passed}/${m.gates_total}` : DASH}
                    </td>
                    <td className="pr-3"><EstadoBadge estado={m.estado} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="Candidatos a promoción"
        icon={<Boxes className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
        info="approval_state en PENDING_APPROVAL. Un modelo requiere aprobación humana (Vote 2) y pasar sus gates antes de operar — la revisión se hace en /dashboard."
        badge={d ? <Badge tone={d.pending_count > 0 ? 'warn' : 'neutral'}>{d.pending_count}</Badge> : null}
        meta={meta} stale={models.stale}
      >
        {models.loading && !d && <SkeletonRows rows={2} cols={3} />}
        {d && d.candidates.length === 0 && (
          <EmptyState icon={<CheckCircle2 className="w-8 h-8" aria-hidden />} cause="No hay candidatos pendientes de promoción." />
        )}
        {d && d.candidates.length > 0 && (
          <div className="space-y-3">
            {d.candidates.map((c) => <CandidateCard key={c.strategy_id} c={c} />)}
          </div>
        )}
      </Card>
    </div>
  );
}
