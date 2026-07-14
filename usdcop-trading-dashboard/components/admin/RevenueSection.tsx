'use client';

/**
 * Ingresos (CTR-ADMIN-CONSOLE-001 · pestaña Revenue del prototipo Var B).
 *
 * Estructura FIEL del prototipo (KPIs mrr/arr/arpu/ltv · ingresos por plan · movimiento
 * de MRR · estado de cobros · ingresos por activo · dunning) con cifras REALES en vivo
 * (`/api/admin/revenue` → lib/admin/revenue.ts sobre sb_users.entitlements + audit_log,
 * precios del SSOT de billing). Sin mocks: 0 suscriptores ⇒ ceros genuinos; lo no medido
 * (reembolsos, contracargos) va "—" (checklist 10.1). Formateo COP por locale (10.2),
 * números tabulares (1.8).
 */
import { Wallet, CreditCard, Coins, TrendingUp, AlertTriangle } from 'lucide-react';

import type { AdminRevenueResponse } from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, TYPE } from '@/lib/ui/tokens';

import { REFRESH, useAdminWidget } from './useAdminWidget';
import { Badge, Card, EmptyState, KpiTile, SkeletonRows, fmtCop, fmtInt, fmtRelative, useNow } from './ui';

/** null = no medido → "—" (checklist 10.1). Un solo lugar para el placeholder. */
const DASH = '—';

export function RevenueSection() {
  const rev = useAdminWidget<AdminRevenueResponse>('/api/admin/revenue', { refreshMs: REFRESH.revenue });
  const now = useNow(30_000);
  const d = rev.data;

  const meta = rev.updatedAt
    ? <span title={new Date(rev.updatedAt).toISOString()}>{fmtRelative(new Date(rev.updatedAt).toISOString(), now)}</span>
    : null;

  const phaseBadge = d ? <Badge tone="warn">{d.phase_note}</Badge> : null;

  return (
    <div className="space-y-4" data-testid="admin-section-ingresos">
      {rev.error && !d && (
        <Card title="Ingresos" icon={<Wallet className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}>
          <EmptyState
            icon={<Wallet className="w-8 h-8" aria-hidden />}
            cause={<>No se pudo cargar ingresos: {rev.error}</>}
            action={<button onClick={rev.reload} className={`${CTA.primary} ${CTA.focusRing} px-3 py-1.5 text-xs`}>Reintentar</button>}
          />
        </Card>
      )}
      {rev.loading && !d && (
        <Card title="Ingresos" icon={<Wallet className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}>
          <SkeletonRows rows={2} cols={4} />
        </Card>
      )}

      {d && (
        <>
          {/* KPI row (§ prototipo adRevKpis) — cifras REALES sobre entitlements + audit_log */}
          <Card
            title="Métricas de ingresos"
            icon={<Wallet className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />}
            badge={phaseBadge}
            info="MRR/ARR/ARPU/LTV en vivo sobre suscripciones reales (sb_users.entitlements) a precios del SSOT de billing. LTV requiere churn medible; hasta entonces «—»."
            meta={meta} stale={rev.stale}
          >
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {([
                ['MRR', d.kpis.mrr], ['ARR', d.kpis.arr], ['ARPU', d.kpis.arpu], ['LTV', d.kpis.ltv],
              ] as Array<[string, number | null]>).map(([label, v]) => (
                <KpiTile key={label} label={label} value={fmtCop(v)} dimmed={v == null} note={v == null ? 'sin datos' : undefined} />
              ))}
            </div>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            {/* Ingresos por plan */}
            <Card title="Ingresos por plan" icon={<Coins className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />} badge={phaseBadge}>
              <table className="w-full text-xs">
                <tbody>
                  {d.por_plan.map((p) => (
                    <tr key={p.plan} className="h-9 border-b border-[var(--gm-border)]">
                      <td className={`pr-3 ${COLOR.textPrimary}`}>{p.plan}</td>
                      <td className={`text-right ${TYPE.mono} tabular-nums ${COLOR.textSecondary}`}>
                        {fmtCop(p.amount)}{p.pct != null && p.pct > 0 && <span className="ml-2 opacity-60">({p.pct}%)</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            {/* Movimiento de MRR (mes) */}
            <Card title="Movimiento de MRR (mes)" icon={<TrendingUp className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />} badge={phaseBadge}>
              <table className="w-full text-xs">
                <tbody>
                  {([
                    ['Nuevo MRR', d.movimiento.nuevo], ['Expansión (upgrades/add-ons)', d.movimiento.expansion],
                    ['Contracción', d.movimiento.contraccion], ['Cancelado (churn)', d.movimiento.churn],
                    ['MRR neto', d.movimiento.neto],
                  ] as Array<[string, number | null]>).map(([label, v], i, arr) => (
                    <tr key={label} className={`h-9 ${i === arr.length - 1 ? 'border-t border-[var(--gm-border)]' : 'border-b border-[var(--gm-border)]'}`}>
                      <td className={`pr-3 ${i === arr.length - 1 ? COLOR.textPrimary : COLOR.textSecondary}`}>{label}</td>
                      <td className={`text-right ${TYPE.mono} tabular-nums ${COLOR.textSecondary}`}>{fmtCop(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            {/* Estado de cobros (mes) */}
            <Card title="Estado de cobros (mes)" icon={<CreditCard className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />} badge={phaseBadge}>
              <table className="w-full text-xs">
                <tbody>
                  {([
                    ['Cobros exitosos', d.cobros.exitosos], ['Pagos fallidos', d.cobros.fallidos],
                    ['Reembolsos', d.cobros.reembolsos], ['Contracargos', d.cobros.contracargos],
                  ] as Array<[string, number | null]>).map(([label, v]) => (
                    <tr key={label} className="h-9 border-b border-[var(--gm-border)]">
                      <td className={`pr-3 ${COLOR.textSecondary}`}>{label}</td>
                      <td className={`text-right ${TYPE.mono} tabular-nums ${COLOR.textSecondary}`}>{fmtInt(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            {/* Ingresos por activo (add-ons) */}
            <Card title="Ingresos por activo (add-ons)" icon={<Coins className={`w-4 h-4 ${COLOR.accent.text}`} aria-hidden />} badge={phaseBadge}>
              <table className="w-full text-xs">
                <tbody>
                  {d.por_activo.map((a) => (
                    <tr key={a.symbol} className="h-9 border-b border-[var(--gm-border)]">
                      <td className={`pr-3 ${TYPE.mono} ${COLOR.textPrimary}`}>{a.symbol}</td>
                      <td className={`text-right ${TYPE.mono} tabular-nums ${COLOR.textSecondary}`}>{fmtCop(a.amount)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>

          {/* Pagos fallidos por recuperar (dunning) */}
          <Card title="Pagos fallidos por recuperar (dunning)" icon={<AlertTriangle className={`w-4 h-4 ${COLOR.warn.text}`} aria-hidden />} badge={phaseBadge}>
            {d.dunning.length === 0 ? (
              <EmptyState
                icon={<AlertTriangle className="w-8 h-8" aria-hidden />}
                cause="Sin pagos fallidos por recuperar en los últimos 30 días."
              />
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className={`text-left ${COLOR.textSecondary} border-b border-[var(--gm-border)]`}>
                    <th className="py-2 pr-3">Usuario</th><th className="pr-3">Plan</th>
                    <th className="pr-3">Monto</th><th className="pr-3">Intentos</th><th className="pr-3">Motivo</th>
                  </tr>
                </thead>
                <tbody>
                  {d.dunning.map((row, i) => (
                    <tr key={`${row.user}-${i}`} className="h-9 border-b border-[var(--gm-border)]">
                      <td className={`pr-3 ${COLOR.textPrimary}`}>{row.user}</td>
                      <td className="pr-3">{row.plan}</td>
                      <td className={`pr-3 ${TYPE.mono} tabular-nums ${COLOR.textSecondary}`}>{fmtCop(row.amount)}</td>
                      <td className={`pr-3 ${TYPE.mono}`}>{row.attempts ?? DASH}</td>
                      <td className="pr-3">{row.reason ?? DASH}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
