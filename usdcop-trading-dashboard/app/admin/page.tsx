'use client';

/**
 * /admin — Admin Console shell (CTR-ADMIN-CONSOLE-001 + UI polish CTR-ADMIN-UI-001).
 *
 * Compact app bar (§2.1): 20px title · tabs (active tab in URL, §3.3) · global
 * StatusDot (worst of freshness+services) + "Actualizado hace Xs" + manual ⟳.
 * The system and queue widgets are LIFTED here so app bar, Overview and Sistema
 * share one poller each; every other widget owns its fetch (C5). Access: admin:all.
 */
import { Suspense } from 'react';
import { RefreshCw } from 'lucide-react';

import {
  ADMIN_SECTIONS, type AdminModelsResponse, type AdminRiskResponse,
  type AdminSectionId, type QueueResponse, type SystemStatus,
} from '@/lib/contracts/admin-console.contract';
import { COLOR, CTA, SURFACE, TYPE, type SemanticTone } from '@/lib/ui/tokens';

import { TerminalShell } from '@/components/gm/TerminalShell';

import { AuditSection } from '@/components/admin/AuditSection';
import { CatalogSection } from '@/components/admin/CatalogSection';
import { ModelsSection } from '@/components/admin/ModelsSection';
import { OverviewSection } from '@/components/admin/OverviewSection';
import { QueueSection } from '@/components/admin/QueueSection';
import { RevenueSection } from '@/components/admin/RevenueSection';
import { RiskSection } from '@/components/admin/RiskSection';
import { RolesSection } from '@/components/admin/RolesSection';
import { SystemSection } from '@/components/admin/SystemSection';
import { UsersSection } from '@/components/admin/UsersSection';
import { REFRESH, useAdminWidget } from '@/components/admin/useAdminWidget';
import { useUrlState } from '@/components/admin/useUrlState';
import { Badge, StatusDot, fmtRelative, useNow } from '@/components/admin/ui';
import { ToastProvider } from '@/components/admin/ui/toast';

const VALID_TABS = new Set(ADMIN_SECTIONS.filter((s) => !s.phaseGate).map((s) => s.id));

function globalTone(s: SystemStatus | null): SemanticTone {
  if (!s) return 'neutral';
  if (s.services.some((x) => !x.ok) || s.freshness.some((f) => f.status === 'stale')) return 'error';
  if (s.freshness.some((f) => f.status === 'warn') || s.partial_errors.length > 0) return 'warn';
  return 'ok';
}

function AdminConsole() {
  const url = useUrlState();
  const tabParam = url.get('tab') as AdminSectionId;
  const section: AdminSectionId = VALID_TABS.has(tabParam) ? tabParam : 'overview';
  const setSection = (s: AdminSectionId) => url.setMany({ tab: s === 'overview' ? null : s });

  // Shared pollers (§3.4): queue feeds Registros + Pendientes KPI + tab badge;
  // system feeds app-bar dot + Overview + Sistema.
  const queue = useAdminWidget<QueueResponse>('/api/admin/queue', { refreshMs: REFRESH.queue });
  const system = useAdminWidget<SystemStatus>('/api/admin/system', { refreshMs: REFRESH.system });
  // Lifted for the tab badges so they show even when the tab is closed (mirrors how
  // queue feeds the Registros badge). Each section still owns its own fetch (C5).
  const models = useAdminWidget<AdminModelsResponse>('/api/admin/models', { refreshMs: REFRESH.models });
  const risk = useAdminWidget<AdminRiskResponse>('/api/admin/risk', { refreshMs: REFRESH.risk });
  const now = useNow(5_000);

  const queueWaitingMax = queue.data?.items.length
    ? Math.max(...queue.data.items.map((i) => i.waiting_hours))
    : null;
  const tone = globalTone(system.data);
  const updatedAt = Math.max(queue.updatedAt ?? 0, system.updatedAt ?? 0) || null;

  return (
    <div className={`${COLOR.textPrimary}`}>
      <div className="space-y-4">
        {/* app bar compacta (§2.1) */}
        <header className="flex items-center gap-4">
          <h1 className={TYPE.pageTitle}>Consola de administración</h1>
          <div className={`ml-auto flex items-center gap-2 ${TYPE.meta}`}>
            <StatusDot tone={tone} label={`estado global: ${tone}`} />
            <span title={updatedAt ? new Date(updatedAt).toISOString() : undefined}>
              {updatedAt ? `Actualizado ${fmtRelative(new Date(updatedAt).toISOString(), now)}` : 'Cargando…'}
            </span>
            <button
              onClick={() => { queue.reload(); system.reload(); }}
              aria-label="actualizar ahora"
              className={`${CTA.ghost} ${CTA.focusRing} p-1.5`}
            >
              <RefreshCw className="w-3.5 h-3.5" aria-hidden />
            </button>
          </div>
        </header>

        {/* tabs — activa en URL (§3.3); fase-gated no parecen clickeables (§2.1) */}
        <nav className="flex flex-wrap gap-1 border-b border-[var(--gm-border)] pb-px" role="tablist">
          {ADMIN_SECTIONS.map((s) => (
            <button
              key={s.id}
              role="tab"
              aria-selected={section === s.id}
              disabled={!!s.phaseGate}
              title={s.phaseGate}
              onClick={() => setSection(s.id)}
              data-testid={`admin-tab-${s.id}`}
              className={`px-3.5 py-2 text-xs font-semibold rounded-t-lg border-b-2 transition-colors ${CTA.focusRing}
                ${section === s.id
                  ? CTA.tabActive
                  : s.phaseGate
                    ? 'border-transparent opacity-45 cursor-not-allowed'
                    : `border-transparent ${COLOR.textSecondary} hover:bg-[rgba(148,163,184,.06)]`}`}
            >
              {s.label}
              {s.id === 'registros' && queue.data && queue.data.count > 0 && (
                <Badge tone="warn" className="ml-1.5">{queue.data.count}</Badge>
              )}
              {s.id === 'modelos' && models.data && models.data.pending_count > 0 && (
                <Badge tone="warn" className="ml-1.5">{models.data.pending_count}</Badge>
              )}
              {s.id === 'riesgo' && risk.data && risk.data.api_pending.length > 0 && (
                <Badge tone="warn" className="ml-1.5">{risk.data.api_pending.length}</Badge>
              )}
            </button>
          ))}
        </nav>

        {section === 'overview' && (
          <OverviewSection
            onNavigate={setSection}
            queueWaitingMax={queueWaitingMax}
            pendingQueue={queue.data ? { count: queue.data.count, test_hidden: queue.data.test_hidden } : null}
            system={system}
          />
        )}
        {section === 'ingresos' && <RevenueSection />}
        {section === 'registros' && <QueueSection queue={queue} />}
        {section === 'usuarios' && <UsersSection />}
        {section === 'roles' && <RolesSection />}
        {section === 'modelos' && <ModelsSection />}
        {section === 'riesgo' && <RiskSection />}
        {section === 'catalogo-admin' && <CatalogSection />}
        {section === 'sistema' && <SystemSection system={system} />}
        {section === 'auditoria' && <AuditSection />}
      </div>
    </div>
  );
}

export default function AdminPage() {
  // Suspense: useSearchParams requires a boundary during prerender (Next 15).
  return (
    <TerminalShell active="admin">
      <ToastProvider>
        <Suspense fallback={<div className={`min-h-screen ${SURFACE.page}`} />}>
          <AdminConsole />
        </Suspense>
      </ToastProvider>
    </TerminalShell>
  );
}
