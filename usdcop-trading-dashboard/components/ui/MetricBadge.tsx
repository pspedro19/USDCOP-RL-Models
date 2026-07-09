'use client';

/**
 * MetricBadge — mandatory phase + provenance badge for every performance figure
 * (ux-navigation §6: "sin badge, no se publica").
 *
 *   <MetricBadge phase="live" provenance={{ strategyId: 'smart_simple_v11',
 *     version: '2.0.0', bundleDate: '2026-07-03' }} />
 *
 * Marketing surfaces lead with LIVE; BACKTEST is visible but never the headline.
 */
import { PHASE_TOKENS, provenanceLabel, type MetricPhase, type MetricProvenance }
  from '@/lib/contracts/ui.contract';

interface MetricBadgeProps {
  phase: MetricPhase;
  provenance?: MetricProvenance;
  className?: string;
}

export function MetricBadge({ phase, provenance, className = '' }: MetricBadgeProps) {
  const t = PHASE_TOKENS[phase];
  return (
    <span
      title={provenance ? provenanceLabel(provenance) : undefined}
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5
        text-[10px] font-bold tracking-wider select-none ${t.className} ${className}`}
      aria-label={`Métrica ${t.label}${provenance ? ` — ${provenanceLabel(provenance)}` : ''}`}
    >
      <span aria-hidden>{t.symbol}</span>
      {t.label}
    </span>
  );
}
