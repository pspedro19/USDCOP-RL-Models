'use client';

/**
 * AsyncBoundary — the envelope-driven data states of CTR-FE-BE-001 (prototype
 * lines 130–159): Skeleton (shape of the content) → Empty ("Sin datos", cause +
 * action) → ErrorRetry (icon, message, `UPSTREAM_*` code + HTTP, Reintentar) →
 * content. Every authenticated GM view wraps its body in one of these.
 */
import type { ReactNode } from 'react';
import { CloudOff, Inbox, RefreshCw } from 'lucide-react';

import { ClientApiError } from '@/lib/api/gm-client';
import { GM, GMT } from '@/lib/ui/gm-tokens';

export interface AsyncState<T> {
  data: T | null;
  error: ClientApiError | Error | null;
  loading: boolean;
  reload: () => void;
}

export function GmSkeleton({ label = 'Cargando datos…' }: { label?: string }) {
  return (
    <div className="motion-safe:animate-in motion-safe:fade-in" aria-busy>
      <div className="h-[30px] w-60 max-w-[60%] rounded-[9px] bg-[rgba(148,163,184,.12)] motion-safe:animate-pulse mb-3" />
      <div className="h-[15px] w-[360px] max-w-[74%] rounded-[7px] bg-[rgba(148,163,184,.08)] motion-safe:animate-pulse mb-6" />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5 mb-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 rounded-[14px] bg-[rgba(148,163,184,.07)] border border-[rgba(148,163,184,.10)] motion-safe:animate-pulse" />
        ))}
      </div>
      <div className="h-[300px] rounded-2xl bg-[rgba(148,163,184,.06)] border border-[rgba(148,163,184,.10)] flex items-center justify-center gap-3">
        <span className="w-[22px] h-[22px] rounded-full border-[2.5px] border-[rgba(34,211,238,.25)] border-t-[var(--gm-accent)] motion-safe:animate-spin" aria-hidden />
        <span className={`${GMT.body} font-semibold ${GM.textMuted}`}>{label}</span>
      </div>
    </div>
  );
}

export function GmEmpty({ title = 'Sin datos', body, action }: {
  title?: string; body?: ReactNode; action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center text-center min-h-[380px] p-5 motion-safe:animate-in motion-safe:fade-in">
      <span className="w-16 h-16 rounded-[18px] bg-[rgba(148,163,184,.08)] border border-[rgba(148,163,184,.14)] flex items-center justify-center mb-4">
        <Inbox className={`w-7 h-7 ${GM.textMuted}`} aria-hidden />
      </span>
      <div className={`text-lg font-bold ${GM.text} mb-1.5`}>{title}</div>
      {body && <div className={`${GMT.body} text-[var(--gm-text-muted)] max-w-[340px] leading-relaxed`}>{body}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function GmErrorRetry({ error, onRetry }: { error: Error; onRetry: () => void }) {
  const api = error instanceof ClientApiError ? error : null;
  const code = api ? `${api.code}${api.status ? ` · HTTP ${api.status}` : ''}` : 'CLIENT_ERROR';
  return (
    <div className="flex flex-col items-center justify-center text-center min-h-[380px] p-5 motion-safe:animate-in motion-safe:fade-in" role="alert">
      <span className="w-16 h-16 rounded-[18px] bg-[rgba(251,113,133,.10)] border border-[rgba(251,113,133,.24)] flex items-center justify-center mb-4">
        <CloudOff className={`w-7 h-7 ${GM.neg}`} aria-hidden />
      </span>
      <div className={`text-lg font-bold ${GM.text} mb-1.5`}>No pudimos cargar los datos</div>
      <div className={`${GMT.body} text-[var(--gm-text-muted)] max-w-[380px] leading-relaxed mb-2`}>{error.message}</div>
      <code className={`${GMT.micro} ${GM.neg} font-mono bg-[rgba(251,113,133,.08)] px-2 py-1 rounded-[7px] mb-5`}>{code}</code>
      <button
        onClick={onRetry}
        className={`${GM.ctaPrimary} ${GM.focus} inline-flex items-center gap-2 h-[42px] px-5 text-[13.5px]`}
      >
        <RefreshCw className="w-4 h-4" aria-hidden /> Reintentar
      </button>
    </div>
  );
}

/**
 * Boundary: pick the state for a widget/view.
 *   <AsyncBoundary state={s} empty={(d) => d.items.length === 0}>{(data) => …}</AsyncBoundary>
 */
export function AsyncBoundary<T>({ state, empty, emptyProps, skeleton, children }: {
  state: AsyncState<T>;
  /** Optional predicate: data present but semantically empty. */
  empty?: (data: T) => boolean;
  emptyProps?: { title?: string; body?: ReactNode; action?: ReactNode };
  skeleton?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (state.loading && state.data == null) return <>{skeleton ?? <GmSkeleton />}</>;
  if (state.error && state.data == null) return <GmErrorRetry error={state.error} onRetry={state.reload} />;
  if (state.data == null) return <GmEmpty {...emptyProps} />;
  if (empty?.(state.data)) return <GmEmpty {...emptyProps} />;
  return <>{children(state.data)}</>;
}
