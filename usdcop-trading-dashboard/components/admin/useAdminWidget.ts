'use client';

/**
 * Widget data hooks (CTR-ADMIN-CONSOLE-001 C5 + CTR-ADMIN-UI-001 §3.4).
 *
 * - One hook per widget ⇒ independent loading/error/retry; a failing endpoint never
 *   blanks its neighbours.
 * - Auto-refresh per widget with per-cadence intervals; each card shows "hace Xs".
 * - stale-while-error: a failed refresh KEEPS the last data (card dims), it never
 *   wipes it — `stale` tells the card to render dimmed with the old timestamp.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export interface WidgetState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  /** Last successful fetch (ms epoch) — drives the "hace Xs" meta. */
  updatedAt: number | null;
  /** True when data is present but the LAST refresh failed (§3.4). */
  stale: boolean;
  reload: () => void;
}

export function useAdminWidget<T>(url: string, opts: { refreshMs?: number } = {}): WidgetState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const hasData = useRef(false);

  const load = useCallback(() => {
    if (!hasData.current) setLoading(true);
    fetch(url, { cache: 'no-store' })
      .then(async (r) => {
        if (r.ok) {
          // Endpoints nuevos usan el envelope { ok, data } (lib/api/envelope.ts);
          // los legacy devuelven el body plano — se aceptan ambos.
          const body = await r.json();
          const isEnvelope = body && typeof body === 'object' && 'ok' in body && ('data' in body || 'error' in body);
          if (isEnvelope && body.ok === false) {
            setError(body.error?.message ?? body.error?.code ?? `HTTP ${r.status}`);
            return;
          }
          setData(isEnvelope ? body.data : body);
          setUpdatedAt(Date.now());
          setError(null);
          hasData.current = true;
          return;
        }
        const body = await r.json().catch(() => ({}));
        const msg = body?.error?.message ?? body?.error;
        setError(msg && typeof msg === 'string' ? `${msg} (HTTP ${r.status})` : `HTTP ${r.status}`);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [url]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (!opts.refreshMs) return;
    const t = setInterval(load, opts.refreshMs);
    return () => clearInterval(t);
  }, [load, opts.refreshMs]);

  return { data, error, loading, updatedAt, stale: !!(data && error), reload: load };
}

/** Auto-refresh cadences (§3.4) — one place, not per-component magic numbers. */
export const REFRESH = {
  system: 30_000,
  queue: 60_000,
  kpis: 300_000,
  audit: 120_000,
  users: 120_000,
  models: 120_000,
  risk: 60_000,
  revenue: 300_000,
  catalog: 300_000,
} as const;
