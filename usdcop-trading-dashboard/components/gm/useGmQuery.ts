'use client';

/**
 * useGmQuery — data hook for GM views (CTR-FE-BE-001 client side).
 * apiFetch (envelope-aware) + abort-on-unmount + optional polling with
 * stale-while-error (a failed refresh keeps the last data). Its return shape
 * plugs 1:1 into <AsyncBoundary state={…}>.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { apiFetch, ClientApiError, goToLogin } from '@/lib/api/gm-client';
import type { AsyncState } from './AsyncBoundary';

export function useGmQuery<T>(
  path: string | null,
  opts: { refreshMs?: number; onUnauthenticated?: () => void } = {},
): AsyncState<T> & { updatedAt: number | null; stale: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ClientApiError | Error | null>(null);
  const [loading, setLoading] = useState(!!path);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const hasData = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const reload = useCallback(() => {
    if (!path) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    if (!hasData.current) setLoading(true);
    apiFetch<T>(path, { signal: ac.signal, onUnauthenticated: opts.onUnauthenticated ?? goToLogin })
      .then(({ data: d }) => {
        setData(d);
        setUpdatedAt(Date.now());
        setError(null);
        hasData.current = true;
      })
      .catch((e: unknown) => {
        if ((e as Error)?.name === 'AbortError') return;
        setError(e instanceof Error ? e : new Error(String(e)));
      })
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);

  useEffect(() => {
    reload();
    return () => abortRef.current?.abort();
  }, [reload]);

  useEffect(() => {
    if (!opts.refreshMs || !path) return;
    const t = setInterval(reload, opts.refreshMs);
    return () => clearInterval(t);
  }, [reload, opts.refreshMs, path]);

  return { data, error, loading, reload, updatedAt, stale: !!(data && error) };
}
