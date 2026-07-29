'use client';

/**
 * useGmQuery — data hook for GM views (CTR-FE-BE-001 client side).
 * apiFetch (envelope-aware) + abort-on-unmount + optional polling with
 * stale-while-error (a failed refresh keeps the last data). Its return shape
 * plugs 1:1 into <AsyncBoundary state={…}>.
 *
 * INVARIANTE (D1, 2026-07-28): `data` PERTENECE a `path`. Cuando `path` cambia, el
 * estado anterior deja de ser una respuesta de esta consulta y se descarta ANTES de
 * pintar. Sin eso, un cambio de URL servía la carga útil de la URL vieja bajo el
 * contrato de la nueva; en /production eso reventaba la página, porque las dos
 * proyecciones del estado de aprobación (`/api/production/status` sanitizada vs
 * `/api/production/approval` íntegra, CXD-057) tienen contratos DISTINTOS y el panel
 * de research leía `gates` de la sanitizada, donde ese campo no existe. El mismo
 * defecto, en silencio, mostraba los KPIs de la estrategia anterior bajo el nombre de
 * la recién seleccionada — números publicados equivocados (quant-constitution §7).
 * `stale-while-error` sigue intacto: es para un REFRESCO fallido del MISMO path.
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

  // Ajuste de estado en RENDER (patrón oficial "resetting state when a prop changes"):
  // se hace aquí y no en un `useEffect` a propósito — un efecto corre DESPUÉS del commit,
  // así que dejaría pintar un frame con el dato de la URL anterior bajo el contrato de la
  // nueva, que es exactamente el fallo que esto cierra. React descarta este render y
  // vuelve a empezar, de modo que ningún render observa `data` de otro `path`.
  const [syncedPath, setSyncedPath] = useState(path);
  if (syncedPath !== path) {
    setSyncedPath(path);
    setData(null);
    setError(null);
    setUpdatedAt(null);
    hasData.current = false;
    setLoading(!!path);
  }

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
