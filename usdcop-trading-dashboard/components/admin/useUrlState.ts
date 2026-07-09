'use client';

/**
 * URL-state (CTR-ADMIN-UI-001 §3.3): filters, active tab and open drawer live in the
 * querystring so refresh and deep-links reproduce the exact view. Uses replaceState
 * (no history spam per keystroke).
 */
import { useCallback } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';

export function useUrlState(): {
  get: (key: string) => string;
  setMany: (patch: Record<string, string | null>) => void;
} {
  const router = useRouter();
  const pathname = usePathname();
  const sp = useSearchParams();

  const get = useCallback((key: string) => sp.get(key) ?? '', [sp]);

  const setMany = useCallback((patch: Record<string, string | null>) => {
    const next = new URLSearchParams(sp.toString());
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === '') next.delete(k);
      else next.set(k, v);
    }
    const qs = next.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [router, pathname, sp]);

  return { get, setMany };
}
