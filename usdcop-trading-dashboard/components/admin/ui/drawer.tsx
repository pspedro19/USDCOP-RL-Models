'use client';

/**
 * Drawer (CTR-ADMIN-UI-001 §1.4/§3.7): 480px right panel, overlay, Esc closes,
 * focus-trap, aria-modal. The caller reflects the open item in the URL (?user=…).
 */
import { useEffect, useRef, type ReactNode } from 'react';
import { X } from 'lucide-react';

import { COLOR, CTA, SURFACE, TYPE } from '@/lib/ui/tokens';

export function Drawer({ open, title, onClose, children }: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose(); }
      if (e.key === 'Tab' && panelRef.current) {
        // Minimal focus trap: cycle within the panel.
        const focusables = panelRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      prev?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70]" role="dialog" aria-modal="true" aria-label={title}>
      <button aria-label="cerrar panel" onClick={onClose} className={`absolute inset-0 w-full ${SURFACE.overlay}`} tabIndex={-1} />
      <div
        ref={panelRef}
        tabIndex={-1}
        className={`absolute right-0 top-0 h-full w-[480px] max-w-[92vw] ${SURFACE.drawer} shadow-2xl
                    flex flex-col outline-none motion-safe:animate-in motion-safe:slide-in-from-right motion-safe:duration-200`}
      >
        <header className="flex items-center justify-between gap-3 px-5 py-4 border-b border-[rgba(148,163,184,.12)]">
          <h2 className={`${TYPE.pageTitle} ${COLOR.textPrimary} truncate`}>{title}</h2>
          <button onClick={onClose} aria-label="cerrar" className={`${CTA.ghost} ${CTA.focusRing} p-1.5`}>
            <X className="w-4 h-4" aria-hidden />
          </button>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </div>
    </div>
  );
}

/** Definition row for drawers: label left, value right. */
export function DrawerField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2 border-b border-[rgba(148,163,184,.08)]">
      <span className={TYPE.meta}>{label}</span>
      <span className={`${TYPE.body} ${COLOR.textPrimary} text-right break-all`}>{children}</span>
    </div>
  );
}

/** Convenience drawer: field list + optional footer actions + free body. */
export function DrawerHost({ open, title, onClose, fields, footer, children }: {
  open: boolean;
  title: string;
  onClose: () => void;
  fields?: Array<[string, ReactNode]>;
  footer?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <Drawer open={open} title={title} onClose={onClose}>
      {fields?.map(([label, value]) => <DrawerField key={label} label={label}>{value}</DrawerField>)}
      {children}
      {footer && <div className="pt-4">{footer}</div>}
    </Drawer>
  );
}
