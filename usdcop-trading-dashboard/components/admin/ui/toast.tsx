'use client';

/**
 * Toast system (CTR-ADMIN-UI-001 §1.4/§3.10): bottom-right, aria-live=polite, max 3
 * stacked, undo variant pauses its timer on hover. The undo variant implements the
 * DEFERRED-COMMIT pattern (§3 decisión): the commit callback fires when the timer
 * expires (or the toast is dismissed); "Deshacer" cancels it entirely.
 */
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { Undo2, X } from 'lucide-react';

import { COLOR, CTA, SURFACE, TYPE, type SemanticTone } from '@/lib/ui/tokens';

export interface ToastSpec {
  id: number;
  message: string;
  tone: SemanticTone;
  testId?: string;
  /** Undo variant: commit() fires after ttlMs unless onUndo cancels it. */
  undo?: { label?: string; ttlMs: number; commit: () => void; onUndo: () => void };
}

interface ToastApi {
  toast: (message: string, tone?: SemanticTone, testId?: string) => void;
  toastUndo: (message: string, opts: { ttlMs?: number; commit: () => void; onUndo: () => void; testId?: string }) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx;
}

const MAX_TOASTS = 3;
const DEFAULT_TTL = 6000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastSpec[]>([]);
  const nextId = useRef(1);

  const remove = useCallback((id: number) => {
    setToasts((ts) => ts.filter((t) => t.id !== id));
  }, []);

  const push = useCallback((spec: Omit<ToastSpec, 'id'>) => {
    const id = nextId.current++;
    setToasts((ts) => [...ts.slice(-(MAX_TOASTS - 1)), { ...spec, id }]);
    return id;
  }, []);

  const toast = useCallback((message: string, tone: SemanticTone = 'info', testId?: string) => {
    const id = push({ message, tone, testId });
    setTimeout(() => remove(id), DEFAULT_TTL);
  }, [push, remove]);

  const toastUndo = useCallback((message: string, opts: { ttlMs?: number; commit: () => void; onUndo: () => void; testId?: string }) => {
    push({ message, tone: 'accent', testId: opts.testId, undo: { ttlMs: opts.ttlMs ?? 5000, commit: opts.commit, onUndo: opts.onUndo } });
  }, [push]);

  return (
    <ToastContext.Provider value={{ toast, toastUndo }}>
      {children}
      <div className="fixed bottom-4 right-4 z-[80] flex flex-col gap-2 w-80" aria-live="polite" role="region" aria-label="notificaciones">
        {toasts.map((t) => <ToastItem key={t.id} spec={t} onDone={() => remove(t.id)} />)}
      </div>
    </ToastContext.Provider>
  );
}

function ToastItem({ spec, onDone }: { spec: ToastSpec; onDone: () => void }) {
  const [paused, setPaused] = useState(false);
  const [remaining, setRemaining] = useState(spec.undo?.ttlMs ?? 0);
  const committed = useRef(false);

  // Undo variant: countdown that pauses on hover (§3.10); commit exactly once.
  useEffect(() => {
    if (!spec.undo) return;
    if (paused) return;
    const started = Date.now();
    const tick = setInterval(() => {
      const left = remaining - (Date.now() - started);
      if (left <= 0) {
        clearInterval(tick);
        if (!committed.current) { committed.current = true; spec.undo!.commit(); }
        onDone();
      } else {
        setRemaining(left);
      }
    }, 100);
    return () => clearInterval(tick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused, spec.undo]);

  const undoNow = () => {
    if (spec.undo && !committed.current) { committed.current = true; spec.undo.onUndo(); }
    onDone();
  };

  const dismiss = () => {
    // Dismissing an undo toast commits immediately (explicit close ≠ undo).
    if (spec.undo && !committed.current) { committed.current = true; spec.undo.commit(); }
    onDone();
  };

  return (
    <div
      data-testid={spec.testId}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      className={`${SURFACE.card} px-3 py-2.5 shadow-lg flex items-center gap-2 motion-safe:animate-in`}
    >
      <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${COLOR[spec.tone].dot}`} aria-hidden />
      <p className={`${TYPE.body} ${COLOR.textPrimary} flex-1 break-words`}>{spec.message}</p>
      {spec.undo && (
        <button onClick={undoNow} className={`${CTA.primary} ${CTA.focusRing} inline-flex items-center gap-1 px-2 py-1 text-xs shrink-0`}>
          <Undo2 className="w-3 h-3" aria-hidden /> {spec.undo.label ?? 'Deshacer'}
          <span className={`${TYPE.mono} opacity-70`}>{Math.ceil(remaining / 1000)}</span>
        </button>
      )}
      <button onClick={dismiss} aria-label="cerrar notificación" className={`${CTA.focusRing} ${COLOR.textSecondary} hover:opacity-80 shrink-0`}>
        <X className="w-3.5 h-3.5" aria-hidden />
      </button>
    </div>
  );
}
