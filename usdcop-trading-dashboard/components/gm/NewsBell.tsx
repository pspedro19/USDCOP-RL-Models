'use client';

/**
 * NewsBell — global top-bar news dropdown (GM chrome).
 *
 * Cross-asset latest headlines from GET /api/analysis/news-feed (real
 * news_intelligence clusters, CTR-NEWS-ENRICH-001). Shows a bell with a count
 * badge; the panel lists the most-recent headlines with asset badge, source,
 * date and a sentiment dot. Data is NEVER invented — the bell hides itself when
 * the feed is empty or errors (graceful degradation), consistent with the ticker.
 */
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Newspaper } from 'lucide-react';

import type { NewsFeedItem, NewsFeedResponse } from '@/lib/contracts/news-feed.contract';
import { useGmLang } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';
import { useGmQuery } from './useGmQuery';

function toneOf(tone: number): 'pos' | 'neg' | 'neutral' {
  if (tone > 0.15) return 'pos';
  if (tone < -0.15) return 'neg';
  return 'neutral';
}

const DOT: Record<string, string> = {
  pos: 'bg-[var(--gm-pos)]',
  neg: 'bg-[var(--gm-neg)]',
  neutral: 'bg-[var(--gm-accent)]',
};

export function NewsBell() {
  const lang = useGmLang();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Poll every 5 min; stale-while-error keeps the last feed on a failed refresh.
  const { data } = useGmQuery<NewsFeedResponse>('/api/analysis/news-feed?limit=14', {
    refreshMs: 300_000,
    onUnauthenticated: () => {},
  });

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onEsc);
    };
  }, [open]);

  const items: NewsFeedItem[] = data?.items ?? [];
  // Graceful: no news → render nothing (never a broken/empty bell).
  if (items.length === 0) return null;

  const t = (es: string, en: string) => (lang === 'es' ? es : en);
  const fmtDate = (d: string) => {
    if (!d) return '';
    try {
      return new Intl.DateTimeFormat(lang === 'es' ? 'es-CO' : 'en-US', {
        month: 'short', day: 'numeric',
      }).format(new Date(d));
    } catch {
      return d;
    }
  };

  return (
    <div ref={wrapRef} className="relative shrink-0">
      <button
        onClick={() => setOpen((v) => !v)}
        title={t('Noticias', 'News')}
        aria-label={t('Abrir noticias', 'Open news')}
        aria-expanded={open}
        data-testid="topbar-news"
        className={`${GM.ctaGhost} ${GM.focus} relative w-11 h-11 flex items-center justify-center`}
      >
        <Newspaper className="w-4.5 h-4.5" aria-hidden />
        <span
          className={`absolute -top-1.5 -right-1.5 min-w-[18px] h-[18px] px-1 rounded-full
            ${GM.ctaGradient} flex items-center justify-center text-[10px] font-extrabold font-mono`}
        >
          {items.length}
        </span>
      </button>

      {open && (
        <div
          role="menu"
          className={`${GM.panel} absolute right-0 mt-2 w-[380px] max-w-[92vw] z-[var(--z-overlay)]
            max-h-[70vh] overflow-y-auto shadow-2xl`}
        >
          <div className="sticky top-0 flex items-center justify-between px-4 py-3 border-b border-[var(--gm-border)]"
               style={{ background: 'var(--gm-surface)' }}>
            <span className={`${GMT.label} ${GM.textStrong} uppercase tracking-wide`}>
              {t('Noticias del mercado', 'Market news')}
            </span>
            {data?.by_asset && data.by_asset.length > 0 && (
              <div className="flex items-center gap-2">
                {data.by_asset.map((a) => {
                  const tn = a.avg_sentiment == null ? 'neutral' : toneOf(a.avg_sentiment);
                  return (
                    <span key={a.asset_id} className={`flex items-center gap-1 ${GMT.micro} ${GM.textMuted}`} title={`${a.symbol} · ${a.week_label}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${DOT[tn]}`} aria-hidden />
                      {a.symbol}
                    </span>
                  );
                })}
              </div>
            )}
          </div>

          <ul className="divide-y divide-[var(--gm-border)]">
            {items.map((it, i) => {
              const tn = toneOf(it.tone);
              const Row = (
                <div className="flex gap-2.5 px-4 py-2.5 hover:bg-[var(--gm-surface-2,rgba(255,255,255,0.03))]">
                  <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${DOT[tn]}`} aria-hidden />
                  <div className="min-w-0 flex-1">
                    <p className={`${GMT.meta} ${GM.textStrong} leading-snug line-clamp-2`}>{it.title}</p>
                    <div className={`mt-1 flex items-center gap-1.5 ${GMT.micro} ${GM.textMuted} ${GMT.mono}`}>
                      <span className={`px-1.5 py-0.5 rounded bg-[var(--gm-border)] ${GM.textSec}`}>{it.symbol}</span>
                      <span className="truncate">{it.source}</span>
                      {it.date && <span className="shrink-0">· {fmtDate(it.date)}</span>}
                    </div>
                  </div>
                </div>
              );
              return (
                <li key={`${it.asset_id}-${i}`} role="menuitem">
                  {it.url ? (
                    <a href={it.url} target="_blank" rel="noopener noreferrer" className={GM.focus}>{Row}</a>
                  ) : (
                    Row
                  )}
                </li>
              );
            })}
          </ul>

          <button
            onClick={() => { setOpen(false); router.push('/analysis'); }}
            className={`${GM.focus} sticky bottom-0 w-full px-4 py-2.5 text-left ${GMT.meta} font-semibold
              text-[var(--gm-accent)] border-t border-[var(--gm-border)]`}
            style={{ background: 'var(--gm-surface)' }}
          >
            {t('Ver análisis completo →', 'View full analysis →')}
          </button>
        </div>
      )}
    </div>
  );
}
