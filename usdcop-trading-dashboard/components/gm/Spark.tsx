'use client';

/**
 * Spark — deterministic SVG sparkline (prototype Var B `seedRand/sparkPoints/spark`,
 * lines 1741-1774, ported verbatim so the same seed yields the same stable curve).
 *
 * Two modes:
 *   - `data` (real series, ≥2 finite points): renders the actual values — used by
 *     surfaces that MUST NOT invent data (e.g. the "Mis activos" ticker).
 *   - seed-generated (no `data`): decorative curve from the seeded RNG, identical
 *     to the prototype's `spark(seed, up, w, h)` — for illustrative cards only.
 *
 * Colors by tone via `--gm-*` primitives (zero hex here); purely decorative
 * (`aria-hidden`). Sizes: sm 56×22 · md 84×34 (prototype watch-card / hub sizes).
 */
import { useId } from 'react';

export type SparkTone = 'pos' | 'neg' | 'accent';
export type SparkSize = 'sm' | 'md';

const SIZES: Record<SparkSize, { w: number; h: number }> = {
  sm: { w: 56, h: 22 },
  md: { w: 84, h: 34 },
};

const TONE_COLOR: Record<SparkTone, string> = {
  pos: 'var(--gm-pos)',
  neg: 'var(--gm-neg)',
  accent: 'var(--gm-accent)',
};

/** Prototype line 1742-1746, verbatim: same seed → same pseudo-random stream. */
function seedRand(seed: string): () => number {
  let s = 0;
  for (let i = 0; i < seed.length; i++) s = (s * 31 + seed.charCodeAt(i)) % 2147483647;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

/** Prototype line 1747-1754, verbatim: bounded random walk with up/down drift. */
function sparkPoints(seed: string, up: boolean, n = 24): number[] {
  const r = seedRand(seed);
  const pts: number[] = [];
  let v = 50;
  for (let i = 0; i < n; i++) {
    v += (r() - (up ? 0.42 : 0.58)) * 16;
    v = Math.max(8, Math.min(92, v));
    pts.push(v);
  }
  return pts;
}

export interface SparkProps {
  /** Stable identity for the generated curve (asset_id, symbol…). */
  seed: string;
  /** Line/area color tone. Default 'accent'. */
  tone?: SparkTone;
  /** Drift of the generated curve; defaults to tone !== 'neg' (prototype `up`). */
  up?: boolean;
  /** Real series — when provided (≥2 finite points) it is drawn instead of the
   *  seeded curve. Callers that must never invent data pass this or don't render. */
  data?: readonly number[] | null;
  /** sm = 56×22 (ticker/watch cards) · md = 84×34 (hub cards). Default 'sm'. */
  size?: SparkSize;
  className?: string;
}

export function Spark({ seed, tone = 'accent', up, data, size = 'sm', className }: SparkProps) {
  // useId tokens carry ':'/'«»' — strip to keep the SVG url(#…) reference valid
  // (prototype line 1765 sanitizes its gradient id the same way).
  const gid = `gm-spark-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;
  const { w, h } = SIZES[size];

  const real = (data ?? []).filter((n) => Number.isFinite(n));
  const pts = real.length >= 2 ? real : sparkPoints(seed, up ?? tone !== 'neg');

  // Prototype lines 1758-1763, verbatim normalization + path building.
  const max = Math.max(...pts);
  const min = Math.min(...pts);
  const rng = max - min || 1;
  const stepX = w / (pts.length - 1);
  const coords: Array<[number, number]> = pts.map((p, i) => [
    i * stepX,
    h - ((p - min) / rng) * (h - 4) - 2,
  ]);
  const line = coords.map((c) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(' ');
  const area = `M0,${h} L${coords.map((c) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(' L')} L${w},${h} Z`;
  const col = TONE_COLOR[tone];

  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      aria-hidden="true"
      focusable="false"
      className={`block overflow-visible shrink-0 ${className ?? ''}`}
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={col} stopOpacity={0.28} />
          <stop offset="100%" stopColor={col} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gid})`} />
      <polyline
        points={line}
        fill="none"
        stroke={col}
        strokeWidth={1.6}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
