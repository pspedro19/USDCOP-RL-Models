'use client';

/**
 * useWeeklyAnalysis — Data fetching hooks for the /analysis page.
 * Uses @tanstack/react-query for caching and deduplication.
 */

import { useQuery } from '@tanstack/react-query';
import type {
  AnalysisIndex,
  WeeklyViewData,
  EconomicEvent,
} from '@/lib/contracts/weekly-analysis.contract';
import {
  ANALYSIS_ASSETS,
  DEFAULT_ANALYSIS_ASSET,
  resolveAnalysisAsset,
  type AnalysisAsset,
} from '@/lib/contracts/analysis-assets';

/** Append ?asset=<id> to a URL (only when non-default, keeping legacy URLs clean). */
function withAsset(url: string, asset: string): string {
  const resolved = resolveAnalysisAsset(asset);
  return `${url}?asset=${encodeURIComponent(resolved)}`;
}

// ---------------------------------------------------------------------------
// Available analysis assets (drives the selector dynamically from the SSOT)
// ---------------------------------------------------------------------------

export function useAnalysisAssets() {
  return useQuery<{ assets: AnalysisAsset[]; default_asset: string }>({
    queryKey: ['analysis', 'assets'],
    queryFn: async () => {
      const res = await fetch('/api/analysis/assets');
      if (!res.ok) throw new Error('Failed to fetch analysis assets');
      return res.json();
    },
    // The SSOT rarely changes; seed with the compiled-in list so the selector
    // renders instantly and still stays correct if the endpoint updates.
    initialData: { assets: ANALYSIS_ASSETS, default_asset: DEFAULT_ANALYSIS_ASSET },
    staleTime: 60 * 60 * 1000, // 1 h
  });
}

// ---------------------------------------------------------------------------
// Week index (list of available weeks) — per asset
// ---------------------------------------------------------------------------

export function useAnalysisIndex(asset: string = DEFAULT_ANALYSIS_ASSET) {
  return useQuery<AnalysisIndex>({
    queryKey: ['analysis', 'index', asset],
    queryFn: async () => {
      const res = await fetch(withAsset('/api/analysis/weeks', asset));
      if (!res.ok) throw new Error('Failed to fetch analysis index');
      return res.json();
    },
    staleTime: 5 * 60 * 1000, // 5 min
  });
}

// ---------------------------------------------------------------------------
// Full weekly view data — per asset
// ---------------------------------------------------------------------------

export function useWeeklyView(
  year: number | null,
  week: number | null,
  asset: string = DEFAULT_ANALYSIS_ASSET,
) {
  return useQuery<WeeklyViewData>({
    queryKey: ['analysis', 'week', asset, year, week],
    queryFn: async () => {
      const res = await fetch(withAsset(`/api/analysis/week/${year}/${week}`, asset));
      if (!res.ok) throw new Error(`No data for ${asset} ${year}-W${week}`);
      return res.json();
    },
    enabled: year !== null && week !== null,
    staleTime: 10 * 60 * 1000, // 10 min — analysis data changes infrequently
  });
}

// ---------------------------------------------------------------------------
// Upcoming economic events — per asset
// ---------------------------------------------------------------------------

export function useUpcomingEvents(asset: string = DEFAULT_ANALYSIS_ASSET) {
  return useQuery<{ events: EconomicEvent[]; generated_at: string | null }>({
    queryKey: ['analysis', 'calendar', asset],
    queryFn: async () => {
      const res = await fetch(withAsset('/api/analysis/calendar', asset));
      if (!res.ok) throw new Error('Failed to fetch calendar');
      return res.json();
    },
    staleTime: 30 * 60 * 1000, // 30 min
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get the current ISO year and week. */
export function getCurrentISOWeek(): { year: number; week: number } {
  const now = new Date();
  const jan4 = new Date(now.getFullYear(), 0, 4);
  const dayOfYear = Math.floor(
    (now.getTime() - new Date(now.getFullYear(), 0, 1).getTime()) / 86400000
  ) + 1;
  const weekDay = (now.getDay() + 6) % 7; // Mon=0 ... Sun=6
  const week = Math.ceil((dayOfYear - weekDay + 10) / 7);

  // Handle year boundary (week 1 may belong to previous year)
  if (week < 1) {
    return { year: now.getFullYear() - 1, week: 52 };
  }
  if (week > 52) {
    // Check if it's really week 1 of next year
    const dec31 = new Date(now.getFullYear(), 11, 31);
    const dec31Weekday = (dec31.getDay() + 6) % 7;
    if (dec31Weekday < 3) {
      return { year: now.getFullYear() + 1, week: 1 };
    }
  }
  return { year: jan4.getFullYear(), week };
}
