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

// ---------------------------------------------------------------------------
// Week index (list of available weeks)
// ---------------------------------------------------------------------------

export function useAnalysisIndex() {
  return useQuery<AnalysisIndex>({
    queryKey: ['analysis', 'index'],
    queryFn: async () => {
      const res = await fetch('/api/analysis/weeks');
      if (!res.ok) throw new Error('Failed to fetch analysis index');
      return res.json();
    },
    staleTime: 5 * 60 * 1000, // 5 min
  });
}

// ---------------------------------------------------------------------------
// Full weekly view data
// ---------------------------------------------------------------------------

export function useWeeklyView(year: number | null, week: number | null) {
  return useQuery<WeeklyViewData>({
    queryKey: ['analysis', 'week', year, week],
    queryFn: async () => {
      const res = await fetch(`/api/analysis/week/${year}/${week}`);
      if (!res.ok) throw new Error(`No data for ${year}-W${week}`);
      return res.json();
    },
    enabled: year !== null && week !== null,
    staleTime: 10 * 60 * 1000, // 10 min — analysis data changes infrequently
  });
}

// ---------------------------------------------------------------------------
// Upcoming economic events
// ---------------------------------------------------------------------------

export function useUpcomingEvents() {
  return useQuery<{ events: EconomicEvent[]; generated_at: string | null }>({
    queryKey: ['analysis', 'calendar'],
    queryFn: async () => {
      const res = await fetch('/api/analysis/calendar');
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
