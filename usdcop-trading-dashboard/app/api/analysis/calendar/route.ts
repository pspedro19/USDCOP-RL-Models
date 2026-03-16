/**
 * GET /api/analysis/calendar
 * Returns upcoming economic events for the next 7 days.
 * Reads from public/data/analysis/upcoming_events.json (file-based).
 */

import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

import type { EconomicEvent } from '@/lib/contracts/weekly-analysis.contract';

const EVENTS_FILE = path.join(process.cwd(), 'public', 'data', 'analysis', 'upcoming_events.json');

interface UpcomingEventsResponse {
  events: EconomicEvent[];
  generated_at: string | null;
}

const DEFAULT_RESPONSE: UpcomingEventsResponse = {
  events: [],
  generated_at: null,
};

export async function GET() {
  try {
    const raw = await fs.readFile(EVENTS_FILE, 'utf-8');
    const data: UpcomingEventsResponse = JSON.parse(raw);
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(DEFAULT_RESPONSE);
  }
}
