/**
 * GET /api/analysis/week/[year]/[week]
 * Returns the full weekly view data for a specific ISO week.
 * Reads from public/data/analysis/weekly_YYYY_WXX.json (file-based).
 */

import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

import type { WeeklyViewData } from '@/lib/contracts/weekly-analysis.contract';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ year: string; week: string }> }
) {
  const { year, week } = await params;

  const yearNum = parseInt(year, 10);
  const weekNum = parseInt(week, 10);

  if (isNaN(yearNum) || isNaN(weekNum) || weekNum < 1 || weekNum > 53) {
    return NextResponse.json(
      { error: 'Invalid year or week parameter' },
      { status: 400 }
    );
  }

  const weekStr = String(weekNum).padStart(2, '0');
  const filename = `weekly_${yearNum}_W${weekStr}.json`;
  const filepath = path.join(process.cwd(), 'public', 'data', 'analysis', filename);

  try {
    const raw = await fs.readFile(filepath, 'utf-8');
    const data: WeeklyViewData = JSON.parse(raw);
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: `No analysis data for ${yearNum}-W${weekStr}` },
      { status: 404 }
    );
  }
}
