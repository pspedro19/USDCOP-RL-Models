/**
 * GET /api/analysis/weeks
 * Returns the analysis index (list of available weeks).
 * Reads from public/data/analysis/analysis_index.json (file-based).
 */

import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

import type { AnalysisIndex } from '@/lib/contracts/weekly-analysis.contract';

const INDEX_FILE = path.join(process.cwd(), 'public', 'data', 'analysis', 'analysis_index.json');

const DEFAULT_INDEX: AnalysisIndex = { weeks: [] };

export async function GET() {
  try {
    const raw = await fs.readFile(INDEX_FILE, 'utf-8');
    const index: AnalysisIndex = JSON.parse(raw);
    return NextResponse.json(index);
  } catch {
    return NextResponse.json(DEFAULT_INDEX);
  }
}
