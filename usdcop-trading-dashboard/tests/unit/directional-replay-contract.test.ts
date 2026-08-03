import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { DirectionalReplayIndexSchema } from '@/lib/contracts/forecasting.contract';


describe('USD/COP directional replay publication contract', () => {
  const source = path.join(
    process.cwd(),
    'public',
    'forecasting',
    'usdcop',
    'directional_replay_index.json',
  );

  it('parses every 2025-2026 week with all seven horizons', () => {
    const parsed = DirectionalReplayIndexSchema.parse(
      JSON.parse(fs.readFileSync(source, 'utf8')),
    );

    const [latestYear, latestWeek] = parsed.latest_week
      .replace('W', '')
      .split('-')
      .map(Number);
    const isoWeeks = (year: number) => {
      const date = new Date(Date.UTC(year, 11, 28));
      const day = date.getUTCDay() || 7;
      date.setUTCDate(date.getUTCDate() + 4 - day);
      const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
      return Math.ceil((((date.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
    };
    let expectedWeeks = latestWeek;
    for (let year = 2025; year < latestYear; year += 1) expectedWeeks += isoWeeks(year);
    expect(parsed.weeks).toHaveLength(expectedWeeks);
    expect(parsed.weeks[0].iso_week).toBe('2025-W01');
    expect(parsed.weeks.at(-1)?.iso_week).toBe(parsed.latest_week);
    expect(parsed.weeks.every((week) => week.horizons.length === 7)).toBe(true);
    expect(parsed.weeks.every((week) => week.horizons.every((horizon) => (
      horizon.forecast_price > 0
      && horizon.forecast_interval_lower <= horizon.forecast_price
      && horizon.forecast_price <= horizon.forecast_interval_upper
      && horizon.direction_price_agree
        === (horizon.prediction === horizon.point_forecast_direction)
    )))).toBe(true);
    expect(parsed.weeks.every((week) => week.decision.signal_authorized === false)).toBe(true);
  });
});
